"""episodes 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import re
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import Integer, String, and_, bindparam, case, cast, func, or_, select, text, true
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import (
    BbotListenerCacheEntry,
    ConversationEpisode,
    DevSession,
    DevTask,
    DevTaskArtifact,
    EpisodeMessage,
    Group,
    GroupPersonaState,
    Job,
    MemoryBackfillRun,
    MemoryItem,
    MemoryItemSemanticVector,
    Message,
    PersonaStyleExample,
    PersonaStyleSyncState,
    RetrievalDocument,
    RetrievalDocumentMessage,
    RetrievalIndexState,
    Summary,
    UsageRecord,
    User,
)
from app.providers.embeddings import hashed_text_embedding
from app.storage.db import validate_retrieval_vector_table_name
from app.core.time_utils import shanghai_naive, shanghai_now_naive

from app.storage.repositories._common import _delete_active_retrieval_vectors, _normalize_utc_sqlite_timestamp

class EpisodeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_episode(
        self,
        *,
        group_id: int,
        start_message_id: int,
        started_at: datetime,
        segmentation_version: str,
        status: str = "open",
        boundary_reason: str = "",
    ) -> ConversationEpisode:
        episode = ConversationEpisode(
            group_id=group_id,
            segmentation_version=segmentation_version,
            status=status,
            is_current=True,
            start_message_id=start_message_id,
            end_message_id=None,
            started_at=_normalize_utc_sqlite_timestamp(started_at),
            ended_at=None,
            boundary_reason=boundary_reason,
            message_count=0,
            token_count=0,
        )
        self.session.add(episode)
        return episode

    def get_open_episode(self, *, group_id: int) -> ConversationEpisode | None:
        return self.session.scalars(
            select(ConversationEpisode)
            .where(
                ConversationEpisode.group_id == group_id,
                ConversationEpisode.status == "open",
                ConversationEpisode.is_current.is_(True),
            )
            .order_by(ConversationEpisode.id.desc())
            .limit(1)
        ).first()

    def get_episode(self, episode_id: int) -> ConversationEpisode | None:
        return self.session.get(ConversationEpisode, episode_id)

    def list_unassigned_messages(
        self,
        *,
        group_id: int | None = None,
        after_message_id: int | None = None,
        watermark_message_id: int | None = None,
        limit: int = 500,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .outerjoin(
                EpisodeMessage,
                (EpisodeMessage.message_id == Message.id)
                & (EpisodeMessage.group_id == Message.group_id),
            )
            .where(
                Message.group_id.is_not(None),
                EpisodeMessage.message_id.is_(None),
                text(
                    "(json_extract(messages.raw_json, '$.delivery_state') IS NULL "
                    "OR json_extract(messages.raw_json, '$.delivery_state') "
                    "NOT IN ('reserved', 'uncertain'))"
                ),
            )
        )
        if group_id is not None:
            stmt = stmt.where(Message.group_id == int(group_id))
        if after_message_id is not None:
            stmt = stmt.where(Message.id > int(after_message_id))
        if watermark_message_id is not None:
            stmt = stmt.where(Message.id <= int(watermark_message_id))
        return list(
            self.session.scalars(
                stmt.order_by(
                    Message.group_id.asc(),
                    Message.timestamp.asc(),
                    Message.id.asc(),
                ).limit(max(1, int(limit)))
            )
        )

    def list_idle_open_episodes(
        self,
        *,
        idle_before: datetime,
        group_id: int | None = None,
        limit: int = 100,
    ) -> list[ConversationEpisode]:
        normalized_idle_before = _normalize_utc_sqlite_timestamp(idle_before)
        last_message_at = (
            select(func.max(Message.timestamp))
            .join(
                EpisodeMessage,
                (EpisodeMessage.message_id == Message.id)
                & (EpisodeMessage.group_id == Message.group_id),
            )
            .where(
                EpisodeMessage.episode_id == ConversationEpisode.id,
                EpisodeMessage.group_id == ConversationEpisode.group_id,
            )
            .correlate(ConversationEpisode)
            .scalar_subquery()
        )
        stmt = select(ConversationEpisode).where(
            ConversationEpisode.status == "open",
            ConversationEpisode.is_current.is_(True),
            func.coalesce(last_message_at, ConversationEpisode.started_at)
            <= normalized_idle_before,
        )
        if group_id is not None:
            stmt = stmt.where(ConversationEpisode.group_id == int(group_id))
        return list(
            self.session.scalars(
                stmt.order_by(
                    ConversationEpisode.group_id.asc(),
                    ConversationEpisode.started_at.asc(),
                    ConversationEpisode.id.asc(),
                ).limit(max(1, int(limit)))
            )
        )

    def list_processable_episodes(
        self,
        *,
        group_id: int | None = None,
        statuses: tuple[str, ...] = ("closed", "failed"),
        compaction_version: str | None = None,
        limit: int = 100,
    ) -> list[ConversationEpisode]:
        normalized_statuses = tuple(dict.fromkeys(str(value) for value in statuses))
        if not normalized_statuses:
            return []
        stmt = select(ConversationEpisode).where(
            ConversationEpisode.is_current.is_(True),
            ConversationEpisode.status.in_(normalized_statuses),
        )
        if group_id is not None:
            stmt = stmt.where(ConversationEpisode.group_id == int(group_id))
        if compaction_version is not None:
            stmt = stmt.where(
                ConversationEpisode.compaction_version == str(compaction_version)
            )
        return list(
            self.session.scalars(
                stmt.order_by(
                    ConversationEpisode.group_id.asc(),
                    ConversationEpisode.ended_at.asc(),
                    ConversationEpisode.id.asc(),
                ).limit(max(1, int(limit)))
            )
        )

    def compare_and_set_status(
        self,
        *,
        episode_id: int,
        group_id: int,
        expected_statuses: tuple[str, ...],
        new_status: str,
        compaction_version: str | None = None,
    ) -> bool:
        normalized_statuses = tuple(
            dict.fromkeys(str(value) for value in expected_statuses)
        )
        if not normalized_statuses:
            return False
        values: dict[str, Any] = {
            "new_status": str(new_status),
            "updated_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            "episode_id": int(episode_id),
            "group_id": int(group_id),
        }
        assignments = "status = :new_status, updated_at = :updated_at"
        if compaction_version is not None:
            assignments += ", compaction_version = :compaction_version"
            values["compaction_version"] = str(compaction_version)
        status_params: list[str] = []
        for index, status in enumerate(normalized_statuses):
            key = f"expected_status_{index}"
            values[key] = status
            status_params.append(f":{key}")
        result = self.session.execute(
            text(
                f"UPDATE conversation_episodes SET {assignments} "
                "WHERE id = :episode_id AND group_id = :group_id "
                f"AND status IN ({','.join(status_params)}) AND is_current = 1"
            ),
            values,
        )
        self.session.expire_all()
        return int(result.rowcount or 0) == 1

    def find_episode_for_late_arrival(
        self,
        *,
        group_id: int,
        timestamp: datetime,
        segmentation_version: str | None = None,
    ) -> ConversationEpisode | None:
        resolved_timestamp = _normalize_utc_sqlite_timestamp(timestamp)
        stmt = select(ConversationEpisode).where(
            ConversationEpisode.group_id == int(group_id),
            ConversationEpisode.is_current.is_(True),
            ConversationEpisode.started_at <= resolved_timestamp,
            or_(
                ConversationEpisode.ended_at.is_(None),
                ConversationEpisode.ended_at >= resolved_timestamp,
            ),
        )
        if segmentation_version is not None:
            normalized_generation = str(segmentation_version)
            stmt = stmt.where(
                or_(
                    ConversationEpisode.segmentation_version
                    == normalized_generation,
                    ConversationEpisode.segmentation_version.like(
                        f"{normalized_generation}:late:%"
                    ),
                )
            )
        containing = self.session.scalars(
            stmt.order_by(
                ConversationEpisode.started_at.desc(),
                ConversationEpisode.id.desc(),
            ).limit(1)
        ).first()
        if containing is not None:
            return containing
        future_stmt = select(ConversationEpisode).where(
            ConversationEpisode.group_id == int(group_id),
            ConversationEpisode.is_current.is_(True),
            ConversationEpisode.started_at > resolved_timestamp,
        )
        if segmentation_version is not None:
            normalized_generation = str(segmentation_version)
            future_stmt = future_stmt.where(
                or_(
                    ConversationEpisode.segmentation_version
                    == normalized_generation,
                    ConversationEpisode.segmentation_version.like(
                        f"{normalized_generation}:late:%"
                    ),
                )
            )
        return self.session.scalars(
            future_stmt.order_by(
                ConversationEpisode.started_at.asc(),
                ConversationEpisode.id.asc(),
            ).limit(1)
        ).first()

    def supersede_episode(
        self,
        *,
        episode_id: int,
        group_id: int,
        expected_current: bool = True,
    ) -> bool:
        result = self.session.execute(
            text(
                "UPDATE conversation_episodes SET status = 'superseded', "
                "is_current = 0, updated_at = :updated_at "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND is_current = :expected_current"
            ),
            {
                "episode_id": int(episode_id),
                "group_id": int(group_id),
                "expected_current": bool(expected_current),
                "updated_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            },
        )
        self.session.expire_all()
        return int(result.rowcount or 0) == 1

    def prepare_late_arrival_resegment(
        self,
        *,
        group_id: int,
        message_id: int,
        timestamp: datetime,
        segmentation_version: str | None = None,
        compaction_version: str | None = None,
    ) -> list[int]:
        """Atomically supersede the affected suffix and release its memberships."""

        resolved_segmentation = str(segmentation_version or "")
        if not resolved_segmentation:
            raise ValueError("late-arrival preparation requires a segmentation generation")
        preparation_id = self.session.execute(
            text(
                "INSERT INTO memory_late_arrival_preparations ("
                "group_id, message_id, segmentation_generation, "
                "compaction_generation, created_at"
                ") VALUES ("
                ":group_id, :message_id, :segmentation_generation, "
                ":compaction_generation, :created_at"
                ") ON CONFLICT(group_id, message_id, segmentation_generation) "
                "DO NOTHING RETURNING id"
            ),
            {
                "group_id": int(group_id),
                "message_id": int(message_id),
                "segmentation_generation": resolved_segmentation,
                "compaction_generation": str(compaction_version or ""),
                "created_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            },
        ).scalar_one_or_none()
        if preparation_id is None:
            return []
        affected = self.find_episode_for_late_arrival(
            group_id=group_id,
            timestamp=timestamp,
            segmentation_version=segmentation_version,
        )
        if affected is None:
            return [int(message_id)]
        suffix_ids = list(
            self.session.scalars(
                select(ConversationEpisode.id)
                .where(
                    ConversationEpisode.group_id == int(group_id),
                    ConversationEpisode.is_current.is_(True),
                    ConversationEpisode.started_at >= affected.started_at,
                )
                .order_by(
                    ConversationEpisode.started_at.asc(),
                    ConversationEpisode.id.asc(),
                )
            )
        )
        if not suffix_ids:
            return [int(message_id)]
        replay_message_ids = list(
            self.session.scalars(
                select(Message.id)
                .join(
                    EpisodeMessage,
                    (EpisodeMessage.message_id == Message.id)
                    & (EpisodeMessage.group_id == Message.group_id),
                )
                .where(
                    EpisodeMessage.group_id == int(group_id),
                    EpisodeMessage.episode_id.in_(suffix_ids),
                    Message.group_id == int(group_id),
                )
                .order_by(Message.timestamp.asc(), Message.id.asc())
            )
        )
        values: dict[str, Any] = {
            "updated_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            "group_id": int(group_id),
            "episode_ids": tuple(int(value) for value in suffix_ids),
        }
        compaction_assignment = ""
        if compaction_version is not None:
            compaction_assignment = ", compaction_version = :compaction_version"
            values["compaction_version"] = str(compaction_version)
        superseded = self.session.execute(
            text(
                "UPDATE conversation_episodes SET status = 'superseded', "
                f"is_current = 0, updated_at = :updated_at{compaction_assignment} "
                "WHERE group_id = :group_id AND is_current = 1 "
                "AND id IN :episode_ids"
            ).bindparams(bindparam("episode_ids", expanding=True)),
            values,
        )
        if int(superseded.rowcount or 0) != len(suffix_ids):
            raise RuntimeError("late-arrival episode generation CAS failed")
        document_ids = list(
            self.session.scalars(
                select(RetrievalDocument.id).where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.episode_id.in_(suffix_ids),
                    RetrievalDocument.status == "active",
                )
            )
        )
        if document_ids:
            self.session.execute(
                text(
                    "UPDATE retrieval_documents SET status = 'inactive', "
                    "embedding_status = 'stale', updated_at = :updated_at "
                    "WHERE group_id = :group_id AND id IN :document_ids"
                ).bindparams(bindparam("document_ids", expanding=True)),
                {
                    "group_id": int(group_id),
                    "document_ids": tuple(int(value) for value in document_ids),
                    "updated_at": values["updated_at"],
                },
            )
            try:
                self.session.execute(
                    text(
                        "DELETE FROM retrieval_documents_fts "
                        "WHERE document_id IN :document_ids"
                    ).bindparams(bindparam("document_ids", expanding=True)),
                    {
                        "document_ids": tuple(
                            str(value) for value in document_ids
                        )
                    },
                )
            except SQLAlchemyError:
                pass
            _delete_active_retrieval_vectors(
                self.session,
                document_ids=[int(value) for value in document_ids],
            )
        self.session.execute(
            text(
                "DELETE FROM episode_messages WHERE group_id = :group_id "
                "AND episode_id IN :episode_ids"
            ).bindparams(bindparam("episode_ids", expanding=True)),
            {
                "group_id": int(group_id),
                "episode_ids": tuple(int(value) for value in suffix_ids),
            },
        )
        self.session.expire_all()
        return [int(value) for value in replay_message_ids]

    def add_message(
        self,
        *,
        episode_id: int,
        group_id: int,
        message_id: int,
        ordinal: int,
        estimated_tokens: int,
    ) -> EpisodeMessage:
        membership = EpisodeMessage(
            episode_id=episode_id,
            group_id=group_id,
            message_id=message_id,
            ordinal=ordinal,
        )
        self.session.add(membership)
        episode = self.session.get(ConversationEpisode, episode_id)
        if episode is not None:
            episode.message_count = max(int(episode.message_count or 0), ordinal + 1)
            episode.token_count = int(episode.token_count or 0) + max(0, int(estimated_tokens))
            episode.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            self.session.add(episode)
        return membership

    def add_message_if_current(
        self,
        *,
        episode_id: int,
        group_id: int,
        message_id: int,
        estimated_tokens: int,
    ) -> bool:
        """Atomically append only while the target remains the current open episode."""
        added_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        inserted = self.session.execute(
            text(
                "INSERT INTO episode_messages ("
                "episode_id, message_id, group_id, ordinal, added_at"
                ") SELECT id, :message_id, group_id, message_count, :added_at "
                "FROM conversation_episodes "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND status = 'open' AND is_current = 1 "
                "ON CONFLICT DO NOTHING"
            ),
            {
                "episode_id": int(episode_id),
                "group_id": int(group_id),
                "message_id": int(message_id),
                "added_at": added_at,
            },
        )
        if int(inserted.rowcount or 0) != 1:
            return False
        updated = self.session.execute(
            text(
                "UPDATE conversation_episodes SET "
                "message_count = message_count + 1, "
                "token_count = token_count + :estimated_tokens, "
                "updated_at = :updated_at "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND status = 'open' AND is_current = 1"
            ),
            {
                "episode_id": int(episode_id),
                "group_id": int(group_id),
                "estimated_tokens": max(0, int(estimated_tokens)),
                "updated_at": added_at,
            },
        )
        if int(updated.rowcount or 0) != 1:
            raise RuntimeError("current episode disappeared after guarded append")
        self.session.expire_all()
        return True

    def close_episode(
        self,
        *,
        episode_id: int,
        ended_at: datetime,
        end_message_id: int,
        boundary_reason: str,
        content_hash: str,
    ) -> ConversationEpisode | None:
        episode = self.session.get(ConversationEpisode, episode_id)
        if (
            episode is None
            or episode.status != "open"
            or not bool(episode.is_current)
        ):
            return None
        closed_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        normalized_ended_at = _normalize_utc_sqlite_timestamp(ended_at)
        segmentation_version = str(episode.segmentation_version or "")
        duplicate_id = self.session.scalar(
            select(ConversationEpisode.id).where(
                ConversationEpisode.id != int(episode_id),
                ConversationEpisode.group_id == int(episode.group_id),
                ConversationEpisode.segmentation_version
                == segmentation_version,
                ConversationEpisode.start_message_id
                == int(episode.start_message_id),
                ConversationEpisode.end_message_id == int(end_message_id),
            )
        )
        if duplicate_id is not None:
            # A late-arrival replay can deterministically reproduce an older
            # superseded boundary. Keep both audit rows, but identify the
            # replay as a derived segmentation generation.
            segmentation_version = (
                f"{segmentation_version}:late:{int(episode_id)}"
            )
        result = self.session.execute(
            text(
                "UPDATE conversation_episodes SET "
                "status = 'closed', segmentation_version = :segmentation_version, "
                "ended_at = :ended_at, end_message_id = :end_message_id, "
                "boundary_reason = :boundary_reason, content_hash = :content_hash, "
                "closed_at = :closed_at, updated_at = :updated_at "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND status = 'open' AND is_current = 1"
            ),
            {
                "segmentation_version": segmentation_version,
                "ended_at": normalized_ended_at,
                "end_message_id": int(end_message_id),
                "boundary_reason": str(boundary_reason),
                "content_hash": str(content_hash),
                "closed_at": closed_at,
                "updated_at": closed_at,
                "episode_id": int(episode_id),
                "group_id": int(episode.group_id),
            },
        )
        self.session.expire_all()
        if int(result.rowcount or 0) != 1:
            return None
        return self.session.get(ConversationEpisode, episode_id)

    def list_episode_messages(self, *, episode_id: int, group_id: int) -> list[Message]:
        return list(
            self.session.scalars(
                select(Message)
                .join(
                    EpisodeMessage,
                    (EpisodeMessage.message_id == Message.id)
                    & (EpisodeMessage.group_id == Message.group_id),
                )
                .where(
                    EpisodeMessage.episode_id == episode_id,
                    EpisodeMessage.group_id == group_id,
                    Message.group_id == group_id,
                )
                .order_by(EpisodeMessage.ordinal.asc(), Message.id.asc())
            )
        )
