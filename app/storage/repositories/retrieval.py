"""retrieval 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

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

from app.storage.repositories.messages import MessageRepository
from app.storage.repositories._common import RetrievalDocumentHit, RetrievalEmbeddingCoverage, _INELIGIBLE_DELIVERY_STATES, _deactivate_memory_retrieval_documents, _delete_active_retrieval_vectors, _fts_search_terms, _initial_vector_fetch_limit, _mentioned_user_ids, _next_vector_fetch_limit, _normalize_optional_integer_filter, _normalize_optional_string_filter, _normalize_utc_sqlite_timestamp, _retrieval_mention_document_ids, _retrieval_source_prefilters, _vector_fetch_ceiling

class RetrievalDocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_document(
        self,
        *,
        scope_type: str,
        scope_id: str,
        group_id: int,
        episode_id: int | None,
        document_kind: str,
        source_table: str,
        source_id: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        metadata_json: dict[str, Any],
        content_hash: str,
        source_message_ids: list[int],
        status: str = "active",
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_version: str = "",
        embedding_dimensions: int | None = None,
        embedding_generation: int | None = None,
        embedding_eligible: bool = False,
        embedding_status: str = "disabled",
    ) -> RetrievalDocument:
        memory_source_id: int | None = None
        if document_kind == "memory" and source_table == "memory_items":
            try:
                memory_source_id = int(source_id)
            except ValueError:
                raise ValueError("memory retrieval document source_id must be numeric") from None
            source_memory = self.session.get(MemoryItem, memory_source_id)
            if (
                source_memory is None
                or scope_type != "group"
                or source_memory.scope_type != "group"
                or source_memory.scope_id != str(scope_id)
                or str(scope_id) != str(int(group_id))
            ):
                raise ValueError("memory retrieval document source scope mismatch")
        document = self.session.scalars(
            select(RetrievalDocument).where(
                RetrievalDocument.scope_type == scope_type,
                RetrievalDocument.scope_id == scope_id,
                RetrievalDocument.group_id == group_id,
                RetrievalDocument.document_kind == document_kind,
                RetrievalDocument.source_table == source_table,
                RetrievalDocument.source_id == source_id,
                RetrievalDocument.content_hash == content_hash,
            )
        ).first()
        if document is None:
            conflicting = self.session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.scope_type == scope_type,
                    RetrievalDocument.scope_id == scope_id,
                    RetrievalDocument.document_kind == document_kind,
                    RetrievalDocument.source_table == source_table,
                    RetrievalDocument.source_id == source_id,
                    RetrievalDocument.content_hash == content_hash,
                )
            ).first()
            if conflicting is not None:
                raise ValueError("retrieval document identity is bound to another group")
            document = RetrievalDocument(
                scope_type=scope_type,
                scope_id=scope_id,
                group_id=group_id,
                episode_id=episode_id,
                document_kind=document_kind,
                source_table=source_table,
                source_id=source_id,
                start_at=_normalize_utc_sqlite_timestamp(start_at),
                end_at=_normalize_utc_sqlite_timestamp(end_at),
                content=content,
                metadata_json=metadata_json,
                content_hash=content_hash,
                status=status,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
                embedding_dimensions=embedding_dimensions,
                embedding_generation=embedding_generation,
                embedding_eligible=bool(embedding_eligible),
                embedding_status=embedding_status,
            )
            self.session.add(document)
            self.session.flush()
        else:
            if document.group_id != group_id or document.episode_id != episode_id:
                raise ValueError("retrieval document identity cannot move across group or episode")
            document.status = status
            document.metadata_json = metadata_json
            document.embedding_provider = embedding_provider
            document.embedding_model = embedding_model
            document.embedding_version = embedding_version
            document.embedding_dimensions = embedding_dimensions
            document.embedding_generation = embedding_generation
            document.embedding_eligible = bool(embedding_eligible)
            document.embedding_status = embedding_status
            document.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            self.session.add(document)

        if document.document_kind == "memory" and document.source_table == "memory_items":
            _deactivate_memory_retrieval_documents(
                self.session,
                memory_id=int(memory_source_id),
                keep_document_id=document.id,
            )

        existing_message_ids = set(
            self.session.scalars(
                select(RetrievalDocumentMessage.message_id).where(
                    RetrievalDocumentMessage.document_id == document.id
                )
            )
        )
        for ordinal, message_id in enumerate(dict.fromkeys(source_message_ids)):
            if message_id in existing_message_ids:
                continue
            self.session.add(
                RetrievalDocumentMessage(
                    document_id=document.id,
                    group_id=group_id,
                    message_id=message_id,
                    ordinal=ordinal,
                    role="source",
                )
            )
        self._sync_fts(document)
        return document

    def project_raw_message_v3(
        self,
        *,
        group_id: int,
        message_id: int,
        embedding_generation: int | None = None,
    ) -> RetrievalDocument | None:
        """Idempotently project one canonical message into immediate V3 retrieval.

        The caller owns the transaction, so the canonical document, provenance,
        and best-effort FTS row become visible together. Unsafe outbound
        placeholders revoke any older projection without mutating the message.
        """
        message = self.session.scalars(
            select(Message).where(
                Message.id == int(message_id),
                Message.group_id == int(group_id),
            )
        ).first()
        if message is None:
            raise ValueError("raw message projection source is missing or out of scope")

        existing = list(
            self.session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.document_kind == "raw_message_v3",
                    RetrievalDocument.source_table == "messages",
                    RetrievalDocument.source_id == str(int(message_id)),
                )
            )
        )
        content = str(message.plain_text or "").strip()
        delivery_state = (
            str(message.raw_json.get("delivery_state") or "")
            .strip()
            .casefold()
            if isinstance(message.raw_json, dict)
            else ""
        )
        unsafe = delivery_state in _INELIGIBLE_DELIVERY_STATES
        if unsafe or not content:
            self._deactivate_documents(existing)
            return None

        generation = (
            int(embedding_generation)
            if embedding_generation is not None
            else None
        )
        hash_input = json.dumps(
            {
                "kind": "raw_message_v3",
                "message_id": int(message.id),
                "group_id": int(message.group_id),
                "platform_msg_id": str(message.platform_msg_id),
                "user_id": int(message.user_id),
                "timestamp": _normalize_utc_sqlite_timestamp(message.timestamp).isoformat(),
                "content": content,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        matching = next(
            (row for row in existing if row.content_hash == content_hash),
            None,
        )
        if (
            generation is not None
            and matching is not None
            and matching.embedding_generation is not None
            and int(matching.embedding_generation) > generation
        ):
            claimed_state = self.session.execute(
                text(
                    "SELECT status, is_active, document_family "
                    "FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND generation = :generation"
                ),
                {"generation": int(matching.embedding_generation)},
            ).one_or_none()
            requested_state = self.session.execute(
                text(
                    "SELECT is_active, document_family "
                    "FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND generation = :generation"
                ),
                {"generation": generation},
            ).one_or_none()
            if (
                claimed_state is not None
                and requested_state is not None
                and str(claimed_state.document_family) == "raw_message_v3"
                and str(requested_state.document_family) == "raw_message_v3"
                and str(claimed_state.status) in {"building", "ready"}
                and not bool(claimed_state.is_active)
                and bool(requested_state.is_active)
            ):
                # A worker pinned to the older active generation must not
                # erase a newer generation's in-progress document claim.
                generation = int(matching.embedding_generation)
        if generation is None and matching is not None:
            embedding_provider = matching.embedding_provider
            embedding_model = matching.embedding_model
            embedding_version = matching.embedding_version
            embedding_dimensions = matching.embedding_dimensions
            effective_generation = matching.embedding_generation
            embedding_eligible = bool(matching.embedding_eligible)
            embedding_status = matching.embedding_status
        elif (
            matching is not None
            and matching.embedding_generation == generation
            and matching.embedding_status == "ready"
        ):
            embedding_provider = matching.embedding_provider
            embedding_model = matching.embedding_model
            embedding_version = matching.embedding_version
            embedding_dimensions = matching.embedding_dimensions
            effective_generation = generation
            embedding_eligible = True
            embedding_status = "ready"
        else:
            embedding_provider = ""
            embedding_model = ""
            embedding_version = ""
            embedding_dimensions = None
            effective_generation = generation
            embedding_eligible = generation is not None
            embedding_status = "pending" if generation is not None else "disabled"
        document = self.upsert_document(
            scope_type="group",
            scope_id=str(int(group_id)),
            group_id=int(group_id),
            episode_id=None,
            document_kind="raw_message_v3",
            source_table="messages",
            source_id=str(int(message.id)),
            start_at=message.timestamp,
            end_at=message.timestamp,
            content=content,
            metadata_json={
                "platform_msg_id": str(message.platform_msg_id),
                "speaker_id": str(message.user_id),
                "reply_to_msg_id": str(message.reply_to_msg_id or ""),
                "mentioned_bot": bool(message.mentioned_bot),
                "index_generation": effective_generation,
            },
            content_hash=content_hash,
            source_message_ids=[int(message.id)],
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            embedding_dimensions=embedding_dimensions,
            embedding_generation=effective_generation,
            embedding_eligible=embedding_eligible,
            embedding_status=embedding_status,
        )
        self._deactivate_documents(
            [row for row in existing if row.id != document.id]
        )
        return document

    def list_raw_message_v3_projection_gaps(
        self,
        *,
        projection_job_type: str,
        embedding_generation: int | None,
        limit: int,
        group_ids: Sequence[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Return eligible ledger rows lacking a projection or live repair job."""
        projection_filters = [
            RetrievalDocumentMessage.message_id == Message.id,
            RetrievalDocumentMessage.group_id == Message.group_id,
            RetrievalDocument.id == RetrievalDocumentMessage.document_id,
            RetrievalDocument.group_id == Message.group_id,
            RetrievalDocument.document_kind == "raw_message_v3",
            RetrievalDocument.source_table == "messages",
            RetrievalDocument.source_id == cast(Message.id, String),
            RetrievalDocument.status == "active",
        ]
        if embedding_generation is not None:
            projection_filters.extend(
                [
                    RetrievalDocument.embedding_generation
                    == int(embedding_generation),
                    RetrievalDocument.embedding_eligible.is_(True),
                ]
            )
        matching_projection = (
            select(RetrievalDocument.id)
            .select_from(RetrievalDocumentMessage)
            .join(
                RetrievalDocument,
                RetrievalDocument.id == RetrievalDocumentMessage.document_id,
            )
            .where(*projection_filters)
            .correlate(Message)
            .exists()
        )
        live_or_terminal_failure_job = (
            select(Job.id)
            .where(
                Job.job_type == str(projection_job_type),
                Job.status.in_(("queued", "running", "failed")),
                cast(
                    func.json_extract(Job.payload_json, "$.group_id"),
                    Integer,
                )
                == Message.group_id,
                cast(
                    func.json_extract(Job.payload_json, "$.message_id"),
                    Integer,
                )
                == Message.id,
            )
            .correlate(Message)
            .exists()
        )
        delivery_state = func.coalesce(
            func.json_extract(Message.raw_json, "$.delivery_state"),
            "",
        )
        filters = [
            Message.group_id.is_not(None),
            func.trim(func.coalesce(Message.plain_text, "")) != "",
            ~delivery_state.in_(_INELIGIBLE_DELIVERY_STATES),
            ~matching_projection,
            ~live_or_terminal_failure_job,
        ]
        if group_ids:
            filters.append(Message.group_id.in_(tuple(int(value) for value in group_ids)))
        rows = self.session.execute(
            select(Message.group_id, Message.id)
            .where(*filters)
            .order_by(Message.id.asc())
            .limit(max(1, int(limit)))
        )
        return [
            (int(group_id), int(message_id))
            for group_id, message_id in rows
            if group_id is not None
        ]

    def revoke_unsafe_raw_message_v3_projections(
        self,
        *,
        limit: int,
        group_ids: Sequence[int] | None = None,
    ) -> int:
        """Deactivate projections whose canonical source became ineligible."""
        delivery_state = func.coalesce(
            func.json_extract(Message.raw_json, "$.delivery_state"),
            "",
        )
        filters = [
            RetrievalDocument.document_kind == "raw_message_v3",
            RetrievalDocument.source_table == "messages",
            RetrievalDocument.status == "active",
            or_(
                func.trim(func.coalesce(Message.plain_text, "")) == "",
                delivery_state.in_(_INELIGIBLE_DELIVERY_STATES),
            ),
        ]
        if group_ids:
            filters.append(
                RetrievalDocument.group_id.in_(
                    tuple(int(value) for value in group_ids)
                )
            )
        documents = list(
            self.session.scalars(
                select(RetrievalDocument)
                .join(
                    RetrievalDocumentMessage,
                    (
                        RetrievalDocumentMessage.document_id
                        == RetrievalDocument.id
                    )
                    & (
                        RetrievalDocumentMessage.group_id
                        == RetrievalDocument.group_id
                    ),
                )
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (
                        Message.group_id
                        == RetrievalDocumentMessage.group_id
                    ),
                )
                .where(*filters)
                .distinct()
                .order_by(RetrievalDocument.id.asc())
                .limit(max(1, int(limit)))
            )
        )
        self._deactivate_documents(documents)
        return len(documents)

    def deactivate_raw_message_v3(
        self,
        *,
        group_id: int,
        message_id: int,
    ) -> int:
        documents = list(
            self.session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.document_kind == "raw_message_v3",
                    RetrievalDocument.source_table == "messages",
                    RetrievalDocument.source_id == str(int(message_id)),
                    RetrievalDocument.status == "active",
                )
            )
        )
        self._deactivate_documents(documents)
        return len(documents)

    def load_raw_message_embedding_document(
        self,
        *,
        group_id: int,
        message_id: int,
        document_id: int,
        embedding_generation: int,
    ) -> RetrievalDocument | None:
        document = self.session.scalars(
            select(RetrievalDocument).where(
                RetrievalDocument.id == int(document_id),
                RetrievalDocument.group_id == int(group_id),
                RetrievalDocument.document_kind == "raw_message_v3",
                RetrievalDocument.source_table == "messages",
                RetrievalDocument.source_id == str(int(message_id)),
                RetrievalDocument.status == "active",
                RetrievalDocument.embedding_eligible.is_(True),
                RetrievalDocument.embedding_generation == int(embedding_generation),
            )
        ).first()
        if document is None:
            return None
        source = self.session.scalars(
            select(Message)
            .join(
                RetrievalDocumentMessage,
                (RetrievalDocumentMessage.message_id == Message.id)
                & (RetrievalDocumentMessage.group_id == Message.group_id),
            )
            .where(
                RetrievalDocumentMessage.document_id == document.id,
                RetrievalDocumentMessage.group_id == int(group_id),
                Message.id == int(message_id),
                Message.group_id == int(group_id),
            )
        ).first()
        if (
            source is None
            or MessageRepository.is_reserved_outbound(source)
            or MessageRepository.is_qq_blocked_outbound(source)
            or MessageRepository.is_delivery_uncertain_outbound(source)
            or (
                isinstance(source.raw_json, dict)
                and str(source.raw_json.get("delivery_state") or "")
                .strip()
                .casefold()
                in _INELIGIBLE_DELIVERY_STATES
            )
        ):
            self._deactivate_documents([document])
            return None
        return document

    def _deactivate_documents(
        self,
        documents: Sequence[RetrievalDocument],
    ) -> None:
        if not documents:
            return
        now = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        for document in documents:
            if document.status != "inactive":
                document.status = "inactive"
                document.embedding_status = "stale"
                document.updated_at = now
                self.session.add(document)
            self._sync_fts(document)
        _delete_active_retrieval_vectors(
            self.session,
            document_ids=[document.id for document in documents],
        )

    def list_source_message_ids(self, *, document_id: int, group_id: int) -> list[int]:
        return list(
            self.session.scalars(
                select(RetrievalDocumentMessage.message_id)
                .join(
                    RetrievalDocument,
                    (RetrievalDocument.id == RetrievalDocumentMessage.document_id)
                    & (RetrievalDocument.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.document_id == document_id,
                    RetrievalDocumentMessage.group_id == group_id,
                    RetrievalDocument.group_id == group_id,
                )
                .order_by(RetrievalDocumentMessage.ordinal.asc())
            )
        )

    def deactivate_episode_documents(
        self,
        *,
        group_id: int,
        episode_id: int,
        embedding_generation: int | None = None,
    ) -> int:
        stmt = select(RetrievalDocument).where(
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.episode_id == int(episode_id),
            RetrievalDocument.status == "active",
        )
        if embedding_generation is not None:
            stmt = stmt.where(
                RetrievalDocument.embedding_generation == int(embedding_generation)
            )
        documents = list(self.session.scalars(stmt))
        for document in documents:
            document.status = "inactive"
            document.embedding_status = "stale"
            document.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            self.session.add(document)
            self._sync_fts(document)
        _delete_active_retrieval_vectors(
            self.session,
            document_ids=[document.id for document in documents],
        )
        return len(documents)

    def embedding_coverage(
        self,
        *,
        group_id: int | None = None,
        generation: int | None = None,
    ) -> RetrievalEmbeddingCoverage:
        filters = [
            RetrievalDocument.status == "active",
            RetrievalDocument.embedding_eligible.is_(True),
        ]
        if group_id is not None:
            filters.append(RetrievalDocument.group_id == int(group_id))
        if generation is not None:
            filters.append(
                RetrievalDocument.embedding_generation == int(generation)
            )
        total = int(
            self.session.scalar(
                select(func.count(RetrievalDocument.id)).where(*filters)
            )
            or 0
        )
        ready = int(
            self.session.scalar(
                select(func.count(RetrievalDocument.id)).where(
                    *filters,
                    RetrievalDocument.embedding_status == "ready",
                )
            )
            or 0
        )
        failed = int(
            self.session.scalar(
                select(func.count(RetrievalDocument.id)).where(
                    *filters,
                    RetrievalDocument.embedding_status == "failed",
                )
            )
            or 0
        )
        return RetrievalEmbeddingCoverage(
            total_documents=total,
            ready_documents=ready,
            failed_documents=failed,
        )

    def search_group_documents_fts_hits(
        self,
        *,
        group_id: int,
        query: str,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        resolved_limit = max(1, int(limit))
        documents = self.search_group_documents_fts(
            group_id=group_id,
            query=query,
            limit=resolved_limit,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )
        ranked = [
            (document.id, float(resolved_limit - rank))
            for rank, document in enumerate(documents)
        ]
        hits = self._validated_hits(
            group_id=group_id,
            ranked_document_ids=ranked,
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )
        normalized_query = str(query or "").strip().casefold()
        exact_ids = {
            int(document.id)
            for document in documents
            if normalized_query
            and normalized_query in str(document.content or "").casefold()
        }
        return [
            replace(hit, lexical_exact=int(hit.document_id) in exact_ids)
            for hit in hits
        ]

    def search_group_documents_temporal_hits(
        self,
        *,
        group_id: int,
        start_at: datetime | None,
        end_at: datetime | None,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
        allow_unbounded: bool = False,
        sample_time_coverage: bool | None = None,
    ) -> list[RetrievalDocumentHit]:
        if start_at is None and end_at is None and not allow_unbounded:
            return []
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if normalized_mentioned_user_ids == ():
            return []
        stmt = select(RetrievalDocument.id).where(
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
            *_retrieval_source_prefilters(
                group_id=group_id,
                speaker_ids=speaker_ids,
                excluded_speaker_ids=excluded_speaker_ids,
            ),
        )
        if normalized_mentioned_user_ids is not None:
            stmt = stmt.where(
                RetrievalDocument.id.in_(
                    _retrieval_mention_document_ids(
                        group_id=group_id,
                        mentioned_user_ids=normalized_mentioned_user_ids,
                    )
                )
            )
        normalized_document_kinds = tuple(
            dict.fromkeys(
                str(kind).strip()
                for kind in (document_kinds or ())
                if str(kind).strip()
            )
        )
        if document_kinds is not None:
            if not normalized_document_kinds:
                return []
            stmt = stmt.where(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            stmt = stmt.where(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            stmt = stmt.where(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        resolved_limit = max(1, int(limit))
        use_time_coverage = (
            bool(allow_unbounded)
            if sample_time_coverage is None
            else bool(sample_time_coverage)
        )
        ordered_stmt = stmt.order_by(
            RetrievalDocument.start_at.asc()
            if use_time_coverage
            else RetrievalDocument.end_at.desc(),
            RetrievalDocument.id.asc()
            if use_time_coverage
            else RetrievalDocument.id.desc(),
        )
        if not use_time_coverage:
            ordered_stmt = ordered_stmt.limit(resolved_limit * 4)
        document_ids = list(self.session.scalars(ordered_stmt))
        hits = self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )
        if len(hits) <= resolved_limit:
            return hits
        if not use_time_coverage or resolved_limit == 1:
            return hits[:resolved_limit]
        # Deterministic coverage sampling over the full eligible chronology.
        # Endpoints are pinned and interior slots are evenly distributed.
        indices = [
            (slot * (len(hits) - 1)) // (resolved_limit - 1)
            for slot in range(resolved_limit)
        ]
        return [hits[index] for index in dict.fromkeys(indices)]

    def search_group_documents_entity_hits(
        self,
        *,
        group_id: int,
        entities: tuple[str, ...],
        speaker_ids: Sequence[str] | None,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        normalized_entities = tuple(
            dict.fromkeys(value.strip() for value in entities if value.strip())
        )[:12]
        normalized_speaker_ids = tuple(
            int(value)
            for value in dict.fromkeys(speaker_ids or ())
            if str(value).strip().lstrip("-").isdigit()
        )
        if not normalized_entities and not normalized_speaker_ids:
            return []
        conditions = []
        for entity in normalized_entities:
            conditions.extend(
                (
                    RetrievalDocument.content.contains(entity),
                    RetrievalDocument.metadata_json.contains(entity),
                )
            )
        identity_conditions = []
        if normalized_speaker_ids:
            identity_conditions.append(Message.user_id.in_(normalized_speaker_ids))
        if normalized_entities and not normalized_speaker_ids:
            identity_user_ids = select(User.user_id).where(
                or_(
                    User.nickname.in_(normalized_entities),
                    User.group_card.in_(normalized_entities),
                )
            )
            identity_conditions.append(Message.user_id.in_(identity_user_ids))
        if identity_conditions:
            identity_document_ids = (
                select(RetrievalDocumentMessage.document_id)
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.group_id == int(group_id),
                    Message.group_id == int(group_id),
                    or_(*identity_conditions),
                )
            )
            conditions.append(RetrievalDocument.id.in_(identity_document_ids))
        stmt = (
            select(RetrievalDocument.id)
            .where(
                RetrievalDocument.group_id == int(group_id),
                RetrievalDocument.status == "active",
                or_(*conditions),
                *_retrieval_source_prefilters(
                    group_id=group_id,
                    speaker_ids=speaker_ids,
                    excluded_speaker_ids=excluded_speaker_ids,
                ),
            )
            .order_by(
                RetrievalDocument.end_at.desc(),
                RetrievalDocument.id.desc(),
            )
        )
        normalized_document_kinds = _normalize_optional_string_filter(
            document_kinds
        )
        if normalized_document_kinds == ():
            return []
        if normalized_document_kinds is not None:
            stmt = stmt.where(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            stmt = stmt.where(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            stmt = stmt.where(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        stmt = stmt.limit(
            2_147_483_647
            if mentioned_user_ids is not None
            else max(1, int(limit))
        )
        document_ids = list(self.session.scalars(stmt))
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )

    def search_group_member_reference_hits(
        self,
        *,
        group_id: int,
        aliases: Sequence[str],
        subject_ids: Sequence[str],
        query_text: str,
        limit: int,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        include_subject_authored: bool = False,
    ) -> list[RetrievalDocumentHit]:
        """Find raw utterances that explicitly name or @ a bound member.

        This route deliberately does not constrain the author to the member:
        third-party observations are useful evidence, with their own provenance.
        """
        normalized_aliases = tuple(dict.fromkeys(
            alias.strip() for alias in aliases if len(alias.strip()) >= 2
        ))[:8]
        normalized_subjects = tuple(dict.fromkeys(
            str(value).strip() for value in subject_ids if str(value).strip()
        ))
        if not normalized_subjects:
            return []
        conditions = [RetrievalDocument.content.contains(alias) for alias in normalized_aliases]
        authored_document_ids = (
            select(RetrievalDocumentMessage.document_id)
            .join(
                Message,
                (Message.id == RetrievalDocumentMessage.message_id)
                & (Message.group_id == RetrievalDocumentMessage.group_id),
            )
            .where(
                RetrievalDocumentMessage.group_id == int(group_id),
                Message.group_id == int(group_id),
                Message.user_id.in_(tuple(int(value) for value in normalized_subjects if value.isdigit())),
            )
        )
        bot_reply_document_ids = (
            select(RetrievalDocumentMessage.document_id)
            .join(
                Message,
                (Message.id == RetrievalDocumentMessage.message_id)
                & (Message.group_id == RetrievalDocumentMessage.group_id),
            )
            .where(
                RetrievalDocumentMessage.group_id == int(group_id),
                Message.group_id == int(group_id),
                Message.platform_msg_id.like("bot-reply-%"),
            )
        )
        scope_filters = [
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
            RetrievalDocument.document_kind == "raw_message_v3",
            RetrievalDocument.id.not_in(bot_reply_document_ids),
            *_retrieval_source_prefilters(
                group_id=group_id,
                speaker_ids=None,
                excluded_speaker_ids=excluded_speaker_ids,
            ),
        ]
        if start_at is not None:
            scope_filters.append(RetrievalDocument.end_at >= _normalize_utc_sqlite_timestamp(start_at))
        if end_at is not None:
            scope_filters.append(RetrievalDocument.start_at < _normalize_utc_sqlite_timestamp(end_at))
        # Inspect a bounded set of recent references, then rank by relation
        # wording before recency. This keeps old but pertinent statements from
        # disappearing behind newer unrelated mentions of the same member.
        rows = []
        search_conditions = list(conditions)
        if include_subject_authored and any(value.isdigit() for value in normalized_subjects):
            search_conditions.append(RetrievalDocument.id.in_(authored_document_ids))
        if search_conditions:
            rows = self.session.execute(
                select(RetrievalDocument.id, RetrievalDocument.content)
                .where(*scope_filters, or_(*search_conditions))
                .order_by(RetrievalDocument.end_at.desc(), RetrievalDocument.id.desc())
                .limit(max(200, int(limit) * 8))
            ).all()
        if not rows:
            rows = self.session.execute(
                select(RetrievalDocument.id, RetrievalDocument.content)
                .where(
                    *scope_filters,
                    RetrievalDocument.id.in_(_retrieval_mention_document_ids(
                        group_id=group_id, mentioned_user_ids=normalized_subjects,
                    )),
                )
                .order_by(RetrievalDocument.end_at.desc(), RetrievalDocument.id.desc())
                .limit(max(200, int(limit) * 8))
            ).all()
        authored_ids: set[int] = set()
        if include_subject_authored and any(value.isdigit() for value in normalized_subjects):
            authored_ids = {
                int(value)
                for value in self.session.execute(
                    select(RetrievalDocumentMessage.document_id)
                    .join(
                        Message,
                        (Message.id == RetrievalDocumentMessage.message_id)
                        & (Message.group_id == RetrievalDocumentMessage.group_id),
                    )
                    .where(
                        RetrievalDocumentMessage.group_id == int(group_id),
                        Message.group_id == int(group_id),
                        Message.user_id.in_(tuple(
                            int(value) for value in normalized_subjects if value.isdigit()
                        )),
                    )
                ).scalars()
            }
        topic = query_text
        for filler in ("最近", "目前", "现在", "什么", "哪里", "哪儿", "如何", "怎么", "多少", "是否", "有没有"):
            topic = topic.replace(filler, "")
        activity_terms = tuple(
            term for term in ("看", "追", "补", "玩", "用", "做", "学", "读", "听")
            if term in topic
        )
        fragments = {
            topic[index:index + width]
            for width in range(2, 7)
            for index in range(max(0, len(topic) - width + 1))
            if topic[index:index + width].strip()
            and not any(char in "？?，,。！!：:、 　" for char in topic[index:index + width])
            and all(topic[index:index + width] not in alias for alias in normalized_aliases)
        }
        ranked = sorted(
            enumerate(rows),
            key=lambda pair: (
                -(
                    sum((len(fragment) - 1) for fragment in fragments if fragment in pair[1].content)
                    + (
                        12
                        if int(pair[1].id) in authored_ids
                        and activity_terms
                        and any(term in pair[1].content for term in activity_terms)
                        else 0
                    )
                ),
                pair[0],
            ),
        )[:max(1, int(limit))]
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(row.id), float(len(ranked) - rank))
                for rank, (_index, row) in enumerate(ranked)
            ],
            document_kinds=("raw_message_v3",),
            start_at=start_at,
            end_at=end_at,
            excluded_speaker_ids=excluded_speaker_ids,
        )

    def search_group_fact_hits(
        self,
        *,
        group_id: int,
        query: str,
        entities: tuple[str, ...],
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        terms = tuple(
            dict.fromkeys(
                [
                    *(value.strip() for value in entities if value.strip()),
                    *(
                        value
                        for value in _fts_search_terms(query)
                        if len(value) >= 2
                    ),
                ]
            )
        )
        instant = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        stmt = (
            select(RetrievalDocument.id)
            .join(
                MemoryItem,
                MemoryItem.id
                == func.cast(RetrievalDocument.source_id, Integer),
            )
            .where(
                RetrievalDocument.group_id == int(group_id),
                RetrievalDocument.status == "active",
                RetrievalDocument.document_kind == "memory",
                RetrievalDocument.source_table == "memory_items",
                RetrievalDocument.scope_type == "group",
                RetrievalDocument.scope_id == str(int(group_id)),
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == str(int(group_id)),
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            )
        )
        if terms:
            stmt = stmt.where(
                or_(
                    *(
                        RetrievalDocument.content.contains(term)
                        for term in terms
                    )
                )
            )
        document_ids = list(
            self.session.scalars(
                stmt.order_by(
                    RetrievalDocument.end_at.desc(),
                    RetrievalDocument.id.desc(),
                ).limit(max(1, int(limit)))
            )
        )
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )

    def search_group_reference_hits(
        self,
        *,
        group_id: int,
        reference_msg_ids: tuple[str, ...],
        include_replies: bool,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        references = tuple(
            dict.fromkeys(value.strip() for value in reference_msg_ids if value.strip())
        )
        if not references:
            return []
        reference_condition = (
            Message.reply_to_msg_id.in_(references)
            if include_replies
            else Message.platform_msg_id.in_(references)
        )
        stmt = (
            select(RetrievalDocument.id)
                .join(
                    RetrievalDocumentMessage,
                    (RetrievalDocumentMessage.document_id == RetrievalDocument.id)
                    & (RetrievalDocumentMessage.group_id == RetrievalDocument.group_id),
                )
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.status == "active",
                    RetrievalDocumentMessage.group_id == int(group_id),
                    Message.group_id == int(group_id),
                    reference_condition,
                    *_retrieval_source_prefilters(
                        group_id=group_id,
                        speaker_ids=speaker_ids,
                        excluded_speaker_ids=excluded_speaker_ids,
                    ),
                )
                .distinct()
                .order_by(
                    RetrievalDocument.end_at.desc(),
                    RetrievalDocument.id.desc(),
                )
            )
        normalized_document_kinds = _normalize_optional_string_filter(
            document_kinds
        )
        if normalized_document_kinds == ():
            return []
        if normalized_document_kinds is not None:
            stmt = stmt.where(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            stmt = stmt.where(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            stmt = stmt.where(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        stmt = stmt.limit(
            2_147_483_647
            if mentioned_user_ids is not None
            else max(1, int(limit))
        )
        document_ids = list(self.session.scalars(stmt))
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )

    def search_group_documents_vector_hits(
        self,
        *,
        group_id: int,
        embedding: list[float],
        provider: str,
        model: str,
        dimensions: int,
        version: str,
        generation: int | None = None,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        if len(embedding) != int(dimensions):
            raise ValueError("query embedding dimensions are incompatible")
        normalized_document_kinds = _normalize_optional_string_filter(document_kinds)
        normalized_subject_ids = _normalize_optional_string_filter(subject_ids)
        normalized_speaker_ids = _normalize_optional_integer_filter(speaker_ids)
        normalized_excluded_speaker_ids = _normalize_optional_integer_filter(
            excluded_speaker_ids
        )
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if (
            normalized_document_kinds == ()
            or normalized_speaker_ids == ()
            or normalized_mentioned_user_ids == ()
            or (
                normalized_subject_ids == ()
                and normalized_document_kinds == ("raw_message_v3",)
            )
        ):
            return []
        generation_document_family = (
            "raw_message_v3"
            if normalized_document_kinds == ("raw_message_v3",)
            else ""
        )
        generation_clause = (
            "AND generation = :generation "
            if generation is not None
            else "AND is_active = 1 "
        )
        state = self.session.execute(
            text(
                "SELECT generation, physical_table "
                "FROM retrieval_index_state "
                "WHERE channel = 'vector' AND status = 'ready' "
                f"{generation_clause}"
                "AND provider = :provider AND model = :model "
                "AND dimensions = :dimensions AND version = :version "
                "AND document_family = :document_family LIMIT 1"
            ),
            {
                "generation": (
                    int(generation) if generation is not None else None
                ),
                "provider": str(provider),
                "model": str(model),
                "dimensions": int(dimensions),
                "version": str(version),
                "document_family": generation_document_family,
            },
        ).one_or_none()
        if state is None:
            return []
        physical_table = validate_retrieval_vector_table_name(
            state.physical_table,
            generation=state.generation,
        )
        resolved_limit = max(1, int(limit))
        available = int(
            self.session.execute(
                text(
                    f"SELECT count(*) FROM {physical_table} "
                    "WHERE group_id = :group_id"
                ),
                {"group_id": int(group_id)},
            ).scalar_one()
            or 0
        )
        if available <= 0:
            return []
        has_sparse_post_filters = any(
            value is not None
            for value in (
                normalized_subject_ids,
                normalized_speaker_ids,
                normalized_mentioned_user_ids,
                start_at,
                end_at,
            )
        )
        fetch_ceiling = _vector_fetch_ceiling(
            requested=resolved_limit,
            available=available,
            has_sparse_post_filters=has_sparse_post_filters,
            has_exclusion_filter=normalized_excluded_speaker_ids is not None,
        )
        fetch_limit = _initial_vector_fetch_limit(
            requested=resolved_limit,
            available=fetch_ceiling,
            has_post_filters=has_sparse_post_filters,
        )
        serialized_embedding = json.dumps(
            [float(value) for value in embedding],
            separators=(",", ":"),
        )
        while True:
            ranked = [
                (int(document_id), -float(distance))
                for document_id, distance in self.session.execute(
                    text(
                        "SELECT document_id, distance "
                        f"FROM {physical_table} "
                        "WHERE embedding MATCH :embedding "
                        "AND group_id = :group_id AND k = :limit"
                    ),
                    {
                        "embedding": serialized_embedding,
                        "group_id": int(group_id),
                        "limit": fetch_limit,
                    },
                )
            ]
            hits = self._validated_hits(
                group_id=group_id,
                ranked_document_ids=ranked,
                subject_ids=normalized_subject_ids,
                document_kinds=normalized_document_kinds,
                start_at=start_at,
                end_at=end_at,
                speaker_ids=(
                    tuple(str(value) for value in normalized_speaker_ids)
                    if normalized_speaker_ids is not None
                    else None
                ),
                excluded_speaker_ids=(
                    tuple(str(value) for value in normalized_excluded_speaker_ids)
                    if normalized_excluded_speaker_ids is not None
                    else None
                ),
                mentioned_user_ids=normalized_mentioned_user_ids,
            )
            if len(hits) >= resolved_limit or fetch_limit >= fetch_ceiling:
                return hits[:resolved_limit]
            fetch_limit = _next_vector_fetch_limit(
                fetch_limit,
                requested=resolved_limit,
                available=fetch_ceiling,
            )

    def _validated_hits(
        self,
        *,
        group_id: int,
        ranked_document_ids: list[tuple[int, float]],
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        if not ranked_document_ids:
            return []
        normalized_document_kinds = _normalize_optional_string_filter(document_kinds)
        normalized_subject_ids = _normalize_optional_string_filter(subject_ids)
        normalized_speaker_ids = _normalize_optional_integer_filter(speaker_ids)
        normalized_excluded_speaker_ids = _normalize_optional_integer_filter(
            excluded_speaker_ids
        )
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if (
            normalized_document_kinds == ()
            or normalized_speaker_ids == ()
            or normalized_mentioned_user_ids == ()
            or (
                normalized_subject_ids == ()
                and normalized_document_kinds == ("raw_message_v3",)
            )
        ):
            return []
        ordered_ids = list(
            dict.fromkeys(int(document_id) for document_id, _ in ranked_document_ids)
        )
        document_filters = [
            RetrievalDocument.id.in_(ordered_ids),
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
        ]
        if normalized_document_kinds is not None:
            document_filters.append(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            document_filters.append(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            document_filters.append(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        documents = {
            document.id: document
            for document in self.session.scalars(
                select(RetrievalDocument).where(*document_filters)
            )
        }
        memory_documents = {
            document_id: document
            for document_id, document in documents.items()
            if document.document_kind == "memory"
        }
        if memory_documents:
            if normalized_subject_ids == ():
                for document_id in memory_documents:
                    documents.pop(document_id, None)
                memory_documents = {}
            memory_ids = {
                int(document.source_id)
                for document in memory_documents.values()
                if document.source_table == "memory_items"
                and str(document.source_id).lstrip("-").isdigit()
            }
            instant = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            current_filters = [
                MemoryItem.id.in_(memory_ids),
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == str(int(group_id)),
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            ]
            if normalized_subject_ids is not None:
                current_filters.append(MemoryItem.subject_id.in_(normalized_subject_ids))
            current_memory_ids = set(
                self.session.scalars(
                    select(MemoryItem.id).where(*current_filters)
                )
            )
            for document_id, document in memory_documents.items():
                source_id = str(document.source_id)
                if (
                    document.source_table != "memory_items"
                    or not source_id.lstrip("-").isdigit()
                    or int(source_id) not in current_memory_ids
                ):
                    documents.pop(document_id, None)
        all_counts = {
            int(document_id): int(count)
            for document_id, count in self.session.execute(
                select(
                    RetrievalDocumentMessage.document_id,
                    func.count(),
                )
                .where(RetrievalDocumentMessage.document_id.in_(ordered_ids))
                .group_by(RetrievalDocumentMessage.document_id)
            )
        }
        source_rows = list(
            self.session.execute(
                select(
                    RetrievalDocumentMessage.document_id,
                    RetrievalDocumentMessage.group_id,
                    Message.group_id,
                    Message.platform_msg_id,
                    Message.user_id,
                    Message.timestamp,
                    Message.raw_json,
                    RetrievalDocumentMessage.ordinal,
                )
                .join(
                    RetrievalDocument,
                    (RetrievalDocument.id == RetrievalDocumentMessage.document_id)
                    & (
                        RetrievalDocument.group_id
                        == RetrievalDocumentMessage.group_id
                    ),
                )
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.document_id.in_(ordered_ids),
                    RetrievalDocumentMessage.group_id == int(group_id),
                    RetrievalDocument.group_id == int(group_id),
                    Message.group_id == int(group_id),
                )
                .order_by(
                    RetrievalDocumentMessage.document_id.asc(),
                    RetrievalDocumentMessage.ordinal.asc(),
                    RetrievalDocumentMessage.message_id.asc(),
                )
            )
        )
        sources: dict[int, list[str]] = {}
        for (
            document_id,
            provenance_group_id,
            message_group_id,
            platform_msg_id,
            user_id,
            message_timestamp,
            raw_json,
            _ordinal,
        ) in source_rows:
            if (
                int(provenance_group_id) != int(group_id)
                or int(message_group_id) != int(group_id)
            ):
                raise ValueError("retrieval provenance failed group validation")
            delivery_state = (
                str(raw_json.get("delivery_state") or "")
                if isinstance(raw_json, dict)
                else ""
            )
            if delivery_state in _INELIGIBLE_DELIVERY_STATES:
                documents.pop(int(document_id), None)
                continue
            if (
                normalized_speaker_ids is not None
                and int(user_id) not in normalized_speaker_ids
            ):
                documents.pop(int(document_id), None)
                continue
            if (
                normalized_excluded_speaker_ids is not None
                and int(user_id) in normalized_excluded_speaker_ids
            ):
                documents.pop(int(document_id), None)
                continue
            if (
                normalized_mentioned_user_ids is not None
                and not set(normalized_mentioned_user_ids).intersection(
                    _mentioned_user_ids(raw_json)
                )
            ):
                documents.pop(int(document_id), None)
                continue
            normalized_timestamp = _normalize_utc_sqlite_timestamp(message_timestamp)
            if (
                start_at is not None
                and normalized_timestamp
                < _normalize_utc_sqlite_timestamp(start_at)
            ):
                documents.pop(int(document_id), None)
                continue
            if (
                end_at is not None
                and normalized_timestamp
                >= _normalize_utc_sqlite_timestamp(end_at)
            ):
                documents.pop(int(document_id), None)
                continue
            sources.setdefault(int(document_id), []).append(str(platform_msg_id))
        score_by_id = {
            int(document_id): float(score)
            for document_id, score in ranked_document_ids
        }
        hits: list[RetrievalDocumentHit] = []
        for document_id in ordered_ids:
            document = documents.get(document_id)
            if document is None:
                continue
            source_msg_ids = tuple(dict.fromkeys(sources.get(document_id, ())))
            if not source_msg_ids or len(sources.get(document_id, ())) != all_counts.get(
                document_id,
                0,
            ):
                raise ValueError("retrieval provenance is incomplete or unscoped")
            hits.append(
                RetrievalDocumentHit(
                    document_id=document.id,
                    group_id=document.group_id,
                    document_kind=document.document_kind,
                    episode_id=document.episode_id,
                    source_msg_ids=source_msg_ids,
                    start_at=document.start_at,
                    end_at=document.end_at,
                    score=score_by_id[document_id],
                )
            )
        return hits

    def search_group_documents_fts(
        self,
        *,
        group_id: int,
        query: str,
        limit: int,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocument]:
        resolved_limit = max(1, int(limit))
        normalized_document_kinds = _normalize_optional_string_filter(document_kinds)
        normalized_speaker_ids = _normalize_optional_integer_filter(speaker_ids)
        normalized_excluded_speaker_ids = _normalize_optional_integer_filter(
            excluded_speaker_ids
        )
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if any(
            values == ()
            for values in (
                normalized_document_kinds,
                normalized_speaker_ids,
                normalized_mentioned_user_ids,
            )
        ):
            return []
        terms = [term for term in _fts_search_terms(query) if len(term) >= 3]
        if terms:
            try:
                clauses = [
                    "retrieval_documents_fts MATCH :match_query",
                    "retrieval_documents_fts.group_id = :group_id",
                    "d.group_id = :group_id_int",
                    "d.status = 'active'",
                    "NOT EXISTS ("
                    "SELECT 1 FROM retrieval_document_messages AS safety_rdm "
                    "JOIN messages AS safety_m "
                    "ON safety_m.id = safety_rdm.message_id "
                    "AND safety_m.group_id = safety_rdm.group_id "
                    "WHERE safety_rdm.document_id = d.id "
                    "AND safety_rdm.group_id = :group_id_int "
                    "AND json_extract(safety_m.raw_json, '$.delivery_state') "
                    "IN ('reserved','blocked','uncertain','deleted'))",
                ]
                parameters: dict[str, Any] = {
                    "match_query": " OR ".join(f'"{term}"' for term in terms),
                    "group_id": str(group_id),
                    "group_id_int": int(group_id),
                    "limit": resolved_limit,
                }
                statement = text(
                    "SELECT d.id FROM retrieval_documents_fts "
                    "JOIN retrieval_documents AS d "
                    "ON d.id = CAST(retrieval_documents_fts.document_id AS INTEGER) "
                    "WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY bm25(retrieval_documents_fts), d.id ASC LIMIT :limit"
                )
                if normalized_document_kinds is not None:
                    clauses.append("d.document_kind IN :document_kinds")
                    parameters["document_kinds"] = normalized_document_kinds
                if start_at is not None:
                    clauses.append("d.end_at >= :start_at")
                    parameters["start_at"] = _normalize_utc_sqlite_timestamp(start_at)
                if end_at is not None:
                    clauses.append("d.start_at < :end_at")
                    parameters["end_at"] = _normalize_utc_sqlite_timestamp(end_at)
                if normalized_speaker_ids is not None:
                    clauses.append(
                        "NOT EXISTS ("
                        "SELECT 1 FROM retrieval_document_messages AS speaker_rdm "
                        "JOIN messages AS speaker_m "
                        "ON speaker_m.id = speaker_rdm.message_id "
                        "AND speaker_m.group_id = speaker_rdm.group_id "
                        "WHERE speaker_rdm.document_id = d.id "
                        "AND speaker_rdm.group_id = :group_id_int "
                        "AND speaker_m.user_id NOT IN :speaker_ids)"
                    )
                    parameters["speaker_ids"] = normalized_speaker_ids
                if normalized_excluded_speaker_ids:
                    clauses.append(
                        "NOT EXISTS ("
                        "SELECT 1 FROM retrieval_document_messages AS excluded_rdm "
                        "JOIN messages AS excluded_m "
                        "ON excluded_m.id = excluded_rdm.message_id "
                        "AND excluded_m.group_id = excluded_rdm.group_id "
                        "WHERE excluded_rdm.document_id = d.id "
                        "AND excluded_rdm.group_id = :group_id_int "
                        "AND excluded_m.user_id IN :excluded_speaker_ids)"
                    )
                    parameters["excluded_speaker_ids"] = normalized_excluded_speaker_ids
                if normalized_mentioned_user_ids is not None:
                    clauses.append(
                        "EXISTS ("
                        "SELECT 1 FROM retrieval_document_messages AS mention_rdm "
                        "JOIN messages AS mention_m "
                        "ON mention_m.id = mention_rdm.message_id "
                        "AND mention_m.group_id = mention_rdm.group_id "
                        "JOIN json_each(CASE "
                        "WHEN json_type(mention_m.raw_json, '$.message') = 'array' "
                        "THEN json_extract(mention_m.raw_json, '$.message') "
                        "ELSE json_extract(mention_m.raw_json, '$.raw_message') END) "
                        "AS mention_segment "
                        "WHERE mention_rdm.document_id = d.id "
                        "AND mention_rdm.group_id = :group_id_int "
                        "AND json_extract(mention_segment.value, '$.type') = 'at' "
                        "AND CAST(coalesce("
                        "json_extract(mention_segment.value, '$.data.qq'), "
                        "json_extract(mention_segment.value, '$.data.uin'), "
                        "json_extract(mention_segment.value, '$.data.target')) AS TEXT) "
                        "IN :mentioned_user_ids)"
                    )
                    parameters["mentioned_user_ids"] = normalized_mentioned_user_ids
                statement = text(
                    "SELECT d.id FROM retrieval_documents_fts "
                    "JOIN retrieval_documents AS d "
                    "ON d.id = CAST(retrieval_documents_fts.document_id AS INTEGER) "
                    "WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY bm25(retrieval_documents_fts), d.id ASC LIMIT :limit"
                )
                expanding = []
                if normalized_document_kinds is not None:
                    expanding.append(bindparam("document_kinds", expanding=True))
                if normalized_speaker_ids is not None:
                    expanding.append(bindparam("speaker_ids", expanding=True))
                if normalized_excluded_speaker_ids:
                    expanding.append(
                        bindparam("excluded_speaker_ids", expanding=True)
                    )
                if normalized_mentioned_user_ids is not None:
                    expanding.append(
                        bindparam("mentioned_user_ids", expanding=True)
                    )
                if expanding:
                    statement = statement.bindparams(*expanding)
                rows = self.session.execute(
                    statement,
                    parameters,
                ).scalars()
                document_ids = [int(document_id) for document_id in rows]
                if document_ids:
                    documents = {
                        document.id: document
                        for document in self.session.scalars(
                            select(RetrievalDocument).where(
                                RetrievalDocument.id.in_(document_ids),
                                RetrievalDocument.group_id == group_id,
                                RetrievalDocument.status == "active",
                            )
                        )
                    }
                    return [
                        documents[hit.document_id]
                        for hit in self._validated_hits(
                            group_id=group_id,
                            ranked_document_ids=[
                                (document_id, float(len(document_ids) - rank))
                                for rank, document_id in enumerate(document_ids)
                            ],
                            document_kinds=normalized_document_kinds,
                            start_at=start_at,
                            end_at=end_at,
                            speaker_ids=(
                                tuple(str(value) for value in normalized_speaker_ids)
                                if normalized_speaker_ids is not None
                                else None
                            ),
                            excluded_speaker_ids=(
                                tuple(str(value) for value in normalized_excluded_speaker_ids)
                                if normalized_excluded_speaker_ids is not None
                                else None
                            ),
                            mentioned_user_ids=normalized_mentioned_user_ids,
                        )
                        if hit.document_id in documents
                    ]
            except SQLAlchemyError:
                # FTS is optional. The canonical group-scoped LIKE fallback
                # preserves availability without weakening isolation.
                pass
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        short_cjk_terms = tuple(
            dict.fromkeys(
                term
                for term in _fts_search_terms(query)
                if len(term) == 2 and re.fullmatch(r"[\u4e00-\u9fff]{2}", term)
            )
        )
        short_match_score = None
        fallback_match_conditions = [
            RetrievalDocument.content.contains(normalized_query),
            *(
                RetrievalDocument.content.contains(term)
                for term in short_cjk_terms
            ),
        ]
        document_filters = [
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
            or_(*fallback_match_conditions),
            *_retrieval_source_prefilters(
                group_id=group_id,
                speaker_ids=(
                    tuple(str(value) for value in normalized_speaker_ids)
                    if normalized_speaker_ids is not None
                    else None
                ),
                excluded_speaker_ids=(
                    tuple(str(value) for value in normalized_excluded_speaker_ids)
                    if normalized_excluded_speaker_ids is not None
                    else None
                ),
            ),
        ]
        if normalized_document_kinds is not None:
            document_filters.append(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            document_filters.append(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            document_filters.append(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        if short_cjk_terms:
            short_match_score = sum(
                case(
                    (RetrievalDocument.content.contains(term), 1),
                    else_=0,
                )
                for term in short_cjk_terms
            )
            document_filters.append(
                short_match_score >= (2 if len(short_cjk_terms) >= 2 else 1)
            )
        order_by = (
            (short_match_score.desc(),)
            if short_match_score is not None
            else ()
        ) + (
            RetrievalDocument.start_at.desc(),
            RetrievalDocument.id.desc(),
        )
        candidates = list(
            self.session.scalars(
                select(RetrievalDocument)
                .where(*document_filters)
                .order_by(*order_by)
                .limit(
                    max(resolved_limit * 4, resolved_limit)
                    if (
                        normalized_speaker_ids is None
                        and normalized_mentioned_user_ids is None
                    )
                    else 2_147_483_647
                )
            )
        )
        candidates_by_id = {document.id: document for document in candidates}
        return [
            candidates_by_id[hit.document_id]
            for hit in self._validated_hits(
                group_id=group_id,
                ranked_document_ids=[
                    (document.id, float(len(candidates) - rank))
                    for rank, document in enumerate(candidates)
                ],
                document_kinds=normalized_document_kinds,
                start_at=start_at,
                end_at=end_at,
                speaker_ids=(
                    tuple(str(value) for value in normalized_speaker_ids)
                    if normalized_speaker_ids is not None
                    else None
                ),
                excluded_speaker_ids=(
                    tuple(str(value) for value in normalized_excluded_speaker_ids)
                    if normalized_excluded_speaker_ids is not None
                    else None
                ),
                mentioned_user_ids=normalized_mentioned_user_ids,
            )
            if hit.document_id in candidates_by_id
        ][:resolved_limit]

    def _sync_fts(self, document: RetrievalDocument) -> None:
        # Canonical document/provenance constraints must propagate. SQLAlchemy
        # 2.1 also autoflushes Core statements, so keep that flush outside the
        # optional FTS error boundary rather than swallowing its IntegrityError.
        self.session.flush()
        try:
            self.session.execute(
                text(
                    "DELETE FROM retrieval_documents_fts "
                    "WHERE document_id = :document_id"
                ),
                {"document_id": str(document.id)},
            )
            if document.status != "active":
                return
            self.session.execute(
                text(
                    "INSERT INTO retrieval_documents_fts "
                    "(content, group_id, document_id, content_hash) "
                    "VALUES (:content, :group_id, :document_id, :content_hash)"
                ),
                {
                    "content": document.content,
                    "group_id": str(document.group_id),
                    "document_id": str(document.id),
                    "content_hash": document.content_hash,
                },
            )
        except SQLAlchemyError:
            return

class RetrievalIndexStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_generation(
        self,
        *,
        channel: str,
        generation: int,
        physical_table: str,
        provider: str,
        model: str,
        dimensions: int | None,
        version: str,
        status: str,
        total_documents: int,
        indexed_documents: int,
    ) -> RetrievalIndexState:
        if channel == "vector":
            validate_retrieval_vector_table_name(
                physical_table,
                generation=generation,
            )
            if dimensions is None or int(dimensions) <= 0:
                raise ValueError("vector generation dimensions are required")
        state = self.session.scalars(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == channel,
                RetrievalIndexState.generation == generation,
            )
        ).first()
        if state is None:
            state = RetrievalIndexState(
                channel=channel,
                generation=generation,
                physical_table=physical_table,
            )
        elif channel == "vector":
            persisted_identity = (
                state.physical_table,
                state.provider,
                state.model,
                state.dimensions,
                state.version,
            )
            requested_identity = (
                physical_table,
                provider,
                model,
                dimensions,
                version,
            )
            if persisted_identity != requested_identity:
                raise ValueError("vector generation identity is immutable")
        state.provider = provider
        state.model = model
        state.dimensions = dimensions
        state.version = version
        state.status = status
        state.total_documents = max(0, int(total_documents))
        state.indexed_documents = max(0, int(indexed_documents))
        state.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        self.session.add(state)
        self.session.flush()
        return state

    def get_active_generation(self, *, channel: str) -> RetrievalIndexState | None:
        return self.session.scalars(
            select(RetrievalIndexState)
            .where(
                RetrievalIndexState.channel == channel,
                RetrievalIndexState.is_active.is_(True),
            )
            .limit(1)
        ).first()

    def activate_generation(
        self,
        *,
        channel: str,
        generation: int,
        expected_active_generation: int | None,
    ) -> bool:
        target = self.session.scalars(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == channel,
                RetrievalIndexState.generation == generation,
                RetrievalIndexState.status == "ready",
            )
        ).first()
        if target is None:
            return False
        if channel == "vector":
            validate_retrieval_vector_table_name(
                target.physical_table,
                generation=target.generation,
            )
            if int(target.total_documents or 0) != int(
                target.indexed_documents or 0
            ):
                return False
        activated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        if expected_active_generation is None:
            result = self.session.execute(
                text(
                    "UPDATE retrieval_index_state "
                    "SET is_active = 1, activated_at = :activated_at, updated_at = :activated_at "
                    "WHERE channel = :channel AND generation = :generation AND status = 'ready' "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM retrieval_index_state AS active "
                    "WHERE active.channel = :channel AND active.is_active = 1"
                    ")"
                ),
                {
                    "channel": channel,
                    "generation": generation,
                    "activated_at": activated_at,
                },
            )
            self.session.expire_all()
            return int(result.rowcount or 0) == 1

        if expected_active_generation == generation:
            active = self.get_active_generation(channel=channel)
            return active is not None and active.generation == generation

        deactivated = self.session.execute(
            text(
                "UPDATE retrieval_index_state SET is_active = 0 "
                "WHERE channel = :channel AND generation = :expected_generation "
                "AND is_active = 1"
            ),
            {
                "channel": channel,
                "expected_generation": expected_active_generation,
            },
        )
        if int(deactivated.rowcount or 0) != 1:
            return False
        activated = self.session.execute(
            text(
                "UPDATE retrieval_index_state "
                "SET is_active = 1, activated_at = :activated_at, updated_at = :activated_at "
                "WHERE channel = :channel AND generation = :generation AND status = 'ready'"
            ),
            {
                "channel": channel,
                "generation": generation,
                "activated_at": activated_at,
            },
        )
        if int(activated.rowcount or 0) != 1:
            raise RuntimeError("ready retrieval generation disappeared during activation")
        self.session.expire_all()
        return True
