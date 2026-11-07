"""messages 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

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

from app.storage.repositories._common import ASIA_SHANGHAI, _INELIGIBLE_DELIVERY_STATES, _as_local_day, _normalize_utc_sqlite_timestamp

class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_user(self, *, user_id: int, nickname: str, group_card: str) -> User:
        user = self.session.get(User, user_id) or User(user_id=user_id)
        user.nickname = nickname
        user.group_card = group_card
        now = shanghai_now_naive()
        user.first_seen_at = user.first_seen_at or now
        user.last_seen_at = now
        self.session.add(user)
        return user

    def get_users_by_ids(self, user_ids: list[int]) -> dict[int, User]:
        unique_ids = list(dict.fromkeys(user_ids))
        if not unique_ids:
            return {}
        stmt = select(User).where(User.user_id.in_(unique_ids))
        return {user.user_id: user for user in self.session.scalars(stmt)}

class MessageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _is_reserved_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return isinstance(raw_json, dict) and raw_json.get("delivery_state") == "reserved"

    @staticmethod
    def is_reserved_outbound(message: Message) -> bool:
        return MessageRepository._is_reserved_outbound(message)

    @staticmethod
    def is_qq_blocked_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return (
            isinstance(raw_json, dict)
            and raw_json.get("delivery_state") == "blocked"
            and raw_json.get("failure_kind") == "qq_sensitive_content"
        )

    @staticmethod
    def is_delivery_uncertain_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return (
            isinstance(raw_json, dict)
            and raw_json.get("delivery_state") == "uncertain"
            and raw_json.get("failure_kind") == "delivery_result_unknown"
        )

    @staticmethod
    def _is_unconfirmed_outbound(message: Message) -> bool:
        return (
            MessageRepository.is_qq_blocked_outbound(message)
            or MessageRepository.is_delivery_uncertain_outbound(message)
        )

    def add_group_message(
        self,
        *,
        platform_msg_id: str,
        group_id: int,
        user_id: int,
        timestamp: datetime,
        plain_text: str,
        raw_json: dict[str, Any],
        msg_type: str,
        reply_to_msg_id: str | None,
        mentioned_bot: bool,
    ) -> Message:
        self.session.flush()
        message = Message(
            platform_msg_id=platform_msg_id,
            group_id=group_id,
            user_id=user_id,
            timestamp=shanghai_naive(timestamp),
            plain_text=plain_text,
            raw_json=raw_json,
            msg_type=msg_type,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
        )
        self.session.add(message)
        return message

    def add_private_message(
        self,
        *,
        platform_msg_id: str,
        user_id: int,
        timestamp: datetime,
        plain_text: str,
        raw_json: dict[str, Any],
        msg_type: str = "text",
        reply_to_msg_id: str | None = None,
        mentioned_bot: bool = False,
    ) -> Message:
        self.session.flush()
        message = Message(
            platform_msg_id=platform_msg_id,
            group_id=None,
            user_id=user_id,
            timestamp=shanghai_naive(timestamp),
            plain_text=plain_text,
            raw_json=raw_json,
            msg_type=msg_type,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
        )
        self.session.add(message)
        return message

    def get_by_platform_msg_id(self, platform_msg_id: str) -> Message | None:
        stmt = select(Message).where(Message.platform_msg_id == platform_msg_id).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def has_group_messages_from(self, *, group_id: int, user_id: int, since) -> bool:
        """本群该用户自 since 起是否发过消息（情绪系统判断"正在对话中"用）。"""
        stmt = select(Message.id).where(
            Message.group_id == group_id,
            Message.user_id == user_id,
            Message.timestamp >= since,
        ).limit(1)
        return self.session.execute(stmt).scalar_one_or_none() is not None

    def has_private_messages_from(self, *, user_id: int, since) -> bool:
        """该用户自 since 起是否发过私聊消息（提醒系统判断"人应了"用）。"""
        stmt = select(Message.id).where(
            Message.group_id.is_(None),
            Message.user_id == user_id,
            Message.timestamp >= since,
        ).limit(1)
        return self.session.execute(stmt).scalar_one_or_none() is not None

    def list_group_messages_between(
        self,
        *,
        group_id: int,
        since,
        until,
        bot_user_id: int | None = None,
        limit: int = 200,
    ) -> list[Message]:
        """本群 (since, until] 时间窗内的消息，按时间正序（沉默学习评估插话结果用）。"""
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.timestamp > since,
                Message.timestamp <= until,
            )
            .order_by(Message.timestamp.asc(), Message.id.asc())
            .limit(max(1, limit))
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message)
            and not self._is_unconfirmed_outbound(message)
            and (bot_user_id is None or message.user_id != bot_user_id)
        ]

    def mark_group_message_deleted(
        self,
        *,
        group_id: int,
        platform_msg_id: str,
        reason: str = "group_recall",
    ) -> Message | None:
        message = self.session.scalars(
            select(Message).where(
                Message.group_id == int(group_id),
                Message.platform_msg_id == str(platform_msg_id),
            )
        ).first()
        if message is None:
            return None
        raw_json = (
            dict(message.raw_json)
            if isinstance(message.raw_json, dict)
            else {}
        )
        raw_json["delivery_state"] = "deleted"
        raw_json["deletion_reason"] = str(reason or "group_recall")[:64]
        message.raw_json = raw_json
        self.session.add(message)
        return message

    def get_group_messages_by_platform_msg_ids(
        self,
        *,
        group_id: int,
        platform_msg_ids: list[str],
    ) -> dict[str, Message]:
        identifiers = list(dict.fromkeys(str(item).strip() for item in platform_msg_ids if str(item).strip()))
        if not identifiers:
            return {}
        rows = self.session.scalars(
            select(Message).where(
                Message.group_id == int(group_id),
                Message.platform_msg_id.in_(identifiers),
            )
        )
        return {str(message.platform_msg_id): message for message in rows}

    def list_direct_group_replies(
        self,
        *,
        group_id: int,
        parent_platform_msg_ids: list[str],
        scan_limit_per_parent: int,
    ) -> list[Message]:
        """Load a bounded direct-reply scan for scoped parents in stable order.

        The reply quota is applied after the query plan's delivery, subject,
        and time eligibility checks. This larger scan bound prevents unbounded
        ORM hydration without consuming the final two-reply eligibility quota.
        """
        if scan_limit_per_parent < 1:
            raise ValueError("direct reply scan limit must be positive")
        parent_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in parent_platform_msg_ids
                if str(item).strip()
            )
        )
        if not parent_ids:
            return []
        ranked = (
            select(
                Message.id.label("message_id"),
                func.row_number()
                .over(
                    partition_by=Message.reply_to_msg_id,
                    order_by=(Message.timestamp.asc(), Message.id.asc()),
                )
                .label("reply_rank"),
            )
            .where(
                Message.group_id == int(group_id),
                Message.reply_to_msg_id.in_(parent_ids),
                ~func.coalesce(
                    func.json_extract(Message.raw_json, "$.delivery_state"),
                    "",
                ).in_(_INELIGIBLE_DELIVERY_STATES),
            )
            .subquery()
        )
        stmt = (
            select(Message)
            .join(ranked, ranked.c.message_id == Message.id)
            .where(ranked.c.reply_rank <= int(scan_limit_per_parent))
            .order_by(
                Message.reply_to_msg_id.asc(),
                Message.timestamp.asc(),
                Message.id.asc(),
            )
        )
        return list(self.session.scalars(stmt))

    def list_preceding_group_message_context(
        self,
        *,
        group_id: int,
        anchor_platform_msg_ids: list[str],
        per_anchor_limit: int = 2,
        max_gap_seconds: int = 120,
        excluded_user_ids: set[int] | None = None,
    ) -> list[Message]:
        """Load a small chronological context prefix for scoped messages.

        Anchors are queried independently so distant historical hits cannot
        become one unbounded time-range scan. The final rows are deduplicated
        after applying the per-anchor limit.
        """
        if per_anchor_limit < 1 or max_gap_seconds < 1:
            raise ValueError("context limits must be positive")
        anchor_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in anchor_platform_msg_ids
                if str(item).strip()
            )
        )
        if not anchor_ids:
            return []
        anchors = tuple(
            self.session.scalars(
                select(Message).where(
                    Message.group_id == int(group_id),
                    Message.platform_msg_id.in_(anchor_ids),
                )
            )
        )
        excluded = {int(value) for value in (excluded_user_ids or set())}
        selected: dict[int, Message] = {}
        for anchor in anchors:
            anchor_time = _normalize_utc_sqlite_timestamp(anchor.timestamp)
            stmt = (
                select(Message)
                .where(
                    Message.group_id == int(group_id),
                    Message.timestamp
                    >= anchor_time - timedelta(seconds=max_gap_seconds),
                    or_(
                        Message.timestamp < anchor_time,
                        and_(
                            Message.timestamp == anchor_time,
                            Message.id < int(anchor.id),
                        ),
                    ),
                    ~func.coalesce(
                        func.json_extract(Message.raw_json, "$.delivery_state"),
                        "",
                    ).in_(_INELIGIBLE_DELIVERY_STATES),
                )
                .order_by(Message.timestamp.desc(), Message.id.desc())
                .limit(int(per_anchor_limit))
            )
            if excluded:
                stmt = stmt.where(Message.user_id.not_in(excluded))
            for row in self.session.scalars(stmt):
                selected.setdefault(int(row.id), row)
        return sorted(
            selected.values(),
            key=lambda row: (row.timestamp, int(row.id)),
        )

    def list_bounded_same_episode_message_context(
        self,
        *,
        group_id: int,
        anchor_platform_msg_ids: list[str],
        per_anchor_limit: int = 12,
        max_gap_seconds: int = 900,
        excluded_user_ids: set[int] | None = None,
    ) -> list[Message]:
        """Load a bounded before/after window from each anchor's own episode.

        This loader intentionally returns raw scoped rows only.  The pure
        clarification-thread resolver decides whether any three rows form an
        unambiguous statement/question/short-answer unit.
        """

        if per_anchor_limit < 1 or max_gap_seconds < 1:
            raise ValueError("context limits must be positive")
        anchor_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in anchor_platform_msg_ids
                if str(item).strip()
            )
        )
        if not anchor_ids:
            return []
        anchor_memberships = tuple(
            self.session.execute(
                select(Message, EpisodeMessage.episode_id)
                .join(
                    EpisodeMessage,
                    (EpisodeMessage.message_id == Message.id)
                    & (EpisodeMessage.group_id == Message.group_id),
                )
                .where(
                    Message.group_id == int(group_id),
                    EpisodeMessage.group_id == int(group_id),
                    Message.platform_msg_id.in_(anchor_ids),
                )
            )
        )
        anchors_by_source: dict[str, list[tuple[Message, int]]] = {}
        for anchor, episode_id in anchor_memberships:
            anchors_by_source.setdefault(str(anchor.platform_msg_id), []).append(
                (anchor, int(episode_id))
            )
        excluded = {int(value) for value in (excluded_user_ids or set())}
        selected: dict[int, Message] = {}
        for source_id in anchor_ids:
            memberships = anchors_by_source.get(source_id, [])
            # A source associated with zero or multiple episodes has no safe
            # conversational boundary for implicit clarification linking.
            if len(memberships) != 1:
                continue
            anchor, episode_id = memberships[0]
            anchor_time = _normalize_utc_sqlite_timestamp(anchor.timestamp)
            stmt = (
                select(Message)
                .join(
                    EpisodeMessage,
                    (EpisodeMessage.message_id == Message.id)
                    & (EpisodeMessage.group_id == Message.group_id),
                )
                .where(
                    Message.group_id == int(group_id),
                    EpisodeMessage.group_id == int(group_id),
                    EpisodeMessage.episode_id == int(episode_id),
                    Message.timestamp
                    >= anchor_time - timedelta(seconds=max_gap_seconds),
                    Message.timestamp
                    <= anchor_time + timedelta(seconds=max_gap_seconds),
                    ~func.coalesce(
                        func.json_extract(Message.raw_json, "$.delivery_state"),
                        "",
                    ).in_(_INELIGIBLE_DELIVERY_STATES),
                )
                .order_by(
                    func.abs(
                        func.julianday(Message.timestamp)
                        - func.julianday(anchor_time)
                    ).asc(),
                    Message.id.asc(),
                )
                .limit(int(per_anchor_limit) * 2 + 1)
            )
            if excluded:
                stmt = stmt.where(Message.user_id.not_in(excluded))
            for row in self.session.scalars(stmt):
                if self.is_qq_blocked_outbound(row):
                    continue
                selected.setdefault(int(row.id), row)
        return sorted(
            selected.values(),
            key=lambda row: (row.timestamp, int(row.id)),
        )

    def is_late_group_message(
        self,
        *,
        group_id: int,
        message_id: int,
        timestamp: datetime,
    ) -> bool:
        normalized_timestamp = _normalize_utc_sqlite_timestamp(timestamp)
        stmt = (
            select(Message.id)
            .where(
                Message.group_id == int(group_id),
                Message.id < int(message_id),
                Message.timestamp > normalized_timestamp,
            )
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none() is not None

    def list_recent_group_messages(self, *, group_id: int, limit: int) -> list[Message]:
        from app.storage.context_resets import automatic_context_floor
        stmt = (
            select(Message)
            .where(Message.group_id == group_id,
                   Message.id > automatic_context_floor(self.session, f'group:{group_id}'))
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        return self._recent_message_pages(stmt, limit=limit)

    def _recent_message_pages(self, stmt, *, limit: int, skip_unconfirmed: bool = False,
                              skip_qq_blocked: bool = False, by_id: bool = False) -> list[Message]:
        # ORM scalars() eagerly materializes the entire result before its first
        # yield. A Python break after 24 rows therefore decodes years of JSON
        # on every dispatch guard. Keyset pages bound both SQL rows and ORM
        # materialization while filling past reserved/unconfirmed outbound rows.
        if limit <= 0:
            return []
        page_size = min(limit, 128)
        selected = []
        observed = 0
        cursor = None
        while observed < limit:
            page_stmt = stmt.limit(page_size)
            if cursor is not None:
                timestamp, identifier = cursor
                page_stmt = page_stmt.where(Message.id < identifier if by_id else
                    or_(Message.timestamp < timestamp,
                        and_(Message.timestamp == timestamp, Message.id < identifier)))
            page = list(self.session.scalars(page_stmt))
            if not page:
                break
            for message in page:
                if (self._is_reserved_outbound(message)
                        or skip_unconfirmed and self._is_unconfirmed_outbound(message)
                        or skip_qq_blocked and self.is_qq_blocked_outbound(message)):
                    continue
                observed += 1
                if not message.context_excluded:
                    selected.append(message)
                if observed >= limit:
                    break
            cursor = (page[-1].timestamp, page[-1].id)
            if len(page) < page_size:
                break
        return list(reversed(selected))

    def list_recent_group_messages_for_summarization(self, *, group_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        return self._recent_message_pages(stmt, limit=limit, skip_unconfirmed=True)

    def list_group_messages_chronological(
        self,
        *,
        group_id: int,
        exclude_platform_msg_id: str | None = None,
    ) -> list[Message]:
        """Return every delivered group message in its original order."""
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message) and message.platform_msg_id != exclude_platform_msg_id
        ]

    def count_group_messages(self, group_id: int) -> int:
        stmt = select(func.count()).select_from(Message).where(Message.group_id == group_id)
        return self.session.scalar(stmt) or 0

    def count_group_inbound_messages(self, *, group_id: int, bot_user_id: int) -> int:
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(
                Message.group_id == group_id,
                Message.user_id != bot_user_id,
                text(
                    "(json_extract(messages.raw_json, '$.delivery_state') IS NULL "
                    "OR json_extract(messages.raw_json, '$.delivery_state') <> 'reserved')"
                ),
            )
        )
        return int(self.session.scalar(stmt) or 0)

    def list_group_messages_for_day(
        self,
        *,
        group_id: int,
        day,
        excluded_user_ids: set[int] | None = None,
    ) -> list[Message]:
        excluded = {int(user_id) for user_id in (excluded_user_ids or set())}
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.id.asc())
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if _as_local_day(message.timestamp) == day
            and message.user_id not in excluded
            and not self._is_reserved_outbound(message)
            and not self._is_unconfirmed_outbound(message)
        ]

    def list_group_ids(self) -> list[int]:
        stmt = select(Message.group_id).where(Message.group_id.is_not(None)).distinct().order_by(Message.group_id.asc())
        return [int(group_id) for group_id in self.session.scalars(stmt) if group_id is not None]

    def list_recent_group_message_windows(
        self,
        *,
        group_id: int,
        batch_size: int,
        limit_windows: int,
        excluded_user_ids: set[int] | None = None,
    ) -> list[list[Message]]:
        excluded = {int(user_id) for user_id in (excluded_user_ids or set())}
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.id.asc())
        )
        rows = [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message)
            and not self._is_unconfirmed_outbound(message)
            and message.user_id not in excluded
        ]
        windows = [
            rows[index : index + batch_size]
            for index in range(0, len(rows), batch_size)
            if len(rows[index : index + batch_size]) == batch_size
        ]
        return windows[-max(1, limit_windows) :]

    def list_recent_group_inbound_messages(
        self,
        *,
        group_id: int,
        bot_user_id: int,
        limit: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id != bot_user_id)
            .order_by(Message.id.desc())
        )
        return self._recent_message_pages(stmt, limit=max(1, limit), skip_unconfirmed=True, by_id=True)

    def list_group_messages_by_id_range(
        self,
        *,
        group_id: int,
        start_id: int,
        end_id: int,
        limit: int = 200,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.id >= start_id,
                Message.id <= end_id,
            )
            .order_by(Message.id.asc())
            .limit(max(1, limit))
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message) and not self._is_unconfirmed_outbound(message)
        ]

    def list_recent_group_messages_for_user(self, *, group_id: int, user_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == user_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        return self._recent_message_pages(stmt, limit=limit, skip_unconfirmed=True)

    def list_recent_group_messages_for_user_since(
        self,
        *,
        group_id: int,
        user_id: int,
        since: datetime,
        limit: int,
    ) -> list[Message]:
        since = _normalize_utc_sqlite_timestamp(since)
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.user_id == user_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        return self._recent_message_pages(stmt, limit=limit)

    def list_recent_private_messages_for_user_since(
        self,
        *,
        user_id: int,
        since: datetime,
        limit: int,
    ) -> list[Message]:
        since = _normalize_utc_sqlite_timestamp(since)
        stmt = (
            select(Message)
            .where(
                Message.group_id.is_(None),
                Message.user_id == user_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        return self._recent_message_pages(stmt, limit=limit)

    def list_group_messages_matching_terms(
        self,
        *,
        group_id: int,
        terms: list[str],
        exclude_platform_msg_ids: set[str],
        limit: int,
    ) -> list[Message]:
        normalized_terms = list(dict.fromkeys(term.strip().lower() for term in terms if len(term.strip()) >= 2))
        if limit <= 0 or not normalized_terms:
            return []
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.plain_text.is_not(None),
                or_(*(Message.plain_text.ilike(f"%{term}%") for term in normalized_terms)),
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
            .limit(max(limit * 4, limit))
        )
        matched: list[Message] = []
        for message in self.session.scalars(stmt):
            if message.platform_msg_id in exclude_platform_msg_ids or self._is_reserved_outbound(message):
                continue
            matched.append(message)
            if len(matched) >= limit:
                break
        return matched

    def list_recent_group_user_ids(self, *, group_id: int, limit: int) -> list[int]:
        latest_message_at = func.max(Message.timestamp).label("latest_message_at")
        stmt = (
            select(Message.user_id, latest_message_at)
            .where(Message.group_id == group_id)
            .group_by(Message.user_id)
            .order_by(latest_message_at.desc())
            .limit(limit)
        )
        return [int(user_id) for user_id, _latest in self.session.execute(stmt)]

    def list_recent_group_member_messages(
        self,
        *,
        group_id: int | None,
        limit: int | None,
    ) -> list[Message]:
        """Return latest eligible sender snapshots, optionally across all groups."""

        if limit is not None and limit <= 0:
            return []
        rank_partition = (
            (Message.group_id, Message.user_id)
            if group_id is None
            else Message.user_id
        )
        filters = (
            [Message.group_id.is_not(None)]
            if group_id is None
            else [Message.group_id == int(group_id)]
        )
        ranked = (
            select(
                Message.id.label("message_id"),
                func.row_number()
                .over(
                    partition_by=rank_partition,
                    order_by=(Message.timestamp.desc(), Message.id.desc()),
                )
                .label("member_rank"),
            )
            .where(
                *filters,
                ~func.coalesce(
                    func.json_extract(Message.raw_json, "$.delivery_state"),
                    "",
                ).in_(_INELIGIBLE_DELIVERY_STATES),
            )
            .subquery()
        )
        stmt = (
            select(Message)
            .join(ranked, ranked.c.message_id == Message.id)
            .where(ranked.c.member_rank == 1)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(int(limit))
        return list(self.session.scalars(stmt))

    def last_bot_reply_at(self, *, group_id: int, bot_user_id: int,
                          exclude_owner_user_id: int | None = None) -> datetime | None:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == bot_user_id,
                   func.coalesce(func.json_extract(Message.raw_json, '$.delivery_state'), '') != 'deleted')
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        if exclude_owner_user_id is not None:
            from app.storage.agent_effects import AgentToolEffect
            owner_effects=select(AgentToolEffect.action_id).where(
                AgentToolEffect.account_id==bot_user_id,AgentToolEffect.scope.startswith('owner:'))
            owner_sources=select(Message.platform_msg_id).where(Message.group_id==group_id,
                Message.user_id==exclude_owner_user_id)
            stmt=stmt.where(func.coalesce(func.json_extract(Message.raw_json,'$.social_action_id'),'').not_in(owner_effects),
                func.coalesce(Message.reply_to_msg_id,'').not_in(owner_sources))
        rows = self._recent_message_pages(stmt, limit=1, skip_qq_blocked=True)
        if not rows:
            return None
        timestamp = rows[0].timestamp
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=ASIA_SHANGHAI).astimezone(UTC)
        return timestamp

class BbotListenerCacheRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_entry(
        self,
        *,
        group_id: int,
        platform: str,
        external_id: str,
        canonical_name: str,
        aliases: list[str],
        source: str,
        updated_at: datetime,
    ) -> BbotListenerCacheEntry:
        stmt = (
            select(BbotListenerCacheEntry)
            .where(
                BbotListenerCacheEntry.group_id == group_id,
                BbotListenerCacheEntry.platform == platform,
                BbotListenerCacheEntry.external_id == external_id,
            )
            .limit(1)
        )
        entry = self.session.execute(stmt).scalar_one_or_none()
        if entry is None:
            entry = BbotListenerCacheEntry(
                group_id=group_id,
                platform=platform,
                external_id=external_id,
            )
        entry.canonical_name = canonical_name
        entry.aliases_json = aliases
        entry.source = source
        entry.updated_at = updated_at
        self.session.add(entry)
        return entry

    def find_best_match(self, *, group_id: int, platform: str, query: str) -> BbotListenerCacheEntry | None:
        stmt = (
            select(BbotListenerCacheEntry)
            .where(
                BbotListenerCacheEntry.group_id == group_id,
                BbotListenerCacheEntry.platform == platform,
            )
            .order_by(BbotListenerCacheEntry.updated_at.desc(), BbotListenerCacheEntry.id.desc())
        )
        normalized_query = self._normalize(query)
        if not normalized_query:
            return None

        best_entry = None
        best_score = -1
        for entry in self.session.scalars(stmt):
            names = [str(entry.canonical_name or "")] + [str(alias) for alias in (entry.aliases_json or [])]
            normalized_candidates = [candidate for candidate in (self._normalize(name) for name in names) if candidate]
            score = self._score_candidates(normalized_query, normalized_candidates)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score <= 0:
            return None
        return best_entry

    def _score_candidates(self, query: str, candidates: list[str]) -> int:
        score = 0
        for candidate in candidates:
            if query == candidate:
                score = max(score, 100)
            elif query in candidate:
                score = max(score, 80)
            elif candidate in query:
                score = max(score, 60)
        return score

    def _normalize(self, value: str) -> str:
        lowered = value.strip().lower()
        return "".join(character for character in lowered if character.isalnum() or "\u4e00" <= character <= "\u9fff")

class SummaryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_summary(
        self,
        *,
        scope_type: str,
        scope_id: str,
        summary_level: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        source_count: int,
    ) -> Summary:
        summary = Summary(
            scope_type=scope_type,
            scope_id=scope_id,
            summary_level=summary_level,
            start_at=start_at,
            end_at=end_at,
            content=content,
            source_count=source_count,
        )
        self.session.add(summary)
        return summary

    def upsert_summary(
        self,
        *,
        scope_type: str,
        scope_id: str,
        summary_level: str,
        summary_key: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        source_count: int,
        source_start_msg_id: str | None = None,
        source_end_msg_id: str | None = None,
        source_summary_ids: list[int] | None = None,
        status: str = "active",
    ) -> Summary:
        """Idempotently replace a recursive summary identified by its stable key."""
        if not summary_key.strip():
            raise ValueError("summary_key is required for upsert_summary")
        stmt = select(Summary).where(
            Summary.scope_type == scope_type,
            Summary.scope_id == scope_id,
            Summary.summary_level == summary_level,
            Summary.summary_key == summary_key,
        )
        summary = self.session.scalars(stmt).first()
        if summary is None:
            summary = Summary(
                scope_type=scope_type,
                scope_id=scope_id,
                summary_level=summary_level,
                summary_key=summary_key,
            )
            self.session.add(summary)
        summary.start_at = start_at
        summary.end_at = end_at
        summary.content = content
        summary.source_count = source_count
        summary.source_start_msg_id = source_start_msg_id
        summary.source_end_msg_id = source_end_msg_id
        summary.source_summary_ids = list(source_summary_ids or [])
        summary.status = status
        return summary

    def list_recent_group_summaries(self, scope_id: str, limit: int) -> list[str]:
        stmt = (
            select(Summary)
            .where(Summary.scope_type == "group", Summary.scope_id == scope_id)
            .order_by(Summary.end_at.desc(), Summary.id.desc())
            .limit(limit)
        )
        summaries = [summary.content for summary in self.session.scalars(stmt)]
        return list(reversed(summaries))

    def list_group_summaries(
        self,
        *,
        scope_id: str,
        limit: int,
        summary_levels: Sequence[str] | None = None,
        summary_key: str | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        require_source_ids: bool = False,
    ) -> list[Summary]:
        if limit <= 0:
            return []
        filters = [Summary.scope_type == "group", Summary.scope_id == scope_id, Summary.status == "active"]
        if summary_levels:
            filters.append(Summary.summary_level.in_(summary_levels))
        if summary_key is not None:
            filters.append(Summary.summary_key == summary_key)
        if require_source_ids:
            filters.append(
                or_(
                    Summary.source_start_msg_id.is_not(None),
                    Summary.source_end_msg_id.is_not(None),
                )
            )
        if start_at is not None:
            filters.append(
                Summary.end_at >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            filters.append(
                Summary.start_at < _normalize_utc_sqlite_timestamp(end_at)
            )
        stmt = select(Summary).where(*filters).order_by(Summary.end_at.desc(), Summary.id.desc()).limit(limit)
        return list(reversed(list(self.session.scalars(stmt))))
