"""groups 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

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

class GroupRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_group(self, *, group_id: int, group_name: str, enabled: bool, speak_enabled: bool) -> Group:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.group_name = group_name
        group.enabled = enabled
        group.speak_enabled = speak_enabled
        self.session.add(group)
        return group

    def set_speak_enabled(self, group_id: int, value: bool) -> None:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.speak_enabled = value
        self.session.add(group)

    def set_enabled(self, group_id: int, value: bool) -> None:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.enabled = value
        self.session.add(group)

    def get_group(self, group_id: int) -> Group | None:
        return self.session.get(Group, group_id)

class GroupPersonaStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def load_all(self) -> dict[int, GroupPersonaState]:
        return {
            int(state.group_id): state
            for state in self.session.scalars(select(GroupPersonaState))
        }

    def get(self, group_id: int) -> GroupPersonaState | None:
        return self.session.get(GroupPersonaState, int(group_id))

    def set_persona_key(self, group_id: int, persona_key: str) -> GroupPersonaState:
        state = self.get(group_id) or GroupPersonaState(group_id=int(group_id))
        state.persona_key = persona_key
        state.updated_at = shanghai_now_naive()
        self.session.add(state)
        return state

    def set_card_snapshot(self, group_id: int, card: str | None) -> GroupPersonaState:
        state = self.get(group_id) or GroupPersonaState(group_id=int(group_id))
        state.card_snapshot = card
        state.updated_at = shanghai_now_naive()
        self.session.add(state)
        return state

    def set_avatar_snapshot(self, group_id: int, avatar: str | None) -> GroupPersonaState:
        state = self.get(group_id) or GroupPersonaState(group_id=int(group_id))
        state.avatar_snapshot = avatar
        state.updated_at = shanghai_now_naive()
        self.session.add(state)
        return state

class PersonaStyleExampleRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def load_active(
        self,
        *,
        user_id: int,
        group_id: int | None = None,
        limit: int = 600,
    ) -> list[PersonaStyleExample]:
        filters = [PersonaStyleExample.user_id == int(user_id)]
        if group_id is not None:
            filters.append(PersonaStyleExample.group_id == int(group_id))
        return list(
            self.session.scalars(
                select(PersonaStyleExample)
                .where(*filters)
                .order_by(PersonaStyleExample.timestamp.desc())
                .limit(limit)
            )
        )

    def load_since(
        self,
        *,
        user_id: int,
        group_id: int | None = None,
        since,
        limit: int = 100,
    ) -> list[PersonaStyleExample]:
        """Samples strictly newer than ``since`` (no overlap with the
        previous refresh window)."""

        filters = [
            PersonaStyleExample.user_id == int(user_id),
            PersonaStyleExample.timestamp > since,
        ]
        if group_id is not None:
            filters.append(PersonaStyleExample.group_id == int(group_id))
        return list(
            self.session.scalars(
                select(PersonaStyleExample)
                .where(*filters)
                .order_by(PersonaStyleExample.timestamp.asc())
                .limit(limit)
            )
        )

    def insert_many(self, rows: list[dict]) -> int:
        inserted = 0
        for row in rows:
            if self.session.get(PersonaStyleExample, row["msg_id"]) is not None:
                continue
            self.session.add(PersonaStyleExample(**row))
            inserted += 1
        return inserted

    def upsert_many(self, rows: list[dict]) -> tuple[int, int]:
        """Insert new samples and enrich overlap rows when context arrives later."""

        inserted = 0
        updated = 0
        mutable_fields = (
            "text",
            "context_before",
            "context_after",
            "reply_target",
            "timestamp",
        )
        for row in rows:
            existing = self.session.get(PersonaStyleExample, row["msg_id"])
            if existing is None:
                self.session.add(PersonaStyleExample(**row))
                inserted += 1
                continue
            if (
                int(existing.user_id) != int(row.get("user_id") or 0)
                or int(existing.group_id) != int(row.get("group_id") or 0)
            ):
                # The legacy table has a global msg_id key. Never overwrite a
                # row belonging to another member/group until a composite-key
                # migration is performed.
                continue
            changed = False
            for field_name in mutable_fields:
                if field_name not in row:
                    continue
                value = row[field_name]
                if getattr(existing, field_name) != value:
                    setattr(existing, field_name, value)
                    changed = True
            if changed:
                self.session.add(existing)
                updated += 1
        return inserted, updated

    def trim_to(
        self,
        *,
        user_id: int,
        group_id: int | None = None,
        keep: int,
    ) -> int:
        filters = [PersonaStyleExample.user_id == int(user_id)]
        if group_id is not None:
            filters.append(PersonaStyleExample.group_id == int(group_id))
        ids = [
            example.msg_id
            for example in self.session.scalars(
                select(PersonaStyleExample)
                .where(*filters)
                .order_by(PersonaStyleExample.timestamp.desc())
                .offset(keep)
            )
        ]
        for msg_id in ids:
            self.session.delete(self.session.get(PersonaStyleExample, msg_id))
        return len(ids)

class PersonaStyleSyncStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, *, group_id: int, user_id: int) -> PersonaStyleSyncState | None:
        return self.session.get(
            PersonaStyleSyncState,
            (int(group_id), int(user_id)),
        )

    def set_watermark(self, *, group_id: int, user_id: int, last_msg_id: str, new_count: int) -> PersonaStyleSyncState:
        state = self.get(group_id=group_id, user_id=user_id) or PersonaStyleSyncState(
            group_id=int(group_id), user_id=int(user_id)
        )
        state.last_msg_id = last_msg_id
        state.new_since_refresh = int(state.new_since_refresh or 0) + max(
            0, int(new_count)
        )
        state.updated_at = shanghai_now_naive()
        self.session.add(state)
        return state

    def mark_refreshed(self, *, group_id: int, user_id: int, when: datetime) -> PersonaStyleSyncState:
        state = self.get(group_id=group_id, user_id=user_id) or PersonaStyleSyncState(
            group_id=int(group_id), user_id=int(user_id)
        )
        state.last_refresh_at = when
        state.new_since_refresh = 0
        state.updated_at = shanghai_now_naive()
        self.session.add(state)
        return state
