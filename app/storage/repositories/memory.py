"""memory 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

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

from app.storage.repositories._common import _deactivate_memory_retrieval_documents, _fts_search_terms, _normalize_utc_sqlite_timestamp

class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_id: str,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        status: str = "active",
    ) -> MemoryItem:
        memory = MemoryItem(
            scope_type=scope_type,
            scope_id=scope_id,
            subject_type=subject_type,
            subject_id=subject_id,
            memory_kind=memory_kind,
            content=content,
            importance=importance,
            confidence=confidence,
            source_msg_id=source_msg_id,
            valid_from=valid_from,
            valid_until=valid_until,
            expires_at=valid_until,
            status=status,
        )
        self.session.add(memory)
        self.session.flush()
        self._sync_memory_indexes(memory)
        return memory

    def upsert_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_id: str,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        status: str = "active",
        supersedes_id: int | None = None,
    ) -> MemoryItem:
        """Idempotently persist one extracted memory while retaining its source message."""
        stmt = select(MemoryItem).where(
            MemoryItem.scope_type == scope_type,
            MemoryItem.scope_id == scope_id,
            MemoryItem.subject_type == subject_type,
            MemoryItem.subject_id == subject_id,
            MemoryItem.memory_kind == memory_kind,
            MemoryItem.content == content,
            MemoryItem.source_msg_id == source_msg_id,
        )
        memory = self.session.scalars(stmt).first()
        if memory is None:
            memory = MemoryItem(
                scope_type=scope_type,
                scope_id=scope_id,
                subject_type=subject_type,
                subject_id=subject_id,
                memory_kind=memory_kind,
                content=content,
                source_msg_id=source_msg_id,
            )
            self.session.add(memory)
        memory.importance = importance
        memory.confidence = confidence
        memory.valid_from = valid_from
        memory.valid_until = valid_until
        memory.expires_at = valid_until
        memory.status = status
        memory.supersedes_id = supersedes_id
        self.session.flush()
        if supersedes_id is not None:
            self.mark_superseded(memory_id=supersedes_id, superseded_by_id=memory.id, valid_until=valid_from)
        self._sync_memory_indexes(memory)
        return memory

    def upsert_canonical_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        canonical_key: str,
        predicate: str,
        object_text: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_ids: list[str],
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        replace_previous: bool = False,
        replacement_predicates: Sequence[str] = (),
    ) -> MemoryItem:
        """Merge repeated evidence into one compact fact and keep its provenance."""
        normalized_sources = list(dict.fromkeys(str(item).strip() for item in source_msg_ids if str(item).strip()))
        if not canonical_key.strip():
            raise ValueError("canonical_key is required")
        memory = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.scope_type == scope_type,
                MemoryItem.scope_id == scope_id,
                MemoryItem.canonical_key == canonical_key,
                MemoryItem.status == "active",
            )
        ).first()
        previous_content: str | None = str(memory.content) if memory is not None else None
        source_at = (
            _normalize_utc_sqlite_timestamp(valid_from)
            if valid_from is not None else None
        )
        if (
            memory_kind == "current"
            and memory is not None
            and source_at is not None
            and memory.last_seen_at is not None
            and _normalize_utc_sqlite_timestamp(memory.last_seen_at) > source_at
        ):
            # A delayed episode must not roll a newer current state backwards.
            return memory
        if memory is None:
            legacy_memory = None
            if normalized_sources:
                legacy_memory = self.session.scalars(
                    select(MemoryItem).where(
                        MemoryItem.scope_type == scope_type,
                        MemoryItem.scope_id == scope_id,
                        MemoryItem.subject_id == subject_id,
                        MemoryItem.memory_kind == memory_kind,
                        MemoryItem.status == "active",
                        MemoryItem.canonical_key == "",
                        MemoryItem.source_msg_id.in_(normalized_sources),
                    )
                ).first()
            primary_source = normalized_sources[0] if normalized_sources else f"canonical:{canonical_key}"
            if legacy_memory is not None:
                memory = legacy_memory
                previous_content = str(memory.content)
                memory.canonical_key = canonical_key
                memory.predicate = predicate
                memory.object_text = object_text
                memory.content = content
                memory.source_msg_ids = normalized_sources
                memory.mention_count = max(1, len(normalized_sources))
            else:
                memory = MemoryItem(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    memory_kind=memory_kind,
                    canonical_key=canonical_key,
                    predicate=predicate,
                    object_text=object_text,
                    content=content,
                    source_msg_id=primary_source,
                    source_msg_ids=normalized_sources,
                    mention_count=max(1, len(normalized_sources)),
                    status="active",
                )
                self.session.add(memory)
        else:
            existing_sources = [str(item) for item in (memory.source_msg_ids or []) if str(item).strip()]
            merged_sources = list(dict.fromkeys([*existing_sources, *normalized_sources]))
            memory.source_msg_ids = merged_sources
            memory.mention_count = max(int(memory.mention_count or 1), len(merged_sources))
            memory.content = content
            memory.predicate = predicate
            memory.object_text = object_text
        memory.importance = max(int(memory.importance or 1), int(importance))
        memory.confidence = max(float(memory.confidence or 0.0), float(confidence))
        memory.valid_from = memory.valid_from or valid_from
        memory.valid_until = valid_until
        memory.expires_at = valid_until
        memory.last_seen_at = valid_from or shanghai_now_naive()
        self.session.flush()
        if previous_content is not None and previous_content != content:
            _deactivate_memory_retrieval_documents(
                self.session,
                memory_id=memory.id,
            )

        legacy_candidates = list(
            self.session.scalars(
                select(MemoryItem).where(
                    MemoryItem.scope_type == scope_type,
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.subject_id == subject_id,
                    MemoryItem.memory_kind == memory_kind,
                    MemoryItem.status == "active",
                    MemoryItem.canonical_key == "",
                    MemoryItem.id != memory.id,
                )
            )
        )
        normalized_object = str(object_text or "").strip().casefold()
        normalized_content = " ".join(str(content or "").casefold().split())
        for duplicate in legacy_candidates:
            duplicate_content = " ".join(str(duplicate.content or "").casefold().split())
            same_source = duplicate.source_msg_id in normalized_sources
            same_object = len(normalized_object) >= 2 and normalized_object in duplicate_content
            same_content = bool(normalized_content) and duplicate_content == normalized_content
            if same_source or same_object or same_content:
                self.mark_superseded(
                    memory_id=duplicate.id,
                    superseded_by_id=memory.id,
                    valid_until=valid_from,
                )

        if replace_previous and predicate.strip():
            equivalent_predicates = tuple(
                dict.fromkeys(
                    str(value).strip().casefold()
                    for value in (*replacement_predicates, predicate)
                    if str(value).strip()
                )
            )
            previous = list(
                self.session.scalars(
                    select(MemoryItem).where(
                        MemoryItem.scope_type == scope_type,
                        MemoryItem.scope_id == scope_id,
                        MemoryItem.subject_type == subject_type,
                        MemoryItem.subject_id == subject_id,
                        MemoryItem.memory_kind == memory_kind,
                        func.lower(MemoryItem.predicate).in_(equivalent_predicates),
                        MemoryItem.status == "active",
                        MemoryItem.id != memory.id,
                    )
                )
            )
            for older in previous:
                if (
                    memory_kind == "current"
                    and source_at is not None
                    and older.last_seen_at is not None
                    and _normalize_utc_sqlite_timestamp(older.last_seen_at) > source_at
                ):
                    continue
                self.mark_superseded(
                    memory_id=older.id,
                    superseded_by_id=memory.id,
                    valid_until=valid_from,
                )
        self._sync_memory_indexes(memory)
        return memory

    def mark_superseded(
        self,
        *,
        memory_id: int,
        superseded_by_id: int | None = None,
        valid_until: datetime | None = None,
    ) -> MemoryItem | None:
        memory = self.session.get(MemoryItem, memory_id)
        if memory is None:
            return None
        memory.status = "superseded"
        memory.superseded_by_id = superseded_by_id
        if valid_until is not None:
            memory.valid_until = valid_until
            memory.expires_at = valid_until
        self._sync_memory_indexes(memory)
        self.delete_memory_item_semantic_vectors([int(memory.id)])
        _deactivate_memory_retrieval_documents(
            self.session,
            memory_id=memory.id,
        )
        return memory

    def invalidate_canonical_memory(
        self,
        *,
        scope_id: str,
        target_canonical_key: str,
        source_msg_ids: list[str],
        valid_until: datetime,
        reason: str = "explicit_denial",
        expected_target_sha256: str | None = None,
    ) -> MemoryItem | None:
        """Retire one exact active fact and persist a non-retrievable receipt."""

        if reason not in {"explicit_denial", "manual_review_rejected"}:
            return None
        target_key = str(target_canonical_key or "").strip()
        sources = list(
            dict.fromkeys(
                str(item).strip() for item in source_msg_ids if str(item).strip()
            )
        )
        if not target_key or not sources:
            return None
        receipt_key = "invalidation|" + hashlib.sha256(
            (target_key + "\n" + "\n".join(sorted(sources))).encode("utf-8")
        ).hexdigest()
        existing_receipt = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.canonical_key == receipt_key,
                MemoryItem.memory_kind == "expired",
            )
        ).first()
        if existing_receipt is not None:
            return existing_receipt

        targets = list(
            self.session.scalars(
                select(MemoryItem).where(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.canonical_key == target_key,
                    MemoryItem.memory_kind.in_(("profile", "preference")),
                    MemoryItem.status == "active",
                )
            )
        )
        if len(targets) != 1:
            return None
        target = targets[0]
        if reason == "manual_review_rejected" and (
            not expected_target_sha256
            or hashlib.sha256(target_key.encode("utf-8")).hexdigest()
            != str(expected_target_sha256).strip().casefold()
        ):
            return None
        try:
            group_id = int(scope_id)
        except (TypeError, ValueError):
            return None
        source_rows = list(
            self.session.scalars(
                select(Message).where(
                    Message.group_id == group_id,
                    Message.platform_msg_id.in_(sources),
                )
            )
        )
        if len(source_rows) != len(sources):
            return None
        if reason == "explicit_denial" and any(
            str(row.user_id) != str(target.subject_id) for row in source_rows
        ):
            return None

        receipt = MemoryItem(
            scope_type="group",
            scope_id=scope_id,
            subject_type=target.subject_type,
            subject_id=target.subject_id,
            memory_kind="expired",
            canonical_key=receipt_key,
            predicate=reason,
            object_text=target_key,
            content=(
                "An explicit denial invalidated a prior memory fact."
                if reason == "explicit_denial"
                else "A manual evidence review rejected a prior memory fact."
            ),
            importance=1,
            confidence=1.0,
            source_msg_id=sources[0],
            source_msg_ids=sources,
            mention_count=len(sources),
            valid_from=valid_until,
            valid_until=valid_until,
            expires_at=valid_until,
            status="superseded",
            supersedes_id=target.id,
        )
        self.session.add(receipt)
        self.session.flush()
        self.mark_superseded(
            memory_id=target.id,
            superseded_by_id=receipt.id,
            valid_until=valid_until,
        )
        return receipt

    def find_unique_correction_candidate(
        self,
        *,
        scope_id: str,
        predicate: str,
        object_text: str,
        replacement_memory_id: int,
        as_of: datetime,
        subject_id: str | None = None,
    ) -> MemoryItem | None:
        instant = _normalize_utc_sqlite_timestamp(as_of)
        memory_kind = "preference" if predicate == "likes" else "taboo" if predicate == "dislikes" else ""
        if not memory_kind:
            return None
        filters = [
            MemoryItem.scope_type == "group",
            MemoryItem.scope_id == scope_id,
            MemoryItem.subject_type == "user",
            MemoryItem.memory_kind == memory_kind,
            or_(MemoryItem.predicate == predicate, MemoryItem.predicate == ""),
            MemoryItem.status == "active",
            MemoryItem.id != int(replacement_memory_id),
            or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
            or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
        ]
        if subject_id is not None:
            filters.append(MemoryItem.subject_id == subject_id)
        rows = list(self.session.scalars(select(MemoryItem).where(*filters)))
        target = " ".join(str(object_text or "").casefold().split())
        if target:
            rows = [
                row
                for row in rows
                if self.correction_objects_are_related(
                    target,
                    row.object_text if str(row.object_text or "").strip() else row.content,
                )
            ]
        return rows[0] if len(rows) == 1 else None

    @staticmethod
    def correction_objects_are_related(target: str, candidate: str) -> bool:
        normalized = " ".join(str(candidate or "").casefold().split())
        if not target or not normalized:
            return False
        if target == normalized:
            return True
        return min(len(target), len(normalized)) >= 4 and (
            target in normalized or normalized in target
        )

    def supersede_current_memories(
        self,
        *,
        scope_id: str,
        subject_id: str,
        predicate: str,
        valid_until: datetime | None,
        object_text: str = "",
    ) -> int:
        rows = list(
            self.session.scalars(
                select(MemoryItem).where(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.subject_id == subject_id,
                    MemoryItem.predicate == predicate,
                    MemoryItem.status == "active",
                )
            )
        )
        normalized_object = str(object_text or "").strip().casefold()
        if normalized_object:
            rows = [
                memory
                for memory in rows
                if str(memory.object_text or "").strip().casefold() == normalized_object
            ]
        for memory in rows:
            self.mark_superseded(memory_id=memory.id, valid_until=valid_until)
        return len(rows)

    def list_group_memories(self, scope_id: str, limit: int) -> list[MemoryItem]:
        stmt = (
            select(MemoryItem)
            .where(MemoryItem.scope_type == "group", MemoryItem.scope_id == scope_id)
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def list_group_memories_by_source_msg_ids(
        self,
        *,
        scope_id: str,
        source_msg_ids: list[str],
    ) -> list[MemoryItem]:
        identifiers = {
            str(item).strip()
            for item in source_msg_ids
            if str(item).strip()
        }
        if not identifiers:
            return []
        rows = self.session.scalars(
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
            )
            .order_by(MemoryItem.id)
        )
        return [
            memory
            for memory in rows
            if str(memory.source_msg_id) in identifiers
            or identifiers.intersection(
                str(item) for item in (memory.source_msg_ids or [])
            )
        ]

    def list_current_group_memories(
        self,
        *,
        scope_id: str,
        limit: int,
        as_of: datetime | None = None,
        subject_id: str | None = None,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []
        instant = _normalize_utc_sqlite_timestamp(as_of or datetime.now(UTC))
        filters = [
            MemoryItem.scope_type == "group",
            MemoryItem.scope_id == scope_id,
            MemoryItem.status == "active",
            or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
            or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
        ]
        if subject_id is not None:
            filters.append(MemoryItem.subject_id == subject_id)
        stmt = (
            select(MemoryItem)
            .where(*filters)
            .order_by(MemoryItem.importance.desc(), MemoryItem.confidence.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def search_group_memories_fts(
        self,
        *,
        scope_id: str,
        query: str,
        limit: int,
        as_of: datetime | None = None,
        subject_ids: Sequence[str] | None = None,
    ) -> list[MemoryItem]:
        """Return group-scoped lexical candidates with FTS5 as an accelerator.

        The SQL fallback is intentional: short Chinese terms cannot be indexed
        by every FTS tokenizer, and missing a source-backed memory is worse
        than spending a bounded query on the source-of-truth table.
        """
        if limit <= 0:
            return []
        normalized_subject_ids = tuple(
            dict.fromkeys(
                str(subject_id).strip()
                for subject_id in (subject_ids or ())
                if str(subject_id).strip()
            )
        )
        if subject_ids is not None and not normalized_subject_ids:
            return []
        terms = _fts_search_terms(query)
        if not terms:
            return []
        # OR candidates tolerate natural-language Chinese questions, whose full
        # token sequence rarely appears verbatim in a stored atomic memory.
        fts_terms = [term for term in terms if len(term) >= 3]
        ids: list[int] = []
        if fts_terms:
            match_query = " OR ".join(f'"{term}"' for term in fts_terms)
            try:
                rows = self.session.execute(
                    text(
                        "SELECT memory_id FROM memory_items_fts "
                        "WHERE memory_items_fts MATCH :query AND scope_type = 'group' AND scope_id = :scope_id "
                        "ORDER BY bm25(memory_items_fts) LIMIT :limit"
                    ),
                    {"query": match_query, "scope_id": scope_id, "limit": limit},
                )
                ids = [int(row[0]) for row in rows]
            except (SQLAlchemyError, ValueError):
                ids = []
        instant = _normalize_utc_sqlite_timestamp(as_of or datetime.now(UTC))
        active_filters = [
            MemoryItem.scope_type == "group",
            MemoryItem.scope_id == scope_id,
            MemoryItem.status == "active",
            or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
            or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
        ]
        if subject_ids is not None:
            active_filters.append(MemoryItem.subject_id.in_(normalized_subject_ids))
        memories = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.id.in_(ids),
                *active_filters,
            )
        ).all()
        by_id = {memory.id: memory for memory in memories}
        ordered = [by_id[memory_id] for memory_id in ids if memory_id in by_id]
        fallback_matches = self.session.scalars(
            select(MemoryItem)
            .where(*active_filters, or_(*(MemoryItem.content.ilike(f"%{term}%") for term in terms)))
            .order_by(MemoryItem.importance.desc(), MemoryItem.confidence.desc(), MemoryItem.id.desc())
            .limit(max(32, limit * 4))
        ).all()
        fallback_matches.sort(
            key=lambda memory: (
                sum(1 for term in terms if term in memory.content.lower()),
                memory.importance,
                memory.confidence,
                memory.id,
            ),
            reverse=True,
        )
        seen = {memory.id for memory in ordered}
        ordered.extend(memory for memory in fallback_matches if memory.id not in seen)
        return ordered[:limit]

    def search_group_memories_vector(
        self,
        *,
        scope_id: str,
        query: str,
        limit: int,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        if limit <= 0 or not str(query or "").strip():
            return []
        instant = _normalize_utc_sqlite_timestamp(as_of or datetime.now(UTC))
        group_ids = list(
            self.session.scalars(
                select(MemoryItem.id).where(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.status == "active",
                    or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                    or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
                )
            )
        )
        if not group_ids:
            return []
        id_filter = ",".join(str(int(memory_id)) for memory_id in group_ids)
        try:
            rows = self.session.execute(
                text(
                    "SELECT memory_id, vec_distance_cosine(embedding, :embedding) AS distance "
                    f"FROM memory_items_vec WHERE memory_id IN ({id_filter}) "
                    "ORDER BY distance LIMIT :limit"
                ),
                {
                    "embedding": json.dumps(hashed_text_embedding(query)),
                    "limit": max(1, min(len(group_ids), int(limit * 3))),
                },
            )
            ids = [int(row[0]) for row in rows]
        except (SQLAlchemyError, ValueError):
            return []
        if not ids:
            return []
        memories = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.id.in_(ids),
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            )
        ).all()
        by_id = {memory.id: memory for memory in memories}
        return [by_id[memory_id] for memory_id in ids if memory_id in by_id][:limit]

    def find_current_memory_for_supersession(
        self,
        *,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        replacement_content: str,
        as_of: datetime | None = None,
    ) -> MemoryItem | None:
        candidates = [
            memory
            for memory in self.list_current_group_memories(
                scope_id=scope_id,
                subject_id=subject_id,
                as_of=as_of,
                limit=20,
            )
            if memory.subject_type == subject_type and memory.memory_kind == memory_kind
        ]
        if not candidates:
            return None
        ignored_terms = {
            "计划", "取消", "决定", "改变", "现在", "之前", "打算", "不打算", "算了",
            "plan", "cancel", "decision", "decided", "planning",
        }
        replacement_body = re.sub(r"^[^:\uff1a]{1,80}[:\uff1a]\s*", "", replacement_content).strip()
        terms = [term for term in _fts_search_terms(replacement_body) if term not in ignored_terms]
        ranked = [
            (
                sum(
                    1
                    for term in terms
                    if term in re.sub(r"^[^:\uff1a]{1,80}[:\uff1a]\s*", "", memory.content).lower()
                ),
                memory.id,
                memory,
            )
            for memory in candidates
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not ranked or ranked[0][0] <= 0:
            return None
        if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
            return None
        return ranked[0][2]

    def list_group_memories_for_subject(self, *, scope_id: str, subject_id: str, limit: int) -> list[MemoryItem]:
        instant = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.subject_id == subject_id,
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            )
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def find_active_memory_by_content(
        self,
        *,
        scope_id: str,
        subject_id: str,
        memory_kind: str,
        content: str,
    ) -> MemoryItem | None:
        return self.session.scalars(
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.subject_id == subject_id,
                MemoryItem.memory_kind == memory_kind,
                MemoryItem.content == content,
                MemoryItem.status == "active",
            )
            .order_by(MemoryItem.id.desc())
            .limit(1)
        ).first()

    def upsert_memory_item_semantic_vectors(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> int:
        now = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        count = 0
        for row in rows:
            memory_id = int(row["memory_id"])
            existing = self.session.get(MemoryItemSemanticVector, memory_id)
            if existing is None:
                existing = MemoryItemSemanticVector(
                    memory_id=memory_id,
                    group_id=int(row["group_id"]),
                )
                self.session.add(existing)
            existing.group_id = int(row["group_id"])
            existing.provider = str(row.get("provider") or "")
            existing.model = str(row.get("model") or "")
            existing.dimensions = int(row.get("dimensions") or 0)
            existing.version = str(row.get("version") or "")
            existing.vector_json = str(row.get("vector_json") or "")
            existing.updated_at = now
            count += 1
        self.session.flush()
        return count

    def load_memory_item_semantic_vectors(
        self,
        memory_ids: Sequence[int],
        *,
        provider: str = "",
        model: str = "",
        dimensions: int = 0,
        version: str = "",
    ) -> dict[int, list[float]]:
        normalized_ids = tuple(
            dict.fromkeys(int(memory_id) for memory_id in memory_ids if memory_id)
        )
        if not normalized_ids:
            return {}
        rows = self.session.scalars(
            select(MemoryItemSemanticVector).where(
                MemoryItemSemanticVector.memory_id.in_(normalized_ids)
            )
        ).all()
        result: dict[int, list[float]] = {}
        for row in rows:
            if provider and row.provider != provider:
                continue
            if model and row.model != model:
                continue
            if dimensions and row.dimensions != dimensions:
                continue
            if version and row.version != version:
                continue
            try:
                vector = json.loads(row.vector_json or "[]")
            except ValueError:
                continue
            if (
                isinstance(vector, list)
                and vector
                and all(isinstance(value, (int, float)) for value in vector)
            ):
                result[int(row.memory_id)] = [float(value) for value in vector]
        return result

    def delete_memory_item_semantic_vectors(
        self,
        memory_ids: Sequence[int],
    ) -> int:
        normalized = [int(memory_id) for memory_id in memory_ids if memory_id]
        if not normalized:
            return 0
        rows = self.session.scalars(
            select(MemoryItemSemanticVector).where(
                MemoryItemSemanticVector.memory_id.in_(normalized)
            )
        ).all()
        for row in rows:
            self.session.delete(row)
        self.session.flush()
        return len(rows)

    def list_active_memory_items_for_indexing(
        self,
        *,
        after_id: int = 0,
        limit: int = 500,
        scope_id: str | None = None,
    ) -> list[MemoryItem]:
        filters = [
            MemoryItem.status == "active",
            MemoryItem.id > int(after_id),
        ]
        if scope_id is not None:
            filters.append(MemoryItem.scope_id == scope_id)
        return list(
            self.session.scalars(
                select(MemoryItem)
                .where(*filters)
                .order_by(MemoryItem.id.asc())
                .limit(max(1, int(limit)))
            )
        )

    def deactivate_memory_items(
        self,
        memory_ids: Sequence[int],
        *,
        valid_until: datetime,
    ) -> int:
        normalized = [int(memory_id) for memory_id in memory_ids if memory_id]
        if not normalized:
            return 0
        rows = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.id.in_(normalized),
                MemoryItem.status == "active",
            )
        ).all()
        for row in rows:
            row.status = "inactive"
            row.valid_until = valid_until
            row.expires_at = valid_until
            self._sync_memory_indexes(row)
            _deactivate_memory_retrieval_documents(
                self.session,
                memory_id=int(row.id),
            )
        self.delete_memory_item_semantic_vectors(
            [int(row.id) for row in rows]
        )
        self.session.flush()
        return len(rows)

    def expire_stale_memories(self, *, now: datetime) -> int:
        """Deactivate every active memory whose validity window has passed."""
        normalized_now = _normalize_utc_sqlite_timestamp(now)
        rows = list(
            self.session.scalars(
                select(MemoryItem.id).where(
                    MemoryItem.status == "active",
                    MemoryItem.valid_until.is_not(None),
                    MemoryItem.valid_until < normalized_now,
                )
            )
        )
        if not rows:
            return 0
        return self.deactivate_memory_items(
            rows,
            valid_until=normalized_now,
        )

    def expire_stale_current_memories(self, *, now: datetime) -> int:
        """Compatibility wrapper for the now kind-agnostic expiry pass."""
        return self.expire_stale_memories(now=now)

    def _sync_memory_indexes(self, memory: MemoryItem) -> None:
        self._sync_fts(memory)
        self._sync_vector(memory)

    def _sync_fts(self, memory: MemoryItem) -> None:
        try:
            self.session.execute(text("DELETE FROM memory_items_fts WHERE memory_id = :memory_id"), {"memory_id": str(memory.id)})
            if memory.status != "active":
                return
            self.session.execute(
                text(
                    "INSERT INTO memory_items_fts (content, scope_type, scope_id, memory_id) "
                    "VALUES (:content, :scope_type, :scope_id, :memory_id)"
                ),
                {
                    "content": memory.content,
                    "scope_type": memory.scope_type,
                    "scope_id": memory.scope_id,
                    "memory_id": str(memory.id),
                },
            )
        except SQLAlchemyError:
            # FTS5 is optional and must not make source-of-truth writes fail.
            return

    def _sync_vector(self, memory: MemoryItem) -> None:
        try:
            self.session.execute(text("DELETE FROM memory_items_vec WHERE memory_id = :memory_id"), {"memory_id": memory.id})
            if memory.status != "active":
                return
            self.session.execute(
                text("INSERT INTO memory_items_vec(memory_id, embedding) VALUES (:memory_id, :embedding)"),
                {
                    "memory_id": memory.id,
                    "embedding": json.dumps(hashed_text_embedding(memory.content)),
                },
            )
        except SQLAlchemyError:
            return

class MemoryBackfillRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        *,
        run_key: str,
        snapshot_watermarks: dict[str, int],
        manifest: dict[str, Any],
        segmentation_generation: str,
        compaction_generation: str,
        index_generation: str,
        created_at: datetime,
    ) -> MemoryBackfillRun:
        frozen_watermarks = {
            str(group_id): int(watermark)
            for group_id, watermark in snapshot_watermarks.items()
        }
        self.session.execute(
            text(
                "INSERT OR IGNORE INTO memory_backfill_runs ("
                "run_key, status, snapshot_watermarks_json, manifest_json, "
                "segmentation_generation, compaction_generation, index_generation, "
                "created_at, last_error_code"
                ") VALUES ("
                ":run_key, 'pending', :snapshot_watermarks_json, :manifest_json, "
                ":segmentation_generation, :compaction_generation, :index_generation, "
                ":created_at, ''"
                ")"
            ),
            {
                "run_key": run_key,
                "snapshot_watermarks_json": json.dumps(frozen_watermarks),
                "manifest_json": json.dumps(manifest),
                "segmentation_generation": segmentation_generation,
                "compaction_generation": compaction_generation,
                "index_generation": index_generation,
                "created_at": _normalize_utc_sqlite_timestamp(created_at),
            },
        )
        self.session.expire_all()
        run = self.session.scalars(
            select(MemoryBackfillRun).where(MemoryBackfillRun.run_key == run_key)
        ).first()
        if run is None:
            raise RuntimeError("backfill run upsert did not return a persisted row")
        return run

    def get_run(self, *, run_id: int) -> MemoryBackfillRun | None:
        return self.session.get(MemoryBackfillRun, run_id)

    def update_status(
        self,
        *,
        run_id: int,
        status: str,
        completed_at: datetime | None,
        last_error_code: str,
    ) -> MemoryBackfillRun | None:
        run = self.session.get(MemoryBackfillRun, run_id)
        if run is None:
            return None
        run.status = status
        run.completed_at = (
            _normalize_utc_sqlite_timestamp(completed_at)
            if completed_at is not None
            else None
        )
        run.last_error_code = str(last_error_code or "")[:96]
        self.session.add(run)
        return run
