"""仓储层共享的模块级常量、辅助函数与数据类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

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


__all__ = [
    "_INELIGIBLE_DELIVERY_STATES",
    "ASIA_SHANGHAI",
    "_as_local_day",
    "_SQLITE_VEC_MAX_K",
    "_next_vector_fetch_limit",
    "_initial_vector_fetch_limit",
    "_vector_fetch_ceiling",
    "RetrievalDocumentHit",
    "RetrievalEmbeddingCoverage",
    "_normalize_utc_sqlite_timestamp",
    "_mentioned_user_ids",
    "_normalize_optional_string_filter",
    "_normalize_optional_integer_filter",
    "_retrieval_source_prefilters",
    "_retrieval_mention_document_ids",
    "_delete_active_retrieval_vectors",
    "_deactivate_memory_retrieval_documents",
    "_fts_search_terms",
]

_INELIGIBLE_DELIVERY_STATES = (
    "reserved",
    "blocked",
    "uncertain",
    "deleted",
)

ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")

def _as_local_day(value: datetime) -> date:
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(ASIA_SHANGHAI).date()

_SQLITE_VEC_MAX_K = 4096

def _next_vector_fetch_limit(
    current: int,
    *,
    requested: int,
    available: int,
) -> int:
    searchable = min(max(0, int(available)), _SQLITE_VEC_MAX_K)
    if searchable <= 0:
        return 0
    return min(
        searchable,
        max(int(current) + max(1, int(requested)), int(current) * 2),
    )

def _initial_vector_fetch_limit(
    *,
    requested: int,
    available: int,
    has_post_filters: bool,
) -> int:
    searchable = min(max(0, int(available)), _SQLITE_VEC_MAX_K)
    if searchable <= 0:
        return 0
    if has_post_filters:
        # sqlite-vec can partition by group but sparse positive filters such
        # as speaker/time/mention are validated against canonical source rows
        # afterwards. Starting at the cap avoids repeated KNN scans for them.
        return searchable
    return min(max(1, int(requested)), searchable)

def _vector_fetch_ceiling(
    *,
    requested: int,
    available: int,
    has_sparse_post_filters: bool,
    has_exclusion_filter: bool,
) -> int:
    searchable = min(max(0, int(available)), _SQLITE_VEC_MAX_K)
    if searchable <= 0:
        return 0
    if has_sparse_post_filters:
        return searchable
    if has_exclusion_filter:
        return min(
            searchable,
            max(max(1, int(requested)) * 4, max(1, int(requested)) + 32),
        )
    return min(max(1, int(requested)), searchable)

@dataclass(frozen=True, slots=True)
class RetrievalDocumentHit:
    document_id: int
    group_id: int
    document_kind: str
    episode_id: int | None
    source_msg_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    score: float
    lexical_exact: bool = False

@dataclass(frozen=True, slots=True)
class RetrievalEmbeddingCoverage:
    total_documents: int
    ready_documents: int
    failed_documents: int

    @property
    def coverage(self) -> float:
        if self.total_documents == 0:
            return 1.0
        return self.ready_documents / self.total_documents

def _normalize_utc_sqlite_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(ASIA_SHANGHAI).replace(tzinfo=None)

def _mentioned_user_ids(raw_json: object) -> frozenset[str]:
    if not isinstance(raw_json, dict):
        return frozenset()
    segments = raw_json.get("message", raw_json.get("raw_message"))
    if not isinstance(segments, list):
        return frozenset()
    mentioned: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict) or str(segment.get("type") or "") != "at":
            continue
        data = segment.get("data")
        if not isinstance(data, dict):
            continue
        for key in ("qq", "uin", "target"):
            value = data.get(key)
            if value is None or isinstance(value, bool):
                continue
            normalized = str(value).strip()
            if normalized:
                mentioned.add(normalized)
    return frozenset(mentioned)

def _normalize_optional_string_filter(
    values: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )

def _normalize_optional_integer_filter(
    values: Sequence[str] | None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(
        dict.fromkeys(
            int(value)
            for value in values
            if str(value).strip().lstrip("-").isdigit()
        )
    )

def _retrieval_source_prefilters(
    *,
    group_id: int,
    speaker_ids: Sequence[str] | None,
    excluded_speaker_ids: Sequence[str] | None = None,
) -> list[Any]:
    """SQL prefilters applied before ranking/limit on document channels."""

    unsafe_document_ids = (
        select(RetrievalDocumentMessage.document_id)
        .join(
            Message,
            (Message.id == RetrievalDocumentMessage.message_id)
            & (Message.group_id == RetrievalDocumentMessage.group_id),
        )
        .where(
            RetrievalDocumentMessage.group_id == int(group_id),
            Message.group_id == int(group_id),
            func.json_extract(Message.raw_json, "$.delivery_state").in_(
                _INELIGIBLE_DELIVERY_STATES
            ),
        )
    )
    filters: list[Any] = [
        RetrievalDocument.id.not_in(unsafe_document_ids),
    ]
    normalized_speakers = _normalize_optional_integer_filter(speaker_ids)
    if normalized_speakers is not None:
        if not normalized_speakers:
            filters.append(RetrievalDocument.id < 0)
        else:
            mismatched_document_ids = (
                select(RetrievalDocumentMessage.document_id)
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.group_id == int(group_id),
                    Message.group_id == int(group_id),
                    Message.user_id.not_in(normalized_speakers),
                )
            )
            filters.append(
                RetrievalDocument.id.not_in(mismatched_document_ids)
            )
    normalized_excluded_speakers = _normalize_optional_integer_filter(
        excluded_speaker_ids
    )
    if normalized_excluded_speakers:
        excluded_document_ids = (
            select(RetrievalDocumentMessage.document_id)
            .join(
                Message,
                (Message.id == RetrievalDocumentMessage.message_id)
                & (Message.group_id == RetrievalDocumentMessage.group_id),
            )
            .where(
                RetrievalDocumentMessage.group_id == int(group_id),
                Message.group_id == int(group_id),
                Message.user_id.in_(normalized_excluded_speakers),
            )
        )
        filters.append(RetrievalDocument.id.not_in(excluded_document_ids))
    return filters

def _retrieval_mention_document_ids(
    *,
    group_id: int,
    mentioned_user_ids: Sequence[str],
):
    """Select documents backed by a real OneBot ``at`` segment."""

    normalized_mentions = _normalize_optional_string_filter(mentioned_user_ids)
    if not normalized_mentions:
        return select(RetrievalDocument.id).where(RetrievalDocument.id < 0)
    segments = func.json_each(
        Message.raw_json,
        "$.message",
    ).table_valued("value").alias("mention_segment")
    return (
        select(RetrievalDocumentMessage.document_id)
        .join(
            Message,
            (Message.id == RetrievalDocumentMessage.message_id)
            & (Message.group_id == RetrievalDocumentMessage.group_id),
        )
        .join(segments, true())
        .where(
            RetrievalDocumentMessage.group_id == int(group_id),
            Message.group_id == int(group_id),
            func.json_extract(segments.c.value, "$.type") == "at",
            cast(
                func.json_extract(segments.c.value, "$.data.qq"),
                String,
            ).in_(normalized_mentions),
        )
    )

def _delete_active_retrieval_vectors(
    session: Session,
    *,
    document_ids: list[int],
) -> None:
    if not document_ids:
        return
    active_states = list(
        session.scalars(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == "vector",
                RetrievalIndexState.is_active.is_(True),
            )
        )
    )
    for state in active_states:
        physical_table = validate_retrieval_vector_table_name(
            state.physical_table,
            generation=state.generation,
        )
        try:
            session.execute(
                text(
                    f"DELETE FROM {physical_table} "
                    "WHERE document_id IN :document_ids"
                ).bindparams(bindparam("document_ids", expanding=True)),
                {
                    "document_ids": tuple(
                        int(document_id) for document_id in document_ids
                    )
                },
            )
        except SQLAlchemyError:
            # The active metadata can outlive optional extension availability
            # in a process. Canonical status still prevents use by fallback
            # channels; vector cleanup can be retried when sqlite-vec returns.
            continue

def _deactivate_memory_retrieval_documents(
    session: Session,
    *,
    memory_id: int,
    keep_document_id: int | None = None,
) -> int:
    filters = [
        RetrievalDocument.document_kind == "memory",
        RetrievalDocument.source_table == "memory_items",
        RetrievalDocument.source_id == str(int(memory_id)),
        RetrievalDocument.status == "active",
    ]
    if keep_document_id is not None:
        filters.append(RetrievalDocument.id != int(keep_document_id))
    documents = list(session.scalars(select(RetrievalDocument).where(*filters)))
    if not documents:
        return 0
    now = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
    for document in documents:
        document.status = "inactive"
        document.embedding_status = "stale"
        document.updated_at = now
        session.add(document)
        try:
            session.execute(
                text("DELETE FROM retrieval_documents_fts WHERE document_id = :document_id"),
                {"document_id": str(document.id)},
            )
        except SQLAlchemyError:
            pass
    _delete_active_retrieval_vectors(
        session,
        document_ids=[document.id for document in documents],
    )
    return len(documents)

def _fts_search_terms(query: str) -> list[str]:
    """Derive FTS-safe lexical candidates for both Latin and Chinese queries."""
    normalized = str(query or "").lower().replace('"', " ")
    terms: list[str] = []
    for chinese_run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        terms.append(chinese_run)
        terms.extend(chinese_run[index : index + 2] for index in range(len(chinese_run) - 1))
        terms.extend(chinese_run[index : index + 3] for index in range(len(chinese_run) - 2))
    terms.extend(re.findall(r"[a-z0-9_]{2,}", normalized))
    return list(dict.fromkeys(term for term in terms if term))[:16]
