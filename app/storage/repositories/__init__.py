"""仓储层包：自 repositories.py 按域拆分（Phase 1a）。

对外表面与拆分前的单文件模块完全一致：所有类、数据类与常量从这里 re-export，
`from app.storage.repositories import X` 的既有用法不受影响。
"""

from app.storage.repositories._common import (
    _INELIGIBLE_DELIVERY_STATES,
    ASIA_SHANGHAI,
    _as_local_day,
    _SQLITE_VEC_MAX_K,
    _next_vector_fetch_limit,
    _initial_vector_fetch_limit,
    _vector_fetch_ceiling,
    RetrievalDocumentHit,
    RetrievalEmbeddingCoverage,
    _normalize_utc_sqlite_timestamp,
    _mentioned_user_ids,
    _normalize_optional_string_filter,
    _normalize_optional_integer_filter,
    _retrieval_source_prefilters,
    _retrieval_mention_document_ids,
    _delete_active_retrieval_vectors,
    _deactivate_memory_retrieval_documents,
    _fts_search_terms,
)
from app.storage.repositories.groups import (
    GroupRepository,
    GroupPersonaStateRepository,
    PersonaStyleExampleRepository,
    PersonaStyleSyncStateRepository,
)
from app.storage.repositories.messages import (
    UserRepository,
    MessageRepository,
    BbotListenerCacheRepository,
    SummaryRepository,
)
from app.storage.repositories.memory import (
    MemoryRepository,
    MemoryBackfillRunRepository,
)
from app.storage.repositories.episodes import (
    EpisodeRepository,
)
from app.storage.repositories.retrieval import (
    RetrievalDocumentRepository,
    RetrievalIndexStateRepository,
)
from app.storage.repositories.jobs import (
    JobRepository,
    UsageRepository,
)
from app.storage.repositories.dev import (
    DevSessionRepository,
    DevTaskRepository,
    DevTaskArtifactRepository,
)

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
    "GroupRepository",
    "GroupPersonaStateRepository",
    "PersonaStyleExampleRepository",
    "PersonaStyleSyncStateRepository",
    "UserRepository",
    "MessageRepository",
    "BbotListenerCacheRepository",
    "SummaryRepository",
    "MemoryRepository",
    "_fts_search_terms",
    "EpisodeRepository",
    "RetrievalDocumentRepository",
    "RetrievalIndexStateRepository",
    "MemoryBackfillRunRepository",
    "JobRepository",
    "UsageRepository",
    "DevSessionRepository",
    "DevTaskRepository",
    "DevTaskArtifactRepository",
]
