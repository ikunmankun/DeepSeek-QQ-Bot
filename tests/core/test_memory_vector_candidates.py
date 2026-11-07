"""事实向量候选通道：query embedding 对群内真向量暴力余弦补候选。"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.runtime.memory_composition import _cosine_similarity, _semantic_vector_candidates
from app.storage.db import session_scope
from app.storage.models import MemoryItem
from app.storage.repositories import MemoryRepository


class _Identity:
    provider = "fake"
    model = "fake-model"
    version = "1"
    dimensions = 2


class _FakeEmbedding:
    """'白饭'→[1,0]，其他→[0,1]；available 可开关。"""

    identity = _Identity()
    available = True

    def __init__(self, available: bool = True):
        self.available = available

    def embed_query(self, query: str):
        return [1.0, 0.0] if "白饭" in str(query) else [0.0, 1.0]


def _memory_row(session, group_id: int, content: str, *, mention: int = 1) -> MemoryItem:
    row = MemoryItem(scope_type="group", scope_id=str(group_id),
        subject_type="group", subject_id=str(group_id),
        memory_kind="fact", content=content, importance=3, confidence=0.8,
        source_msg_id="m-1", mention_count=mention,
        valid_from=datetime(2026, 11, 1))
    session.add(row)
    session.flush()
    return row


def test_cosine_similarity_basics():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], []) == 0.0
    assert _cosine_similarity([], [1.0, 0.0]) == 0.0


def test_semantic_vector_candidates_returns_relevant_only(sqlite_engine):
    import json

    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        rice = _memory_row(session, 123, "群友管小町叫大白饭，梗已经传开")
        unrelated = _memory_row(session, 123, "群主养了一只橘猫")
        memories.upsert_memory_item_semantic_vectors([
            {"memory_id": rice.id, "group_id": 123,
             "vector_json": json.dumps([1.0, 0.0]),
             "provider": "fake", "model": "fake-model", "dimensions": 2, "version": "1"},
            {"memory_id": unrelated.id, "group_id": 123,
             "vector_json": json.dumps([0.0, 1.0]),
             "provider": "fake", "model": "fake-model", "dimensions": 2, "version": "1"},
        ])
    provider = _FakeEmbedding()
    with session_scope(sqlite_engine) as session:
        rows = _semantic_vector_candidates(
            session, MemoryRepository(session), provider,
            group_id=123, query="白饭 梗", exclude_ids=set(), limit=10)
    ids = {row.id for row in rows}
    assert rice.id in ids and unrelated.id not in ids


def test_semantic_vector_candidates_unavailable_provider(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        rows = _semantic_vector_candidates(
            session, MemoryRepository(session), _FakeEmbedding(available=False),
            group_id=123, query="随便", exclude_ids=set(), limit=10)
    assert rows == []
