"""模糊记忆检索单元：多通道检索、RRF 合并、LLM 扩词、无 LLM/无向量降级。"""
from datetime import datetime

import pytest

from app.core.fuzzy_recall import (
    FuzzyRecallService, expand_recall_queries, fuzzy_recall,
)
from app.storage.db import build_engine, create_all
from app.storage.models import Group, MemoryItem, Message, User


@pytest.fixture()
def db(tmp_path):
    engine = build_engine(tmp_path / "recall.db")
    create_all(engine)
    from sqlalchemy import select
    from app.storage.db import session_scope
    with session_scope(engine) as session:
        session.add(Group(group_id=101, group_name="测试群"))
        session.add(Group(group_id=102, group_name="另一个群"))
        session.add(User(user_id=42, nickname="主人"))
        session.add(User(user_id=43, nickname="群友"))
        session.flush()  # 模型间无 relationship，父表必须先落
        session.add(Message(platform_msg_id="g1", group_id=101, user_id=43,
            timestamp=datetime(2026, 8, 12, 21, 0, 0), raw_json={},
            plain_text="最近在肝星穹列车，白毛那个角色强度离谱"))
        session.add(Message(platform_msg_id="g2", group_id=102, user_id=43,
            timestamp=datetime(2026, 8, 13, 21, 0, 0), raw_json={},
            plain_text="那个画风超好的独立游戏叫空洞骑士"))
        session.add(Message(platform_msg_id="p1", group_id=None, user_id=42,
            timestamp=datetime(2026, 9, 1, 23, 0, 0), raw_json={},
            plain_text="白毛红眼的游戏想不起来了，急死"))
        session.add(MemoryItem(scope_type="group", scope_id="101", subject_type="member",
            subject_id="43", memory_kind="fact", canonical_key="game:starrail",
            source_msg_id="g1",
            content="群友43在玩星穹列车，最喜欢白毛角色", importance=3, confidence=0.8))
    return engine


class _FakeLLM:
    def __init__(self, text=""):
        self._text = text
        self.calls = []

    def generate_text(self, prompt_lines, **kwargs):
        self.calls.append(list(prompt_lines))
        return self._text


def test_expand_parses_keywords_and_web_hints():
    keywords, web_hints = expand_recall_queries(
        "描述", llm=_FakeLLM("星穹列车\n崩铁\n白毛\n猜:崩坏:星穹铁道、星穹铁道"))
    assert keywords == ["星穹列车", "崩铁", "白毛"]
    assert web_hints == ["崩坏:星穹铁道", "星穹铁道"]


def test_expand_tolerates_llm_failure():
    class _Boom:
        def generate_text(self, *args, **kwargs):
            raise RuntimeError("boom")
    assert expand_recall_queries("描述", llm=_Boom()) == ([], [])
    assert expand_recall_queries("描述", llm=None) == ([], [])


def test_finds_message_via_description_substring(db):
    result = fuzzy_recall(db, description="白毛那个角色", group_ids=(101,))
    sources = {(hit.source, hit.content) for hit in result.hits}
    assert any(source == "group_message" and "星穹列车" in content for source, content in sources)
    hit = next(hit for hit in result.hits if hit.source == "group_message")
    assert hit.when == "2026-08-12"
    assert hit.group_id == 101


def test_llm_keywords_unlock_memory_and_messages(db):
    result = fuzzy_recall(db, description="有个游戏只记得白毛角色",
        group_ids=(101,), llm=_FakeLLM("星穹列车\n猜:崩坏:星穹铁道"))
    kinds = {hit.source for hit in result.hits}
    assert "memory" in kinds
    assert "group_message" in kinds
    memory_hit = next(hit for hit in result.hits if hit.source == "memory")
    assert "星穹列车" in memory_hit.content
    assert result.web_hints == ["崩坏:星穹铁道"]


def test_private_channel_only_when_user_given(db):
    scoped = fuzzy_recall(db, description="白毛红眼", group_ids=(101,))
    assert all(hit.source != "private_message" for hit in scoped.hits)
    with_private = fuzzy_recall(db, description="白毛红眼", group_ids=(101,), private_user_id=42)
    private_hits = [hit for hit in with_private.hits if hit.source == "private_message"]
    assert private_hits and "急死" in private_hits[0].content


def test_service_scopes_groups_or_all(db):
    service = FuzzyRecallService(engine=db, settings=None, llm_client=None)
    own_group = service.recall("白毛", group_ids=(101,))
    assert all(hit["group_id"] in (101, None) for hit in own_group["hits"])
    everything = service.recall("空洞骑士", group_ids=None)
    assert any(hit["group_id"] == 102 for hit in everything["hits"])
    none_group = service.recall("星穹列车", group_ids=())
    assert all(hit["source"] != "group_message" for hit in none_group["hits"])


def test_empty_description_returns_empty(db):
    assert fuzzy_recall(db, description="  ", group_ids=(101,)).hits == []


def test_limit_caps_hits(db):
    from app.storage.db import session_scope
    from app.storage.models import Message as M
    with session_scope(db) as session:
        for index in range(6):
            session.add(M(platform_msg_id=f"m{index}", group_id=101, user_id=43,
                timestamp=datetime(2026, 8, 20, 12, index, 0), raw_json={},
                plain_text="白毛角色评测系列第" + str(index) + "期"))
    result = fuzzy_recall(db, description="白毛角色", group_ids=(101,), limit=3)
    assert len(result.hits) <= 3


def test_broken_embedding_provider_degrades_gracefully(db):
    class _Broken:
        available = True
        identity = None

        def embed_query(self, query):
            raise RuntimeError("model missing")

    result = fuzzy_recall(db, description="白毛角色", group_ids=(101,),
        embedding_provider=_Broken())
    assert result.hits  # 向量挂了，FTS/LIKE 通道照常出结果
