"""人脑式记忆衰减：强度公式、淡忘/归档判定、排序沉底与后台归档服务。"""

from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import select

from app.core.memory_decay import (
    ARCHIVE_HALF_LIVES,
    half_life_days,
    is_archivable,
    is_faded,
    is_immutable,
    is_promoted,
    strength,
)
from app.core.memory_decay_service import MemoryDecayService
from app.core.memory_fact_ranking import rank_member_facts
from app.storage.db import session_scope
from app.storage.models import MemoryItem
from app.storage.repositories import MemoryRepository

NOW = datetime(2026, 10, 25, 12, 0, 0)


def make_memory(**overrides):
    defaults = dict(
        memory_kind="fact",
        content="随便一条记忆",
        predicate="",
        object_text="",
        importance=1,
        mention_count=1,
        last_seen_at=NOW,
        valid_from=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_fresh_memory_is_not_faded_and_strength_decays_by_half_life():
    memory = make_memory(memory_kind="preference")  # 基础半衰期 180 天
    assert strength(memory, NOW + timedelta(days=180)) == 0.5
    assert not is_faded(memory, NOW + timedelta(days=180))
    assert is_faded(memory, NOW + timedelta(days=181))


def test_mention_count_and_importance_extend_half_life():
    plain = make_memory(memory_kind="event")  # 60 天
    assert half_life_days(plain) == 60.0
    # 间隔效应：提 1 次不强化，2 次 ×2，4 次 ×4，8 次 ×5
    often = make_memory(memory_kind="event", mention_count=8)
    assert half_life_days(often) == 240.0
    assert not is_faded(often, NOW + timedelta(days=200))
    assert is_faded(plain, NOW + timedelta(days=200))
    # 深刻（importance 高）的记忆衰减慢
    deep = make_memory(memory_kind="event", importance=3)
    assert not is_faded(deep, NOW + timedelta(days=170))


def test_immutable_kinds_never_fade_or_archive():
    for memory in (
        make_memory(memory_kind="taboo", last_seen_at=NOW - timedelta(days=4000)),
        make_memory(memory_kind="relationship", last_seen_at=NOW - timedelta(days=4000)),
        make_memory(memory_kind="profile", last_seen_at=NOW - timedelta(days=4000)),
        make_memory(memory_kind="preference", predicate="addressing rule: 叫我老板",
                    last_seen_at=NOW - timedelta(days=4000)),
    ):
        assert is_immutable(memory)
        assert not is_faded(memory, NOW)
        assert not is_archivable(memory, NOW)


def test_archivable_only_after_archive_half_lives():
    memory = make_memory(memory_kind="event")  # 半衰期 60 天
    boundary = 60.0 * ARCHIVE_HALF_LIVES
    assert not is_archivable(memory, NOW + timedelta(days=boundary))
    assert is_archivable(memory, NOW + timedelta(days=boundary + 1))


def test_memory_without_timestamps_is_treated_as_fresh():
    memory = make_memory(last_seen_at=None, valid_from=None)
    assert strength(memory, NOW) == 1.0
    assert not is_faded(memory, NOW)
    assert not is_archivable(memory, NOW)


def test_promoted_memory_becomes_immutable():
    # 反复（>=3 次）跨天（>=2 天）被提起的长期认知 → 晋升为身份级记忆
    promoted = make_memory(mention_count=3,
                           valid_from=NOW - timedelta(days=5),
                           last_seen_at=NOW - timedelta(days=1))
    assert is_promoted(promoted)
    assert is_immutable(promoted)
    assert not is_faded(promoted, NOW + timedelta(days=4000))
    assert not is_archivable(promoted, NOW + timedelta(days=4000))


def test_promotion_requires_recurring_mentions_across_days():
    # 提了 3 次但都在同一天：一次刷屏不算稳定认知
    same_day = make_memory(mention_count=3,
                           valid_from=NOW - timedelta(hours=2), last_seen_at=NOW)
    assert not is_promoted(same_day)
    # 跨了 10 天但只提了 2 次：还不够"反复"
    few_mentions = make_memory(mention_count=2,
                               valid_from=NOW - timedelta(days=10), last_seen_at=NOW)
    assert not is_promoted(few_mentions)
    # 短命类型（current/plan）提起再多也只是过眼云烟
    transient = make_memory(memory_kind="current", mention_count=5,
                            valid_from=NOW - timedelta(days=10), last_seen_at=NOW)
    plan = make_memory(memory_kind="plan", mention_count=5,
                       valid_from=NOW - timedelta(days=10), last_seen_at=NOW)
    assert not is_promoted(transient)
    assert not is_promoted(plan)
    # 没有提及元数据的旧记忆不参与晋升（宁可正常淡忘，不凭空豁免）
    legacy = make_memory(mention_count=1, valid_from=None)
    assert not is_promoted(legacy)
    # taboo 本来就走 kind 豁免，不属于晋升通道
    assert not is_promoted(make_memory(memory_kind="taboo", mention_count=5,
                                       valid_from=NOW - timedelta(days=10), last_seen_at=NOW))


def test_promotion_never_fades_even_when_very_old():
    # 晋升后哪怕 2000 天没提起也不再淡忘/归档
    old_promoted = make_memory(memory_kind="preference", mention_count=4,
                               valid_from=NOW - timedelta(days=2010),
                               last_seen_at=NOW - timedelta(days=2000))
    assert is_promoted(old_promoted)
    assert not is_faded(old_promoted, NOW)
    assert not is_archivable(old_promoted, NOW)


def _fact(**overrides):
    defaults = dict(content="", predicate="", object_text="", memory_kind="fact",
                    importance=1, confidence=0.5, id=0, mention_count=1, last_seen_at=NOW)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_stale_unmatched_fact_sinks_below_fresh_but_matched_fact_stays():
    query_features = ("钓鱼",)
    fresh_matched = _fact(id=1, content="上周刚去水库钓鱼", last_seen_at=NOW - timedelta(days=3))
    stale_important = _fact(id=2, content="他旧乐队巡演的事", memory_kind="event", importance=5,
                            last_seen_at=NOW - timedelta(days=400))
    fresh_unmatched = _fact(id=3, content="最近在学日语", last_seen_at=NOW - timedelta(days=1))
    ranked = rank_member_facts(
        [stale_important, fresh_unmatched, fresh_matched],
        query_features=query_features, limit=10, now=NOW,
    )
    # 淡忘且未命中的事实沉底：importance 再高也排到新鲜事实后面
    assert ranked[0] is fresh_matched
    assert ranked[1] is fresh_unmatched
    assert ranked[-1] is stale_important

    # 但显式问起（查询词命中淡忘记忆内容）时不受衰减影响：
    old_preference = _fact(id=4, content="他钓鱼最喜欢用蚯蚓", memory_kind="preference",
                           importance=1, last_seen_at=NOW - timedelta(days=400))
    ranked = rank_member_facts(
        [old_preference, fresh_unmatched],
        query_features=query_features, limit=10, now=NOW,
    )
    assert ranked[0] is old_preference


def _memory_row(**overrides):
    defaults = dict(
        scope_type="group", scope_id="900000001", subject_type="user", subject_id="900000103",
        memory_kind="event", canonical_key="", predicate="", object_text="",
        content="一年前聊过的旧事", importance=1, confidence=0.5,
        source_msg_id="legacy-1", mention_count=1,
        last_seen_at=NOW - timedelta(days=800), status="active",
    )
    defaults.update(overrides)
    return MemoryItem(**defaults)


def test_decay_service_archives_stale_memories_and_spares_immune_ones(sqlite_engine, tmp_path):
    with session_scope(sqlite_engine) as session:
        stale = _memory_row()
        immune = _memory_row(memory_kind="taboo", content="他的禁忌话题")
        fresh = _memory_row(content="昨天刚聊的新事", last_seen_at=NOW - timedelta(days=2))
        session.add_all([stale, immune, fresh])
    service = MemoryDecayService(engine=sqlite_engine,
                                 settings=SimpleNamespace(log_dir=tmp_path))
    assert service._tick() == 1
    with session_scope(sqlite_engine) as session:
        rows = {row.content: row for row in session.scalars(select(MemoryItem))}
    assert rows["一年前聊过的旧事"].status == "inactive"
    assert rows["一年前聊过的旧事"].valid_until is not None
    assert rows["他的禁忌话题"].status == "active"
    assert rows["昨天刚聊的新事"].status == "active"
    # 再跑一轮不会重复处理
    assert service._tick() == 0


def test_decay_service_spares_promoted_memory(sqlite_engine, tmp_path):
    # 同样 2000 天没人提起，但反复跨天提过的记忆已晋升为身份级，不再归档
    with session_scope(sqlite_engine) as session:
        promoted = _memory_row(
            content="他反复说过的事",
            mention_count=3,
            valid_from=NOW - timedelta(days=2010),
            last_seen_at=NOW - timedelta(days=2000),
        )
        stale = _memory_row()
        session.add_all([promoted, stale])
    service = MemoryDecayService(engine=sqlite_engine,
                                 settings=SimpleNamespace(log_dir=tmp_path))
    assert service._tick() == 1
    with session_scope(sqlite_engine) as session:
        rows = {row.content: row for row in session.scalars(select(MemoryItem))}
    assert rows["他反复说过的事"].status == "active"
    assert rows["一年前聊过的旧事"].status == "inactive"
