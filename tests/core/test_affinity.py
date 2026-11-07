"""好感度：事件加减分、限幅、半衰期衰减、定性档位与提示词行。"""

from datetime import datetime, timedelta
from types import SimpleNamespace

from app.core.affinity import (
    apply_affinity_event,
    affinity_level,
    affinity_prompt_line,
    clamp_score,
    decayed_score,
    load_affinity,
)
from app.agent.plugins.affinity import AffinityFeature
from app.agent.plugins.base import InboundContext
from app.core.group_mood import evaluate_inbound_trigger
from app.storage.db import session_scope
from app.storage.models import UserAffinityState

NOW = datetime(2026, 10, 29, 12, 0, 0)


def test_apply_clamps_and_counts(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        for _ in range(30):
            apply_affinity_event(session, group_id=1, user_id=100,
                                 delta=1.0, now=NOW)
    row = load_affinity(sqlite_engine, group_id=1, user_id=100)
    assert row.score == 10.0 and row.interactions == 30


def test_decay_toward_zero(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        apply_affinity_event(session, group_id=1, user_id=101, delta=4.0, now=NOW)
    row = load_affinity(sqlite_engine, group_id=1, user_id=101)
    assert decayed_score(row, NOW + timedelta(days=14)) == 2.0
    assert 0.0 < decayed_score(row, NOW + timedelta(days=90)) < 1.0


def test_levels_qualitative_order():
    assert affinity_level(7.0) == "老朋友"
    assert affinity_level(4.0) == "熟人"
    assert affinity_level(1.0) == "认识"
    assert affinity_level(-1.0) == "有点相处不快"
    assert clamp_score(-99) == -10.0


def _inbound_ctx(sqlite_engine, *, text, mentioned=False, user_id=900000103):
    event = SimpleNamespace(
        user_id=user_id, group_id=900000001, plain_text=text,
        mentioned_bot=mentioned, reply_to_msg_id=None,
        raw_payload={"message": [{"type": "text", "data": {"text": text}}]},
        timestamp=NOW, images=[],
    )
    messages = SimpleNamespace(
        get_by_platform_msg_id=lambda platform_msg_id: None,
        has_group_messages_from=lambda **kwargs: True,
    )
    router = SimpleNamespace(runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=900000101)))
    return InboundContext(router=router, session=None, messages=messages, event=event, now=NOW)


def test_feature_inbound_deltas(sqlite_engine):
    feature = AffinityFeature()
    feature.attach_engine(sqlite_engine)
    from app.storage.db import session_scope as scope

    # 被夸 → +1.0
    ctx = _inbound_ctx(sqlite_engine, text="你也太厉害了吧", mentioned=True)
    with scope(sqlite_engine) as session:
        ctx = InboundContext(**{**ctx.__dict__, "session": session})
        feature.on_inbound(ctx)
    # 普通点名闲聊（无情绪词）→ +0.2
    ctx = _inbound_ctx(sqlite_engine, text="在吗", mentioned=True)
    with scope(sqlite_engine) as session:
        ctx = InboundContext(**{**ctx.__dict__, "session": session})
        feature.on_inbound(ctx)
    # 被怼 → -0.3
    ctx = _inbound_ctx(sqlite_engine, text="杂鱼")
    with scope(sqlite_engine) as session:
        ctx = InboundContext(**{**ctx.__dict__, "session": session})
        feature.on_inbound(ctx)
    row = load_affinity(sqlite_engine, group_id=900000001, user_id=900000103)
    assert abs(row.score - (1.0 + 0.2 - 0.3)) < 1e-6


def test_undirected_non_exchange_does_not_change_affinity(sqlite_engine):
    feature = AffinityFeature()
    feature.attach_engine(sqlite_engine)
    ctx = _inbound_ctx(sqlite_engine, text="这个游戏好菜啊")
    messages = SimpleNamespace(
        get_by_platform_msg_id=lambda platform_msg_id: None,
        has_group_messages_from=lambda **kwargs: False,
    )
    ctx = InboundContext(**{**ctx.__dict__, "messages": messages})
    feature.on_inbound(ctx)
    assert load_affinity(sqlite_engine, group_id=900000001, user_id=900000103) is None


def test_prompt_line_requires_two_interactions(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        apply_affinity_event(session, group_id=1, user_id=102, delta=1.0, now=NOW)
    assert affinity_prompt_line(sqlite_engine, group_id=1, user_id=102, now=NOW) is None
    with session_scope(sqlite_engine) as session:
        apply_affinity_event(session, group_id=1, user_id=102, delta=1.0, now=NOW)
    line = affinity_prompt_line(sqlite_engine, group_id=1, user_id=102, now=NOW)
    assert line is not None and "认识" in line
