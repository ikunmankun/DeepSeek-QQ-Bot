"""戳一戳/欢迎插件与聊天总结工具。"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.agent.plugins.base import FeatureRegistry, NoticeContext
from app.agent.plugins.interactions import PokeFeature, WelcomeFeature
from app.agent.plugins.summary import ChatSummaryFeature
from app.core.interactions import (
    InteractionState,
    build_poke_prompt,
    group_increase_info,
    poke_payload_info,
)
from app.storage.db import session_scope
from app.storage.models import Summary

# 注意：日上限按"自然日"计数，NOW 必须取真实的今天——写死日期会让
# 测试过一天就翻车（固定 10-30 的版本在 10-31 就挂了）。
NOW = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)


def _poke_payload(*, target=900000101, user=900000103, sub="poke"):
    return {"post_type": "notice", "notice_type": "notify", "sub_type": sub,
            "group_id": 900000001, "user_id": user, "target_id": target}


def test_poke_payload_recognition():
    assert poke_payload_info(_poke_payload(), bot_qq=900000101) == (900000001, 900000103)
    # 戳的不是 bot
    assert poke_payload_info(_poke_payload(target=900000103), bot_qq=900000101) is None
    # bot 戳别人（自己的动作回显）
    assert poke_payload_info(_poke_payload(user=900000101), bot_qq=900000101) is None
    # 普通群荣誉通知之类的不是戳
    assert poke_payload_info(_poke_payload(sub="honor"), bot_qq=900000101) is None


def test_group_increase_recognition():
    join = {"post_type": "notice", "notice_type": "group_increase", "sub_type": "approve",
            "group_id": 900000001, "user_id": 900000103}
    assert group_increase_info(join, bot_qq=900000101) == (900000001, 900000103)
    # 新人是 bot 自己
    assert group_increase_info({**join, "user_id": 900000101}, bot_qq=900000101) is None
    # 退群事件不触发欢迎
    assert group_increase_info({**join, "sub_type": "kick"}, bot_qq=900000101) is None


def test_interaction_state_cooldown_and_daily_cap():
    state = InteractionState()
    assert state.poke_allowed(1, NOW)
    state.record_poke_react(1, NOW)
    assert not state.poke_allowed(1, NOW + timedelta(seconds=10))
    assert state.poke_allowed(1, NOW + timedelta(seconds=61))
    for i in range(9):
        state.record_poke_react(1, NOW + timedelta(minutes=i + 2))
    late = NOW + timedelta(minutes=11)
    assert not state.poke_allowed(1, late)          # 每日 10 次已满
    assert state.poke_allowed(2, late)              # 别的群独立计数
    # 跨天重置
    assert state.poke_allowed(1, late + timedelta(days=1))


def test_welcome_cooldown():
    state = InteractionState()
    assert state.welcome_allowed(1, NOW)
    state.record_welcome(1, NOW)
    assert not state.welcome_allowed(1, NOW + timedelta(seconds=5))
    assert state.welcome_allowed(1, NOW + timedelta(seconds=31))


class _FakeLlm:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_text(self, prompt_lines, **kwargs):
        self.calls.append(list(prompt_lines))
        return self.reply


class _FakeSender:
    def __init__(self):
        self.texts = []
        self.pokes = []

    async def send_group_text(self, outbound):
        self.texts.append(outbound.text)

    async def send_group_poke(self, *, group_id, user_id):
        self.pokes.append((group_id, user_id))


def _notice_ctx(router, payload):
    return NoticeContext(router=router, payload=payload, now=NOW)


def test_poke_feature_reacts_and_sometimes_pokes_back(monkeypatch):
    monkeypatch.setattr("app.agent.plugins.interactions.should_poke_back", lambda seed: True)
    llm = _FakeLlm("戳什么戳，说事")
    sender = _FakeSender()
    router = SimpleNamespace(
        runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=900000101),
                                persona={"identity": "大肥鱼，傲娇"}),
        llm_client=llm, sender=sender)

    feature = PokeFeature()
    registry = FeatureRegistry([feature])

    async def main():
        registry.dispatch_notice(_notice_ctx(router, _poke_payload()))
        await asyncio.sleep(0)  # 让 deliver 任务跑完

    asyncio.run(main())
    assert sender.texts == ["戳什么戳，说事"]
    assert sender.pokes == [(900000001, 900000103)]
    # 提示词带人设和戳人者信息
    assert "大肥鱼" in llm.calls[0][0]
    assert not any("900000103" in line for line in llm.calls[0])  # 号码不进提示词
    assert any("一位群友" in line for line in llm.calls[0])  # 查不到名片时的中性称呼


def test_poke_feature_silenced_by_daily_cap(monkeypatch):
    sender = _FakeSender()
    llm = _FakeLlm("哼")
    router = SimpleNamespace(
        runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=900000101),
                                persona={"identity": "x"}),
        llm_client=llm, sender=sender)
    feature = PokeFeature()
    now = NOW
    for _ in range(10):
        feature.state.record_poke_react(900000001, now)
    ctx = _notice_ctx(router, _poke_payload())
    feature.on_notice(ctx)
    assert sender.texts == [] and llm.calls == []


def test_welcome_feature_greets_join():
    llm = _FakeLlm("来了？从哪刷到这个群的")
    sender = _FakeSender()
    router = SimpleNamespace(
        runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=900000101),
                                persona={"identity": "大肥鱼"}),
        llm_client=llm, sender=sender)
    feature = WelcomeFeature()
    registry = FeatureRegistry([feature])

    join = {"post_type": "notice", "notice_type": "group_increase", "sub_type": "approve",
            "group_id": 900000001, "user_id": 900000103}

    async def main():
        registry.dispatch_notice(_notice_ctx(router, join))
        await asyncio.sleep(0)

    asyncio.run(main())
    assert sender.texts == ["[@qq:900000103] 来了？从哪刷到这个群的"]


def test_summary_tool_returns_recent_summaries(sqlite_engine):
    # 工具按真实时钟取 24h 窗口，种数据必须相对 NOW（今天）构造，
    # 写死日期的版本过了窗口就翻车（10-30 的版本在 10-31 挂过）。
    now = NOW
    with session_scope(sqlite_engine) as session:
        session.add(Summary(
            scope_type="group", scope_id="900000001", summary_level="episode",
            start_at=now - timedelta(hours=3), end_at=now - timedelta(hours=1),
            content="大家聊了猫猫和新表情包", source_count=12, status="active"))

    feature = ChatSummaryFeature()
    turn = SimpleNamespace(
        router=SimpleNamespace(engine=sqlite_engine),
        job=SimpleNamespace(scope_id=900000001), guard=None, store=None,
        blocking=None, interactive=True, now=NOW)

    class Guard:
        def require_valid(self):
            return None

    turn.guard = Guard()
    tool = feature.build_tools(turn)[0]
    result = asyncio.run(tool.handler({}))
    assert result.status == "ok"
    assert any("猫猫" in entry for entry in result.data["entries"])


def test_summary_tool_empty_window(sqlite_engine):
    feature = ChatSummaryFeature()
    turn = SimpleNamespace(
        router=SimpleNamespace(engine=sqlite_engine),
        job=SimpleNamespace(scope_id=1), guard=None, store=None,
        blocking=None, interactive=True, now=NOW)

    class Guard:
        def require_valid(self):
            return None

    turn.guard = Guard()
    tool = feature.build_tools(turn)[0]
    result = asyncio.run(tool.handler({}))
    assert result.status == "ok" and result.data["entries"] == []
