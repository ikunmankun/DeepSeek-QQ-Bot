"""语音插件回归：MoodFeature 签名、VoiceFeature 提示行节奏、私聊语音工具权限。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.agent.plugins.builtin import MoodFeature
from app.agent.plugins.voice import VoiceFeature
from app.agent.tool_agent import AgentTool, ToolOutcome


async def _unused(args):
    return ToolOutcome({})


def test_mood_prompt_lines_accepts_target_user_id():
    # 回归：registry 统一带 target_user_id 调 prompt_lines，
    # MoodFeature 缺参数会让情绪提示词整轮丢失（日志 plugin_prompt_lines_failed）。
    feature = MoodFeature()
    feature._engine = None
    assert feature.prompt_lines(kind="group", group_id=1,
        now=datetime(2026, 11, 5, 12, 0), observing=False, target_user_id=42) == ()
    assert feature.prompt_lines(kind="group", group_id=1,
        now=datetime(2026, 11, 5, 12, 0), observing=False) == ()


class _FakeTts:
    pass


def _voice_feature() -> VoiceFeature:
    feature = VoiceFeature()
    feature._settings = object()
    feature._tts = _FakeTts()
    return feature


def test_voice_prompt_lines_inject_every_eight_direct_replies(monkeypatch):
    # 概率主动开口提示关掉（random 恒 > 概率），只验证工具提醒的固定节奏。
    monkeypatch.setattr("random.random", lambda: 0.99)
    feature = _voice_feature()
    group_id = 138890665
    now = datetime(2026, 11, 5, 12, 0)
    injected = []
    for turn in range(1, 17):
        lines = feature.prompt_lines(kind="group", group_id=group_id,
            now=now, observing=False, target_user_id=42)
        injected.append(bool(lines))
    # 注入窗口固定在 count % 8 == 2，即第 2、10 轮两次。
    assert injected == [turn in (2, 10) for turn in range(1, 17)]


def test_voice_prompt_lines_silent_for_observation_turns_and_other_groups():
    feature = _voice_feature()
    now = datetime(2026, 11, 5, 12, 0)
    for _ in range(10):
        assert feature.prompt_lines(kind="group", group_id=1,
            now=now, observing=True, target_user_id=42) == ()
        assert feature.prompt_lines(kind="private", group_id=0,
            now=now, observing=False) == ()
    # 未配置 TTS 时整体静默。
    disabled = VoiceFeature()
    disabled._tts = None
    assert disabled.prompt_lines(kind="group", group_id=1,
        now=now, observing=False) == ()


def test_voice_proactive_nudge_fires_then_respects_cooldown(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.0)  # 概率必中
    feature = _voice_feature()
    group_id = 713687091
    now = datetime(2026, 11, 5, 12, 0)
    first = feature.prompt_lines(kind="group", group_id=group_id,
        now=now, observing=False, target_user_id=42)
    assert any("speak" in line for line in first)
    assert group_id in feature._last_nudge
    # 冷却期内不再提示。
    within = feature.prompt_lines(kind="group", group_id=group_id,
        now=now + timedelta(hours=1), observing=False, target_user_id=42)
    assert not any("直接用 speak" in line for line in within)
    # 冷却过后再次放行。
    later = feature.prompt_lines(kind="group", group_id=group_id,
        now=now + timedelta(hours=3), observing=False, target_user_id=42)
    assert any("直接用 speak" in line for line in later)


def test_voice_proactive_nudge_never_fires_when_probability_misses(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.99)
    feature = _voice_feature()
    now = datetime(2026, 11, 5, 12, 0)
    for turn in range(1, 12):
        lines = feature.prompt_lines(kind="group", group_id=1,
            now=now, observing=False, target_user_id=42)
        assert not any("直接用 speak" in line for line in lines)


def test_voice_proactive_nudge_blocked_by_exhausted_quota(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.0)
    feature = _voice_feature()
    from app.core.time_utils import shanghai_now_naive

    feature._usage_day = f"{shanghai_now_naive():%Y-%m-%d}"
    feature._usage_count = int(getattr(feature._settings, "tts_daily_limit", 20) or 20)
    lines = feature.prompt_lines(kind="group", group_id=1,
        now=shanghai_now_naive(), observing=False, target_user_id=42)
    assert not any("直接用 speak" in line for line in lines)
    assert 1 not in feature._last_nudge


def test_voice_proactive_nudge_silent_on_observation_turns(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.0)
    feature = _voice_feature()
    now = datetime(2026, 11, 5, 12, 0)
    for _ in range(5):
        lines = feature.prompt_lines(kind="group", group_id=1,
            now=now, observing=True, target_user_id=42)
        assert lines == ()
    assert feature._last_nudge == {}


@pytest.mark.asyncio
async def test_speak_private_survives_nonowner_restriction():
    from app.agent.owner_actions import restrict_directed_tools

    tools = [AgentTool("speak_private", "语音条", {}, _unused, "write"),
        AgentTool("set_nickname", "改名", {}, _unused, "write")]
    restricted = restrict_directed_tools(tools, owner=False)
    speak = next(tool for tool in restricted if tool.name == "speak_private")
    nickname = next(tool for tool in restricted if tool.name == "set_nickname")
    # 语音条对非主人可用（就是一次普通投递），管理工具保持封锁。
    assert speak.authorization is None or speak.authorization() is not False
    assert (await speak.handler({})).status == "ok"
    assert nickname.authorization() is False
    assert (await nickname.handler({})).status == "blocked"
