"""日记 read_recent、read_diary 的 owner 门、群黑话学习。"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.agent.owner_actions import OWNER_CONTROL_TOOLS, restrict_directed_tools
from app.agent.tool_agent import AgentTool
from app.core.diary import DiaryWriter
from app.core.slang import SlangLearner
from app.storage.db import session_scope
from app.storage.models import GroupSlangTerm, Message


def test_read_recent_spans_days_and_caps(tmp_path):
    # read_recent 的"今天"取上海钟面（shanghai_now_naive），写日记的日期
    # 必须用同一时钟相对构造——用 datetime.now() 在 UTC 容器里会差出一天
    # （固定 10-30 的版本翻车过，datetime.now() 版本在容器里也翻车过）。
    from app.core.time_utils import shanghai_now_naive

    today = shanghai_now_naive().replace(hour=9, minute=30, second=0, microsecond=0)
    earlier = today - timedelta(days=2)
    writer = DiaryWriter(tmp_path)
    writer.write(text="第一天的事", now=earlier.replace(hour=21, minute=0))
    writer.write(text="今天的事", now=today)
    content = writer.read_recent()
    assert "【今天】" in content and "今天的事" in content
    assert f"{earlier:%m-%d}" in content and "第一天的事" in content
    # max_chars=1 时什么都装不下 → 空串
    assert writer.read_recent(max_chars=1) == ""


def test_read_diary_is_owner_control_tool(tmp_path):
    import asyncio

    from app.agent.plugins.builtin import DiaryFeature
    from app.agent.tool_agent import ToolOutcome

    assert "read_diary" in OWNER_CONTROL_TOOLS

    class Guard:
        def require_valid(self):
            return None

    async def unused(args):
        return ToolOutcome({})

    feature = DiaryFeature()
    turn = SimpleNamespace(
        router=SimpleNamespace(runtime=SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path))),
        job=SimpleNamespace(scope_id=900000001), guard=Guard(), store=None,
        blocking=None, interactive=True, now=datetime(2026, 10, 30, 12, 0))
    tools = feature.build_tools(turn)
    assert {t.name for t in tools} == {"write_diary", "read_diary"}

    # 非主人交互轮：read_diary 被 restrict_directed_tools 拒掉
    restricted = restrict_directed_tools(
        [AgentTool("read_diary", "d", {}, unused, "read"),
         AgentTool("send_group", "s", {}, unused, "write")],
        owner=False, allow_group_images=False)
    denied = next(t for t in restricted if t.name == "read_diary")
    assert asyncio.run(denied.handler({})).status == "blocked"


def test_slang_learner_filters_stopwords_and_ranks(sqlite_engine):
    now = datetime(2026, 10, 30, 12, 0)
    from app.storage.repositories import GroupRepository, UserRepository

    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=900000001, group_name="g",
                                              enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=900000103, nickname="tester",
                                            group_card="")
    with session_scope(sqlite_engine) as session:
        for i in range(8):
            session.add(Message(
                platform_msg_id=f"slang-{i}", group_id=900000001, user_id=900000103,
                timestamp=now - timedelta(hours=i), plain_text="绝绝子，真的绝绝子，太绝绝子了",
                raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False))
        session.add(Message(
            platform_msg_id="slang-noise", group_id=900000001, user_id=900000103,
            timestamp=now, plain_text="这个那个的知道了", raw_json={}, msg_type="text",
            reply_to_msg_id=None, mentioned_bot=False))
    learner = SlangLearner(sqlite_engine)
    learned = learner.learn(group_ids=[900000001], now=now)
    assert learned > 0
    terms = learner.top_terms(900000001)
    top = [t for t, _c in terms]
    assert "绝绝子" in top                      # 高频黑话上榜
    assert "这个" not in top and "知道" not in top  # 停用词不上榜


def test_slang_prompt_line_formats_terms(sqlite_engine):
    from app.storage.repositories import GroupRepository, UserRepository

    now = datetime(2026, 10, 30, 12, 0)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=900000001, group_name="g",
                                              enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=900000103, nickname="tester",
                                            group_card="")
    with session_scope(sqlite_engine) as session:
        for i in range(8):
            session.add(Message(
                platform_msg_id=f"slang-p-{i}", group_id=900000001, user_id=900000103,
                timestamp=now - timedelta(hours=i),
                plain_text="绝绝子，真的绝绝子，太绝绝子了",
                raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False))
    learner = SlangLearner(sqlite_engine)
    learner.learn(group_ids=[900000001], now=now)
    from app.agent.plugins.builtin import SlangFeature

    feature = SlangFeature()
    feature.attach(engine=sqlite_engine, settings=None, group_ids=(900000001,))
    lines = feature.prompt_lines(kind="group", group_id=900000001, now=now, observing=False)
    assert lines and "高频词" in lines[0] and "绝绝子" in lines[0]
    # 私聊不注入
    assert feature.prompt_lines(kind="private", group_id=1, now=now, observing=False) == ()
