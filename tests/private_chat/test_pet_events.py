"""桌宠事件导出：主人私聊/日程待办/愿望完成 → agent_monitor 快照。"""
from datetime import datetime, timedelta

import pytest

from app.monitor.pet_events import PetEventsReader
from app.private_chat.pet_events import PetEventsWorker
from app.storage.db import session_scope
from app.storage.models import AgentAgendaItem, Message, User

NOW = datetime(2026, 11, 13, 0, 30, 0)
OWNER = 900000101


@pytest.fixture()
def engine_with_events(tmp_path):
    from app.storage.db import build_engine, create_all

    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    with session_scope(engine) as session:
        session.add(User(user_id=OWNER, nickname="主人"))
        session.flush()
        session.add(Message(platform_msg_id="pm1", group_id=None, user_id=OWNER,
                            timestamp=NOW - timedelta(minutes=3), raw_json={},
                            plain_text="大肥鱼，看看这个"))
        session.add(Message(platform_msg_id="pm-old", group_id=None, user_id=OWNER,
                            timestamp=NOW - timedelta(hours=2), raw_json={},
                            plain_text="旧消息不该进快照"))
        session.flush()
        session.add(AgentAgendaItem(kind="todo", status="pending",
                                    content="查隔夜饭为什么粒粒分明"))
        session.add(AgentAgendaItem(kind="todo", status="pending",
                                    content="给主人写周报"))
        session.add(AgentAgendaItem(kind="todo", status="done",
                                    content="已完成的不该出现"))
        session.add(AgentAgendaItem(kind="wish", status="done",
                                    content="想要的能力：语音转文字",
                                    note="nightly", done_at=NOW - timedelta(hours=1)))
    return engine


def test_worker_exports_messages_todos_and_wishes(engine_with_events, tmp_path):
    worker = PetEventsWorker(engine=engine_with_events, data_dir=tmp_path,
                             owner_qq=OWNER, now_fn=lambda: NOW)
    target = worker.run_once()
    assert target is not None and target.is_file()
    payload = __import__("json").loads(target.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["owner_messages"]) == 1
    assert payload["owner_messages"][0]["text"] == "大肥鱼，看看这个"
    todo_texts = {item["text"] for item in payload["todos"]}
    assert todo_texts == {"查隔夜饭为什么粒粒分明", "给主人写周报"}
    assert len(payload["wishes_done"]) == 1
    assert "语音转文字" in payload["wishes_done"][0]["content"]

    reader = PetEventsReader(tmp_path / "agent_monitor").read()
    assert reader["available"] is True and reader["read_only"] is True
    assert len(reader["todos"]) == 2 and len(reader["owner_messages"]) == 1


def test_worker_without_owner_messages_exports_empty_lists(tmp_path):
    from app.storage.db import build_engine, create_all

    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    worker = PetEventsWorker(engine=engine, data_dir=tmp_path,
                             owner_qq=OWNER, now_fn=lambda: NOW)
    assert worker.run_once() is not None
    reader = PetEventsReader(tmp_path / "agent_monitor").read()
    assert reader["available"] is True
    assert reader["owner_messages"] == [] and reader["todos"] == []


def test_worker_ignores_other_users_messages(tmp_path):
    from app.storage.db import build_engine, create_all

    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    with session_scope(engine) as session:
        session.add(User(user_id=888, nickname="路人"))
        session.flush()
        session.add(Message(platform_msg_id="p-other", group_id=None, user_id=888,
                            timestamp=NOW - timedelta(minutes=1), raw_json={},
                            plain_text="这不是主人的消息"))
    worker = PetEventsWorker(engine=engine, data_dir=tmp_path,
                             owner_qq=OWNER, now_fn=lambda: NOW)
    worker.run_once()
    reader = PetEventsReader(tmp_path / "agent_monitor").read()
    assert reader["owner_messages"] == []


def test_worker_exports_group_activity_empty(tmp_path):
    from app.storage.db import build_engine, create_all

    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    worker = PetEventsWorker(engine=engine, data_dir=tmp_path,
                             owner_qq=OWNER, bot_qq=2900000001, now_fn=lambda: NOW)
    worker.run_once()
    reader = PetEventsReader(tmp_path / "agent_monitor").read()
    assert reader["group_activity"] == []


def test_worker_exports_group_bot_reply_and_mention(tmp_path):
    """群聊动静导出：bot 自己的回复 + 被 @ 的消息都进快照。"""
    BOT = 2900000001
    from app.storage.db import build_engine, create_all
    from app.storage.models import Group

    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    with session_scope(engine) as session:
        session.add(User(user_id=OWNER, nickname="主人"))
        session.add(User(user_id=BOT, nickname="大肥鱼"))
        session.add(Group(group_id=900000001, group_name="测试群"))
        session.flush()
        session.add(Message(platform_msg_id="bot-reply", group_id=900000001, user_id=BOT,
                            timestamp=NOW - timedelta(minutes=2), raw_json={},
                            plain_text="那咋了，牛马都一样，先炫饭"))
        session.add(Message(platform_msg_id="mention", group_id=900000001, user_id=OWNER,
                            timestamp=NOW - timedelta(minutes=1), raw_json={},
                            plain_text="@大肥鱼 看看这个", mentioned_bot=True))
    worker = PetEventsWorker(engine=engine, data_dir=tmp_path,
                             owner_qq=OWNER, bot_qq=BOT, now_fn=lambda: NOW)
    worker.run_once()
    reader = PetEventsReader(tmp_path / "agent_monitor").read()
    entries = reader["group_activity"]
    assert len(entries) == 2
    bot_entries = [e for e in entries if e["by_bot"]]
    assert len(bot_entries) == 1 and "牛马" in bot_entries[0]["text"]


def test_worker_exports_group_chat_with_cards(tmp_path):
    """群消息播报导出：任何群友的发言都进快照，带群名片/昵称，bot 自己的除外。"""
    BOT = 2900000001
    from app.storage.db import build_engine, create_all
    from app.storage.models import Group

    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    with session_scope(engine) as session:
        session.add(User(user_id=OWNER, nickname="主人", group_card="群主大人"))
        session.add(User(user_id=BOT, nickname="大肥鱼"))
        session.add(User(user_id=777, nickname="老王"))
        session.add(Group(group_id=900000001, group_name="测试群"))
        session.flush()
        session.add(Message(platform_msg_id="chat-1", group_id=900000001, user_id=777,
                            timestamp=NOW - timedelta(minutes=5), raw_json={},
                            plain_text="今天中午吃什么"))
        session.add(Message(platform_msg_id="chat-2", group_id=900000001, user_id=OWNER,
                            timestamp=NOW - timedelta(minutes=4), raw_json={},
                            plain_text="随便，别问我是不是又吃白饭"))
        session.add(Message(platform_msg_id="chat-bot", group_id=900000001, user_id=BOT,
                            timestamp=NOW - timedelta(minutes=3), raw_json={},
                            plain_text="bot 自己的消息不进播报"))
        session.add(Message(platform_msg_id="chat-old", group_id=900000001, user_id=777,
                            timestamp=NOW - timedelta(hours=2), raw_json={},
                            plain_text="窗口外的旧消息不进播报"))
    worker = PetEventsWorker(engine=engine, data_dir=tmp_path,
                             owner_qq=OWNER, bot_qq=BOT, now_fn=lambda: NOW)
    worker.run_once()
    reader = PetEventsReader(tmp_path / "agent_monitor").read()
    entries = reader["group_chat"]
    assert len(entries) == 2
    by_id = {e["id"]: e for e in entries}
    owner_entry = next(e for e in entries if "白饭" in e["text"])
    assert owner_entry["user"] == "群主大人"          # 群名片优先于昵称
    assert by_id and all(e["user"] for e in entries)  # 都带称呼
    assert not any("不进播报" in e["text"] for e in entries)
