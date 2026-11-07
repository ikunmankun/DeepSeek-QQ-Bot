"""提醒/闹钟：时间解析、存取上限、到点发送、贪睡催促、每日顺延。"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from types import SimpleNamespace

from sqlalchemy import select

from app.core.reminders import (
    NAG_INTERVAL_MINUTES,
    ReminderService,
    cancel_user_reminder,
    create_reminder,
    list_user_reminders,
    parse_reminder_command,
    pick_alarm_voice,
)
from app.storage.db import session_scope
from app.storage.models import Reminder
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

NOW = datetime(2026, 11, 1, 10, 0, 0)
GROUP = 900000001
USER = 900000103


class FakeSender:
    def __init__(self):
        self.calls: list[tuple] = []

    async def send_group_at_text(self, *, group_id, user_id, text):
        self.calls.append(("at", group_id, user_id, text))

    async def send_group_voice(self, *, group_id, voice_file):
        self.calls.append(("voice", group_id, voice_file))

    async def send_private_text(self, outbound):
        self.calls.append(("private", outbound.user_id, outbound.text))

    async def send_private_voice(self, *, user_id, voice_file):
        self.calls.append(("pvoice", user_id, voice_file))


def test_parse_relative_and_absolute_times():
    parsed = parse_reminder_command("半小时后提醒我喝水", now=NOW)
    assert parsed.action == "set"
    assert parsed.parsed.due_at == NOW + timedelta(minutes=30)
    assert parsed.parsed.content == "喝水"

    parsed = parse_reminder_command("提醒我 20分钟后 交作业", now=NOW)
    assert parsed.parsed.due_at == NOW + timedelta(minutes=20)
    assert parsed.parsed.content == "交作业"

    parsed = parse_reminder_command("一个半小时后叫我", now=NOW)
    assert parsed.parsed.due_at == NOW + timedelta(minutes=90)

    # 明早/今晚/今天 系
    parsed = parse_reminder_command("明早8点叫我起床", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 2, 8, 0)
    assert parsed.parsed.content == "起床"

    parsed = parse_reminder_command("今晚11点提醒我吃褪黑素", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 1, 23, 0)

    # 下午/晚上折 24 小时制；今天的 15:30 还没到 → 就是今天
    parsed = parse_reminder_command("提醒我 下午3点半 开会", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 1, 15, 30)
    # 已过时刻（今天 9 点）没带"明天" → 默认算明天
    parsed = parse_reminder_command("提醒我 早上9点 交表", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 2, 9, 0)

    parsed = parse_reminder_command("21:30 提醒我关窗", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 1, 21, 30)

    parsed = parse_reminder_command("大后天早上9点提醒我交报告", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 4, 9, 0)


def test_parse_daily_reminders():
    parsed = parse_reminder_command("每天早上8点叫我起床", now=NOW)
    assert parsed.parsed.repeat == "daily"
    assert parsed.parsed.due_at == datetime(2026, 11, 2, 8, 0)  # 今天8点已过→明天

    parsed = parse_reminder_command("每天叫我起床", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 2, 8, 0)  # 缺省早八

    parsed = parse_reminder_command("每天晚上11点30分提醒我睡觉", now=NOW)
    assert parsed.parsed.due_at == datetime(2026, 11, 1, 23, 30)  # 今天还没到→今晚


def test_parse_errors_and_non_commands():
    assert parse_reminder_command("你好呀", now=NOW) is None
    assert parse_reminder_command("谢谢提醒", now=NOW) is None
    assert parse_reminder_command("你怎么不提醒我", now=NOW) is None
    # 是意图但缺时间/时间没看懂 → error（router 会回复引导文案）
    assert parse_reminder_command("提醒我喝水", now=NOW).action == "error"
    assert parse_reminder_command("叫我起床", now=NOW).action == "error"
    assert parse_reminder_command("提醒我 明天大概吧 出门", now=NOW).action == "error"


def test_parse_cancel_and_list():
    assert parse_reminder_command("取消提醒").action == "cancel"
    assert parse_reminder_command("取消提醒 3").target_id == 3
    assert parse_reminder_command("我的提醒").action == "list"
    assert parse_reminder_command("闹钟列表").action == "list"
    assert parse_reminder_command("在吗") is None


def _seed_group_and_user(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=GROUP, group_name="测试群",
                                              enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=USER, nickname="主人",
                                            group_card="主人")


def _insert_due(sqlite_engine, *, now, repeat="once", channel="group", content="起床"):
    parsed = SimpleNamespace(due_at=now, repeat=repeat, content=content)
    with session_scope(sqlite_engine) as session:
        row = create_reminder(session, group_id=GROUP, user_id=USER, channel=channel,
                              parsed=parsed, now=now)
        return int(row.id)


def _get_row(sqlite_engine, reminder_id):
    with session_scope(sqlite_engine) as session:
        row = session.get(Reminder, reminder_id)
        session.expunge(row)
        return row


def _make_service(sqlite_engine, sender, *, alarms_dir=None, nag_minutes=NAG_INTERVAL_MINUTES):
    return ReminderService(
        engine=sqlite_engine, sender=sender, alarms_dir=alarms_dir,
        nag_interval_minutes=nag_minutes, now_fn=lambda: NOW,
    )


def test_fire_sends_at_and_enters_nagging(sqlite_engine):
    _seed_group_and_user(sqlite_engine)
    rid = _insert_due(sqlite_engine, now=NOW)
    sender = FakeSender()
    service = _make_service(sqlite_engine, sender)
    snapshot = service._collect_due()[0]
    assert snapshot.id == rid
    asyncio.run(service._process(snapshot))
    assert sender.calls[0][0] == "at" and sender.calls[0][2] == USER
    assert "起床" in sender.calls[0][3]
    row = _get_row(sqlite_engine, rid)
    assert row.status == "nagging"
    assert row.last_sent_at is not None and row.next_nag_at is not None


def test_fire_sends_voice_from_alarms_dir(sqlite_engine, tmp_path):
    _seed_group_and_user(sqlite_engine)
    clip = tmp_path / "wake01.mp3"
    clip.write_bytes(b"fake-mp3-bytes")
    rid = _insert_due(sqlite_engine, now=NOW)
    sender = FakeSender()
    service = _make_service(sqlite_engine, sender, alarms_dir=tmp_path)
    asyncio.run(service._process(service._collect_due()[0]))
    kinds = [call[0] for call in sender.calls]
    assert kinds[0] == "voice" and Path(sender.calls[0][2]).read_bytes() == b"fake-mp3-bytes"
    assert kinds[1] == "at"


def test_pick_alarm_voice_skips_non_audio(tmp_path):
    (tmp_path / "readme.txt").write_text("x")
    assert pick_alarm_voice(tmp_path) is None
    assert pick_alarm_voice(tmp_path / "missing") is None
    assert pick_alarm_voice(None) is None


def test_nag_escalates_then_gives_up(sqlite_engine):
    _seed_group_and_user(sqlite_engine)
    rid = _insert_due(sqlite_engine, now=NOW)
    sender = FakeSender()
    service = _make_service(sqlite_engine, sender)
    asyncio.run(service._process(service._collect_due()[0]))

    # 时间快进到该催了：没人应 → 催第一次
    with session_scope(sqlite_engine) as session:
        row = session.get(Reminder, rid)
        row.next_nag_at = NOW
    snapshot = service._collect_due()[0]
    asyncio.run(service._process(snapshot))
    row = _get_row(sqlite_engine, rid)
    assert row.nag_count == 1 and len([c for c in sender.calls if c[0] == "at"]) == 2

    # 催满 max_nags 次还没人应 → 放弃（done）
    with session_scope(sqlite_engine) as session:
        row = session.get(Reminder, rid)
        row.next_nag_at = NOW
    asyncio.run(service._process(service._collect_due()[0]))
    row = _get_row(sqlite_engine, rid)
    assert row.status == "done"
    total_at = len([c for c in sender.calls if c[0] == "at"])
    assert total_at == 3  # 首发 + 2 次催，到此为止


def test_response_settles_and_daily_reschedules(sqlite_engine):
    _seed_group_and_user(sqlite_engine)
    rid = _insert_due(sqlite_engine, now=NOW, repeat="daily", content="起床")
    sender = FakeSender()
    service = _make_service(sqlite_engine, sender)
    asyncio.run(service._process(service._collect_due()[0]))

    # 主人在群里冒了个泡 → 算应了：每日提醒顺延到明天同一时刻
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).add_group_message(
            platform_msg_id="resp-1", group_id=GROUP, user_id=USER,
            timestamp=datetime(2026, 11, 1, 10, 1), plain_text="起了起了",
            raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
        )
    with session_scope(sqlite_engine) as session:
        row = session.get(Reminder, rid)
        row.next_nag_at = NOW
    asyncio.run(service._process(service._collect_due()[0]))
    row = _get_row(sqlite_engine, rid)
    assert row.status == "pending"
    assert row.due_at == datetime(2026, 11, 2, 10, 0)
    assert row.nag_count == 0

    # 一次性提醒：有人应 → done
    rid2 = _insert_due(sqlite_engine, now=NOW, content="喝水")
    sender.calls.clear()
    asyncio.run(service._process(service._collect_due()[-1]))
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).add_group_message(
            platform_msg_id="resp-2", group_id=GROUP, user_id=USER,
            timestamp=datetime(2026, 11, 1, 10, 2), plain_text="喝了",
            raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
        )
    with session_scope(sqlite_engine) as session:
        row = session.get(Reminder, rid2)
        row.next_nag_at = NOW
    asyncio.run(service._process(service._collect_due()[-1]))
    assert _get_row(sqlite_engine, rid2).status == "done"


def test_private_channel_uses_private_send_and_detects_private_reply(sqlite_engine):
    _seed_group_and_user(sqlite_engine)
    rid = _insert_due(sqlite_engine, now=NOW, channel="private", content="吃药")
    sender = FakeSender()
    service = _make_service(sqlite_engine, sender)
    asyncio.run(service._process(service._collect_due()[0]))
    assert sender.calls and sender.calls[0][0] in ("private", "pvoice")
    assert sender.calls[-1][0] == "private" and "吃药" in sender.calls[-1][2]

    # 私聊回了消息 → 结算
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).add_private_message(
            platform_msg_id="p-resp-1", user_id=USER,
            timestamp=datetime(2026, 11, 1, 10, 1), plain_text="吃过了",
            raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
        )
    with session_scope(sqlite_engine) as session:
        row = session.get(Reminder, rid)
        row.next_nag_at = NOW
    asyncio.run(service._process(service._collect_due()[0]))
    assert _get_row(sqlite_engine, rid).status == "done"


def test_cancel_and_pending_cap(sqlite_engine):
    _seed_group_and_user(sqlite_engine)
    parsed = SimpleNamespace(due_at=NOW + timedelta(hours=1), repeat="once", content="喝水")
    with session_scope(sqlite_engine) as session:
        rid = create_reminder(session, group_id=GROUP, user_id=USER, channel="group",
                              parsed=parsed, now=NOW).id
        assert len(list_user_reminders(session, user_id=USER)) == 1
        # 别人的提醒取消不到
        assert cancel_user_reminder(session, user_id=999, reminder_id=rid) == 0
        assert cancel_user_reminder(session, user_id=USER, reminder_id=rid) == 1
        assert len(list_user_reminders(session, user_id=USER)) == 0

    from app.core.reminders import MAX_PENDING_PER_USER
    with session_scope(sqlite_engine) as session:
        for index in range(MAX_PENDING_PER_USER):
            create_reminder(session, group_id=GROUP, user_id=USER, channel="group",
                            parsed=parsed, now=NOW)
        with pytest.raises(ValueError):
            create_reminder(session, group_id=GROUP, user_id=USER, channel="group",
                            parsed=parsed, now=NOW)
