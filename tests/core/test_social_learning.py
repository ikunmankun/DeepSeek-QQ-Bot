"""沉默学习：插话结果记录/回填、时段聚合与硬闸、成绩线。"""

from datetime import datetime, timedelta

from sqlalchemy import select

from app.core.social_learning import (
    BUCKET_HOURS,
    FOLLOWUP_WINDOW_MINUTES,
    GATE_MIN_SAMPLES,
    OUTCOME_IGNORED,
    OUTCOME_NEGATIVE,
    OUTCOME_NOT_SENT,
    OUTCOME_RESPONDED,
    evaluate_pending,
    hour_bucket,
    record_interjection,
    silence_gate,
)
from app.storage.db import session_scope
from app.storage.models import GroupInterjectionOutcome
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

NOW = datetime(2026, 10, 31, 21, 0, 0)
GROUP = 900000001
BOT = 900000101
MEMBER = 900000103

BOT_MSG_ID = f"bot-reply-{NOW:%Y%m%d}-1"


def _add_message(session, *, platform_msg_id, user_id, text, timestamp,
                 delivery_state="sent", mentioned_bot=False, reply_to=None):
    _ensure_group_and_users(session)
    session.add(MessageRepository(session).add_group_message(
        platform_msg_id=platform_msg_id,
        group_id=GROUP,
        user_id=user_id,
        timestamp=timestamp,
        plain_text=text,
        raw_json={"delivery_state": delivery_state},
        msg_type="text",
        reply_to_msg_id=reply_to,
        mentioned_bot=mentioned_bot,
    ))


def _ensure_group_and_users(session) -> None:
    """messages 表对 groups/users 有外键：播种时先把群和人都建好。"""

    GroupRepository(session).upsert_group(group_id=GROUP, group_name="测试群",
                                          enabled=True, speak_enabled=True)
    for user_id, nickname in ((BOT, "bot"), (MEMBER, "群友"), (900000104, "路人")):
        UserRepository(session).upsert_user(user_id=user_id, nickname=nickname,
                                            group_card=nickname)


def _seed_interjection(sqlite_engine, *, bot_msg_id=BOT_MSG_ID, sent_at=NOW,
                       delivery_state="sent"):
    with session_scope(sqlite_engine) as session:
        if delivery_state is not None:
            _add_message(session, platform_msg_id=bot_msg_id, user_id=BOT,
                         text="就这？", timestamp=sent_at,
                         delivery_state=delivery_state)
    record_interjection(sqlite_engine, group_id=GROUP,
                        bot_msg_id=bot_msg_id, sent_at=sent_at)


def _outcome_of(sqlite_engine, bot_msg_id=BOT_MSG_ID) -> str | None:
    with session_scope(sqlite_engine) as session:
        row = session.scalars(
            select(GroupInterjectionOutcome).where(
                GroupInterjectionOutcome.bot_msg_id == bot_msg_id)
        ).first()
        return None if row is None else row.outcome


def test_hour_bucket_and_label_boundaries():
    assert hour_bucket(datetime(2026, 10, 31, 0, 30)) == 0
    assert hour_bucket(datetime(2026, 10, 31, 1, 59)) == 0
    assert hour_bucket(datetime(2026, 10, 31, 2, 0)) == 1
    assert hour_bucket(datetime(2026, 10, 31, 23, 0)) == 11
    assert BUCKET_HOURS == 2


def test_record_is_idempotent(sqlite_engine):
    assert record_interjection(sqlite_engine, group_id=GROUP,
                               bot_msg_id="bot-reply-x", sent_at=NOW)
    assert not record_interjection(sqlite_engine, group_id=GROUP,
                                   bot_msg_id="bot-reply-x", sent_at=NOW)
    with session_scope(sqlite_engine) as session:
        rows = list(session.scalars(select(GroupInterjectionOutcome)))
    assert len(rows) == 1 and rows[0].outcome is None


def test_evaluate_responded_when_conversation_continues(sqlite_engine):
    _seed_interjection(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        _add_message(session, platform_msg_id="m1", user_id=MEMBER,
                     text="哈哈哈", timestamp=NOW + timedelta(minutes=2))
        _add_message(session, platform_msg_id="m2", user_id=900000104,
                     text="笑死", timestamp=NOW + timedelta(minutes=4))
    filled = evaluate_pending(sqlite_engine, now=NOW + timedelta(
        minutes=FOLLOWUP_WINDOW_MINUTES + 1))
    assert filled == 1
    assert _outcome_of(sqlite_engine) == OUTCOME_RESPONDED


def test_evaluate_direct_reply_counts_as_responded(sqlite_engine):
    _seed_interjection(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        _add_message(session, platform_msg_id="m1", user_id=MEMBER,
                     text="你说得对", timestamp=NOW + timedelta(minutes=1),
                     reply_to=BOT_MSG_ID)
    evaluate_pending(sqlite_engine, now=NOW + timedelta(
        minutes=FOLLOWUP_WINDOW_MINUTES + 1))
    assert _outcome_of(sqlite_engine) == OUTCOME_RESPONDED


def test_evaluate_negative_reaction(sqlite_engine):
    _seed_interjection(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        _add_message(session, platform_msg_id="m1", user_id=MEMBER,
                     text="别插嘴 行吗", timestamp=NOW + timedelta(minutes=3))
    evaluate_pending(sqlite_engine, now=NOW + timedelta(
        minutes=FOLLOWUP_WINDOW_MINUTES + 1))
    assert _outcome_of(sqlite_engine) == OUTCOME_NEGATIVE


def test_evaluate_ignored_when_silent(sqlite_engine):
    _seed_interjection(sqlite_engine)
    evaluate_pending(sqlite_engine, now=NOW + timedelta(
        minutes=FOLLOWUP_WINDOW_MINUTES + 1))
    assert _outcome_of(sqlite_engine) == OUTCOME_IGNORED


def test_evaluate_not_sent_when_blocked_or_missing(sqlite_engine):
    # 预留态（没发出去）→ not_sent
    _seed_interjection(sqlite_engine, delivery_state="reserved")
    evaluate_pending(sqlite_engine, now=NOW + timedelta(
        minutes=FOLLOWUP_WINDOW_MINUTES + 1))
    assert _outcome_of(sqlite_engine) == OUTCOME_NOT_SENT
    # 完全没有 bot 消息行 → not_sent
    _seed_interjection(sqlite_engine, bot_msg_id="bot-reply-y",
                       delivery_state=None)
    evaluate_pending(sqlite_engine, now=NOW + timedelta(
        minutes=FOLLOWUP_WINDOW_MINUTES + 1))
    assert _outcome_of(sqlite_engine, "bot-reply-y") == OUTCOME_NOT_SENT


def test_evaluate_skips_rows_inside_observation_window(sqlite_engine):
    _seed_interjection(sqlite_engine)
    evaluate_pending(sqlite_engine, now=NOW + timedelta(minutes=1))
    assert _outcome_of(sqlite_engine) is None


def test_gate_blocks_bucket_with_repeated_ignored_outcomes(sqlite_engine):
    sent_at = datetime(2026, 10, 31, 21, 0, 0)  # 21 点 → bucket 10
    with session_scope(sqlite_engine) as session:
        session.add(GroupInterjectionOutcome(
            group_id=GROUP, bot_msg_id="b-old", sent_at=sent_at,
            hour_bucket=hour_bucket(sent_at), outcome=OUTCOME_IGNORED,
            evaluated_at=sent_at))
    # 已回填的 ignored 不足以闸（只有 1 条样本）
    gate = silence_gate(sqlite_engine, group_id=GROUP, now=sent_at + timedelta(days=1))
    assert gate.allowed
    # 凑够 GATE_MIN_SAMPLES 条全 ignored → 硬闸
    with session_scope(sqlite_engine) as session:
        existing = list(session.scalars(select(GroupInterjectionOutcome)))
        session.delete(existing[0])
        for index in range(GATE_MIN_SAMPLES):
            at = sent_at - timedelta(days=index + 1)
            session.add(GroupInterjectionOutcome(
                group_id=GROUP, bot_msg_id=f"b-{index}", sent_at=at,
                hour_bucket=hour_bucket(at), outcome=OUTCOME_IGNORED,
                evaluated_at=at))
    gate = silence_gate(sqlite_engine, group_id=GROUP,
                        now=sent_at + timedelta(days=1))
    assert not gate.allowed
    assert "bucket_silence" in gate.reason


def test_gate_stats_line_reports_recent_record(sqlite_engine):
    at = datetime(2026, 10, 31, 12, 0, 0)  # 与 NOW 同一天但不在深夜桶
    with session_scope(sqlite_engine) as session:
        outcomes = [OUTCOME_RESPONDED, OUTCOME_RESPONDED, OUTCOME_NEGATIVE,
                    OUTCOME_IGNORED]
        for index, outcome in enumerate(outcomes):
            moment = at - timedelta(days=index + 1)
            session.add(GroupInterjectionOutcome(
                group_id=GROUP, bot_msg_id=f"r-{index}", sent_at=moment,
                hour_bucket=hour_bucket(moment), outcome=outcome,
                evaluated_at=moment))
    gate = silence_gate(sqlite_engine, group_id=GROUP, now=NOW)
    assert gate.allowed
    assert gate.stats_line is not None
    assert "4 次" in gate.stats_line
    assert "2 次有人接" in gate.stats_line
    assert "1 次被怼" in gate.stats_line


def test_gate_allows_when_no_history(sqlite_engine):
    assert silence_gate(sqlite_engine, group_id=GROUP, now=NOW).allowed
    assert silence_gate(sqlite_engine, group_id=GROUP, now=NOW).stats_line is None
