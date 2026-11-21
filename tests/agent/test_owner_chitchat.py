"""主人主动搭话 worker：窗口解析、排程去重、打扰检查、生成发送全链路。"""
import random
from datetime import datetime, time, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.agent.owner_chitchat import (
    JOB_TYPE,
    KIND_CHAT,
    KIND_MORNING,
    OwnerChitChatWorker,
    _cap_reply,
    parse_windows,
    render_owner_persona,
)
from app.core.diary import DiaryWriter
from app.core.weather import WeatherReport
from app.storage.db import session_scope
from app.storage.models import Job
from app.storage.repositories import JobRepository, MessageRepository, UserRepository

OWNER, BOT = 3510000001, 2900000001
BASE_NOW = datetime(2026, 11, 18, 13, 0)  # 下午，非安静时段


def _settings(tmp_path, **overrides):
    # 默认把窗口设在深夜（对 13:00 的基准时钟是未来时间），让行为测试里
    # ensure_scheduled 造出的任务永远不到点，只有测试自己种的种子任务到期。
    base = dict(owner_qq=OWNER, bot_qq=BOT, data_dir=tmp_path,
                owner_chitchat_enabled=True,
                owner_chitchat_morning_window="22:51-22:55",
                owner_chitchat_windows="22:50-22:55",
                owner_chitchat_max_daily=2,
                owner_chitchat_min_gap_minutes=150.0,
                owner_chitchat_idle_minutes=45.0,
                weather_city="", weather_lat=0.0, weather_lon=0.0,
                vpn_proxy_url="", vpn_proxy_hosts="")
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeLlm:
    def __init__(self, text="主人主人，我刚刷B站看到个好笑的，听我讲嘛"):
        self.text = text
        self.prompts = []

    def generate_text(self, prompt, **kwargs):
        self.prompts.append(list(prompt))
        return self.text


class _FakeSender:
    def __init__(self):
        self.sent = []

    async def send_private_text(self, outbound):
        self.sent.append(outbound)
        return ("receipt",)


def _persona():
    return {"name": "鲸鱼娘", "identity": "你是主人私聊里的甜心肥鱼。",
            "speech_habits": ["说话软软的"], "favorite_owner": True}


def _worker(sqlite_engine, tmp_path, *, now=BASE_NOW, llm=None, sender=None,
            settings=None, weather=None):
    holder = {"value": now}
    settings = settings or _settings(tmp_path)
    worker = OwnerChitChatWorker(
        engine=sqlite_engine, sender=sender or _FakeSender(), settings=settings,
        llm_client=llm or _FakeLlm(), owner_persona=_persona(),
        connected=lambda: True, rng=random.Random(7),
        now_fn=lambda: holder["value"],
        weather_fetcher=lambda s: weather)
    return worker, holder


def _seed_job(engine, *, kind, run_at, status="queued", job_key=None,
              payload=None, window_end=None):
    body = {"kind": kind}
    if window_end is not None:
        body["window_end"] = window_end.isoformat(timespec="seconds")
    if payload:
        body.update(payload)
    with session_scope(engine) as session:
        row = JobRepository(session).add_job(
            job_type=JOB_TYPE,
            job_key=job_key or f"{JOB_TYPE}:{kind}:seed:{random.randint(0, 10**9)}",
            run_at=run_at, status=status, payload_json=body)
        session.flush()
        return row.id


def _job_status(engine, job_id):
    with session_scope(engine) as session:
        row = session.get(Job, job_id)
        return row.status, row.last_error_code


def _seed_private_message(engine, *, user_id, minutes_ago, now=BASE_NOW, text="想你"):
    with session_scope(engine) as session:
        # messages.user_id 对 users 表有外键，先补用户行
        UserRepository(session).upsert_user(user_id=user_id, nickname="probe", group_card="")
        MessageRepository(session).add_private_message(
            platform_msg_id=f"probe-{user_id}-{minutes_ago}-{random.randint(0, 10**9)}",
            user_id=user_id, timestamp=now - timedelta(minutes=minutes_ago),
            plain_text=text, raw_json={"sender": {"nickname": "n"}})


def test_parse_windows_mixed_separators_and_invalid():
    windows = parse_windows("08:10-08:50，19:00-20:40,垃圾,x:y")
    assert windows == [(time(8, 10), time(8, 50)), (time(19, 0), time(20, 40))]
    assert parse_windows("") == []
    assert parse_windows("22:30-20:40") == []  # 倒挂窗口跳过


def test_cap_reply_cuts_at_sentence_boundary():
    assert _cap_reply("短的回复") == "短的回复"
    long_text = "第一句很重要！" + "填充" * 60 + "。尾巴被截掉"
    capped = _cap_reply(long_text, limit=60)
    assert len(capped) <= 61
    assert capped.endswith("！")


def test_render_owner_persona_includes_owner_line():
    text = render_owner_persona(_persona(), owner_qq=OWNER)
    assert f"QQ 号 {OWNER}" in text
    assert "排第一" in text


def test_ensure_scheduled_creates_each_window_once(sqlite_engine, tmp_path):
    settings = _settings(tmp_path, owner_chitchat_morning_window="08:10-08:50",
                         owner_chitchat_windows="12:40-14:30,19:00-20:40")
    worker, holder = _worker(sqlite_engine, tmp_path,
                             now=datetime(2026, 11, 18, 7, 30), settings=settings)
    assert worker.ensure_scheduled() == 3  # 1 早报 + 2 闲聊
    assert worker.ensure_scheduled() == 0  # 幂等
    with session_scope(sqlite_engine) as session:
        rows = list(session.scalars(select(Job).where(Job.job_type == JOB_TYPE)))
        assert len(rows) == 3
        assert all(row.status == "queued" for row in rows)
        kinds = {row.payload_json["kind"] for row in rows}
        assert kinds == {KIND_MORNING, KIND_CHAT}


def test_ensure_scheduled_skips_passed_windows(sqlite_engine, tmp_path):
    settings = _settings(tmp_path, owner_chitchat_morning_window="08:10-08:50",
                         owner_chitchat_windows="12:40-14:30,19:00-20:40")
    # 13:00 时早报窗口（08:10-08:50）已过，不该补发过期话题
    worker, _ = _worker(sqlite_engine, tmp_path, now=BASE_NOW, settings=settings)
    created = worker.ensure_scheduled()
    assert created == 2
    with session_scope(sqlite_engine) as session:
        kinds = {row.payload_json["kind"] for row in
                 session.scalars(select(Job).where(Job.job_type == JOB_TYPE))}
        assert KIND_MORNING not in kinds


def test_claim_postpones_during_quiet_hours(sqlite_engine, tmp_path):
    worker, holder = _worker(sqlite_engine, tmp_path, now=datetime(2026, 11, 18, 23, 30))
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=datetime(2026, 11, 18, 21, 0),
                       window_end=datetime(2026, 11, 18, 22, 30))
    assert worker.claim() is None
    with session_scope(sqlite_engine) as session:
        row = session.get(Job, job_id)
        assert row.status == "queued"
        assert row.run_at >= datetime(2026, 11, 19, 8, 0)


def test_claim_cancels_stale_window(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path, now=BASE_NOW)
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(minutes=10),
                       window_end=BASE_NOW - timedelta(hours=12))
    assert worker.claim() is None
    assert _job_status(sqlite_engine, job_id) == ("cancelled", "stale_window")


async def test_chat_skips_when_owner_recently_active(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path)
    _seed_private_message(sqlite_engine, user_id=OWNER, minutes_ago=10)
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(seconds=30))
    assert await worker.run_once() == "skipped"
    assert _job_status(sqlite_engine, job_id) == ("cancelled", "owner_recently_active")


async def test_chat_skips_at_daily_cap(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path,
                        settings=_settings(tmp_path, owner_chitchat_max_daily=2))
    for index in range(2):
        _seed_job(sqlite_engine, kind=KIND_CHAT, run_at=BASE_NOW - timedelta(hours=3),
                  status="done", job_key=f"{JOB_TYPE}:chat:2026-11-18:9{index}")
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(seconds=30))
    assert await worker.run_once() == "skipped"
    assert _job_status(sqlite_engine, job_id) == ("cancelled", "daily_cap_reached")


async def test_chat_skips_within_min_gap_after_bot_outbound(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path)
    _seed_private_message(sqlite_engine, user_id=BOT, minutes_ago=30, text="晚安")
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(seconds=30))
    assert await worker.run_once() == "skipped"
    assert _job_status(sqlite_engine, job_id) == ("cancelled", "min_gap_not_reached")


async def test_morning_sends_weather_and_bili_topic(sqlite_engine, tmp_path):
    llm, sender = _FakeLlm("主人早呀，今天有小雨记得带伞，淋湿了我心疼"), _FakeSender()
    DiaryWriter(tmp_path).write(text="刷B站看到《测试视频》：评论区吵翻了",
                                now=BASE_NOW, tag="逛B站")
    weather = WeatherReport(city="天河", description="小雨", temp_now=18.0,
                            feels_like=16.0, temp_min=14.0, temp_max=22.0,
                            precip_probability=70, precip_sum=3.0,
                            wind_kmh=10.0, humidity=85)
    worker, _ = _worker(sqlite_engine, tmp_path, llm=llm, sender=sender, weather=weather)
    job_id = _seed_job(sqlite_engine, kind=KIND_MORNING,
                       run_at=BASE_NOW - timedelta(seconds=30))
    assert await worker.run_once() == "done"
    assert len(sender.sent) == 1
    assert sender.sent[0].user_id == OWNER
    assert sender.sent[0].text.startswith("主人早呀")
    assert _job_status(sqlite_engine, job_id) == ("done", "")
    prompt = "\n".join(llm.prompts[0])
    assert "天河小雨" in prompt          # 天气实况进提示词
    assert "《测试视频》" in prompt      # B站发现进提示词
    assert f"QQ 号 {OWNER}" in prompt    # 主人动态行进提示词


async def test_chat_sends_bili_topic_without_weather(sqlite_engine, tmp_path):
    llm, sender = _FakeLlm("主人主人，我今天刷到个好笑的"), _FakeSender()
    DiaryWriter(tmp_path).write(text="刷B站看到《另一条》：很有意思",
                                now=BASE_NOW - timedelta(days=1), tag="逛B站")
    worker, _ = _worker(sqlite_engine, tmp_path, llm=llm, sender=sender)
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(seconds=30))
    assert await worker.run_once() == "done"
    assert len(sender.sent) == 1
    prompt = "\n".join(llm.prompts[0])
    assert "《另一条》" in prompt
    assert "天河小雨" not in prompt  # 闲聊窗口不拉天气
    assert "主动发起" in prompt      # 主动开场指令在提示词里


async def test_chat_failed_when_reply_empty(sqlite_engine, tmp_path):
    llm, sender = _FakeLlm(""), _FakeSender()
    worker, _ = _worker(sqlite_engine, tmp_path, llm=llm, sender=sender)
    job_id = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(seconds=30))
    assert await worker.run_once() == "failed"
    assert _job_status(sqlite_engine, job_id) == ("failed", "empty_reply")
    assert sender.sent == []


async def test_done_chat_jobs_count_toward_cap(sqlite_engine, tmp_path):
    """发一条 → done 计数生效；第二次到点直接卡上限。"""
    settings = _settings(tmp_path, owner_chitchat_max_daily=1,
                         owner_chitchat_min_gap_minutes=0.0)
    worker, _ = _worker(sqlite_engine, tmp_path, settings=settings)
    first = _seed_job(sqlite_engine, kind=KIND_CHAT,
                      run_at=BASE_NOW - timedelta(seconds=30),
                      job_key=f"{JOB_TYPE}:chat:2026-11-18:90")
    assert await worker.run_once() == "done"
    second = _seed_job(sqlite_engine, kind=KIND_CHAT,
                       run_at=BASE_NOW - timedelta(seconds=5),
                       job_key=f"{JOB_TYPE}:chat:2026-11-18:91")
    assert await worker.run_once() == "skipped"
    assert _job_status(sqlite_engine, second) == ("cancelled", "daily_cap_reached")
    assert _job_status(sqlite_engine, first) == ("done", "")
