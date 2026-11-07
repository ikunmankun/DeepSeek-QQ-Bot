"""WishNoticeWorker：认领→发送→标记、安静时段顺延、语音概率与配额。"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.agent.wish_notice import WishNoticeWorker
from app.core import agent_agenda
from app.storage.db import session_scope
from app.storage.models import Job

NOON = datetime(2026, 11, 7, 12, 0)          # 白天，可发送
LATE_NIGHT = datetime(2026, 11, 7, 23, 30)   # 安静时段


class _FakeSender:
    def __init__(self):
        self.texts: list[tuple[int, str]] = []
        self.voices: list[int] = []

    async def send_private_text(self, message):
        self.texts.append((message.user_id, message.text))
        return ["receipt-1"]

    async def send_private_voice(self, *, user_id, voice_file):
        self.voices.append(user_id)
        return ["voice-receipt-1"]


class _FakeTts:
    def synthesize(self, text, workdir):
        out = Path(workdir) / "wish.mp3"
        out.write_bytes(b"mp3-bytes")
        return out


def _settings():
    return SimpleNamespace(owner_qq=10086, bot_qq=2900000001)


def _enqueue(engine, *, text="主人，我想要一个听歌识曲的能力", wish_ids=(1,)):
    with session_scope(engine) as session:
        JobRepository_add(session, text=text, wish_ids=list(wish_ids))


def JobRepository_add(session, *, text, wish_ids):
    from app.storage.repositories.jobs import JobRepository

    JobRepository(session).add_job(
        job_type="owner_wish_notice",
        job_key=f"owner-wish-notice:2900000001:test-{text[:12]}",
        payload_json={"wish_ids": wish_ids, "text": text, "date": "2026-11-07"},
        run_at=NOON)


def _worker(engine, sender, *, now_fn=lambda: NOON, rng=None, voice_probability=0.3,
            max_voice_daily=2, tts_builder=None):
    class _Rng:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    return WishNoticeWorker(engine=engine, sender=sender, settings=_settings(),
        connected=lambda: True, voice_probability=voice_probability,
        max_voice_daily=max_voice_daily, tts_builder=tts_builder,
        rng=_Rng(0.0) if rng is None else rng, now_fn=now_fn)


@pytest.mark.asyncio
async def test_worker_sends_text_and_marks_wishes(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        wish = agent_agenda.add_wish(session, content="想要听歌识曲的能力：认不出歌",
                                     now=NOON)
    _enqueue(sqlite_engine, wish_ids=[wish.id])
    sender = _FakeSender()
    worker = _worker(sqlite_engine, sender)
    result = await worker.run_once()
    assert result == "done"
    assert sender.texts and sender.texts[0][0] == 10086
    assert "听歌识曲" in sender.texts[0][1]
    with session_scope(sqlite_engine) as session:
        row = session.get(agent_agenda.AgentAgendaItem, wish.id)
        assert "notified:" in str(row.note or "")
        jobs = list(session.scalars(select(Job).where(
            Job.job_type == "owner_wish_notice")))
    assert jobs[0].status == "done"
    # 已发送的不再重发。
    worker2 = _worker(sqlite_engine, _FakeSender())
    assert await worker2.run_once() is None


@pytest.mark.asyncio
async def test_worker_defers_during_quiet_hours(sqlite_engine):
    _enqueue(sqlite_engine, text="深夜想出来的愿望")
    sender = _FakeSender()
    worker = _worker(sqlite_engine, sender, now_fn=lambda: LATE_NIGHT)
    assert await worker.run_once() is None
    assert sender.texts == []
    with session_scope(sqlite_engine) as session:
        row = session.scalars(select(Job).where(
            Job.job_type == "owner_wish_notice")).first()
        # run_at 被顺延出安静时段，状态仍是 queued。
        assert row.status == "queued" and row.run_at.hour == 8


@pytest.mark.asyncio
async def test_worker_voice_branch_probability_and_quota(sqlite_engine):
    _enqueue(sqlite_engine, text="想要画图的能力")
    sender = _FakeSender()

    def tts_builder(settings):
        return _FakeTts()

    worker = _worker(sqlite_engine, sender, rng=_Value(0.0),  # 概率必中
                     tts_builder=tts_builder)
    await worker.run_once()
    assert len(sender.voices) == 1

    # 概率不中 → 只有文字。
    sender2 = _FakeSender()
    _enqueue(sqlite_engine, text="想要搜题的能力")
    worker2 = _worker(sqlite_engine, sender2, rng=_Value(0.99),
                      tts_builder=tts_builder)
    await worker2.run_once()
    assert sender2.texts and sender2.voices == []


class _Value:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.mark.asyncio
async def test_worker_voice_quota_exhausted(sqlite_engine):
    sender = _FakeSender()

    def tts_builder(settings):
        return _FakeTts()

    worker = _worker(sqlite_engine, sender, rng=_Value(0.0),
                     max_voice_daily=1, tts_builder=tts_builder)
    _enqueue(sqlite_engine, text="第一条愿望想语音说")
    await worker.run_once()
    assert len(sender.voices) == 1
    _enqueue(sqlite_engine, text="第二条愿望当天不再语音")
    await worker.run_once()
    assert len(sender.voices) == 1  # 配额用完，只发文字


@pytest.mark.asyncio
async def test_worker_empty_text_finishes_without_send(sqlite_engine):
    _enqueue(sqlite_engine, text="")
    sender = _FakeSender()
    worker = _worker(sqlite_engine, sender)
    assert await worker.run_once() == "done"
    assert sender.texts == []
