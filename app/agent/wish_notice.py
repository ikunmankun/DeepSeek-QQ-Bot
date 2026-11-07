"""功能愿望通知 worker（私聊进程）：bot 想要新功能时主动私信主人。

夜间反思钩子提炼出新愿望后往共享 jobs 表写一条 `owner_wish_notice`
（job_key 天级去重），本 worker 在私聊进程认领并发送：安静时段（23:00-08:00
上海）把 run_at 顺延到安静期结束，文字为主，偶尔（概率 + 独立配额）用
TTS 萝莉音说给主人听。发送至多一次：不确定回执不重试，避免反复打扰。
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, time, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.adapters.sender import (
    OutboundPrivateMessage,
    QQMessageDeliveryUncertainError,
)
from app.core.time_utils import shanghai_naive, shanghai_now_naive
from app.storage.db import session_scope
from app.storage.models import Job

logger = logging.getLogger(__name__)

#: 安静时段：深夜不打扰主人，顺延到早上。
QUIET_START, QUIET_END = time(23, 0), time(8, 0)
#: 无 TTS / 非第一条时的兜底轮询间隔（秒）。
POLL_SECONDS = 60.0
#: 认领租约时长（秒）；worker 崩溃后租约过期，下一轮重认领。
LEASE_SECONDS = 300


def _in_quiet_hours(now: datetime) -> bool:
    current = shanghai_naive(now).time() if now is not None else shanghai_now_naive().time()
    return _time_in_quiet(current)


def _time_in_quiet(current: time) -> bool:
    if QUIET_START <= QUIET_END:
        return QUIET_START <= current < QUIET_END
    return current >= QUIET_START or current < QUIET_END


def _next_quiet_end(now: datetime) -> datetime:
    """安静时段结束的下一次钟点（用于把 run_at 顺延出去）。"""

    base = shanghai_naive(now)
    end_today = base.replace(hour=QUIET_END.hour, minute=QUIET_END.minute,
                             second=0, microsecond=0)
    if base < end_today and _time_in_quiet(base.time()):
        return end_today
    return end_today + timedelta(days=1)


class WishNoticeWorker:
    """认领 owner_wish_notice 任务并私信主人；至多一次投递。"""

    def __init__(self, *, engine, sender, settings, connected=lambda: False,
                 voice_probability: float = 0.3, max_voice_daily: int = 2,
                 tts_builder=None, rng=random, now_fn=shanghai_now_naive) -> None:
        self.engine = engine
        self.sender = sender
        self.settings = settings
        self.connected = connected
        self.voice_probability = float(voice_probability)
        self.max_voice_daily = int(max_voice_daily)
        self._tts_builder = tts_builder
        self._rng = rng
        self._now_fn = now_fn
        self.worker = uuid4().hex
        self.running = False
        self._task: asyncio.Task | None = None
        self._tts = None
        self._voice_day = ""
        self._voice_count = 0

    async def start(self) -> None:
        self.running = True
        self._task = asyncio.create_task(self._run(), name="owner-wish-notices")

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        while self.running:
            if self.connected():
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("wish_notice_failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(POLL_SECONDS)

    # -- 认领 ------------------------------------------------------------

    def claim(self) -> dict | None:
        """认领一条到期愿望通知；安静时段把 run_at 顺延到早上。"""

        now = self._now_fn()
        try:
            with session_scope(self.engine) as session:
                row = session.scalars(
                    select(Job).where(Job.job_type == "owner_wish_notice",
                                      Job.status == "queued", Job.run_at <= now)
                    .order_by(Job.id.asc()).limit(1)).first()
                if row is None:
                    return None
                if _in_quiet_hours(now):
                    row.run_at = _next_quiet_end(now)
                    return None
                row.status = "running"
                row.locked_by = self.worker
                row.lease_until = now + timedelta(seconds=LEASE_SECONDS)
                return {"task_id": row.id, **dict(row.payload_json or {})}
        except SQLAlchemyError:
            logger.exception("wish_notice_claim_failed")
            return None

    def finish(self, data: dict, *, status: str, error: str = "") -> None:
        try:
            with session_scope(self.engine) as session:
                row = session.get(Job, int(data["task_id"]))
                if row is None or row.status != "running":
                    return
                row.status = status
                row.last_error_code = str(error or "")[:96]
                if status == "done":
                    self._mark_wishes_notified(session, data)
        except SQLAlchemyError:
            logger.exception("wish_notice_finish_failed task_id=%s", data.get("task_id"))

    def _mark_wishes_notified(self, session, data: dict) -> None:
        from app.core import agent_agenda

        for item_id in data.get("wish_ids") or ():
            try:
                agent_agenda.mark_wish_notified(session, int(item_id), now=self._now_fn())
            except (TypeError, ValueError):
                continue

    # -- 发送 ------------------------------------------------------------

    def _tts_available(self) -> bool:
        if self._tts is None:
            if self._tts_builder is None:
                from app.core.voice import build_tts

                self._tts_builder = build_tts
            try:
                self._tts = self._tts_builder(self.settings)
            except Exception:
                logger.exception("wish_notice_tts_build_failed")
                self._tts = None
        return self._tts is not None

    def _voice_allowed_today(self) -> bool:
        today = f"{self._now_fn():%Y-%m-%d}"
        if self._voice_day != today:
            self._voice_day, self._voice_count = today, 0
        return self._voice_count < self.max_voice_daily

    async def _speak_notice(self, text: str) -> None:
        """合成并发送语音版愿望；失败自吞（文字版已在同一轮发出）。"""

        import tempfile
        from pathlib import Path

        speech = str(text or "").strip()[:150]
        if not speech or not self._tts_available() or not self._voice_allowed_today():
            return
        if self._rng.random() >= self.voice_probability:
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                mp3 = self._tts.synthesize(speech, Path(tmp))
                await self.sender.send_private_voice(user_id=self.settings.owner_qq,
                                                     voice_file=str(mp3))
            self._voice_count += 1
        except Exception as exc:
            logger.warning("wish_notice_voice_failed error_type=%s", type(exc).__name__)

    async def run_once(self) -> str | None:
        data = self.claim()
        if data is None:
            return None
        text = str(data.get("text") or "").strip()
        if not text:
            self.finish(data, status="done", error="empty_notice_text")
            return "done"
        try:
            await self.sender.send_private_text(
                OutboundPrivateMessage(self.settings.owner_qq, text))
        except QQMessageDeliveryUncertainError:
            # 至多一次：回执未知也不重发，等主人来问。
            logger.warning("wish_notice_delivery_uncertain task_id=%s", data.get("task_id"))
            self.finish(data, status="uncertain", error="delivery_result_unknown")
            return "uncertain"
        except Exception as exc:
            logger.exception("wish_notice_send_failed task_id=%s", data.get("task_id"))
            self.finish(data, status="failed", error=type(exc).__name__)
            return "failed"
        await self._speak_notice(text)
        self.finish(data, status="done")
        return "done"
