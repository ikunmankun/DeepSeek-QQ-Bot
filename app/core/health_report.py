"""每日自检报告：每晚私聊主人一份简短的运行健康摘要。

内容全部来自本地可证事实（DB 计数 + 端口探测 + 配置状态），不编造。
默认 22:00 发送（上海钟面），每天最多一条；发送失败只记日志不重试——
报告本身不能成为新的故障源。
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
from datetime import datetime, time, timedelta
from urllib.parse import urlsplit

from sqlalchemy import func, select

from app.adapters.sender import OutboundPrivateMessage, QQMessageDeliveryUncertainError
from app.core.time_utils import shanghai_now_naive
from app.storage.models import Message

logger = logging.getLogger(__name__)

REPORT_TIME = time(22, 0)


def _proxy_probe() -> str:
    """探测容器代理环境变量指向的端口；没配代理就报直连模式。"""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    if not proxy:
        return "直连模式（未配代理）"
    parsed = urlsplit(proxy)
    try:
        conn = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=3)
        conn.close()
        return "代理在线"
    except OSError as exc:
        return f"代理挂了（{type(exc).__name__}）——境外下载类功能会受影响"


def _direct_probe() -> str:
    try:
        conn = socket.create_connection(("www.bilibili.com", 443), timeout=5)
        conn.close()
        return "B站直连通"
    except OSError:
        return "B站直连异常"


class HealthReportService:
    def __init__(self, *, engine, settings, sender, check_interval: float = 300.0) -> None:
        self.engine = engine
        self.settings = settings
        self.sender = sender
        self.check_interval = float(check_interval)
        self._last_sent_date: str = ""
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="health-report-worker")

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    async def run(self) -> None:
        while True:
            try:
                now = shanghai_now_naive()
                today = now.strftime("%Y-%m-%d")
                if self._last_sent_date != today and now.time() >= REPORT_TIME:
                    self._last_sent_date = today
                    await self._send_report(now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("health_report_tick_failed")
            await asyncio.sleep(self.check_interval)

    async def _send_report(self, now: datetime) -> None:
        owner = int(getattr(self.settings, "owner_qq", 0) or 0)
        if not owner:
            return
        text = self._compose(now)
        try:
            await self.sender.send_private_text(OutboundPrivateMessage(user_id=owner, text=text))
            logger.info("health_report_sent date=%s chars=%s", now.strftime("%Y-%m-%d"), len(text))
        except QQMessageDeliveryUncertainError:
            logger.warning("health_report_delivery_uncertain")
        except Exception:
            logger.exception("health_report_send_failed")

    def _compose(self, now: datetime) -> str:
        from app.storage.db import session_scope
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        bot_qq = int(self.settings.bot_qq)
        bot_sent = user_msgs = delivery_bad = 0
        with session_scope(self.engine) as session:
            rows = session.execute(
                select(Message.user_id, func.count(Message.id)).where(
                    Message.timestamp >= day_start,
                ).group_by(Message.user_id)
            ).all()
            by_user = {int(uid): int(count) for uid, count in rows}
            user_msgs = sum(count for uid, count in by_user.items() if uid != bot_qq)
            bot_sent = by_user.get(bot_qq, 0)
            bad_rows = session.execute(
                select(func.count(Message.id)).where(
                    Message.user_id == bot_qq,
                    Message.timestamp >= day_start,
                    Message.raw_json["delivery_state"].as_string().in_(["failed", "uncertain"]),
                )
            ).scalar()
            delivery_bad = int(bad_rows or 0)
        learned = meme_meanings = 0
        stickers_dir = self.settings.data_dir / "sticker_library" / "learned"
        meanings_dir = self.settings.data_dir / "meme_meanings"
        if stickers_dir.exists():
            learned = sum(1 for p in stickers_dir.glob("*.json"))
        if meanings_dir.exists():
            meme_meanings = sum(1 for p in meanings_dir.glob("*.json"))
        provider = str(getattr(self.settings, "music_id_provider", "audd") or "audd")
        music_on = bool(str(getattr(self.settings, "music_id_api_token", "") or "").strip()) \
            if provider != "xfyun" else all(
            bool(str(getattr(self.settings, name, "") or "").strip())
            for name in ("music_id_app_id", "music_id_api_key", "music_id_api_secret"))
        lines = [
            "【每日自检】" + now.strftime("%m-%d"),
            f"今天：群友 {user_msgs} 条，我发了 {bot_sent} 条（投递失败/不确定 {delivery_bad}）",
            f"网络：{_proxy_probe()}，{_direct_probe()}",
            f"表情库：学过 {learned} 张，梗含义已补 {meme_meanings} 张",
            f"识曲：{provider} {'在线' if music_on else '未配置'}",
            "自改提案：见 data/selfmod/（有 ready 未应用的我会主动提）",
        ]
        return "\n".join(lines)
