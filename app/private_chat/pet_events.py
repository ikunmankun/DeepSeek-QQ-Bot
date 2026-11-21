"""桌宠事件导出 worker（私聊进程）：把主人私聊/日程待办/愿望完成写成监控快照。

桥脚本（宿主 pet/bridge/pet_bridge.py）轮询监控 API 的 /api/pet-events，
驱动 dsh-pet 桌宠做反应（挥手表私聊、头顶挂待办、愿望完成跳舞）。
只读查询 + 原子写 JSON，不碰任何发送链路。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select

from app.core.time_utils import shanghai_now_naive
from app.monitor.pet_events import PET_EVENTS_VERSION, pet_events_path
from app.storage.db import session_scope
from app.storage.models import AgentAgendaItem, Message

logger = logging.getLogger(__name__)

#: 主人私聊事件的回看窗口
OWNER_MESSAGE_WINDOW = timedelta(minutes=30)
#: 愿望完成的回看窗口
WISH_DONE_WINDOW = timedelta(days=1)
#: 群聊活动的回看窗口（bot 自己的回复 + 被 @ 的消息）
GROUP_ACTIVITY_WINDOW = timedelta(minutes=15)
#: 群消息播报的回看窗口（任何群友的发言，桌宠"听到群里热闹"）
GROUP_CHAT_WINDOW = timedelta(minutes=15)
OWNER_MESSAGE_LIMIT = 10
TODO_LIMIT = 12
WISH_LIMIT = 5
GROUP_ACTIVITY_LIMIT = 6
GROUP_CHAT_LIMIT = 8


class PetEventsWorker:
    """后台轮询导出器：启动即写一次，之后按间隔刷新。"""

    def __init__(self, *, engine, data_dir, owner_qq: int, bot_qq: int | None = None,
                 interval_seconds: float = 300.0, now_fn=None) -> None:
        self._engine = engine
        self._data_dir = Path(data_dir)
        self._owner_qq = int(owner_qq)
        self._bot_qq = int(bot_qq) if bot_qq else None
        self._interval = max(60.0, float(interval_seconds))
        self._now_fn = now_fn or shanghai_now_naive

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run(), name="pet-events-exporter")

    async def stop(self) -> None:
        self._running = False
        task = getattr(self, "_task", None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.run_once)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("pet_events_export_failed")
            await asyncio.sleep(self._interval)

    def run_once(self) -> Path | None:
        now = self._now_fn()
        with session_scope(self._engine) as session:
            messages = list(session.scalars(
                select(Message)
                .where(Message.user_id == self._owner_qq,
                       Message.group_id.is_(None),
                       Message.timestamp >= now - OWNER_MESSAGE_WINDOW)
                .order_by(Message.id.desc())
                .limit(OWNER_MESSAGE_LIMIT)))
            todos = list(session.scalars(
                select(AgentAgendaItem)
                .where(AgentAgendaItem.kind == "todo",
                       AgentAgendaItem.status == "pending")
                .order_by(AgentAgendaItem.id.desc())
                .limit(TODO_LIMIT)))
            wishes = list(session.scalars(
                select(AgentAgendaItem)
                .where(AgentAgendaItem.kind == "wish",
                       AgentAgendaItem.status == "done",
                       AgentAgendaItem.done_at >= now - WISH_DONE_WINDOW)
                .order_by(AgentAgendaItem.id.desc())
                .limit(WISH_LIMIT)))
            group_rows = []
            if self._bot_qq:
                group_rows = list(session.scalars(
                    select(Message)
                    .where(Message.group_id.is_not(None),
                           Message.plain_text.is_not(None),
                           Message.timestamp >= now - GROUP_ACTIVITY_WINDOW,
                           or_(Message.user_id == self._bot_qq,
                               Message.mentioned_bot.is_(True)))
                    .order_by(Message.id.desc())
                    .limit(GROUP_ACTIVITY_LIMIT)))
            # 群消息播报素材：任何群友的文字发言（bot 自己的除外），带群名片/昵称
            chat_rows = []
            if self._bot_qq:
                from app.storage.models import User

                chat_rows = list(session.execute(
                    select(Message, User)
                    .join(User, User.user_id == Message.user_id)
                    .where(Message.group_id.is_not(None),
                           Message.plain_text.is_not(None),
                           Message.user_id != self._bot_qq,
                           Message.timestamp >= now - GROUP_CHAT_WINDOW)
                    .order_by(Message.id.desc())
                    .limit(GROUP_CHAT_LIMIT)).all())
        payload = {
            "version": PET_EVENTS_VERSION,
            "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "owner_messages": [
                {"id": int(m.id), "at": m.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                 "text": str(m.plain_text or "")[:120]}
                for m in reversed(messages)  # 时间正序，桥按 id 判新
            ],
            "todos": [
                {"text": str(item.content or "")[:120], "due_at": None}
                for item in todos
            ],
            "wishes_done": [
                {"id": int(item.id), "at": item.done_at.strftime("%Y-%m-%d %H:%M") if item.done_at else "",
                 "content": str(item.content or "")[:120]}
                for item in wishes
            ],
            "group_activity": [
                {"id": int(m.id), "at": m.timestamp.strftime("%H:%M"),
                 "group_id": int(m.group_id) if m.group_id else 0,
                 "by_bot": bool(m.user_id == self._bot_qq),
                 "text": str(m.plain_text or "")[:80]}
                for m in reversed(group_rows)
            ],
            "group_chat": [
                {"id": int(m.id), "at": m.timestamp.strftime("%H:%M"),
                 "user": (str(user.group_card or "").strip()
                          or str(user.nickname or "").strip())[:12],
                 "text": str(m.plain_text or "")[:80]}
                for m, user in reversed(chat_rows)
            ],
        }
        target = pet_events_path(self._data_dir)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, target)
            return target
        except Exception:
            logger.exception("pet_events_write_failed")
            return None
