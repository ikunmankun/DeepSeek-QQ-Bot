"""后台归档任务：把淡忘到接近遗忘的记忆正式停用。

淡忘（memory_decay.is_faded）只是排序沉底，记忆还在库里、被提醒还能想起；
归档（is_archivable）之后由本任务定期调用既有 deactivate_memory_items 停用——
检索文档同步下线，"彻底忘了"。豁免类（禁忌/关系/画像/称呼规则）永不归档。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from sqlalchemy import select

from app.core.memory_decay import ARCHIVE_BATCH_LIMIT, is_archivable, is_promoted
from app.core.time_utils import shanghai_now_naive
from app.core.worker_status import write_worker_status
from app.storage.db import session_scope
from app.storage.models import MemoryItem
from app.storage.repositories import MemoryRepository

logger = logging.getLogger(__name__)

#: 每 6 小时巡检一轮即可：归档线本身以"天"为单位，实时性没有意义。
TICK_INTERVAL_SECONDS = 6 * 60 * 60


class MemoryDecayService:
    """周期性把超过归档线的 active 记忆停用（软删除，保留审计链）。"""

    def __init__(self, *, engine, settings, interval_seconds: float = TICK_INTERVAL_SECONDS) -> None:
        self._engine = engine
        self._settings = settings
        self.interval_seconds = float(interval_seconds)

    async def run(self) -> None:
        while True:
            write_worker_status(
                self._settings.log_dir, "memory_decay", "running", self.interval_seconds
            )
            try:
                archived = await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("memory_decay_tick_failed")
                write_worker_status(
                    self._settings.log_dir, "memory_decay", "error", self.interval_seconds
                )
            else:
                write_worker_status(
                    self._settings.log_dir, "memory_decay", "idle", self.interval_seconds
                )
                if archived:
                    logger.info("memory_decay_archived count=%s", archived)
            await asyncio.sleep(self.interval_seconds)

    def _tick(self) -> int:
        now: datetime = shanghai_now_naive()
        with session_scope(self._engine) as session:
            rows = list(
                session.scalars(
                    select(MemoryItem)
                    .where(MemoryItem.status == "active")
                    .limit(2000)
                )
            )
            # 晋升是读取时按证据现算的（见 memory_decay.is_promoted），这里
            # 只做观测：每轮报一次当前库存里有多少条已是身份级记忆。
            promoted = sum(1 for row in rows if is_promoted(row))
            if promoted:
                logger.info("memory_decay_promoted count=%s", promoted)
            stale_ids = [
                int(row.id) for row in rows if is_archivable(row, now)
            ][:ARCHIVE_BATCH_LIMIT]
            if not stale_ids:
                return 0
            return MemoryRepository(session).deactivate_memory_items(
                stale_ids,
                valid_until=now,
            )
