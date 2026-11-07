"""沉默学习插件：周期回填主动插话的结果（有没有人接话）。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule

logger = logging.getLogger(__name__)


class SilenceLearningService:
    """周期 worker：把观察窗已过的插话记录回填成结果。"""

    def __init__(self, *, engine, log_dir, interval_seconds: float = 900.0) -> None:
        self._engine = engine
        self._log_dir = log_dir
        self.interval_seconds = float(interval_seconds)

    async def run(self) -> None:
        from app.core.social_learning import evaluate_pending
        from app.core.time_utils import shanghai_now_naive
        from app.core.worker_status import write_worker_status

        while True:
            write_worker_status(self._log_dir, "silence_learning", "running", self.interval_seconds)
            try:
                filled = await asyncio.to_thread(
                    evaluate_pending, self._engine, now=shanghai_now_naive()
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("silence_learning_tick_failed")
                write_worker_status(self._log_dir, "silence_learning", "error", self.interval_seconds)
            else:
                write_worker_status(self._log_dir, "silence_learning", "idle", self.interval_seconds)
                if filled:
                    logger.info("silence_learning_evaluated count=%s", filled)
            await asyncio.sleep(self.interval_seconds)


class SilenceLearningFeature(FeatureModule):
    """沉默学习：主动插话的结果沉淀，反哺"什么时候该闭嘴"。"""

    name = "silence_learning"

    _engine = None
    _settings = None

    def attach(self, *, engine, settings) -> None:
        self._engine = engine
        self._settings = settings

    def create_workers(self) -> Sequence[Any]:
        if self._engine is None or self._settings is None:
            return ()
        if not getattr(self._settings, "silence_learning_enabled", True):
            return ()
        return [SilenceLearningService(engine=self._engine, log_dir=self._settings.log_dir)]
