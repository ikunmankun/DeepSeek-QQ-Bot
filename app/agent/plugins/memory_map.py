"""记忆星图插件：把记忆衰减状态定期导出成监控页可读的快照。

只写 ``data/agent_monitor/memory_map.json`` 一个文件，不参与对话、不注册工具。
快照存在与否只影响监控页能不能画图，不影响任何聊天或记忆链路。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule

logger = logging.getLogger(__name__)


class MemoryMapWorker:
    """后台快照 worker：启动即写一次，之后按配置间隔刷新。"""

    def __init__(self, *, engine, data_dir, interval_seconds: float = 300.0) -> None:
        self._engine = engine
        self._data_dir = data_dir
        self._interval = max(30.0, float(interval_seconds))

    async def run(self) -> None:
        from app.monitor.memory_map import write_snapshot

        while True:
            try:
                await asyncio.to_thread(write_snapshot, self._engine, self._data_dir)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("memory_map_snapshot_worker_failed")
            await asyncio.sleep(self._interval)


class MemoryMapFeature(FeatureModule):
    """记忆星图：只读快照导出，无工具、无提示词。"""

    name = "memory_map"

    _engine = None
    _settings = None

    def attach(self, *, engine, settings) -> None:
        self._engine = engine
        self._settings = settings

    def create_workers(self) -> Sequence[Any]:
        if self._engine is None or self._settings is None:
            return ()
        if not bool(getattr(self._settings, "memory_map_enabled", True)):
            return ()
        interval = float(getattr(self._settings, "memory_map_interval_seconds", 300) or 300)
        return [MemoryMapWorker(engine=self._engine, data_dir=self._settings.data_dir,
                                interval_seconds=interval)]
