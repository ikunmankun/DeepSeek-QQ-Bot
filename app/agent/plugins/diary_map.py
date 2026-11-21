"""日记本插件：把 data/diary 的近期日记定期导出成监控页可读的快照。

只写 ``data/agent_monitor/diary_map.json`` 一个文件，不参与对话、不注册工具。
快照存在与否只影响监控页能不能翻日记，不影响任何日记或聊天链路。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule

logger = logging.getLogger(__name__)


class DiaryMapWorker:
    """后台快照 worker：启动即写一次，之后按配置间隔刷新。"""

    def __init__(self, *, data_dir, interval_seconds: float = 300.0) -> None:
        self._data_dir = data_dir
        self._interval = max(30.0, float(interval_seconds))

    async def run(self) -> None:
        from app.monitor.diary_map import write_snapshot

        while True:
            try:
                await asyncio.to_thread(write_snapshot, self._data_dir)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("diary_map_snapshot_worker_failed")
            await asyncio.sleep(self._interval)


class DiaryMapFeature(FeatureModule):
    """日记本：只读快照导出，无工具、无提示词。"""

    name = "diary_map"

    _settings = None

    def attach(self, *, settings) -> None:
        self._settings = settings

    def create_workers(self) -> Sequence[Any]:
        if self._settings is None:
            return ()
        if not bool(getattr(self._settings, "diary_map_enabled", True)):
            return ()
        interval = float(getattr(self._settings, "diary_map_interval_seconds", 300) or 300)
        return [DiaryMapWorker(data_dir=self._settings.data_dir,
                               interval_seconds=interval)]
