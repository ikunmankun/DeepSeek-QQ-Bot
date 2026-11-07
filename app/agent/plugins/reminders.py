"""提醒/闹钟插件：周期 worker 扫到期提醒，到点 @人（带可选语音条）。

指令本身走 router 的确定性快速通道（_try_handle_reminder），这里只负责
后台的到点发送与贪睡催促。语音素材从 alarms 目录随机抽——目录空就纯
文字+@，不报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule


class ReminderFeature(FeatureModule):
    """提醒/闹钟：worker 到点发送，语音素材读 alarms 目录。"""

    name = "reminders"

    _engine = None
    _sender = None

    def attach(self, *, engine, sender) -> None:
        self._engine = engine
        self._sender = sender

    def create_workers(self) -> Sequence[Any]:
        from app.core.reminders import ReminderService

        if self._engine is None or self._sender is None:
            return ()
        # compose 把宿主机 alarms/ 挂到 /workspace/alarms；目录不存在时
        # 纯文字提醒，不报错。
        return [ReminderService(
            engine=self._engine,
            sender=self._sender,
            alarms_dir=Path("alarms"),
        )]
