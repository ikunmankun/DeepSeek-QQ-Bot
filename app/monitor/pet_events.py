"""桌宠事件快照：bot 侧导出主人私聊/日程待办/愿望完成，监控端与桥只读。

与 memory_map 同一模式：机器人进程把状态写进 agent_monitor 目录的
JSON 快照（原子替换），监控容器与宿主桥脚本只读文件，不碰数据库。
"""
from __future__ import annotations

import json
from pathlib import Path

PET_EVENTS_FILENAME = "pet_events.json"
PET_EVENTS_VERSION = 1
MAX_EVENTS_BYTES = 512 * 1024


def pet_events_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "agent_monitor" / PET_EVENTS_FILENAME


class PetEventsReader:
    """监控/桥侧只读读取器：缺失/损坏/超限都返回可解释的空结果。"""

    def __init__(self, root: Path):
        self._path = Path(root) / PET_EVENTS_FILENAME

    def read(self) -> dict:
        empty = {"available": False, "reason": "", "read_only": True,
                 "updated_at": "", "owner_messages": [], "todos": [],
                 "wishes_done": [], "group_activity": [], "group_chat": []}
        try:
            if not self._path.is_file():
                return {**empty, "reason": "events_missing"}
            if self._path.stat().st_size > MAX_EVENTS_BYTES:
                return {**empty, "reason": "events_too_large"}
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {**empty, "reason": "events_unreadable"}
        if not isinstance(payload, dict) or payload.get("version") != PET_EVENTS_VERSION \
                or not isinstance(payload.get("owner_messages"), list) \
                or not isinstance(payload.get("todos"), list) \
                or not isinstance(payload.get("wishes_done"), list) \
                or not isinstance(payload.get("group_activity"), list):
            return {**empty, "reason": "events_incompatible"}
        return {**payload, "available": True, "read_only": True}
