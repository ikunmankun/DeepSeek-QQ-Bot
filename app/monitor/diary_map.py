"""日记本快照：把 data/diary/*.md 的近期日记固化成只读 JSON。

机器人进程按固定间隔把最近 N 天的日记条目（随手记、逛B站、睡前记）
写进 ``data/agent_monitor/diary_map.json``；监控容器（只读、只挂
agent_monitor 子目录、看不到 diary 目录）从磁盘读这份快照渲染日记页。
与 /memory 星图同一套隔离思路：监控端不接触日记原文件，机器人端
不为看日记多开端口。

解析复用 :func:`app.core.diary.parse_diary_file`，不另造一套。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from app.core.diary import REFLECTION_TAG, WEEKDAY_NAMES, parse_diary_file
from app.core.time_utils import shanghai_now_naive

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "diary_map.json"
SNAPSHOT_VERSION = 1
#: 快照覆盖的天数（回看一个月足够）。
SNAPSHOT_DAY_LIMIT = 30
#: 单条日记的导出上限：睡前记升到 800 字后仍完整可读。
MAX_ENTRY_CHARS = 4000
#: 监控端读取上限，防止坏文件把页面拖死。
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024


def snapshot_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "agent_monitor" / SNAPSHOT_FILENAME


def build_snapshot(data_dir: Path | str, *, now: datetime | None = None,
                   days: int = SNAPSHOT_DAY_LIMIT) -> dict:
    """生成一份日记快照。没有日记时返回空列表而不是报错。"""

    instant = now or shanghai_now_naive()
    diary_dir = Path(data_dir) / "diary"
    day_items: list[dict] = []
    for offset in range(max(1, min(SNAPSHOT_DAY_LIMIT, int(days)))):
        day = (instant - timedelta(days=offset)).date()
        path = diary_dir / f"{day:%Y-%m-%d}.md"
        if not path.exists():
            continue
        entries = []
        for entry in parse_diary_file(path):
            if not entry.text:
                continue
            entries.append({
                "time": entry.time,
                "mood": entry.mood,
                "tag": entry.tag,
                "text": entry.text[:MAX_ENTRY_CHARS],
                "chars": len(entry.text),
                "reflection": entry.tag == REFLECTION_TAG,
            })
        if not entries:
            continue
        entries.sort(key=lambda item: item["time"])
        day_items.append({
            "date": f"{day:%Y-%m-%d}",
            "weekday": WEEKDAY_NAMES.get(day.weekday(), ""),
            "reflected": any(item["reflection"] for item in entries),
            "entries": entries,
            "total_chars": sum(item["chars"] for item in entries),
            "bili_count": sum(1 for item in entries if item["tag"] == "逛B站"),
        })
    stats = {
        "days": len(day_items),
        "total_chars": sum(item["total_chars"] for item in day_items),
        "reflection_days": sum(1 for item in day_items if item["reflected"]),
        "bili_count": sum(item["bili_count"] for item in day_items),
    }
    return {
        "version": SNAPSHOT_VERSION,
        "generated_at": instant.strftime("%Y-%m-%d %H:%M:%S"),
        "now": instant.isoformat(timespec="seconds"),
        "stats": stats,
        "days": day_items,
    }


def write_snapshot(data_dir: Path | str, *, now: datetime | None = None,
                   days: int = SNAPSHOT_DAY_LIMIT) -> Path | None:
    """原子写入快照：先写同目录临时文件再 replace，避免监控端读到半个 JSON。"""

    target = snapshot_path(data_dir)
    try:
        payload = build_snapshot(data_dir, now=now, days=days)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
        return target
    except Exception:
        logger.exception("diary_map_snapshot_failed")
        return None


class DiaryMapReader:
    """监控侧只读读取器：坏文件/缺失/超限都返回可解释的空结果。"""

    def __init__(self, root: Path):
        self._path = Path(root) / SNAPSHOT_FILENAME

    def read(self) -> dict:
        try:
            if not self._path.is_file():
                return {"available": False, "reason": "snapshot_missing",
                        "read_only": True, "stats": {}, "days": []}
            if self._path.stat().st_size > MAX_SNAPSHOT_BYTES:
                return {"available": False, "reason": "snapshot_too_large",
                        "read_only": True, "stats": {}, "days": []}
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"available": False, "reason": "snapshot_unreadable",
                    "read_only": True, "stats": {}, "days": []}
        if (not isinstance(payload, dict) or payload.get("version") != SNAPSHOT_VERSION
                or not isinstance(payload.get("days"), list)
                or not isinstance(payload.get("stats"), dict)):
            return {"available": False, "reason": "snapshot_incompatible",
                    "read_only": True, "stats": {}, "days": []}
        return {**payload, "available": True, "read_only": True}
