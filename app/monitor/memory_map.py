"""记忆星图快照：把记忆的衰减状态固化成只读 JSON。

机器人进程按固定间隔把 ``memory_items`` 里的活跃记忆算成"星星"写进
``data/agent_monitor/memory_map.json``；监控容器（只读、只挂 agent_monitor
子目录、看不到 bot.db）从磁盘读这份快照画图。这样星图与主链路隔离：
监控端不接触数据库，机器人端不为看图画图多开端口。

强度公式直接复用 :mod:`app.core.memory_decay`，不另造一套：
``strength = 2 ** (-age_days / half_life_days)``，半衰期
``= 基础值 × importance × (1 + log2(mention_count))``；
禁忌/关系/画像/称呼规则与已晋升的身份级记忆永不衰减。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app.core.memory_decay import (
    ARCHIVE_HALF_LIVES, half_life_days, is_faded, is_immutable, is_promoted,
    memory_age_days, strength,
)
from app.core.time_utils import shanghai_now_naive, stored_as_utc
from app.storage.db import session_scope
from app.storage.models import MemoryItem

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "memory_map.json"
SNAPSHOT_VERSION = 1
#: 单次快照最多导出的记忆条数（活跃记忆通常远小于这个上限）。
SNAPSHOT_NODE_LIMIT = 2000
#: 内容截断长度：星图只用于"认得出来"，不承载全文。
CONTENT_CHARS = 140
#: 监控端读取上限，防止坏文件把页面拖死。
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024


def snapshot_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "agent_monitor" / SNAPSHOT_FILENAME


def _node(memory: MemoryItem, now: datetime) -> dict:
    immutable = bool(is_immutable(memory))
    faded = bool(is_faded(memory, now))
    value = max(0.0, min(1.0, float(strength(memory, now))))
    age_days = memory_age_days(memory, now)
    half_life = float(half_life_days(memory))
    forget_in_days = None if immutable or age_days is None else max(0.0, half_life - age_days)
    archive_in_days = (None if immutable or age_days is None
                       else max(0.0, half_life * ARCHIVE_HALF_LIVES - age_days))

    def day(value_: datetime | None) -> str:
        if value_ is None:
            return ""
        try:
            return stored_as_utc(value_).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            return ""

    return {
        "id": int(memory.id),
        "kind": str(memory.memory_kind or ""),
        "scope_type": str(memory.scope_type or ""),
        "scope_id": str(memory.scope_id or ""),
        "subject_id": str(memory.subject_id or ""),
        "content": str(memory.content or "")[:CONTENT_CHARS],
        "importance": int(memory.importance or 1),
        "mention_count": int(memory.mention_count or 1),
        "strength": round(value, 4),
        "half_life_days": round(half_life, 1),
        "age_days": None if age_days is None else round(max(0.0, age_days), 1),
        "immutable": immutable,
        "promoted": bool(is_promoted(memory)),
        "faded": faded,
        "forget_in_days": None if forget_in_days is None else round(forget_in_days, 1),
        "archive_in_days": None if archive_in_days is None else round(archive_in_days, 1),
        "last_seen_at": day(memory.last_seen_at),
        "valid_from": day(memory.valid_from),
    }


def build_snapshot(engine, *, now: datetime | None = None,
                   limit: int = SNAPSHOT_NODE_LIMIT) -> dict:
    """生成一份星图快照。只读查询；无记忆时返回空星图而不是报错。"""
    instant = now or shanghai_now_naive()
    limit = max(1, min(SNAPSHOT_NODE_LIMIT, int(limit)))
    with session_scope(engine) as session:
        rows = list(session.scalars(
            select(MemoryItem)
            .where(MemoryItem.status == "active")
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)))
    nodes = [_node(memory, instant) for memory in rows]
    by_kind: dict[str, int] = {}
    for node in nodes:
        by_kind[node["kind"]] = by_kind.get(node["kind"], 0) + 1
    # 越接近归档越靠前，供"即将遗忘"列表直接使用。
    nodes.sort(key=lambda node: (
        1 if node["immutable"] else 0,
        -1 if node["archive_in_days"] is None else node["archive_in_days"],
        node["strength"],
        node["id"],
    ))
    stats = {
        "total": len(nodes),
        "immutable": sum(1 for node in nodes if node["immutable"]),
        "promoted": sum(1 for node in nodes if node["promoted"]),
        "faded": sum(1 for node in nodes if node["faded"]),
        "archiving_soon": sum(1 for node in nodes
                              if not node["immutable"] and node["archive_in_days"] is not None
                              and node["archive_in_days"] <= 7.0),
        "by_kind": by_kind,
        "strongest": round(max((node["strength"] for node in nodes), default=0.0), 4),
        "weakest": round(min((node["strength"] for node in nodes), default=0.0), 4),
    }
    return {
        "version": SNAPSHOT_VERSION,
        "generated_at": instant.strftime("%Y-%m-%d %H:%M:%S"),
        "now": instant.isoformat(timespec="seconds"),
        "stats": stats,
        "nodes": nodes,
    }


def write_snapshot(engine, data_dir: Path | str, *, now: datetime | None = None,
                   limit: int = SNAPSHOT_NODE_LIMIT) -> Path | None:
    """原子写入快照：先写同目录临时文件再 replace，避免监控端读到半个 JSON。"""
    target = snapshot_path(data_dir)
    try:
        payload = build_snapshot(engine, now=now, limit=limit)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
        return target
    except Exception:
        logger.exception("memory_map_snapshot_failed")
        return None


class MemoryMapReader:
    """监控侧只读读取器：坏文件/缺失/超限都返回可解释的空结果。"""

    def __init__(self, root: Path):
        self._path = Path(root) / SNAPSHOT_FILENAME

    def read(self) -> dict:
        try:
            if not self._path.is_file():
                return {"available": False, "reason": "snapshot_missing",
                        "read_only": True, "stats": {}, "nodes": []}
            if self._path.stat().st_size > MAX_SNAPSHOT_BYTES:
                return {"available": False, "reason": "snapshot_too_large",
                        "read_only": True, "stats": {}, "nodes": []}
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"available": False, "reason": "snapshot_unreadable",
                    "read_only": True, "stats": {}, "nodes": []}
        if (not isinstance(payload, dict) or payload.get("version") != SNAPSHOT_VERSION
                or not isinstance(payload.get("nodes"), list)
                or not isinstance(payload.get("stats"), dict)):
            return {"available": False, "reason": "snapshot_incompatible",
                    "read_only": True, "stats": {}, "nodes": []}
        return {**payload, "available": True, "read_only": True}
