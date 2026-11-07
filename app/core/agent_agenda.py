"""agent 自我层：议程、项目与"想法→行动"闭环。

这是把机器人从"响应式"变成"自主"的核心：
* 议程（AgentAgendaItem）——想做的具体的事。夜间反思从当天素材里生成
  明天的议程（0~3 条），观察轮里注入并推进，做完打勾；
* 项目（AgentProject）——跨天的长时程自主项目（如"这周整理群友爱看的
  视频类型"），每晚推进一点，完成后可交付；
* 主人可以交代事项（source=owner），优先级高于自生成。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from app.core.time_utils import shanghai_naive, shanghai_now_naive
from app.storage.models import AgentAgendaItem, AgentProject

AGENDA_MAX_PENDING = 6
PROJECT_MAX_ACTIVE = 2
_MAX_ITEMS_FROM_LLM = 3


def add_item(session, *, kind: str, content: str, source: str = "self",
             group_id: int = 0, now: datetime) -> AgentAgendaItem | None:
    """新增议程项；与现有 pending 内容重复时跳过（返回 None）。"""

    content = str(content or "").strip()[:300]
    if len(content) < 2:  # 中文两字也是合法议程（“买猫粮”），单字才拒
        return None
    existing = session.scalars(
        select(AgentAgendaItem).where(
            AgentAgendaItem.status == "pending",
            AgentAgendaItem.content == content,
        ).limit(1)
    ).first()
    if existing is not None:
        return None
    row = AgentAgendaItem(kind=str(kind or "todo")[:16], content=content,
                          source=str(source or "self")[:16], group_id=int(group_id or 0),
                          created_at=now)
    session.add(row)
    return row


# ---------------------------------------------------------------- 功能愿望

#: pending 愿望上限：愿望是"想要新功能"，攒太多说明该清了。
WISH_MAX_PENDING = 5


def add_wish(session, *, content: str, now: datetime,
             origin: str = "") -> AgentAgendaItem | None:
    """记一条功能愿望（kind='wish'）；重复/超量跳过。愿望不过期——

    cleanup_old_pending 只清 self 源的 todo，wish 单独处理（由通知
    worker 在投递后 mark_wish_notified，主人没回应的愿望继续挂着）。
    """

    content = str(content or "").strip()[:300]
    if len(content) < 4:  # "想要一个画图的" 这种短句也要说得清
        return None
    existing = session.scalars(
        select(AgentAgendaItem).where(
            AgentAgendaItem.status == "pending",
            AgentAgendaItem.kind == "wish",
            AgentAgendaItem.content == content,
        ).limit(1)
    ).first()
    if existing is not None:
        return None
    pending = list_pending_wishes(session)
    if len(pending) >= WISH_MAX_PENDING:
        return None
    row = AgentAgendaItem(kind="wish", content=content, source="self",
                          group_id=0, created_at=now,
                          note=str(origin or "")[:300])
    session.add(row)
    session.flush()  # 调用方要拿 row.id 入通知 payload
    return row


def list_pending_wishes(session) -> list[AgentAgendaItem]:
    return list(session.scalars(
        select(AgentAgendaItem)
        .where(AgentAgendaItem.status == "pending", AgentAgendaItem.kind == "wish")
        .order_by(AgentAgendaItem.id.asc())))


def mark_wish_notified(session, item_id: int, *, now: datetime) -> bool:
    """愿望已私信主人：note 打上时间戳；条目保持 pending 等主人拍板。"""

    row = session.get(AgentAgendaItem, int(item_id))
    if row is None or row.status != "pending" or row.kind != "wish":
        return False
    stamp = f"notified:{now:%Y-%m-%d %H:%M}" if now is not None else "notified"
    row.note = (str(row.note or "") + " " + stamp).strip()[:300]
    return True


def list_pending(session, *, limit: int = AGENDA_MAX_PENDING) -> list[AgentAgendaItem]:
    rows = list(session.scalars(
        select(AgentAgendaItem)
        .where(AgentAgendaItem.status == "pending")
        .order_by(AgentAgendaItem.source.desc(), AgentAgendaItem.id.asc())  # owner 优先
        .limit(limit)
    ))
    return rows


def complete_item(session, item_id: int, *, note: str = "", now: datetime) -> bool:
    row = session.get(AgentAgendaItem, int(item_id))
    if row is None or row.status != "pending":
        return False
    row.status = "done"
    row.done_at = now
    if note:
        row.note = str(note)[:300]
    return True


def drop_item(session, item_id: int) -> bool:
    row = session.get(AgentAgendaItem, int(item_id))
    if row is None or row.status != "pending":
        return False
    row.status = "dropped"
    return True


def agenda_prompt_line(session, *, now: datetime) -> str | None:
    rows = list_pending(session, limit=5)
    if not rows:
        return None
    current = shanghai_naive(now) if now is not None else shanghai_now_naive()
    items = []
    for row in rows:
        marker = ""
        if row.created_at is not None:
            age_days = int(max(0.0, (current - shanghai_naive(row.created_at)).total_seconds())) // 86400
            if age_days >= 1:
                marker = f"（已放{age_days}天）"
        items.append(f"{row.id}. {row.content}{marker}"
                     + ("（主人交代）" if row.source == "owner" else ""))
    return ("你最近的日程（想做但还没做的事）："
            + "；".join(items)
            + "。挂着的旧日程要清：这轮能推进就真推进一点，"
            "做完用 agent_agenda 工具 done 打勾；明显做不了或过时了就 drop 掉，别一直挂着。")


# ---------------------------------------------------------------- 项目

def start_project(session, *, title: str, plan: str = "", now: datetime) -> AgentProject:
    active = session.scalars(
        select(AgentProject).where(AgentProject.status == "active")
    ).all()
    if len(active) >= PROJECT_MAX_ACTIVE:
        oldest = min(active, key=lambda r: r.last_worked_at or r.started_at)
        oldest.status = "shelved"
    row = AgentProject(title=str(title or "").strip()[:120],
                       plan=str(plan or "").strip()[:600], status="active")
    session.add(row)
    return row


def progress_project(session, project_id: int, *, note: str, now: datetime) -> bool:
    row = session.get(AgentProject, int(project_id))
    if row is None or row.status != "active":
        return False
    log = list(row.work_log or [])
    log.append({"date": f"{now:%m-%d}", "note": str(note or "")[:200]})
    row.work_log = log[-12:]
    row.last_worked_at = now
    return True


def finish_project(session, project_id: int, *, deliverable: str, now: datetime) -> bool:
    row = session.get(AgentProject, int(project_id))
    if row is None or row.status != "active":
        return False
    row.status = "done"
    row.deliverable = str(deliverable or "")[:400]
    row.last_worked_at = now
    return True


def list_active_projects(session) -> list[AgentProject]:
    return list(session.scalars(
        select(AgentProject).where(AgentProject.status == "active")
        .order_by(AgentProject.last_worked_at.desc().nulls_last())
    ))


def projects_prompt_line(session, *, now: datetime) -> str | None:
    rows = list_active_projects(session)
    if not rows:
        return None
    bits = []
    for row in rows[:PROJECT_MAX_ACTIVE]:
        log = (row.work_log or [])
        last = str(log[-1]["note"])[:80] if log and isinstance(log[-1], dict) else ""
        bits.append(f"{row.id}. {row.title}" + (f"（上次：{last}）" if last else ""))
    return ("你在推进的自主项目（观察轮里可以顺手推进一点并记录，用 agent_project 工具更新；"
            "不必每天都要有进展）：" + "；".join(bits))


# ---------------------------------------------------------------- 夜间生成

def parse_agenda_json(raw: str) -> list[dict[str, str]]:
    """从 LLM 输出里防御式解析议程 JSON 数组；坏了返回空。"""

    match = re.search(r"\[.*\]", str(raw or ""), re.S)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return []
    items = []
    for entry in payload[:_MAX_ITEMS_FROM_LLM]:
        if isinstance(entry, dict):
            content = str(entry.get("content", "")).strip()
            if len(content) >= 2:
                items.append({"kind": str(entry.get("kind", "todo"))[:16],
                              "content": content[:300]})
        elif isinstance(entry, str) and len(entry.strip()) >= 2:
            items.append({"kind": "todo", "content": entry.strip()[:300]})
    return items


def stale_cutoff(now: datetime) -> datetime:
    return now - timedelta(days=3)


def cleanup_old_pending(session, *, now: datetime) -> int:
    """三天前生成仍未做的 pending 项标记 dropped（议程不是愿望清单，过期就放）。

    wish 不参与过期清理：愿望等主人拍板，投递状态由通知 worker 维护。
    """

    cutoff = stale_cutoff(now)
    rows = list(session.scalars(
        select(AgentAgendaItem).where(
            AgentAgendaItem.status == "pending",
            AgentAgendaItem.source == "self",
            AgentAgendaItem.kind != "wish",
            AgentAgendaItem.created_at < cutoff,
        )))
    dropped = 0
    for row in rows:
        row.status = "dropped"
        dropped += 1
    return dropped
