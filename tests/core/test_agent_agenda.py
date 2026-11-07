"""Agent 自我层：议程增删改查、项目生命周期、夜间议程生成解析。"""

from datetime import datetime, timedelta

from app.core import agent_agenda
from app.storage.db import session_scope
from app.storage.models import AgentProject

NOW = datetime(2026, 11, 2, 12, 0)


def test_add_and_list_and_dedup(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        first = agent_agenda.add_item(session, kind="todo", content="去查一下新番评价", now=NOW)
        dup = agent_agenda.add_item(session, kind="todo", content="去查一下新番评价", now=NOW)
        agent_agenda.add_item(session, kind="todo", content="主动问候小甲",
                              source="owner", now=NOW)
        rows = agent_agenda.list_pending(session)
    assert first is not None and dup is None
    assert [r.content for r in rows] == ["去查一下新番评价", "主动问候小甲"]  # owner 优先


def test_complete_and_drop(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        agent_agenda.add_item(session, kind="todo", content="做A", now=NOW)
        agent_agenda.add_item(session, kind="todo", content="做B", now=NOW)
        rows = agent_agenda.list_pending(session)
    a_id, b_id = rows[0].id, rows[1].id
    with session_scope(sqlite_engine) as session:
        assert agent_agenda.complete_item(session, a_id, note="搞定了", now=NOW) is True
        assert agent_agenda.drop_item(session, b_id) is True
        assert agent_agenda.complete_item(session, a_id, now=NOW) is False  # 已 done
    with session_scope(sqlite_engine) as session:
        assert agent_agenda.list_pending(session) == []


def test_short_content_rejected(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        assert agent_agenda.add_item(session, kind="todo", content="嗯", now=NOW) is None


def test_agenda_prompt_line_marks_owner(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        agent_agenda.add_item(session, kind="todo", content="主人交代的事",
                              source="owner", now=NOW)
        line = agent_agenda.agenda_prompt_line(session, now=NOW)
    assert line is not None and "主人交代" in line


def test_agenda_prompt_line_marks_stale_items_and_pushes_cleanup(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        agent_agenda.add_item(session, kind="todo", content="刚想的事", now=NOW)
        agent_agenda.add_item(session, kind="todo", content="很久之前想的事",
                              now=NOW - timedelta(days=3))
        line = agent_agenda.agenda_prompt_line(session, now=NOW)
    assert line is not None
    # 只有放了一天的条目带天数标记；新条目不带。
    assert "很久之前想的事（已放3天）" in line
    assert "刚想的事" in line and line.count("（已放") == 1
    assert "done" in line and "drop" in line


def test_project_lifecycle_and_cap(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        p1 = agent_agenda.start_project(session, title="项目一", plan="计划A", now=NOW)
        agent_agenda.start_project(session, title="项目二", now=NOW)
        agent_agenda.start_project(session, title="项目三", now=NOW)  # 超上限 → 最旧搁置
        rows = agent_agenda.list_active_projects(session)
        assert len(rows) == 2
        assert all(r.status == "active" for r in rows)
        rows = agent_agenda.list_active_projects(session)
        target_id = rows[0].id        # 项目二（项目一已被上限搁置）
        agent_agenda.progress_project(session, target_id, note="推进了第一步", now=NOW)
        agent_agenda.finish_project(session, target_id, deliverable="成果交付", now=NOW)
        rows = agent_agenda.list_active_projects(session)
        assert len(rows) == 1
    with session_scope(sqlite_engine) as session:
        done = session.get(AgentProject, target_id)
        assert done.status == "done" and done.deliverable == "成果交付"
        assert done.work_log[-1]["note"] == "推进了第一步"


def test_parse_agenda_json_defensive():
    raw = ('好的，明天的议程：[{"kind": "todo", "content": "查一下新番评价"},'
           ' {"content": "问候小甲"}, {"content": "嗯"}]')
    items = agent_agenda.parse_agenda_json(raw)
    assert items == [{"kind": "todo", "content": "查一下新番评价"},
                     {"kind": "todo", "content": "问候小甲"}]
    assert agent_agenda.parse_agenda_json("没有JSON") == []
    assert agent_agenda.parse_agenda_json('[{"content": "嗯"}]') == []  # 单字拒绝


def test_stale_pending_cleanup(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        agent_agenda.add_item(session, kind="todo", content="三天前的事",
                              now=NOW - timedelta(days=4))
        agent_agenda.add_item(session, kind="todo", content="今天的事", now=NOW)
    now2 = NOW + timedelta(days=1)
    with session_scope(sqlite_engine) as session:
        dropped = agent_agenda.cleanup_old_pending(session, now=now2)
    assert dropped == 1
    with session_scope(sqlite_engine) as session:
        rows = agent_agenda.list_pending(session)
    assert [r.content for r in rows] == ["今天的事"]
