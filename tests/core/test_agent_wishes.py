"""功能愿望：add_wish 去重/上限/通知标记、过期清理豁免、夜间提取解析。"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.agent.plugins.agent_self import _parse_wish_json
from app.core import agent_agenda
from app.storage.db import session_scope

NOW = datetime(2026, 11, 7, 12, 0)


def test_add_wish_dedup_and_min_length(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        first = agent_agenda.add_wish(session, content="想要听歌识曲的能力：今天认不出歌",
                                      now=NOW)
        dup = agent_agenda.add_wish(session, content="想要听歌识曲的能力：今天认不出歌",
                                    now=NOW)
        short = agent_agenda.add_wish(session, content="太短", now=NOW)
    assert first is not None and dup is None and short is None


def test_add_wish_cap_and_mark_notified(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        ids = []
        for index in range(6):
            row = agent_agenda.add_wish(session, content=f"想要能力{index}：场景{index}",
                                        now=NOW)
            if row is not None:
                ids.append(row.id)
        assert len(ids) == agent_agenda.WISH_MAX_PENDING  # 第 6 条被拒
        assert agent_agenda.mark_wish_notified(session, ids[0], now=NOW) is True
        row = session.get(agent_agenda.AgentAgendaItem, ids[0])
        assert "notified:" in str(row.note or "")
        assert row.status == "pending"  # 等主人拍板，不自动 done
        # 重复标记与未知 id 安全
        assert agent_agenda.mark_wish_notified(session, ids[0], now=NOW) is True
        assert agent_agenda.mark_wish_notified(session, 99999, now=NOW) is False


def test_cleanup_keeps_wishes(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        wish = agent_agenda.add_wish(session, content="想要看视频的能力：想接梗",
                                     now=NOW - timedelta(days=5))
        todo = agent_agenda.add_item(session, kind="todo", content="过期议程",
                                     source="self", now=NOW - timedelta(days=5))
    with session_scope(sqlite_engine) as session:
        dropped = agent_agenda.cleanup_old_pending(session, now=NOW)
        wish_row = session.get(agent_agenda.AgentAgendaItem, wish.id)
        todo_row = session.get(agent_agenda.AgentAgendaItem, todo.id)
    assert dropped >= 1
    assert wish_row.status == "pending"  # 愿望不过期
    assert todo_row.status == "dropped"


def test_parse_wish_json_defensive():
    assert _parse_wish_json("不是json") == []
    assert _parse_wish_json('[{"content": "想要听歌识曲：认不出歌"}]') == ["想要听歌识曲：认不出歌"]
    assert _parse_wish_json('["纯字符串愿望也可以"]') == ["纯字符串愿望也可以"]
    assert _parse_wish_json('[{"content": "短"}]') == []
    assert _parse_wish_json("[]") == []


def test_list_pending_wishes_scope(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        agent_agenda.add_wish(session, content="想要生成图表的能力：想装懂行", now=NOW)
        agent_agenda.add_item(session, kind="todo", content="普通议程不是愿望",
                              source="self", now=NOW)
        rows = agent_agenda.list_pending_wishes(session)
    assert len(rows) == 1 and rows[0].kind == "wish"
