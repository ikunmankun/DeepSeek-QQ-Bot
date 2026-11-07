"""Agent 自我插件：议程/项目管理工具 + 观察轮注入 + 夜间议程生成。

挂在日记反思的钩子上：每晚写完日记后，基于当天素材生成明天的议程
（0~3 条）、提炼"想要的新功能"愿望（私信主人）、凝练人生笔记——
这就是"想法→行动"闭环与自传体长记忆的生成端。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core import agent_agenda
from app.core.time_utils import shanghai_now_naive

logger = logging.getLogger(__name__)

_WEEKDAY = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}


class AgentSelfFeature(FeatureModule):
    name = "agent_self"

    _engine = None
    _llm_client = None
    _persona_text = ""
    _bot_qq = 0
    _hook = None
    _settings = None
    _data_dir = None

    def attach(self, *, engine, llm_client, persona_text, bot_qq,
               reflect_hook=None, settings=None, data_dir=None) -> None:
        self._engine = engine
        self._llm_client = llm_client
        self._persona_text = persona_text
        self._bot_qq = int(bot_qq)
        self._hook = reflect_hook
        self._settings = settings
        self._data_dir = data_dir

    def bind_reflection(self, service) -> None:
        """把夜间议程生成挂到日记反思服务上。"""
        service._reflect_hook = self._nightly_agenda

    # -- 夜间议程生成 ----------------------------------------------------

    def _nightly_agenda(self, now) -> None:
        from app.storage.db import session_scope

        diary_material = ""
        # 复用当天的反思素材（简化：直接让 LLM 基于议程现状+自由发挥）
        with session_scope(self._engine) as session:
            existing = agent_agenda.list_pending(session, limit=8)
            existing_text = "；".join(r.content for r in existing) if existing else "（空）"
            prompt = [
                "System persona: " + self._persona_text.strip(),
                "Group policy: 你刚写完今天的日记。现在给明天排议程：想做的具体的事，"
                "0到3条。可以是：跟进日记里提到的某件事、主动找某人聊、查一个好奇的东西、"
                "推进某个想法。别写日常应酬，别重复已有议程。"
                "输出 JSON 数组：[{\"kind\": \"todo\", \"content\": \"...\"}]，"
                "没有想做的就输出 []。",
                "已有议程（别重复）：" + existing_text,
            ]
            raw = self._llm_client.generate_text(prompt, conversation_key=f"agenda:{now:%Y-%m-%d}")
            items = agent_agenda.parse_agenda_json(raw)
            added = 0
            for item in items:
                row = agent_agenda.add_item(session, kind=item["kind"],
                                            content=item["content"], source="self", now=now)
                if row is not None:
                    added += 1
            dropped = agent_agenda.cleanup_old_pending(session, now=now)
            if added or dropped:
                logger.info("nightly_agenda_generated added=%s dropped=%s", added, dropped)

    # -- 夜间愿望提取：想要新功能就私信主人 ------------------------------

    _WISH_PROMPT = (
        "你刚写完今天的日记。回想今天有没有哪种时刻：你想做一件事但发现自己没有那个能力——"
        "比如认不出一首歌、查不了某个东西、看不懂某种图、记不住该记的事。"
        "把这种「想要一个新功能」的时刻提炼成 0 到 2 条愿望，说清楚：想要什么能力、"
        "为什么想要（用今天的具体场景）。只从日记里提取，别凭空造愿望。"
        "输出 JSON 数组：[{\"content\": \"想要的能力：原因（含具体场景）\"}]，"
        "没有就输出 []。"
    )

    def _nightly_wishes(self, now) -> None:
        from app.storage.db import session_scope
        from app.storage.repositories.jobs import JobRepository

        if self._settings is not None and not getattr(self._settings, "owner_wish_notice_enabled", True):
            return
        with session_scope(self._engine) as session:
            pending = agent_agenda.list_pending_wishes(session)
            pending_text = "；".join(row.content for row in pending) if pending else "（空）"
            prompt = [
                "System persona: " + self._persona_text.strip(),
                self._WISH_PROMPT,
                "已有愿望（别重复）：" + pending_text,
            ]
            raw = self._llm_client.generate_text(prompt, conversation_key=f"wish:{now:%Y-%m-%d}")
            entries = _parse_wish_json(raw)
            new_ids: list[int] = []
            with session_scope(self._engine) as session:
                for content in entries:
                    row = agent_agenda.add_wish(session, content=content, now=now,
                                                origin="nightly")
                    if row is not None:
                        new_ids.append(row.id)
            if not new_ids:
                return
            # 有一批新愿望 → 组装一条私信文本投进 jobs，由私聊进程的
            # WishNoticeWorker 在非安静时段发给主人（job_key 天级去重）。
            with session_scope(self._engine) as session:
                fresh = [session.get(agent_agenda.AgentAgendaItem, item_id)
                         for item_id in new_ids]
                wish_lines = "\n".join(
                    f"- {row.content}" for row in fresh if row is not None)
                notice = self._llm_client.generate_text([
                    "System persona: " + self._persona_text.strip(),
                    "Group policy: 你最近想要几个自己还没有的能力，现在用一条私信告诉主人。"
                    "用你自己的口吻（傲娇、直接、别客套），把下面的愿望合自然地讲清楚，"
                    "顺便说说为什么想要。200 字以内，只发这一条，别撒娇过头。"
                    "直接输出私信正文，别加引号或说明。",
                    "愿望列表：\n" + wish_lines,
                ], conversation_key=f"wish-notice:{now:%Y-%m-%d}")
                JobRepository(session).add_job(
                    job_type="owner_wish_notice",
                    job_key=f"owner-wish-notice:{self._bot_qq}:{now:%Y-%m-%d}",
                    payload_json={"wish_ids": new_ids,
                                  "text": str(notice or "").strip()[:600],
                                  "date": f"{now:%Y-%m-%d}"},
                    run_at=now,
                )
            logger.info("nightly_wishes_extracted new=%s", len(new_ids))

    # -- 夜间人生笔记凝练 -------------------------------------------------

    def _nightly_life_notes(self, now) -> None:
        if self._settings is not None and not getattr(self._settings, "life_note_enabled", True):
            return
        from app.core.life_notes import consolidate_life_notes

        provider = None
        try:
            if self._settings is not None:
                from app.providers.semantic_embeddings import build_embedding_provider

                provider = build_embedding_provider(
                    provider=self._settings.memory_embedding_provider,
                    device=self._settings.memory_embedding_device,
                    model=self._settings.memory_embedding_model,
                    dimensions=self._settings.memory_embedding_dimensions,
                    cache_dir=self._settings.memory_embedding_cache_dir,
                    local_files_only=self._settings.memory_embedding_local_files_only,
                    version=self._settings.memory_embedding_version,
                    base_url=self._settings.memory_embedding_base_url,
                    api_key=self._settings.memory_embedding_api_key,
                    timeout_seconds=self._settings.memory_embedding_timeout_seconds,
                )
        except Exception:
            logger.exception("life_notes_embedding_provider_failed")
            provider = None
        consolidate_life_notes(
            self._engine, llm_client=self._llm_client,
            persona_text=self._persona_text, data_dir=self._data_dir, now=now,
            min_interval_days=int(getattr(self._settings, "life_note_min_interval_days", 3) or 3),
            max_notes=int(getattr(self._settings, "life_note_max_count", 200) or 200),
            embedding_provider=provider,
        )

    # -- 工具 ------------------------------------------------------------

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        async def agenda_manage(args):
            turn.guard.require_valid()
            action = str(args.get("action") or "list").strip()
            now = shanghai_now_naive()
            from app.storage.db import session_scope

            if action == "add":
                content = str(args.get("content") or "").strip()
                if len(content) < 2:
                    return ToolOutcome({"error": "content_too_short"}, "blocked")
                with session_scope(self._engine) as session:
                    row = agent_agenda.add_item(session, kind="todo", content=content,
                                                source="owner", now=now)
                return ToolOutcome({"added": row is not None,
                    "note": "" if row is not None else "已有相同议程，未重复添加。"})
            if action in {"done", "drop"}:
                item_id = int(args.get("item_id") or 0)
                with session_scope(self._engine) as session:
                    ok = (agent_agenda.complete_item(session, item_id, now=now)
                          if action == "done" else agent_agenda.drop_item(session, item_id))
                return ToolOutcome({"updated": ok})
            # list
            with session_scope(self._engine) as session:
                rows = agent_agenda.list_pending(session, limit=8)
                return ToolOutcome({"items": [
                    {"id": r.id, "content": r.content, "source": r.source} for r in rows]})

        async def project_manage(args):
            turn.guard.require_valid()
            action = str(args.get("action") or "").strip()
            now = shanghai_now_naive()
            from app.storage.db import session_scope

            project_id = int(args.get("project_id") or 0)
            if action == "start":
                title = str(args.get("title") or "").strip()
                if len(title) < 2:
                    return ToolOutcome({"error": "title_too_short"}, "blocked")
                with session_scope(self._engine) as session:
                    row = agent_agenda.start_project(session, title=title,
                        plan=str(args.get("plan") or ""), now=now)
                    return ToolOutcome({"started": True, "project_id": row.id,
                        "note": "项目已立项。观察轮里会提醒你推进；完成时用 finish 交付。"})
            if action == "progress":
                with session_scope(self._engine) as session:
                    ok = agent_agenda.progress_project(session, project_id,
                        note=str(args.get("note") or ""), now=now)
                return ToolOutcome({"updated": ok})
            if action == "finish":
                with session_scope(self._engine) as session:
                    ok = agent_agenda.finish_project(session, project_id,
                        deliverable=str(args.get("deliverable") or ""), now=now)
                return ToolOutcome({"finished": ok,
                    "note": "项目完成。想分享成果就在群里说，或写进日记。" if ok else ""})
            if action == "shelve":
                with session_scope(self._engine) as session:
                    ok = agent_agenda.finish_project(session, project_id,
                        deliverable="搁置", now=now)
                return ToolOutcome({"shelved": ok})
            with session_scope(self._engine) as session:
                rows = agent_agenda.list_active_projects(session)
                return ToolOutcome({"projects": [
                    {"id": r.id, "title": r.title, "plan": r.plan,
                     "log": (r.work_log or [])[-3:]} for r in rows]})

        async def record_wish(args):
            turn.guard.require_valid()
            content = str(args.get("content") or "").strip()
            now = shanghai_now_naive()
            from app.storage.db import session_scope

            with session_scope(self._engine) as session:
                row = agent_agenda.add_wish(session, content=content, now=now,
                                            origin="tool")
            if row is None:
                return ToolOutcome({"added": False,
                    "note": "重复愿望或愿望太多（上限5条），没重复记。"})
            return ToolOutcome({"added": True, "wish_id": row.id,
                "note": "记下了。夜间反思会汇总，主人会收到你的私信。"})

        return [
            AgentTool("agent_agenda",
                "管理你自己的日程/待办（add 加一条、done 完成一条、drop 划掉、list 看全部）。"
                "主人交代你做什么事时用 add 记下来（source 会标记主人）；完成观察轮里推进的事后用 done 打勾。",
                {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["add", "done", "drop", "list"]},
                    "content": {"type": "string", "maxLength": 300},
                    "item_id": {"type": "integer"},
                }, "required": ["action"], "additionalProperties": False},
                agenda_manage, "write"),
            AgentTool("agent_project",
                "管理你的长时程自主项目（start 立项、progress 记进展、finish 交付完成、shelve 搁置、list 看进行中）。"
                "想做一个需要跨天推进的事就立项；每天推进一点；完成后把成果说清楚。",
                {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["start", "progress", "finish", "shelve", "list"]},
                    "title": {"type": "string", "maxLength": 120},
                    "plan": {"type": "string", "maxLength": 600},
                    "project_id": {"type": "integer"},
                    "note": {"type": "string", "maxLength": 200},
                    "deliverable": {"type": "string", "maxLength": 400},
                }, "required": ["action"], "additionalProperties": False},
                project_manage, "write"),
            AgentTool("record_wish",
                "记一条「想要的新功能」愿望。什么时候用：你发现自己想做某件事但没有那个能力——"
                "认不出歌、查不了东西、看不懂某类内容、记不住该记的事——当时就记下来想跟主人要什么，"
                "写清楚想要什么能力+今天的具体场景。别记闲聊念头，只记能力缺口。攒着的愿望会私信主人。",
                {"type": "object", "properties": {
                    "content": {"type": "string", "maxLength": 300},
                }, "required": ["content"], "additionalProperties": False},
                record_wish, "write"),
        ]

    # -- 提示注入 --------------------------------------------------------

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group":
            return ()
        from app.storage.db import session_scope

        try:
            with session_scope(self._engine) as session:
                agenda = agent_agenda.agenda_prompt_line(session, now=now)
                projects = agent_agenda.projects_prompt_line(session, now=now)
                from app.core import life_notes

                life_line = life_notes.prompt_line(session, now=now)
        except Exception:
            logger.exception("agent_self_prompt_failed")
            return ()
        lines = []
        if agenda:
            lines.append(agenda)
        if projects:
            lines.append(projects)
        if life_line:
            lines.append(life_line)
        return lines


def _parse_wish_json(raw: str) -> list[str]:
    """防御式解析夜间愿望提取的 JSON 数组；坏了返回空。"""

    import re

    match = re.search(r"\[.*\]", str(raw or ""), re.S)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return []
    entries: list[str] = []
    for entry in payload[:2]:
        if isinstance(entry, dict):
            content = str(entry.get("content", "")).strip()
        elif isinstance(entry, str):
            content = entry.strip()
        else:
            continue
        if len(content) >= 4:
            entries.append(content[:300])
    return entries
