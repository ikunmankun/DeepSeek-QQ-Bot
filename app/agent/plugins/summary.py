"""聊天总结工具：基于既有 episode 摘要回答"今天群里聊了啥"。"""

from __future__ import annotations

from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome


class ChatSummaryFeature(FeatureModule):
    """把记忆系统的 Summary 暴露成一个全员可用的查询工具。"""

    name = "chat_summary"

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        async def summarize(args):
            turn.guard.require_valid()
            hours = 24
            from datetime import timedelta

            from app.core.time_utils import shanghai_now_naive

            now = shanghai_now_naive()
            start = now - timedelta(hours=hours)
            with turn.router.engine.connect() as connection:
                from sqlalchemy import select

                from app.storage.models import Summary

                stmt = (
                    select(Summary.content)
                    .where(
                        Summary.scope_type == "group",
                        Summary.scope_id == str(turn.job.scope_id),
                        Summary.status == "active",
                        Summary.end_at >= start,
                    )
                    .order_by(Summary.end_at.asc())
                    .limit(10)
                )
                rows = connection.execute(stmt).scalars().all()
            if not rows:
                return ToolOutcome({"entries": [],
                    "note": "最近24小时没有已生成的聊天摘要。如实告诉对方即可。"})
            entries = [str(row)[:400] for row in rows]
            return ToolOutcome({"entries": entries,
                "guidance": "用你自己的语气把这几段概括给对方，别整段照念、别列点。"})

        return [AgentTool(
            "summarize_group_chat",
            "总结最近24小时本群的聊天内容（基于已生成的记忆摘要）。"
            "有人问'今天群里聊了啥''刚才在聊什么'时用。回答要用你的语气口语化转述。",
            {"type": "object", "properties": {}, "required": [],
             "additionalProperties": False},
            summarize, "read")]
