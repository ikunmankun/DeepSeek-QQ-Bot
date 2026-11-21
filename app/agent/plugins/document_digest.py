"""长文档要点提取插件：summarize_document 工具的群聊接线。

朋友甩来一篇超长文档/一整屏长文本，让机器人"看重点"——靠模型在
上下文里自由发挥不行：长消息进上下文会被预算截断，模型看到的都不全。
本工具把全文（从消息库原文读，不经上下文）交给分块 map-reduce 提取
（见 app/core/document_digest.py），返回结构化要点，模型用自己的口吻
转述。配额按天计；LLM client 与特质进化同款（主模型、独立输出预算），
未配置 LLM 时整个功能自动隐藏。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Sequence

from sqlalchemy import select

from app.agent.plugins.base import FeatureModule, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.document_digest import digest_client_enabled, digest_points
from app.core.memory_source_guard import is_usable_memory_source
from app.core.time_utils import shanghai_naive, shanghai_now_naive
from app.storage.db import session_scope
from app.storage.models import Message

logger = logging.getLogger(__name__)

#: 文档原文只从最近 7 天的本群消息里读——提炼重点针对的是"刚甩过来"
#: 的文档，不是翻历史记录的检索工具。
SOURCE_MAX_AGE_DAYS = 7.0
#: 单轮工具调用的来源上限（对齐 fuzzy_recall 的 source 预算习惯）。
MAX_SOURCE_IDS = 5
#: 每天工具调用配额：一次调用最多 9 次 LLM 请求（8 块 map + 1 reduce）。
DOCUMENT_DAILY_LIMIT = 12

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "source_ids": {"type": "array", "items": {"type": "integer", "minimum": 1,
                                                  "maximum": 2**63 - 1},
                       "minItems": 1, "maxItems": MAX_SOURCE_IDS},
        "text": {"type": "string", "minLength": 20, "maxLength": 60000},
        "focus": {"type": "string", "minLength": 1, "maxLength": 120},
    },
    "required": [],
    "additionalProperties": False,
}


class DocumentDigestFeature(FeatureModule):
    name = "document_digest"

    def __init__(self) -> None:
        # 每实例状态：配额按天重置。类属性会跨实例/测试污染。
        self._settings = None
        self._client = None
        self._usage_day: str = ""
        self._usage_count = 0

    def attach(self, *, settings) -> None:
        self._settings = settings
        self._client = digest_client_enabled(settings)

    def available(self) -> bool:
        return self._client is not None

    def _quota_ok(self) -> bool:
        now = shanghai_now_naive()
        today = f"{now:%Y-%m-%d}"
        if self._usage_day != today:
            self._usage_day = today
            self._usage_count = 0
        return self._usage_count < DOCUMENT_DAILY_LIMIT

    def _collect_source_text(self, turn: TurnContext, source_ids: Sequence[int]) -> str:
        """从本群消息库读原文全文；跨群/隔离/撤回来源一律读不到。"""
        ids = tuple(dict.fromkeys(int(value) for value in source_ids
                                  if type(value) is int and value > 0))[:MAX_SOURCE_IDS]
        if not ids:
            return ""
        cutoff = shanghai_naive(turn.now) - timedelta(days=SOURCE_MAX_AGE_DAYS)
        # 注意必须走 Session 层（session.scalars）：Connection.execute 的
        # Core 结果里 context_excluded 这个 exists column_property 会成为
        # 首列 anon_1，scalars() 取到的是布尔而不是实体。
        with session_scope(turn.router.engine) as session:
            rows = list(session.scalars(
                select(Message).where(
                    Message.group_id == turn.job.scope_id,
                    Message.id.in_(ids),
                )
            ))
        parts: list[str] = []
        for row in sorted(rows, key=lambda item: item.id):
            if not is_usable_memory_source(row, group_id=turn.job.scope_id):
                continue
            if row.timestamp is not None and shanghai_naive(row.timestamp) < cutoff:
                continue
            if str(row.plain_text or "").strip():
                parts.append(str(row.plain_text).strip())
        return "\n\n".join(parts)

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        if not self.available():
            return ()

        async def summarize_document(args: dict[str, Any]) -> ToolOutcome:
            turn.guard.require_valid()
            text = str(args.get("text") or "").strip()
            source_ids = args.get("source_ids") or []
            focus = str(args.get("focus") or "").strip()
            if not text and source_ids:
                def read_sources() -> str:
                    return self._collect_source_text(turn, source_ids)

                text = await turn.blocking.run(read_sources)
            text = (text or "").strip()
            if not text:
                return ToolOutcome({"error": "no_document_text",
                    "guidance": "没拿到文档内容。对方发的长文要是引用消息，"
                                "用引用消息的来源ID；是纯文本就让对方整段发来。"}, "blocked")
            if not self._quota_ok():
                return ToolOutcome({"error": "document_digest_daily_limit",
                    "note": "今天的文档提炼额度用完了，凭记忆简述+如实说明即可。"}, "blocked")

            def run_digest() -> dict[str, Any]:
                return digest_points(client=self._client, text=text, focus=focus)

            value = await asyncio.to_thread(run_digest)
            turn.guard.require_valid()
            if value.get("error"):
                logger.warning("document_digest_failed error=%s", value["error"])
                return ToolOutcome({"error": value["error"],
                    "note": "提炼失败了，别编重点；如实说明读不了这份文档。"}, "failed")
            self._usage_count += 1
            guidance = ("要点是文档里真有的内容，用你的口吻挑着讲：先一句话主旨，"
                        "再讲最相关的几条；别整列照念，别加文档没有的结论。")
            if value.get("truncated_note"):
                guidance += f"注意：{value['truncated_note']}，转述时要说清楚。"
            return ToolOutcome({**value, "guidance": guidance}, "ok")

        return [AgentTool(
            "summarize_document",
            "长文档要点提取：有人甩来一篇超长文本（测试文档、攻略、公告、聊天记录导出）"
            "让你看重点、划重点、总结一下时用。全文分块精读后给出主旨+要点清单，"
            "不靠你上下文里的残缺片段。对方刚发的长文本消息用其来源ID（source_ids），"
            "或直接把整段文本传进 text；focus 可以写对方关心的方向（比如'重点看第几章'、"
            "'跟支付有关的部分'）。同步出结果，稍等几秒正常。",
            SUMMARIZE_SCHEMA,
            summarize_document, "read")]
