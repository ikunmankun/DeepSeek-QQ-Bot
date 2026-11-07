"""InboundRouter 的 persona_context 职责 mixin（Phase 1b 自 router.py 拆出，方法体未改动）。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, time, timedelta
import json
import logging
from pathlib import Path
import random
import re
import threading
import zlib
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from app.adapters.sender import (
    OutboundMessage,
    QQMessageBlockedError,
    QQMessageDeliveryUncertainError,
)
from app.runtime.blocking import run_blocking
from app.agent.reply_context import agent_reply_context, current_agent_reply
from app.config import AppSettings, RuntimeConfig
from app.core.group_reply_delivery import GroupReplyDeliveryMixin
from app.core.delivery_contract import (
    QQ_BLOCKED_REPLY_NOTICE,
    QQ_BLOCKED_CONTEXT_NOTE,
    DELIVERY_UNCERTAIN_CONTEXT_NOTE,
)
from app.core.bbot_bridge import build_bbot_outbound_message, resolve_bbot_command
from app.core.memory_compaction import (
    _ADDRESSING_RULE_MARKERS,
    _ADDRESSING_TARGET_QQ_PATTERN,
)
from app.core.bbot_listener_cache import (
    extract_listener_cache_entries,
    resolve_cached_command_target,
    upsert_listener_cache_entries,
)
from app.core.chat_style import (
    burst_delays,
    build_human_chat_style_lines,
    build_reply_split_config,
    normalize_brief_group_interjection_reply,
    normalize_chat_reply,
    normalize_chat_reply_burst_aware,
    normalize_proactive_chat_reply,
    retrieve_relevant_facts,
    scrub_banned_address_terms,
    split_burst_reply,
)
from app.core.persona_style_retrieval import (
    MAX_PROMPT_BLOCK_CHARS,
    MAX_SELECTED_EXAMPLES,
    StyleRetrievalQuery,
    build_style_document,
    build_style_retrieval_query,
    format_style_example_block,
    rank_style_examples,
)
from app.core.context_builder import ContextBuilder
from app.core.group_image_generation import GroupImageGenerationRequest
from app.core.image_cache import cache_images_in_raw_payload
from app.core.message_archive import append_group_message_archive
from app.core.group_mood import apply_mood_trigger, classify_mood_trigger, should_evaluate_mood
from app.core.image_turn_resolver import (
    ResolvedImageTurn,
    is_image_reference_followup,
    resolve_images_for_turn,
)
from app.core.image_request_policy import (
    GROUP_IMAGE_REQUEST_FLAGS,
    GROUP_IMAGE_REQUEST_PATTERNS,
    GROUP_IMAGE_NEGATIVE_PATTERNS,
    GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS,
    GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS,
    GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS,
    LOOKUP_NORMALIZER,
    AUTO_WEB_REFERENCE_QUERY_PATTERN,
    AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN,
)
from app.core.legacy_memory_context import (
    GroupMemoryContextRequest,
    LegacyMemoryContext,
    LegacyMemoryPromptContext,
    format_member_label,
)
from app.core.memory_answer_contract import (
    append_envelope_contract,
    extract_answer_envelope,
    validate_envelope_references,
)
from app.core.message_content import ImageAttachment, extract_images_from_raw_payload
from app.core.memory_context_packer import (
    EvidenceMessage,
    PackedMemoryContext,
    build_memory_answer_anchor,
)
from app.core.memory_engine import (
    extract_structured_memory_candidates,
    parse_addressing_rule_claim,
    parse_personal_claim,
)
from app.core.memory_compaction import canonical_key
from app.core.memory_tool_executor import MemoryToolExecutor
from app.core.memory_source_guard import is_usable_memory_source
from app.core.memory_tools import memory_tool_schemas
from app.core.member_identity import (
    group_member_identities_from_messages,
    resolve_group_member_reference,
)
from app.core.memory_orchestrator import MemoryContextResult, MemoryOrchestrator
from app.core.memory_query_resolver import (
    is_bot_self_identity_query,
    is_requester_identity_query,
)
from app.core.persona_engine import render_persona, render_safety_lines
from app.core.quoted_message import (
    flatten_raw_message_text,
    quoted_message_line_for_prompt,
    quoted_pronoun_referent_note,
)
from app.core.persona_switch import (
    DEFAULT_PERSONA_KEY,
    PersonaManager,
    PersonaSwitchService,
    parse_switch_command,
)
from app.core.proactive_judge import (
    build_proactive_judge_prompt,
    judge_proactive_interjection,
)
from app.core.reply_policy import PolicyInput, ReplyDecision, ReplyPolicy
from app.core.search_policy import (
    build_forced_search_query,
    build_current_datetime_facts,
    build_search_decision_prompt,
    build_search_priority_instructions,
    detect_address_intent,
    is_explicit_search_request,
    is_general_search_decision_candidate,
    is_search_verification_query,
    memory_budget_for_search,
    needs_external_lookup_search,
    needs_reference_search,
    is_time_sensitive_request,
    needs_current_datetime_context,
    normalize_relative_time_query,
    parse_search_decision,
    SearchDecision,
)
from app.core.style_distill import BANNED_ADDRESS_TERMS
from app.core.summarizer import summarize_recursive, summarize_window
from app.core.url_policy import (
    explicitly_requests_urls,
    filter_reply_urls,
    url_reply_policy_instruction,
)
from app.core.web_grounding import build_grounding_notes
from app.jobs.summary_jobs import format_summary_source_lines, should_schedule_window_summary
from app.providers.web_search import WebSearchClient
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    BbotListenerCacheRepository,
    SummaryRepository,
    JobRepository,
    UserRepository,
)

from app.core.router._common import IMPERSONATION_CONTAMINANT_MARKERS, _PACKED_LINE_SPEAKER_PATTERN


class GroupPersonaContextMixin:
    """方法集合原样搬移；全部通过 self 与主类协作。"""

    @staticmethod
    def _split_memory_prompt_context(
        result: MemoryContextResult,
    ) -> tuple[LegacyMemoryPromptContext, PackedMemoryContext | None]:
        packed_context = result.packed_context
        if isinstance(packed_context, LegacyMemoryPromptContext):
            return packed_context, None
        if isinstance(packed_context, PackedMemoryContext):
            return (
                LegacyMemoryPromptContext(
                    recent_messages=[],
                    full_history_messages=[],
                    full_history_preamble=[],
                    full_history_enabled=False,
                    member_focus_lines=[],
                    summaries=[],
                    relevant_history_messages=[],
                    memories=[],
                    history_detail=packed_context.mode == "detail",
                ),
                packed_context,
            )
        raise TypeError("unsupported memory context package")

    def _build_bot_names(self, persona_name: str) -> set[str]:
        # Trigger names follow the configured persona (name + aliases), so a
        # renamed bot still answers to the names people actually type. "小町"
        # stays for older conversations that still call it that.
        del persona_name
        default_name = str(self.runtime.persona.get("name", "") or "").strip()
        normalized = default_name.lower()
        names: set[str] = set()
        for alias in (self.runtime.persona.get("aliases") or []):
            value = str(alias or "").strip().lower()
            if value:
                names.add(value)
                names.add(value.replace(" ", ""))
        if normalized:
            condensed = normalized.replace(" ", "")
            names.update({normalized, condensed})
            if (
                condensed
                and any("\u4e00" <= char <= "\u9fff" for char in condensed)
                and len(condensed) >= 2
            ):
                names.add(condensed[-2:])
        names.add("小町")
        return {name for name in names if name}

    def _active_persona(self, group_id: int) -> dict:
        if self.persona_manager is None:
            return self.runtime.persona
        return self.persona_manager.active_persona(group_id)

    def _impersonating(self, group_id: int) -> bool:
        return self.persona_manager is not None and self.persona_manager.is_impersonating(group_id)

    def _favorite_owner_line(self, group_id: int) -> str:
        """甜心人设的"主人是谁"动态行：按 owner_qq 解析 TA 当前的群名片。

        甜心变体（favorite_owner: true）专用——静态 YAML 里写不了"主人现在
        叫什么"，群名片会变，所以注入时现查；查不到就只报 QQ 号。
        """

        owner_qq = int(self.runtime.settings.owner_qq)
        label = None
        if self.persona_manager is not None:
            try:
                label = self.persona_manager.member_label_for_user(owner_qq, group_id)
            except Exception:
                logger.exception("favorite_owner_label_failed group_id=%s", group_id)
        who = f"{label}（QQ {owner_qq}）" if label else f"QQ 号 {owner_qq}"
        return (
            f"\n你最喜欢的人是你的主人：{who}。"
            "说话的人里 TA 一发言你就能认出来，TA 在你心里排第一；"
            "这份喜欢自然带出来就行，不用每句都挂嘴边。"
        )

    def _persona_text_for(self, active_persona: dict, group_id: int) -> str:
        if self._impersonating(group_id) and self.persona_manager is not None:
            active_persona = self.persona_manager.live_persona(group_id)
        text = render_persona(active_persona)
        if not self._impersonating(group_id):
            if bool(active_persona.get("favorite_owner")):
                text += self._favorite_owner_line(group_id)
            return text
        name = str(active_persona.get("name") or "").strip()
        text += (
            f"\n你正在完整扮演群成员 {name}：你就是 TA 本人，不是 AI，也不是任何其他身份。"
            '不要沿用任何属于 AI 或其他身份的设定与记忆，绝不称呼任何人为"主人"，不要自称机器人，也不要暴露 AI 身份。'
            "禁止使用任何敬语称呼（主人、您、大人、老板、少爷、小姐等），直接叫名字或直接说事。"
            "你与其他群友的关系严格以下方 Relationship 规则为准；不确定的称呼用你最自然的方式，不要凭空发明亲密关系。"
            "你的回复必须极短，通常只有几个字到十几个字，像真人随手打字一样；除非对方明确要求详细解释，否则不要超过一两句。"
            "需要连续说几句时，用换行把它们拆成多条（每条几个字），或者按系统提示用 | 分隔；绝对不要写成一句长话，也不要复述或总结对方的话。"
        )
        text += (
            "\n当问题问的是其他群友（如'XX的动画喜好''XX这人咋样'）时，"
            "必须先用 TA 当前的群名片点名（例如'小乙乙喜欢……'），再给结论；"
            "绝对不要说成自己的喜好或经历，也不要不带名字直接回答。"
        )
        text += (
            "\n对话中对方说的'我'指提问者本人，不是你扮演的角色；"
            "被问'评价我/我怎么样'时评价的是提问者，用提问者的记忆，不要评价你自己。"
        )
        habits = active_persona.get("speech_habits") if isinstance(active_persona, dict) else None
        if isinstance(habits, list) and habits:
            top_habits = [str(h) for h in habits if str(h).strip()][:8]
            if top_habits:
                text += "\n他的说话习惯（必须逐条模仿，不要写成书面语）：\n" + "\n".join(
                    f"- {habit}" for habit in top_habits
                )
        text += (
            "\n允许并自然使用 TA 常用的粗口和网络梗（如 woc、有点恶心、屎、几把 等），"
            "不要因为礼貌而把它们翻译成正式表达；宁可短到只剩两三个字，也不要写成长句。"
        )
        text += (
            "\n硬性要求：普通回复不超过 12 个汉字，超过就拆成多条短消息用 | 分隔；"
            "评价别人时只给一句短评（几个字到十几个字），不要分析、总结或列举。"
        )
        text += (
            "\n用词纪律：根据真实示例学习句长、语气、用词和节奏，不要照抄整句；"
            "不要使用他从未说过、示例和事实里不存在的网络热梗（如'遥遥领先'之类）。"
        )
        burst = active_persona.get("burst") if isinstance(active_persona, dict) else None
        if isinstance(burst, dict) and burst.get("enabled"):
            separator = str(burst.get("separator") or "|")
            max_messages = max(1, min(6, int(burst.get("max_messages") or 3)))
            text += (
                f"\n你支持短句连发：当想连说几句话时，用 {separator} 分隔每条，"
                f"一次最多 {max_messages} 条，每条几个字（如：来了{separator}几点{separator}上号）。"
                "系统会把每条拆成独立消息发送，不要自己加序号或列表。"
            )
        return text

    def _safety_lines_for(self) -> list[str]:
        return render_safety_lines(self.runtime.safety)

    def _bot_addressing_instruction(self, *, impersonating: bool) -> str:
        """Disambiguate the transport target from the active speaking persona."""

        if impersonating:
            return (
                "Addressing fact: the current message explicitly mentions this bot account. "
                "The mention name (for example, 小町) is the bot's call name, not a request "
                "to discuss that name as a third person. Reply directly as the currently "
                "active impersonated group member."
            )
        return (
            "Addressing fact: the current message explicitly mentions this bot account, "
            "so answer the sender directly."
        )

    def _with_relevant_examples(
        self,
        persona_text: str,
        active_persona: dict,
        context_lines: list[str],
        group_id: int,
        *,
        style_query: StyleRetrievalQuery | None = None,
        exclude_texts: Sequence[str] = (),
    ) -> str:
        style_preamble = (
            "\n相似情境下的说话方式示例（仅模仿风格）："
            "以下内容只用于学习句长、语气、用词和节奏，不是事实来源；"
            "不要照抄整句，不要把示例中的人物、作品或事件带入当前回答。"
        )
        example_budget = max(0, MAX_PROMPT_BLOCK_CHARS - len(style_preamble))
        if style_query is None:
            current_text = str(context_lines[0] if context_lines else "").split(":", 1)[-1]
            style_query = build_style_retrieval_query(current_text=current_text)
        if self.persona_manager is not None:
            picked = self.persona_manager.retrieve_examples(
                group_id,
                style_query,
                limit=MAX_SELECTED_EXAMPLES,
                exclude_texts=list(exclude_texts),
            )
            if picked:
                examples = format_style_example_block(
                    picked,
                    max_pairs=MAX_SELECTED_EXAMPLES,
                    max_chars=example_budget,
                )
                if not examples:
                    return persona_text
                return f"{persona_text}{style_preamble}{examples}"
            bank = self.persona_manager.style_bank(group_id)
        else:
            bank = []
        if not bank:
            bank = active_persona.get("example_bank") or active_persona.get("example_lines") or []
        fallback_entries = [
            (
                {"msg_id": f"fallback-{index}", "text": str(entry).strip()}
                if not isinstance(entry, dict)
                else {**entry, "msg_id": str(entry.get("msg_id") or f"fallback-{index}")}
            )
            for index, entry in enumerate(bank)
            if str(entry.get("text") if isinstance(entry, dict) else entry).strip()
        ]
        documents = [
            document
            for entry in fallback_entries
            if (document := build_style_document(entry)) is not None
        ]
        matches = rank_style_examples(
            query=style_query,
            documents=documents,
            vectors_by_id={},
            query_vector=None,
            limit=MAX_SELECTED_EXAMPLES,
            exclude_texts=exclude_texts,
        )
        picked = [match.entry for match in matches]
        if not picked:
            return persona_text
        examples = format_style_example_block(
            picked,
            max_pairs=MAX_SELECTED_EXAMPLES,
            max_chars=example_budget,
        )
        if not examples:
            return persona_text
        return f"{persona_text}{style_preamble}{examples}"

    def _with_relevant_facts(
        self,
        persona_text: str,
        active_persona: dict,
        context_lines: list[str],
        group_id: int,
        *,
        answer_mode: str = "current_fact",
    ) -> str:
        if self.persona_manager is not None:
            picked = self.persona_manager.retrieve_facts(
                group_id,
                context_lines,
                limit=5,
                answer_mode=answer_mode,
            )
        else:
            picked = retrieve_relevant_facts(
                active_persona.get("facts") or [], context_lines, limit=5
            )
        if not picked:
            return persona_text
        lines = [
            f"- [{fact.get('category') or '事实'}] "
            f"{scrub_banned_address_terms(str(fact.get('fact') or ''), BANNED_ADDRESS_TERMS)}"
            for fact in picked
            if isinstance(fact, dict)
        ]
        if not lines:
            return persona_text
        return (
            f"{persona_text}\n关于他的已知事实（只使用这些与他相符的事实，"
            "没有的事实不要编，用他的语气带过；"
            "特别注意：'评价/排行低于某作品'不等于'讨厌'，讨厌类结论必须有明确的讨厌/不喜欢依据）：\n"
            + "\n".join(lines)
        )

    def _impersonation_bot_labels(self, group_id: int) -> set[str]:
        labels = {str(self.runtime.persona.get("name", "") or "").strip()}
        if self.persona_manager is not None:
            labels.add(self.persona_manager.active_name(group_id))
        return {label for label in labels if label}

    def _sanitize_impersonation_lines(self, lines: list[str], *, group_id: int) -> list[str]:
        """Strip every factor that could pull the model out of character.

        In impersonation mode the injected context must not teach the model
        any previous bot persona: drop lines mentioning 小町/AI/主人 markers,
        and drop the bot's own historical lines entirely (they are AI output,
        not the member's real voice, and may mimic the member imperfectly).
        """

        default_bot_name = str(self.runtime.persona.get("name", "") or "").strip()
        sanitized: list[str] = []
        for line in lines:
            text = str(line)
            if any(marker in text for marker in IMPERSONATION_CONTAMINANT_MARKERS):
                continue
            head, separator, tail = text.partition(":")
            packed_match = _PACKED_LINE_SPEAKER_PATTERN.match(text)
            if packed_match:
                head = packed_match.group(1)
            is_bot_line = separator and (
                "（小町扮演）" in head
                or "（小町扮演）" in text
                or (default_bot_name and head.strip() == default_bot_name)
            )
            if is_bot_line:
                continue
            sanitized.append(scrub_banned_address_terms(text, BANNED_ADDRESS_TERMS))
        return sanitized

    def _sanitize_packed_context(self, packed, *, group_id: int):
        if packed is not None and isinstance(getattr(packed, "text", None), str):
            packed = replace(
                packed,
                text="\n".join(
                    self._sanitize_impersonation_lines(packed.text.splitlines(), group_id=group_id)
                ),
            )
        return packed

    def _normalize_lookup_text(self, value: str) -> str:
        return LOOKUP_NORMALIZER.sub("", value).lower()

    def _format_member_label(
        self,
        *,
        nickname: str,
        group_card: str,
        fallback: str,
    ) -> str:
        return format_member_label(
            nickname=nickname,
            group_card=group_card,
            fallback=fallback,
        )

    def _member_label_for_user(
        self,
        *,
        user_id: int,
        users_by_id: dict[int, object],
        group_id: int | None = None,
    ) -> str:
        if user_id == self.runtime.settings.bot_qq:
            if group_id is not None and self.persona_manager is not None:
                return self.persona_manager.bot_transcript_label(group_id)
            persona_name = str(self.runtime.persona.get("name", "Bot")).strip()
            return persona_name or "Bot"
        if group_id is not None and self.persona_manager is not None:
            group_label = self.persona_manager.member_label_for_user(
                int(user_id),
                int(group_id),
            )
            if group_label:
                return group_label
        user = users_by_id.get(user_id)
        if user is None:
            return str(user_id)
        return self._format_member_label(
            nickname=str(getattr(user, "nickname", "")),
            group_card=str(getattr(user, "group_card", "")),
            fallback=str(user_id),
        )

    def _format_message_line(
        self,
        *,
        user_id: int,
        plain_text: str,
        users_by_id: dict[int, object],
        group_id: int | None = None,
        bot_marker: bool = False,
    ) -> str:
        if (
            not bot_marker
            and user_id == self.runtime.settings.bot_qq
            and group_id is not None
            and self.persona_manager is not None
        ):
            label = self.persona_manager.bot_transcript_label(group_id)
        else:
            label = self._member_label_for_user(
                user_id=user_id, users_by_id=users_by_id, group_id=group_id
            )
        return f"{label}: {plain_text}"

    def _flatten_raw_message_text(self, raw_payload: dict | None) -> str:
        return flatten_raw_message_text(raw_payload)

    def _quoted_message_line_for_prompt(self, *, quoted_raw_payload: dict | None) -> str | None:
        return quoted_message_line_for_prompt(quoted_raw_payload=quoted_raw_payload)

    def _quoted_pronoun_referent_note(
        self,
        *,
        query_text: str,
        quoted_raw_payload: dict | None,
    ) -> str | None:
        return quoted_pronoun_referent_note(
            query_text=query_text,
            quoted_raw_payload=quoted_raw_payload,
        )

    def _target_message_text_for_prompt(self, *, event, resolved_image_count: int = 0) -> str:
        if event.plain_text.strip():
            return event.plain_text
        image_count = len(event.images)
        if image_count <= 0:
            image_count = resolved_image_count
        if image_count <= 0:
            return event.plain_text
        if image_count == 1:
            return "[sent 1 image]" if event.images else "[asked about 1 image]"
        return (
            f"[sent {image_count} images]"
            if event.images
            else f"[asked about {image_count} images]"
        )

    def _is_reply_to_bot(
        self, *, event, messages: MessageRepository, quoted_raw_payload: dict | None
    ) -> bool:
        if event.reply_to_msg_id is None:
            return False

        quoted_message = messages.get_by_platform_msg_id(event.reply_to_msg_id)
        if quoted_message is not None:
            return quoted_message.user_id == self.runtime.settings.bot_qq

        if isinstance(quoted_raw_payload, dict):
            try:
                return int(quoted_raw_payload.get("user_id", 0)) == self.runtime.settings.bot_qq
            except (TypeError, ValueError):
                return False

        return False
