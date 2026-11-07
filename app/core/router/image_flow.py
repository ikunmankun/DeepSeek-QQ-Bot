"""InboundRouter 的 image_flow 职责 mixin（Phase 1b 自 router.py 拆出，方法体未改动）。"""

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

from app.core.router._common import GROUP_IMAGE_REFERENCE_PROMPT_PREFIX, IMAGE_SENDER_GROUNDING_INSTRUCTION, logger


class GroupImageFlowMixin:
    """方法集合原样搬移；全部通过 self 与主类协作。"""

    def _should_hold_group_image_for_followup(self, event) -> bool:
        return not event.plain_text.strip() and len(event.images) == 1

    def _remember_group_image_for_followup(self, event) -> None:
        self.pending_group_image_turns[(event.group_id, event.user_id)] = (
            self._normalize_timestamp(event.timestamp),
            list(event.images),
        )

    def _consume_group_image_for_followup(
        self,
        event,
        *,
        addressed_turn: bool,
        bot_names: set[str],
    ) -> list[ImageAttachment] | None:
        key = (event.group_id, event.user_id)
        pending = self.pending_group_image_turns.get(key)
        if pending is None:
            return None
        pending_timestamp, images = pending
        if self._normalize_timestamp(event.timestamp) - pending_timestamp > timedelta(minutes=3):
            self.pending_group_image_turns.pop(key, None)
            return None
        if event.images:
            self.pending_group_image_turns.pop(key, None)
            return None
        if not event.plain_text.strip():
            return None
        self.pending_group_image_turns.pop(key, None)
        if not addressed_turn or not is_image_reference_followup(
            event.plain_text,
            bot_names=bot_names,
        ):
            return None
        return list(images)

    def _build_local_generation_failure_reply(
        self, *, target_images: list[ImageAttachment] | None
    ) -> str:
        if target_images:
            return "我这边刚想回你的时候卡了一下，图还没顾上看，你再叫我一次。"
        return "我这边刚刚卡了一下，结果没拿到。你再叫我一次，我马上接上。"

    def _build_vision_unavailable_reply(self) -> str:
        return "我这边这路模型现在还看不了图，得换支持识图的模型才行。"

    def _strip_group_image_prefix(self, text: str) -> str:
        persona_name = str(self.runtime.persona.get("name", "")).strip()
        stripped = text.strip()
        if persona_name:
            stripped = re.sub(
                rf"^(?:@?{re.escape(persona_name)}[\s,，:：]*)+",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
        stripped = re.sub(r"^@[A-Za-z0-9_\-\u4e00-\u9fff]+\s*", "", stripped)
        return stripped.strip()

    def _extract_auto_web_reference_query(self, *, stripped_text: str) -> str | None:
        match = AUTO_WEB_REFERENCE_QUERY_PATTERN.search(stripped_text)
        if match is None:
            return None
        query = str(match.group("query") or "").strip(" \t,，。.!?？；;:：")
        if query.endswith("的"):
            query = query[:-1].rstrip()
        return query or None

    def _build_auto_web_reference_prompt(self, *, stripped_text: str, query: str) -> str:
        prompt = AUTO_WEB_REFERENCE_QUERY_PATTERN.sub("", stripped_text, count=1)
        prompt = AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN.sub("", prompt).strip(
            " \t,，。.!?？；;:："
        )
        if not prompt:
            return f"参考搜索到的{query}人设图生成一张图"
        return f"参考搜索到的{query}人设图，{prompt}"

    def _strip_group_image_request_prefix(self, text: str) -> str:
        stripped = GROUP_IMAGE_REFERENCE_PROMPT_PREFIX.sub("", text, count=1)
        return stripped.strip(" \t,，。.!！?？")

    def _build_image_attribution_lines(
        self,
        *,
        images: list[ImageAttachment],
        users_by_id: dict[int, object],
    ) -> list[str]:
        lines: list[str] = []
        for index, image in enumerate(images, start=1):
            source_user_id = image.source_user_id
            if source_user_id is not None and source_user_id in users_by_id:
                sender_label = self._member_label_for_user(
                    user_id=source_user_id,
                    users_by_id=users_by_id,
                )
            else:
                sender_label = self._format_member_label(
                    nickname=image.source_nickname or "",
                    group_card=image.source_group_card or "",
                    fallback="an unidentified group member",
                )
            lines.append(f"Attached image {index} was sent by {sender_label}.")
        if lines:
            lines.append(IMAGE_SENDER_GROUNDING_INSTRUCTION)
        return lines

    def _looks_like_reference_image_generation_request(
        self,
        *,
        stripped_text: str,
        resolved_image_turn: ResolvedImageTurn | None,
    ) -> bool:
        if resolved_image_turn is None or not resolved_image_turn.images:
            return False
        normalized_text = self._normalize_lookup_text(stripped_text)
        if not normalized_text:
            return False
        has_transform_intent = any(
            self._normalize_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS
        )
        has_reference_context = any(
            self._normalize_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS
        )
        has_generation_intent = any(
            self._normalize_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS
        )
        return has_transform_intent or (has_reference_context and has_generation_intent)

    def _build_group_image_request(
        self,
        *,
        event,
        addressed_turn: bool,
        resolved_image_turn: ResolvedImageTurn | None = None,
    ) -> GroupImageGenerationRequest | None:
        if self.group_image_service is None:
            return None
        if not self._group_policy_bool(
            group_id=event.group_id, key="image_generation", default=True
        ):
            return None
        stripped = self._strip_group_image_prefix(event.plain_text)
        if not stripped:
            return None
        if any(pattern.search(stripped) for pattern in GROUP_IMAGE_NEGATIVE_PATTERNS):
            return None
        reference_images = (
            list(resolved_image_turn.images) if resolved_image_turn is not None else []
        )
        auto_web_reference_query = self._extract_auto_web_reference_query(stripped_text=stripped)
        explicit_prompt: str | None = None
        for pattern in GROUP_IMAGE_REQUEST_PATTERNS:
            match = pattern.match(stripped)
            if match is None:
                continue
            prompt = match.group("prompt").strip(" \t,，。.!！?？")
            if not prompt:
                return None
            explicit_prompt = prompt
            break
        reference_request = self._looks_like_reference_image_generation_request(
            stripped_text=stripped,
            resolved_image_turn=resolved_image_turn,
        )
        implicitly_addressed_image_request = (
            event.reply_to_msg_id is not None
            and bool(reference_images)
            and (
                auto_web_reference_query is not None
                or explicit_prompt is not None
                or reference_request
            )
        )
        if not addressed_turn and not implicitly_addressed_image_request:
            return None
        if auto_web_reference_query is not None:
            return GroupImageGenerationRequest(
                group_id=event.group_id,
                trigger_message_id=event.platform_msg_id,
                prompt=self._build_auto_web_reference_prompt(
                    stripped_text=stripped,
                    query=auto_web_reference_query,
                ),
                requester_user_id=event.user_id,
                reference_images=reference_images,
                web_search_query=auto_web_reference_query,
            )
        if explicit_prompt is not None:
            return GroupImageGenerationRequest(
                group_id=event.group_id,
                trigger_message_id=event.platform_msg_id,
                prompt=explicit_prompt,
                requester_user_id=event.user_id,
                reference_images=reference_images,
            )
        if not reference_request:
            return None
        prompt = self._strip_group_image_request_prefix(stripped)
        if not prompt:
            return None
        return GroupImageGenerationRequest(
            group_id=event.group_id,
            trigger_message_id=event.platform_msg_id,
            prompt=prompt,
            requester_user_id=event.user_id,
            reference_images=reference_images,
        )

    def _group_image_enqueue_reply_text(self, enqueue_result) -> str:
        if not getattr(enqueue_result, "accepted", False):
            return "先等等，出图队列满了"
        queue_position = int(getattr(enqueue_result, "queue_position", 1) or 1)
        if queue_position <= 1:
            return "行，我画"
        return f"收到了，排队第 {queue_position}"

    async def _handle_group_image_request(
        self, event, request: GroupImageGenerationRequest
    ) -> None:
        if self.group_image_service is None:
            return
        try:
            enqueue_result = await self.group_image_service.enqueue(request)
        except Exception:
            logger.exception(
                "group_image_enqueue_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
            await self._send_prebuilt_reply(event, "出图队列刚卡了一下，你再叫我一次")
            return
        logger.info(
            "group_image_enqueue group_id=%s msg_id=%s accepted=%s queue_position=%s reason=%s",
            event.group_id,
            event.platform_msg_id,
            getattr(enqueue_result, "accepted", False),
            getattr(enqueue_result, "queue_position", None),
            getattr(enqueue_result, "reason", ""),
        )
        await self._send_prebuilt_reply(event, self._group_image_enqueue_reply_text(enqueue_result))
