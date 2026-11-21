"""router 共享的模块级常量与辅助函数（Phase 1b 自 router.py 拆出，内容未改动）。"""

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


__all__ = [
    "logger",
    "ASIA_SHANGHAI",
    "PROACTIVE_RESERVATION_TTL_SECONDS",
    "_PROACTIVE_CANDIDATE_REASONS",
    "MEMORY_TOOL_EFFICIENCY_INSTRUCTION",
    "MEMORY_ATTRIBUTE_MATCHING_INSTRUCTION",
    "IMAGE_SENDER_GROUNDING_INSTRUCTION",
    "BOT_SELF_IDENTITY_INSTRUCTION",
    "IMPERSONATION_CONTAMINANT_MARKERS",
    "_PACKED_LINE_SPEAKER_PATTERN",
    "_scrub_impersonation_reply",
    "REQUESTER_IDENTITY_INSTRUCTION",
    "GROUP_IMAGE_REFERENCE_PROMPT_PREFIX",
    "PreparedGroupReply",
]

logger = logging.getLogger(__name__)

ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")

PROACTIVE_RESERVATION_TTL_SECONDS = 300

_PROACTIVE_CANDIDATE_REASONS = frozenset({"proactive_candidate", "proactive_local_candidate"})

MEMORY_TOOL_EFFICIENCY_INSTRUCTION = (
    "If the injected memory context is already enough to answer, do not call "
    "memory_search again; only search when information is clearly missing."
)

MEMORY_ATTRIBUTE_MATCHING_INSTRUCTION = (
    "Match the memory question's requested subject, attribute, activity, and time exactly. "
    "One directly relevant fact or quoted hit is enough for an incomplete answer. Use plan "
    "evidence for plans, explicit preference evidence for preferences, and only a direct "
    "current activity for what someone is playing or doing; related discussion is not a "
    "substitute. For first-person questions use only the bound requester and requested "
    "attribute, never a nearby speaker. For an exact historical phrase, answer from its "
    "matching quoted source instead of abstaining over missing surrounding context."
)

IMAGE_SENDER_GROUNDING_INSTRUCTION = (
    "Image sender metadata identifies who posted each attached image in this "
    "group. If the user asks who sent or posted an image, use that metadata. "
    "Do not claim that the sender is the person shown, owns an item, ate a meal, "
    "or performed another visible activity unless the image or chat establishes "
    "it; distinguish the known sender from any uncertain real-world actor."
)

BOT_SELF_IDENTITY_INSTRUCTION = (
    "For this target identity question, 'you' means the assistant described by "
    "System persona, while 'I/me' means the current human requester. Answer who "
    "the assistant is from System persona. Do not describe, profile, name, or "
    "identify the requester, even if recent or historical context discusses them."
)

IMPERSONATION_CONTAMINANT_MARKERS = (
    "主人",
    "小町",
    "比企谷",
    "机器人",
    "AI助手",
    "AI群友",
    "mesugaki",
    "笨蛋",
    "毒舌",
)

_PACKED_LINE_SPEAKER_PATTERN = re.compile(r"^\[[^\]]*\]\s*([^(\[]+?)\s*\(uin:")

def _scrub_impersonation_reply(
    text: str,
    *,
    speaker_labels: Iterable[str] = (),
) -> str:
    """Deterministic backstop for impersonation replies.

    Removes maid-style address terms (主人/您/大人...) that leak from model
    habit or memory, replacing them with "你" so the line still reads as
    direct speech to the group.
    """

    cleaned = scrub_banned_address_terms(str(text or ""), BANNED_ADDRESS_TERMS)
    for label in sorted(
        {str(value).strip() for value in speaker_labels if str(value).strip()},
        key=len,
        reverse=True,
    ):
        without_prefix = re.sub(
            rf"^\s*@?{re.escape(label)}\s*[:：]\s*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
        if without_prefix != cleaned:
            cleaned = without_prefix
            break
    return cleaned.strip()

REQUESTER_IDENTITY_INSTRUCTION = (
    "For this requester identity question, 'I/me' means the current human requester. "
    "Treat the literal question 'who am I' as a request for a remembered portrait, not "
    "as a demand for a legal name. The current Target message sender label is direct "
    "identity evidence: when it contains a nickname or group card, answer at minimum "
    "'you are <that display name> in this group' and never abstain. If the injected "
    "memory contains even one direct, "
    "self-authored identity, profile, preference, taboo, relationship, or durable fact "
    "about that requester, you must answer concisely with one to three such attributes; "
    "choose the newest eligible fact and do not abstain merely because the portrait is "
    "incomplete. Never copy a nearby assistant "
    "reply about another member, and never infer age, nationality, occupation, location, "
    "or another attribute that the requester did not directly establish."
)

GROUP_IMAGE_REFERENCE_PROMPT_PREFIX = re.compile(r"^(?:请|麻烦|拜托)?(?:帮我)?", re.IGNORECASE)

@dataclass(slots=True)
class PreparedGroupReply:
    should_reply: bool
    prompt_lines: list[str] | None = None
    prebuilt_reply_text: str | None = None
    group_image_request: GroupImageGenerationRequest | None = None
    target_images: list[ImageAttachment] | None = None
    requires_user_visible_failure_reply: bool = False
    proactive_turn: bool = False
    force_web_search: bool = False
    allow_web_search: bool = False
    use_memory_tools: bool = False
    memory_tool_executor: object | None = None
    memory_source_ids: tuple[str, ...] = ()
    memory_has_evidence: bool = False
    # 思考深度路由：本回合按消息复杂度动态选的 reasoning effort。
    # None=沿用 client 固化档位；minimal/medium/high=覆盖档位。
    think_effort: str | None = None
