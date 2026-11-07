"""InboundRouter 的 policy 职责 mixin（Phase 1b 自 router.py 拆出，方法体未改动）。"""

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

from app.core.router._common import ASIA_SHANGHAI


class GroupPolicyMixin:
    """方法集合原样搬移；全部通过 self 与主类协作。"""

    def _group_runtime_policy(
        self,
        *,
        group_id: int,
    ) -> tuple[bool, bool, bool, tuple[int, int], tuple[time, time] | None, list[str]]:
        defaults = self.runtime.group_policy.get("default_group_behavior", {})
        configured = self.runtime.group_policy.get("groups", {}).get(str(group_id), {})

        enabled = bool(configured.get("enabled", defaults.get("enabled", False)))
        speak_enabled = bool(configured.get("speak", defaults.get("speak", False)))
        proactive_enabled = bool(
            configured.get("proactive_reply", defaults.get("proactive_reply", True))
        )
        proactive_interval = self._parse_interval_range(
            configured.get(
                "proactive_interval_seconds", defaults.get("proactive_interval_seconds", "180-480")
            )
        )
        quiet_hours = self._parse_quiet_hours(
            configured.get("quiet_hours", defaults.get("quiet_hours"))
        )
        if not enabled:
            speak_enabled = False

        group_policy_lines = [
            "Speak only in allowlisted groups.",
            "Keep replies short in group chat.",
            "Only use web search when the service has marked the turn as eligible.",
        ]
        return (
            enabled,
            speak_enabled,
            proactive_enabled,
            proactive_interval,
            quiet_hours,
            group_policy_lines,
        )

    def _parse_interval_range(self, value: object) -> tuple[int, int]:
        if not isinstance(value, str) or "-" not in value:
            return (180, 480)
        minimum_text, maximum_text = value.split("-", maxsplit=1)
        try:
            minimum = int(minimum_text)
            maximum = int(maximum_text)
        except ValueError:
            return (180, 480)
        if minimum <= 0 or maximum <= 0:
            return (180, 480)
        return minimum, max(minimum, maximum)

    def _parse_quiet_hours(self, value: object) -> tuple[time, time] | None:
        if not isinstance(value, str) or "-" not in value:
            return None
        start_text, end_text = value.split("-", maxsplit=1)
        try:
            return time.fromisoformat(start_text), time.fromisoformat(end_text)
        except ValueError:
            return None

    def _group_policy_bool(self, *, group_id: int, key: str, default: bool) -> bool:
        defaults = self.runtime.group_policy.get("default_group_behavior", {})
        configured = self.runtime.group_policy.get("groups", {}).get(str(group_id), {})
        return bool(configured.get(key, defaults.get(key, default)))

    def _group_policy_int(self, *, group_id: int, key: str, default: int) -> int:
        defaults = self.runtime.group_policy.get("default_group_behavior", {})
        configured = self.runtime.group_policy.get("groups", {}).get(str(group_id), {})
        value = configured.get(key, defaults.get(key, default))
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _group_policy_float(self, *, group_id: int, key: str, default: float) -> float:
        defaults = self.runtime.group_policy.get("default_group_behavior", {})
        configured = self.runtime.group_policy.get("groups", {}).get(str(group_id), {})
        value = configured.get(key, defaults.get(key, default))
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _group_memory_enabled(self, *, group_id: int) -> bool:
        return self._group_policy_bool(
            group_id=group_id,
            key="memory_enabled",
            default=False,
        )

    def _active_addressing_rules_for_user(
        self,
        *,
        group_id: int,
        user_id: int,
    ) -> list[str]:
        """Active addressing rules that apply to this user, fail-closed."""
        lines: list[str] = []
        with session_scope(self.engine) as session:
            rows = MemoryRepository(session).list_group_memories_for_subject(
                scope_id=str(group_id),
                subject_id=str(user_id),
                limit=10,
            )
            for row in rows:
                if row.memory_kind != "preference":
                    continue
                text = f"{row.content} {row.object_text} {row.predicate}"
                if _ADDRESSING_RULE_MARKERS.search(text) is None:
                    continue
                target = _ADDRESSING_TARGET_QQ_PATTERN.search(text)
                if target is not None and target.group(1) != str(user_id):
                    continue
                lines.append(
                    f"Active addressing rule for this user (source: {row.source_msg_id or ''}): "
                    f"{row.content}"
                )
                if len(lines) >= 5:
                    break
        return lines

    def _normalize_timestamp(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=ASIA_SHANGHAI).astimezone(UTC)
        return value.astimezone(UTC)

    def _resolve_group_policy(
        self,
        *,
        groups: GroupRepository,
        group_id: int,
    ) -> tuple[bool, bool, bool, tuple[int, int], tuple[time, time] | None, list[str]]:
        (
            runtime_enabled,
            runtime_speak_enabled,
            proactive_enabled,
            proactive_interval,
            quiet_hours,
            group_policy_lines,
        ) = self._group_runtime_policy(group_id=group_id)
        stored_group = groups.get_group(group_id)
        if stored_group is None:
            return (
                runtime_enabled,
                runtime_speak_enabled,
                proactive_enabled,
                proactive_interval,
                quiet_hours,
                group_policy_lines,
            )

        enabled = bool(stored_group.enabled)
        speak_enabled = bool(stored_group.speak_enabled) if enabled else False
        return (
            enabled,
            speak_enabled,
            proactive_enabled,
            proactive_interval,
            quiet_hours,
            group_policy_lines,
        )

    def _private_inbound_platform_msg_id(self, event) -> str:
        return f"private-inbound-{event.user_id}-{event.platform_msg_id}"
