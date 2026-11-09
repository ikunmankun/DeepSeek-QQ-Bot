"""InboundRouter 的 ingest 职责 mixin（Phase 1b 自 router.py 拆出，方法体未改动）。"""

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

from app.core.router._common import ASIA_SHANGHAI, logger


class GroupIngestMixin:
    """方法集合原样搬移；全部通过 self 与主类协作。"""

    def ingest_historical_group_message(self, event) -> bool:
        persisted = self._persist_inbound_message(event, cache_images=False)
        if not persisted:
            return False
        self._enqueue_episode_message(
            group_id=event.group_id,
            platform_msg_id=event.platform_msg_id,
            timestamp=event.timestamp,
            allow_late_arrival=False,
        )
        self._archive_inbound_message(event)
        self._ingest_bbot_listener_cache(event)
        return True

    def ingest_live_group_message(self, event) -> bool:
        persisted = self._persist_inbound_message(event, cache_images=True)
        if not persisted:
            return False
        self._enqueue_episode_message(
            group_id=event.group_id,
            platform_msg_id=event.platform_msg_id,
            timestamp=event.timestamp,
        )
        self._archive_inbound_message(event)
        self._ingest_bbot_listener_cache(event)
        from app.agent.participation_feedback import record_feedback
        try:
            record_feedback(self, event)
        except Exception:
            logger.exception('group_participation_feedback_failed group_id=%s', event.group_id)
        return True

    def _merge_own_message_echo(self, session, event, echo_row) -> None:
        """自己的网关回执行与早到的管线记录行合并，避免上下文里自己刚说的话出现两份。

        social_executor._finish_group_output 只在 echo 先到时合并；echo 晚一拍时
        两行都带文本，模型在最近窗口会看到自己刚说的话两遍（强化复读、污染计数）。
        这里按 文本+30 秒窗 找到管线行（带 social_action_id 且未 canonical 化），
        把它标记 deleted，回执行保留为 canonical（带真实 message_id）。
        """
        if not (event.plain_text or "").strip():
            return
        from sqlalchemy import select as _select
        from app.storage.models import Message
        try:
            cutoff = event.timestamp - timedelta(seconds=30)
            candidates = session.scalars(_select(Message).where(
                Message.group_id == event.group_id,
                Message.user_id == event.user_id,
                Message.id != echo_row.id,
                Message.plain_text == event.plain_text,
                Message.timestamp >= cutoff,
            ).order_by(Message.id.desc()).limit(5))
            for row in candidates:
                raw = row.raw_json if isinstance(row.raw_json, dict) else {}
                if not raw.get("social_action_id"):
                    continue
                if raw.get("delivery_state") not in ("sent", "uncertain"):
                    continue
                row.raw_json = {**raw, "delivery_state": "deleted",
                                "canonical_platform_msg_id": echo_row.platform_msg_id}
                row.plain_text = ""
                logger.info("own_message_echo_merged echo=%s pipeline_row=%s",
                            echo_row.platform_msg_id, row.id)
                return
        except Exception:
            logger.exception("own_message_echo_merge_failed msg_id=%s", event.platform_msg_id)

    def _update_group_mood(self, session, messages, event) -> None:
        """入站情绪钩子：委托插件注册表（MoodFeature），无注册表时为 no-op。

        情绪是氛围不是功能：钩子内部自吞异常，绝不影响消息入库。
        """

        registry = getattr(self, "plugin_registry", None)
        if registry is None:
            return
        from app.agent.plugins.base import InboundContext
        from app.core.time_utils import shanghai_now_naive

        registry.dispatch_inbound(InboundContext(
            router=self, session=session, messages=messages, event=event,
            now=shanghai_now_naive()))

    def _archive_inbound_message(self, event) -> None:
        if not self._group_policy_bool(group_id=event.group_id, key="archive", default=False):
            return
        try:
            append_group_message_archive(
                history_dir=self.runtime.settings.data_dir / "history",
                group_id=event.group_id,
                timestamp=event.timestamp,
                platform_msg_id=event.platform_msg_id,
                user_id=event.user_id,
                nickname=event.nickname,
                group_card=event.group_card,
                plain_text=event.plain_text,
                msg_type=event.msg_type,
                mentioned_bot=event.mentioned_bot,
                reply_to_msg_id=event.reply_to_msg_id,
                direction="inbound",
                image_local_paths=[image.local_path for image in event.images if image.local_path],
            )
        except Exception:
            logger.exception(
                "history_archive_inbound_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )

    def _persist_inbound_message(self, event, *, cache_images: bool) -> bool:
        with session_scope(self.engine) as session:
            groups = GroupRepository(session)
            users = UserRepository(session)
            messages = MessageRepository(session)
            summaries = SummaryRepository(session)
            memories = MemoryRepository(session)

            inbound_message = messages.get_by_platform_msg_id(event.platform_msg_id)
            if inbound_message is not None:
                outbound_message = messages.get_by_platform_msg_id(
                    self._outbound_platform_msg_id(event.platform_msg_id)
                )
                return outbound_message is None

            (
                enabled,
                speak_enabled,
                _proactive_enabled,
                _proactive_interval,
                _quiet_hours,
                _group_policy_lines,
            ) = self._resolve_group_policy(
                groups=groups,
                group_id=event.group_id,
            )
            group = groups.get_group(event.group_id)
            if group is None:
                groups.upsert_group(
                    group_id=event.group_id,
                    group_name=str(event.group_id),
                    enabled=enabled,
                    speak_enabled=speak_enabled,
                )
            else:
                group.group_name = str(event.group_id)
                groups.session.add(group)
            current_user = users.upsert_user(
                user_id=event.user_id, nickname=event.nickname, group_card=event.group_card
            )
            current_users_by_id = {event.user_id: current_user}
            if event.images:
                if cache_images:
                    cache_images_in_raw_payload(
                        event.raw_payload,
                        cache_dir=self.runtime.settings.data_dir / "image_cache",
                    )
                event.images = extract_images_from_raw_payload(event.raw_payload)
            inbound_message = messages.add_group_message(
                platform_msg_id=event.platform_msg_id,
                group_id=event.group_id,
                user_id=event.user_id,
                timestamp=event.timestamp,
                plain_text=event.plain_text,
                raw_json=event.raw_payload,
                msg_type=event.msg_type,
                reply_to_msg_id=event.reply_to_msg_id,
                mentioned_bot=event.mentioned_bot,
            )
            if event.user_id == self.runtime.settings.bot_qq:
                self._merge_own_message_echo(session, event, inbound_message)
            self._update_group_mood(session, messages, event)

            current_lines = format_summary_source_lines(
                [
                    self._format_message_line(
                        user_id=event.user_id,
                        plain_text=event.plain_text,
                        users_by_id=current_users_by_id,
                        group_id=event.group_id,
                        bot_marker=True,
                    )
                ]
            )
            claim = parse_personal_claim(event.plain_text)
            if claim is not None and event.user_id != self.runtime.settings.bot_qq:
                member_messages = messages.list_recent_group_member_messages(
                    group_id=event.group_id,
                    limit=200,
                )
                members = group_member_identities_from_messages(member_messages)
                subject_id: int | None = event.user_id if claim.subject_mode == "sender" else None
                subject_display = str(event.group_card or event.nickname or event.user_id).strip()
                if claim.subject_alias is not None:
                    resolved_subject = resolve_group_member_reference(
                        claim.subject_alias,
                        members,
                        match_mode="exact",
                        exclude_user_ids={self.runtime.settings.bot_qq},
                    )
                    subject_id = resolved_subject.user_id if resolved_subject is not None else None
                    if resolved_subject is not None:
                        subject_display = resolved_subject.matched_alias
                if subject_id is not None:
                    observed_at = self._normalize_timestamp(event.timestamp)
                    stable_subject_id = str(subject_id)
                    memory = memories.upsert_canonical_memory(
                        scope_type="group",
                        scope_id=str(event.group_id),
                        subject_type="user",
                        subject_id=stable_subject_id,
                        memory_kind=claim.memory_kind,
                        canonical_key=canonical_key(
                            claim.memory_kind,
                            stable_subject_id,
                            claim.predicate,
                            claim.object_text,
                        ),
                        predicate=claim.predicate,
                        object_text=claim.object_text,
                        content=f"{subject_display} {claim.predicate} {claim.object_text}.",
                        importance=4,
                        confidence=0.9 if claim.is_correction else 0.8,
                        source_msg_ids=[event.platform_msg_id],
                        valid_from=observed_at,
                    )
                    if claim.is_correction:
                        old_subject_id: str | None = None
                        old_subject_resolved = claim.old_subject_alias is None
                        if claim.old_subject_alias is not None:
                            old_subject = resolve_group_member_reference(
                                claim.old_subject_alias,
                                members,
                                match_mode="exact",
                                exclude_user_ids={self.runtime.settings.bot_qq},
                            )
                            if old_subject is not None:
                                old_subject_id = str(old_subject.user_id)
                                old_subject_resolved = True
                        if old_subject_resolved:
                            previous = memories.find_unique_correction_candidate(
                                scope_id=str(event.group_id),
                                predicate=claim.predicate,
                                object_text=claim.object_text,
                                replacement_memory_id=memory.id,
                                as_of=observed_at,
                                subject_id=old_subject_id,
                            )
                            if previous is not None:
                                memories.mark_superseded(
                                    memory_id=previous.id,
                                    superseded_by_id=memory.id,
                                    valid_until=observed_at,
                                )
                                memory.supersedes_id = previous.id
                                session.add(memory)

            addressing_rule = (
                parse_addressing_rule_claim(event.plain_text)
                if event.mentioned_bot and event.user_id != self.runtime.settings.bot_qq
                else None
            )
            if addressing_rule is not None:
                memories.upsert_canonical_memory(
                    scope_type="group",
                    scope_id=str(event.group_id),
                    subject_type="user",
                    subject_id=str(event.user_id),
                    memory_kind="preference",
                    canonical_key=canonical_key(
                        "preference",
                        str(event.user_id),
                        addressing_rule.predicate,
                        addressing_rule.role,
                    ),
                    predicate=addressing_rule.predicate,
                    object_text=addressing_rule.role,
                    content=addressing_rule.content,
                    importance=5,
                    confidence=1.0,
                    source_msg_ids=[event.platform_msg_id],
                    valid_from=self._normalize_timestamp(event.timestamp),
                    replace_previous=True,
                )
                logger.info(
                    "addressing_rule_persisted group_id=%s msg_id=%s user_id=%s",
                    event.group_id,
                    event.platform_msg_id,
                    event.user_id,
                )

            for candidate in extract_structured_memory_candidates(
                scope_id=str(event.group_id),
                source_msg_id=event.platform_msg_id,
                lines=current_lines,
                observed_at=self._normalize_timestamp(event.timestamp),
            ):
                if candidate["subject_type"] == "user":
                    candidate["subject_id"] = str(event.user_id)
                supersedes_kind = candidate.pop("supersedes_kind", None)
                if supersedes_kind:
                    previous_memory = memories.find_current_memory_for_supersession(
                        scope_id=str(event.group_id),
                        subject_type=str(candidate["subject_type"]),
                        subject_id=str(candidate["subject_id"]),
                        memory_kind=str(supersedes_kind),
                        replacement_content=str(candidate["content"]),
                        as_of=self._normalize_timestamp(event.timestamp),
                    )
                    if previous_memory is not None:
                        candidate["supersedes_id"] = previous_memory.id
                memories.upsert_memory(**candidate)

            message_count = messages.count_group_inbound_messages(
                group_id=event.group_id,
                bot_user_id=self.runtime.settings.bot_qq,
            )
            if should_schedule_window_summary(message_count=message_count):
                window_messages = messages.list_recent_group_messages_for_summarization(
                    group_id=event.group_id,
                    limit=25,
                )
                window_users_by_id = users.get_users_by_ids(
                    [message.user_id for message in window_messages]
                )
                source_lines = format_summary_source_lines(
                    [
                        self._format_message_line(
                            user_id=item.user_id,
                            plain_text=item.plain_text,
                            users_by_id=window_users_by_id,
                            group_id=event.group_id,
                            bot_marker=True,
                        )
                        for item in window_messages
                    ]
                )
                if source_lines:
                    window_summary = summaries.upsert_summary(
                        scope_type="group",
                        scope_id=str(event.group_id),
                        summary_level="window",
                        summary_key=f"window:{window_messages[0].platform_msg_id}:{window_messages[-1].platform_msg_id}",
                        start_at=window_messages[0].timestamp,
                        end_at=window_messages[-1].timestamp,
                        content=summarize_window(source_lines),
                        source_count=len(source_lines),
                        source_start_msg_id=window_messages[0].platform_msg_id,
                        source_end_msg_id=window_messages[-1].platform_msg_id,
                    )
                    session.flush()
                    daily_key = (
                        "daily:"
                        f"{self._normalize_timestamp(inbound_message.timestamp).astimezone(ASIA_SHANGHAI).date().isoformat()}"
                    )
                    daily_day = self._normalize_timestamp(inbound_message.timestamp).astimezone(
                        ASIA_SHANGHAI
                    )
                    day_start = daily_day.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ).replace(tzinfo=None)
                    existing_daily = summaries.list_group_summaries(
                        scope_id=str(event.group_id),
                        limit=1,
                        summary_levels=["daily"],
                        summary_key=daily_key,
                    )
                    previous_daily = existing_daily[-1] if existing_daily else None
                    summaries.upsert_summary(
                        scope_type="group",
                        scope_id=str(event.group_id),
                        summary_level="daily",
                        summary_key=daily_key,
                        start_at=(
                            max(previous_daily.start_at, day_start)
                            if previous_daily is not None
                            else day_start
                        ),
                        end_at=window_messages[-1].timestamp,
                        content=summarize_recursive(
                            previous_summary=(
                                previous_daily.content if previous_daily is not None else ""
                            ),
                            new_window_summary=window_summary.content,
                        ),
                        source_count=(
                            previous_daily.source_count if previous_daily is not None else 0
                        )
                        + len(source_lines),
                        source_start_msg_id=(
                            previous_daily.source_start_msg_id
                            if previous_daily is not None
                            else window_messages[0].platform_msg_id
                        ),
                        source_end_msg_id=window_messages[-1].platform_msg_id,
                        source_summary_ids=list(
                            dict.fromkeys(
                                [
                                    *(
                                        previous_daily.source_summary_ids
                                        if previous_daily is not None
                                        else []
                                    ),
                                    window_summary.id,
                                ]
                            )
                        ),
                    )
            # V3 owns episode derivation through MemoryBackgroundService.  The
            # old batch compaction queue must only be fed when its legacy
            # worker is actually enabled; otherwise jobs accumulate forever.
            legacy_compaction_enabled = bool(
                self.runtime.settings.memory_compaction_enabled
                and getattr(
                    self.memory_compaction_service,
                    "legacy_enabled",
                    False,
                )
            )
            if legacy_compaction_enabled:
                batch_size = max(10, int(self.runtime.settings.memory_compaction_batch_size))
                if message_count > 0 and message_count % batch_size == 0:
                    compaction_messages = messages.list_recent_group_inbound_messages(
                        group_id=event.group_id,
                        bot_user_id=self.runtime.settings.bot_qq,
                        limit=batch_size,
                    )
                    if compaction_messages:
                        start_id = compaction_messages[0].id
                        end_id = compaction_messages[-1].id
                        job_key = f"memory:{event.group_id}:{start_id}:{end_id}"
                        JobRepository(session).add_job(
                            job_type="memory_compaction",
                            job_key=job_key,
                            payload_json={
                                "group_id": event.group_id,
                                "start_id": start_id,
                                "end_id": end_id,
                                "attempts": 0,
                            },
                            run_at=datetime.now(UTC),
                            status="queued",
                        )
            return True

    def _persist_private_inbound_message(self, event) -> bool:
        platform_msg_id = str(getattr(event, "platform_msg_id", "")).strip()
        if not platform_msg_id:
            return True

        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)
            private_platform_msg_id = self._private_inbound_platform_msg_id(event)

            if messages.get_by_platform_msg_id(private_platform_msg_id) is not None:
                return False

            users.upsert_user(user_id=event.user_id, nickname=event.nickname, group_card="")
            if event.images:
                cache_images_in_raw_payload(
                    event.raw_payload,
                    cache_dir=self.runtime.settings.data_dir / "image_cache",
                )
                event.images = extract_images_from_raw_payload(event.raw_payload)
            messages.add_private_message(
                platform_msg_id=private_platform_msg_id,
                user_id=event.user_id,
                timestamp=event.timestamp,
                plain_text=event.plain_text,
                raw_json=event.raw_payload,
                msg_type=getattr(event, "msg_type", "text"),
                reply_to_msg_id=getattr(event, "reply_to_msg_id", None),
            )
            return True

    def _ingest_bbot_listener_cache(self, event) -> None:
        entries = extract_listener_cache_entries(
            group_id=event.group_id,
            user_id=event.user_id,
            plain_text=event.plain_text,
        )
        if not entries:
            return
        with session_scope(self.engine) as session:
            upsert_listener_cache_entries(
                cache_repo=BbotListenerCacheRepository(session),
                group_id=event.group_id,
                entries=entries,
                now=event.timestamp,
            )

    def _resolve_bbot_cached_command(self, *, event, command_text: str) -> str:
        with session_scope(self.engine) as session:
            return resolve_cached_command_target(
                command_text=command_text,
                group_id=event.group_id,
                cache_repo=BbotListenerCacheRepository(session),
            )
