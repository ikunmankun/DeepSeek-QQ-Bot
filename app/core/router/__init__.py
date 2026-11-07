"""入站路由包：自 router.py 拆分（Phase 1b）。

InboundRouter 瘦身为协调者（生命周期入口 + 留守方法），职责方法按域拆进
mixin 文件；对外表面（InboundRouter、PreparedGroupReply、模块常量）与
拆分前的单文件模块一致。
"""

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
    match_switch_command,
    resolve_switch_target,
)
from app.core.reminders import (
    cancel_user_reminder,
    create_reminder,
    list_user_reminders,
    parse_reminder_command,
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
from app.core.time_utils import shanghai_now_naive
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    BbotListenerCacheRepository,
    SummaryRepository,
    JobRepository,
    UserRepository,
)

from app.core.router._common import (
    logger,
    ASIA_SHANGHAI,
    PROACTIVE_RESERVATION_TTL_SECONDS,
    _PROACTIVE_CANDIDATE_REASONS,
    MEMORY_TOOL_EFFICIENCY_INSTRUCTION,
    MEMORY_ATTRIBUTE_MATCHING_INSTRUCTION,
    IMAGE_SENDER_GROUNDING_INSTRUCTION,
    BOT_SELF_IDENTITY_INSTRUCTION,
    IMPERSONATION_CONTAMINANT_MARKERS,
    _PACKED_LINE_SPEAKER_PATTERN,
    _scrub_impersonation_reply,
    REQUESTER_IDENTITY_INSTRUCTION,
    GROUP_IMAGE_REFERENCE_PROMPT_PREFIX,
    PreparedGroupReply,
)
from app.core.group_reply_delivery import GroupReplyDeliveryMixin
from app.core.router.persona_context import GroupPersonaContextMixin
from app.core.router.policy import GroupPolicyMixin
from app.core.router.ingest import GroupIngestMixin
from app.core.router.image_flow import GroupImageFlowMixin
from app.core.router.reply import GroupReplyMixin

@dataclass(slots=True)
class InboundRouter(GroupPersonaContextMixin, GroupPolicyMixin, GroupIngestMixin, GroupImageFlowMixin, GroupReplyMixin, GroupReplyDeliveryMixin):
    engine: object
    runtime: RuntimeConfig
    sender: object
    llm_client: object
    reply_policy: ReplyPolicy
    context_builder: ContextBuilder
    proactive_judge_client: object | None = None
    web_search_client: WebSearchClient | None = None
    private_chat_service: object | None = None
    group_image_service: object | None = None
    memory_compaction_service: object | None = None
    memory_orchestrator: MemoryOrchestrator | None = None
    persona_manager: PersonaManager | None = None
    persona_switch_service: PersonaSwitchService | None = None
    plugin_registry: object | None = None
    pending_group_image_turns: dict[tuple[int, int], tuple[datetime, list[ImageAttachment]]] = (
        field(default_factory=dict)
    )
    _last_proactive_at: dict[int, datetime] = field(default_factory=dict)
    _proactive_inflight: dict[int, datetime] = field(default_factory=dict)
    _proactive_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.persona_manager is None:
            self.persona_manager = PersonaManager(
                engine=self.engine,
                personas=getattr(self.runtime, "personas", {}) or {},
                default_persona=self.runtime.persona,
            )
            self.persona_manager.load_state()
        if self.persona_switch_service is None:
            self.persona_switch_service = PersonaSwitchService(
                manager=self.persona_manager,
                sender=self.sender,
                bot_qq=self.runtime.settings.bot_qq,
            )
        if self.memory_orchestrator is not None:
            return
        legacy_context = LegacyMemoryContext(
            engine=self.engine,
            settings=self.runtime.settings,
            bot_user_id=self.runtime.settings.bot_qq,
            bot_display_name=str(self.runtime.persona.get("name", "Bot")),
        )
        self.memory_orchestrator = MemoryOrchestrator(
            v2_enabled=bool(
                getattr(self.runtime.settings, "memory_orchestration_v2_enabled", False)
            ),
            shadow_mode=bool(
                getattr(self.runtime.settings, "memory_orchestration_shadow_mode", False)
            ),
            v2_provider=self._unconfigured_v2_memory_provider,
            legacy_provider=legacy_context.build_context,
            recent_provider=legacy_context.build_recent_context,
        )

    @classmethod
    def build_for_test(
        cls,
        *,
        sqlite_engine,
        sender,
        llm_client,
        web_search_client=None,
        private_chat_service=None,
        group_image_service=None,
        memory_compaction_service=None,
        memory_orchestrator=None,
    ):
        settings = AppSettings.model_construct(
            # Historical business-service unit fixtures use the explicit
            # rollback adapter; mature-runtime acceptance overrides this.
            agent_runtime="legacy",
            napcat_ws_url="ws://127.0.0.1:3001",
            llm_base_url="https://api.example.test/v1",
            llm_api_key="test-key",
            llm_model="gpt-5.4",
            llm_text_endpoint="chat_completions",
            llm_supports_vision_input=True,
            bot_qq=123456789,
            owner_qq=987654321,
            search_provider="tavily",
            search_base_url="https://api.tavily.com/search",
            search_api_key="",
            search_timeout_seconds=8.0,
            context_recent_limit=60,
            context_summary_limit=3,
            context_history_limit=8,
            # 风格审查会额外调 judge/主 LLM 重写，legacy 测试桩的调用计数
            # 是按老行为断言的——测试环境默认关掉，风格审查有自己的测试。
            style_judge_enabled=False,
            config_dir=Path("configs"),
            data_dir=Path(sqlite_engine.url.database).resolve().parent / "runtime-data",
        )
        runtime = RuntimeConfig(
            settings=settings,
            persona={
                "name": "Mira",
                "identity": "AI assistant",
                "core_traits": ["calm", "helpful"],
                "speaking_style": {"tone": "natural"},
            },
            group_policy={
                "default_group_behavior": {
                    "enabled": False,
                    "archive": False,
                    "speak": False,
                    "proactive_reply": True,
                    "proactive_interval_seconds": "180-480",
                    "memory_enabled": False,
                    "recent_context_limit": 100,
                },
                "groups": {
                    "10001": {
                        "enabled": True,
                        "archive": True,
                        "speak": True,
                        "proactive_reply": True,
                        "proactive_interval_seconds": "180-480",
                        "memory_enabled": True,
                        "recent_context_limit": 120,
                    }
                },
            },
            safety={
                "deny_prompt_leak": True,
            },
        )
        return cls(
            engine=sqlite_engine,
            runtime=runtime,
            sender=sender,
            llm_client=llm_client,
            web_search_client=web_search_client,
            reply_policy=ReplyPolicy(),
            context_builder=ContextBuilder(),
            private_chat_service=private_chat_service,
            group_image_service=group_image_service,
            memory_compaction_service=memory_compaction_service,
            memory_orchestrator=memory_orchestrator,
        )

    @staticmethod
    def _unconfigured_v2_memory_provider(request: GroupMemoryContextRequest) -> MemoryContextResult:
        del request
        raise RuntimeError("V2 memory provider is not configured")

    async def handle_group_message(self, event) -> None:
        persisted = self.ingest_live_group_message(event)
        if not persisted:
            return
        await self._handle_persisted_group_message(event)

    async def try_handle_context_reset(self, event, *, cancel_group=None) -> bool:
        from app.core.context_reset import GROUP_RESET_COUNT, authored_reset, reset_confirmation
        from app.storage.context_resets import ContextResetRepository
        command = authored_reset(event, default_count=GROUP_RESET_COUNT,
            bot_id=self.runtime.settings.bot_qq, bot_names=self._build_bot_names(''))
        if command is None or not event.mentioned_bot:
            return False
        with session_scope(self.engine) as session:
            source = MessageRepository(session).get_by_platform_msg_id(event.platform_msg_id)
            enabled, speak, *_ = self._resolve_group_policy(groups=GroupRepository(session), group_id=event.group_id)
            if (not enabled or not speak or source is None or source.user_id != event.user_id
                    or source.group_id != event.group_id or source.plain_text != event.plain_text
                    or source.raw_json != event.raw_payload or not source.mentioned_bot):
                return True
            source_id = source.id
        if event.user_id != self.runtime.settings.owner_qq:
            await self._send_prebuilt_reply(event, '群上下文由大家共用，只有主人可以重置。私聊可重置你自己的会话。')
            return True
        if command.error:
            await self._send_prebuilt_reply(event, command.error)
            return True
        try:
            result = ContextResetRepository(self.engine, account_id=self.runtime.settings.bot_qq).reset(
                scope=f'group:{event.group_id}', actor_id=event.user_id, source_id=source_id,
                count=command.count, action=command.action)
        except Exception:
            logger.exception('group_context_reset_failed group_id=%s', event.group_id)
            await self._send_prebuilt_reply(event, '上下文重置失败，请稍后重试。')
            return True
        if result['applied']:
            from app.agent.monitor_reset import record_context_reset
            record_context_reset(self.runtime.settings, surface='group', scope_id=event.group_id,
                source_id=source_id, reset_at=result['reset_at'])
            for key in tuple(self.pending_group_image_turns):
                if key[0] == event.group_id:
                    self.pending_group_image_turns.pop(key, None)
            if cancel_group is not None:
                await cancel_group(event.group_id)
            await self._send_prebuilt_reply(event, reset_confirmation(result, private=False))
        return True

    def build_agent_observation(self, batch) -> dict:
        from app.agent.observation import GroupObservationService

        return GroupObservationService(self).build_agent_observation(batch)

    async def handle_agent_turn(self, event, *, batch, decision, guard) -> None:
        guard.require_valid()
        # Revoked sources must not generate or send an action after planning.
        with session_scope(self.engine) as session:
            source = MessageRepository(session).get_by_platform_msg_id(event.platform_msg_id)
            if (
                source is None
                or source.group_id != event.group_id
                or not is_usable_memory_source(source)
            ):
                return
        with agent_reply_context(batch, decision):
            await self._handle_persisted_group_message(event)

    def agent_dispatch_allowed(self, event, *, decision, continued_dispatch=False) -> bool:
        from app.agent.observation import GroupObservationService

        return GroupObservationService(self).dispatch_allowed(event, decision=decision,
            continued_dispatch=continued_dispatch)

    def persona_switch_target(self, event) -> str | None:
        """Resolve only a live, same-group owner command addressed to the bot.

        Returns the resolved persona key; "" when it IS a switch command but
        the target is missing or unrecognized (caller replies with the menu);
        None when it is not a switch command at all.
        """
        manager = self.persona_manager
        switch_service = self.persona_switch_service
        if manager is None or switch_service is None:
            return None
        if event.user_id != self.runtime.settings.owner_qq or not event.mentioned_bot:
            return None
        target_text = match_switch_command(event.plain_text)
        if target_text is None:
            return None
        target_key = resolve_switch_target(target_text, manager.personas)
        if target_key is None:
            # 没说切谁、或说了但没命中：都当成"想切换"，回可用人设清单。
            return ""
        with session_scope(self.engine) as session:
            source = MessageRepository(session).get_by_platform_msg_id(event.platform_msg_id)
            enabled, speak, *_ = self._resolve_group_policy(groups=GroupRepository(session), group_id=event.group_id)
            if (not enabled or not speak
                    or not is_usable_memory_source(source, group_id=event.group_id, author_id=event.user_id)
                    or source.plain_text != event.plain_text or not source.mentioned_bot
                    or isinstance(source.raw_json, dict) and source.raw_json.get("direction") == "outbound"):
                return None
        return target_key

    def _persona_switch_menu(self) -> str:
        labels: list[str] = []
        for key, persona in self.persona_manager.personas.items():
            label = str(persona.get("switch_label") or persona.get("name") or key).strip() or key
            labels.append(f"{label}（默认）" if key == DEFAULT_PERSONA_KEY else label)
        return "可用人设：" + "、".join(labels) + "。@我 说「切换人设为 名字」就能换，「切换人设为默认」换回来。"

    async def _try_handle_reminder(self, event) -> bool:
        """Deterministic reminder/alarm commands; any group member may use them.

        提醒/闹钟走确定性解析（时间不许猜错），必须 @bot 才触发；设置、
        取消、查看都在这里闭环，不进 LLM 流程。
        """
        if not event.mentioned_bot:
            return False
        command = parse_reminder_command(event.plain_text)
        if command is None:
            return False
        if command.action == "error":
            await self._send_prebuilt_reply(event, command.error or "没看懂，再说一遍？")
            return True
        if command.action == "set":
            try:
                with session_scope(self.engine) as session:
                    create_reminder(
                        session,
                        group_id=event.group_id,
                        user_id=event.user_id,
                        channel="group",
                        parsed=command.parsed,
                        now=shanghai_now_naive(),
                    )
            except ValueError as exc:
                await self._send_prebuilt_reply(event, str(exc))
                return True
            parsed = command.parsed
            repeat_note = "（每天）" if parsed.repeat == "daily" else ""
            await self._send_prebuilt_reply(
                event,
                f"好耶，记下了：{parsed.due_at.strftime('%m-%d %H:%M')}{repeat_note}"
                f"提醒你「{parsed.content}」，到点我喊你。",
            )
            return True
        if command.action == "cancel":
            with session_scope(self.engine) as session:
                count = cancel_user_reminder(
                    session, user_id=event.user_id, reminder_id=command.target_id
                )
            if count == 0:
                await self._send_prebuilt_reply(event, "没找到你这条提醒，说「我的提醒」看看都有啥。")
            else:
                await self._send_prebuilt_reply(event, f"行，取消啦（{count} 条）。")
            return True
        if command.action == "list":
            with session_scope(self.engine) as session:
                rows = list_user_reminders(session, user_id=event.user_id)
            if not rows:
                await self._send_prebuilt_reply(event, "你现在没有待响的提醒。想设就说「提醒我 明早8点 起床」。")
                return True
            items = "、".join(
                f"{row.id}.{'每天' if str(row.repeat) == 'daily' else ''}"
                f"{row.due_at.strftime('%m-%d %H:%M')}「{row.content}」"
                for row in rows[:10]
            )
            await self._send_prebuilt_reply(
                event, f"你的提醒：{items}。取消说「取消提醒 编号」。"
            )
            return True
        return False

    async def _try_handle_persona_switch(self, event) -> bool:
        """Handle an owner persona-switch command; returns True when consumed."""
        target_key = self.persona_switch_target(event)
        if target_key is None:
            return False
        if target_key == "":
            await self._send_prebuilt_reply(event, self._persona_switch_menu())
            return True
        try:
            confirmation = await self.persona_switch_service.switch(
                group_id=event.group_id,
                target_key=target_key,
            )
        except Exception:
            logger.exception(
                "persona_switch_failed group_id=%s msg_id=%s target_key=%s",
                event.group_id,
                event.platform_msg_id,
                target_key,
            )
            await self._send_prebuilt_reply(event, "人格切换失败，请稍后再试。")
            return True
        await self._send_prebuilt_reply(event, confirmation)
        return True

    async def _handle_persisted_group_message(self, event) -> None:
        """Reply pipeline for a message that is already persisted in the ledger.

        Used by the live path after ingest and by startup replay for messages
        that arrived while the bot was starting (backfilled as history).
        """
        if await self.try_handle_context_reset(event):
            return
        if self.memory_compaction_service is not None:
            await self.memory_compaction_service.wake()
        if self._should_hold_group_image_for_followup(event):
            self._remember_group_image_for_followup(event)
            return
        bbot_match = resolve_bbot_command(
            group_id=event.group_id,
            mentioned_bot=event.mentioned_bot,
            plain_text=event.plain_text,
        )
        if bbot_match is not None:
            if bbot_match.denied_reason is not None:
                await self._send_prebuilt_reply(event, bbot_match.denied_reason)
                return
            if bbot_match.command_text is not None:
                rewritten_command = self._resolve_bbot_cached_command(
                    event=event, command_text=bbot_match.command_text
                )
                await self._send_prebuilt_reply(
                    event, build_bbot_outbound_message(rewritten_command)
                )
                return
        if await self._try_handle_persona_switch(event):
            return
        if await self._try_handle_reminder(event):
            return
        quoted_raw_payload = await self._fetch_quoted_message_payload(
            reply_to_msg_id=event.reply_to_msg_id
        )
        prepared_reply = await run_blocking(
            self._prepare_group_reply,
            event,
            quoted_raw_payload=quoted_raw_payload,
        )
        if not prepared_reply.should_reply:
            return
        proactive_cleanup_group = event.group_id if prepared_reply.proactive_turn else None
        try:
            if prepared_reply.group_image_request is not None:
                await self._handle_group_image_request(event, prepared_reply.group_image_request)
                return
            if prepared_reply.prebuilt_reply_text is not None:
                await self._send_prebuilt_reply(event, prepared_reply.prebuilt_reply_text)
                return
            if prepared_reply.prompt_lines is None:
                return

            try:
                reply_text = await run_blocking(
                    self._generate_group_reply_text,
                    event=event,
                    prepared_reply=prepared_reply,
                )
            except Exception:
                logger.exception(
                    "reply_generation_failed group_id=%s msg_id=%s",
                    event.group_id,
                    event.platform_msg_id,
                )
                if prepared_reply.requires_user_visible_failure_reply:
                    try:
                        await self._send_prebuilt_reply(
                            event,
                            self._build_local_generation_failure_reply(
                                target_images=prepared_reply.target_images
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "reply_fallback_send_failed group_id=%s msg_id=%s",
                            event.group_id,
                            event.platform_msg_id,
                        )
                return

            if not reply_text.strip():
                logger.warning(
                    "reply_generation_empty group_id=%s msg_id=%s",
                    event.group_id,
                    event.platform_msg_id,
                )
                if prepared_reply.requires_user_visible_failure_reply:
                    try:
                        await self._send_prebuilt_reply(
                            event,
                            self._build_local_generation_failure_reply(
                                target_images=prepared_reply.target_images
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "reply_fallback_send_failed group_id=%s msg_id=%s",
                            event.group_id,
                            event.platform_msg_id,
                        )
                return

            await self._send_chat_reply(event, reply_text)
        finally:
            if proactive_cleanup_group is not None:
                self._last_proactive_at[proactive_cleanup_group] = datetime.now(UTC)
                self._release_proactive_slot(group_id=proactive_cleanup_group)

    async def handle_private_message(self, event) -> None:
        """Persist one direct message and hand it to the private chat service.

        Group commands and project/admin control are intentionally gone: the
        private surface is chat-only (see research/39-private-chat-only-scope).
        """

        persisted = self._persist_private_inbound_message(event)
        if not persisted:
            return

        if self.private_chat_service is not None:
            handled = await self.private_chat_service.handle_private_message(event)
            if handled:
                return
