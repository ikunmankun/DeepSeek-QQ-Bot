"""戳一戳反应与新人欢迎插件：notice 事件驱动的真人感互动。

两个插件都通过 ctx.router 拿到 sender / llm_client / persona，反应文案
用一次轻量 LLM 调用按人设现说（不预置台词池），冷却与限额见
app/core/interactions.py。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, NoticeContext
from app.core.interactions import (
    InteractionState,
    build_poke_prompt,
    build_welcome_prompt,
    group_increase_info,
    poke_payload_info,
    should_poke_back,
)
from app.core.time_utils import shanghai_naive, shanghai_now_naive

logger = logging.getLogger(__name__)


def _event_now(ctx: NoticeContext):
    """notice 事件时钟：优先用 ctx.now（可测试），兜底真实上海时间。"""

    return shanghai_naive(ctx.now) if ctx.now is not None else shanghai_now_naive()

MAX_REACTION_CHARS = 60


class PokeFeature(FeatureModule):
    """被戳有反应：按人设现说一句，30% 概率戳回去。"""

    name = "poke"

    def __init__(self) -> None:
        self.state = InteractionState()

    def _resolve_display_name(self, engine, group_id: int, user_id: int) -> str | None:
        """群成员 QQ 号 → 群名片/昵称（查不到返回 None）。"""
        if engine is None:
            return None
        try:
            from app.core.member_identity import group_member_identities_from_messages
            from app.storage.db import session_scope
            from app.storage.repositories import MessageRepository

            with session_scope(engine) as session:
                msgs = MessageRepository(session).list_recent_group_member_messages(
                    group_id=group_id, limit=200)
            for m in group_member_identities_from_messages(msgs, target_group_id=group_id):
                if int(m.user_id) == int(user_id):
                    return str(m.group_card or "").strip() or str(m.nickname or "").strip() or None
        except Exception:
            pass
        return None

    def _relation_hint(self, engine, group_id: int, user_id: int) -> str | None:
        """好感度定性提示（老朋友/熟人/认识），没有记录返回 None。"""
        if engine is None:
            return None
        try:
            from app.core.affinity import affinity_level, decayed_score, load_affinity

            row = load_affinity(engine, group_id=group_id, user_id=user_id)
            if row is None:
                return None
            return affinity_level(decayed_score(row, shanghai_now_naive()))
        except Exception:
            return None

    def on_notice(self, ctx: NoticeContext) -> None:
        import asyncio

        info = poke_payload_info(ctx.payload, bot_qq=ctx.router.runtime.settings.bot_qq)
        if info is None:
            return
        group_id, poker_user_id = info
        # 时钟以 ctx.now 为准（可测试），归一化成 naive 上海钟面存储。
        now = _event_now(ctx)
        if not self.state.poke_allowed(group_id, now):
            return
        self.state.record_poke_react(group_id, now)

        persona_text = str(ctx.router.runtime.persona.get("identity", "") or "")
        engine = getattr(ctx.router, "engine", None)
        poker_name = self._resolve_display_name(engine, group_id, poker_user_id)
        relation_hint = self._relation_hint(engine, group_id, poker_user_id)
        reply = ctx.router.llm_client.generate_text(
            build_poke_prompt(persona_text=persona_text, poker_name=poker_name,
                              relation_hint=relation_hint),
            conversation_key=f"poke:{group_id}",
        )
        text = str(reply or "").strip()[:MAX_REACTION_CHARS]
        if not text:
            return

        async def deliver() -> None:
            from app.adapters.sender import OutboundMessage

            try:
                await ctx.router.sender.send_group_text(
                    OutboundMessage(group_id, text))
                if should_poke_back(now.timestamp()):
                    await ctx.router.sender.send_group_poke(
                        group_id=group_id, user_id=poker_user_id)
            except Exception:
                logger.exception("poke_reaction_send_failed group=%s", group_id)

        asyncio.get_running_loop().create_task(deliver())


class WelcomeFeature(FeatureModule):
    """新人进群欢迎：一句话热乎招呼，30 秒冷却防连发。"""

    name = "welcome"

    def __init__(self) -> None:
        self.state = InteractionState()

    def on_notice(self, ctx: NoticeContext) -> None:
        import asyncio

        info = group_increase_info(ctx.payload, bot_qq=ctx.router.runtime.settings.bot_qq)
        if info is None:
            return
        group_id, new_user_id = info
        now = _event_now(ctx)
        if not self.state.welcome_allowed(group_id, now):
            return
        self.state.record_welcome(group_id, now)

        persona_text = str(ctx.router.runtime.persona.get("identity", "") or "")
        reply = ctx.router.llm_client.generate_text(
            build_welcome_prompt(persona_text=persona_text),
            conversation_key=f"welcome:{group_id}",
        )
        text = str(reply or "").strip()[:MAX_REACTION_CHARS]
        if not text:
            return

        async def deliver() -> None:
            from app.adapters.sender import OutboundMessage

            try:
                # 新人 QQ 号来自 notice 本身：直接前缀 at 记号，@ 得又准又自然。
                await ctx.router.sender.send_group_text(
                    OutboundMessage(group_id, f"[@qq:{new_user_id}] " + text))
            except Exception:
                logger.exception("welcome_send_failed group=%s", group_id)

        asyncio.get_running_loop().create_task(deliver())
