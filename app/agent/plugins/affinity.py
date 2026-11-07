"""好感度插件：入站互动积累好感，直接互动时注入当前关系行。"""

from __future__ import annotations

from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, InboundContext, TurnContext
from app.agent.tool_agent import AgentTool
from app.core.affinity import EVENT_DELTAS, affinity_prompt_line, apply_affinity_event
from app.core.group_mood import evaluate_inbound_trigger


class AffinityFeature(FeatureModule):
    """麦麦式关系系统的轻量实现（详见 app/core/affinity.py）。"""

    name = "affinity"

    _engine = None

    def attach_engine(self, engine) -> None:
        self._engine = engine

    def on_inbound(self, ctx: InboundContext) -> None:
        from app.core.time_utils import shanghai_now_naive

        event = ctx.event
        bot_qq = ctx.router.runtime.settings.bot_qq
        trigger, explicit = evaluate_inbound_trigger(
            messages=ctx.messages, event=event, bot_qq=bot_qq)
        if trigger is not None:
            delta = EVENT_DELTAS.get(trigger[0], 0.2)
        elif explicit:
            delta = EVENT_DELTAS["chat"]
        else:
            return
        apply_affinity_event(ctx.session, group_id=event.group_id, user_id=event.user_id,
                             delta=delta, now=shanghai_now_naive())

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group" or self._engine is None or not target_user_id:
            return ()
        from app.core.time_utils import shanghai_now_naive

        line = affinity_prompt_line(self._engine, group_id=group_id,
                                    user_id=int(target_user_id), now=shanghai_now_naive())
        return [line] if line else ()
