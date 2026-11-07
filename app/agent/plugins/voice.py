"""语音插件：agent 想说话时可以真的"开口"——TTS 成语音条发群里/私聊。

模型通过 speak 工具使用：用自己的语气现说（不预置台词），TTS 成语音后
以 record 段发出。配额 TTS_DAILY_LIMIT/天；未配置 TTS_API_KEY 时整个
功能自动隐藏。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.time_utils import shanghai_now_naive

logger = logging.getLogger(__name__)

MAX_SPEAK_CHARS = 150


class VoiceFeature(FeatureModule):
    name = "voice"

    #: 主动开口提示的概率与冷却：情绪到位偶尔直接开口才有"活人感"，
    #: 但不能刷屏——每群至少隔 N 小时才提示一次，且只在有配额时。
    PROACTIVE_NUDGE_PROBABILITY = 0.1
    PROACTIVE_NUDGE_COOLDOWN_SECONDS = 7200

    def __init__(self) -> None:
        # 每实例状态：轮次计数、主动开口冷却、配额。类属性会让多个实例
        # 共享同一个 dict（测试互相污染，生产也会跨生命周期残留）。
        self._settings = None
        self._tts = None
        self._usage_day: str = ""
        self._usage_count = 0
        self._prompt_turns: dict = {}
        self._last_nudge: dict = {}

    def attach(self, *, settings) -> None:
        self._settings = settings
        from app.core.voice import build_tts

        self._tts = build_tts(settings)
        if self._tts is None:
            logger.info("voice_feature_disabled reason=no_tts_api_key")

    def available(self) -> bool:
        return self._tts is not None

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        # 只在直接回应群友的轮注入（observing=True 是主动观察轮，不鼓励
        # 纯主动发语音）。两种提示：每 8 轮一次的工具提醒，加小概率的
        # "这轮情绪到位可以直接开口"。
        if kind != "group" or observing or self._tts is None:
            return ()
        count = self._prompt_turns.get(group_id, 0) + 1
        self._prompt_turns[group_id] = count
        lines = []
        if count % 8 == 2:
            lines.append("（提示：你有 speak 工具，可以把要说的话变成真语音条；"
                         "别人提到想听你声音、让你说话或唱一句时优先用它。）")
        if self._proactive_nudge_ok(group_id, now):
            lines.append("（这轮如果情绪到位、语气适合开口，可以直接用 speak "
                         "把要说的话变成语音条发出去，不用打字。）")
        return lines

    def _proactive_nudge_ok(self, group_id: int, now) -> bool:
        import random

        if random.random() >= self.PROACTIVE_NUDGE_PROBABILITY:
            return False
        last = self._last_nudge.get(group_id)
        if last is not None:
            try:
                from app.core.time_utils import shanghai_naive

                elapsed = (shanghai_naive(now) - shanghai_naive(last)).total_seconds()
                if elapsed < self.PROACTIVE_NUDGE_COOLDOWN_SECONDS:
                    return False
            except (TypeError, ValueError):
                return False
        if not self._quota_ok():
            return False
        self._last_nudge[group_id] = now
        return True

    def _quota_ok(self) -> bool:
        now = shanghai_now_naive()
        today = f"{now:%Y-%m-%d}"
        if self._usage_day != today:
            self._usage_day = today
            self._usage_count = 0
        limit = int(getattr(self._settings, "tts_daily_limit", 20) or 20)
        return self._usage_count < limit

    def _spend_quota(self) -> None:
        self._usage_count += 1

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        if not self.available():
            return ()

        async def speak(args):
            turn.guard.require_valid()
            text = str(args.get("text") or "").strip()
            if not text:
                return ToolOutcome({"error": "empty_text"}, "blocked")
            if len(text) > MAX_SPEAK_CHARS:
                return ToolOutcome({"error": "text_too_long",
                    "guidance": f"语音最多 {MAX_SPEAK_CHARS} 字；长内容请发文字。"}, "blocked")
            if not self._quota_ok():
                return ToolOutcome({"error": "voice_daily_limit",
                    "note": "今天的语音额度用完了，改发文字。"}, "blocked")

            import tempfile
            workdir = Path(tempfile.mkdtemp())
            try:
                mp3 = await asyncio.to_thread(self._tts.synthesize, text, workdir)
                self._spend_quota()
            except Exception as exc:
                logger.warning("tts_synthesize_failed error=%s", exc)
                return ToolOutcome({"error": "tts_failed",
                    "note": "语音生成失败；改发文字，别提语音的事。"}, "blocked")

            from app.adapters.sender import OutboundMessage

            await turn.router.sender.send_group_voice(
                group_id=turn.job.scope_id, voice_file=str(mp3))
            return ToolOutcome({"spoken": True, "chars": len(text),
                "note": "语音已发出。说得自然，像开口说话不是念稿。"})

        return [AgentTool(
            "speak",
            "真的开口说话：把你要说的话变成一条语音发到群里。什么时候用："
            "有人想听你声音、让你说话/唱一句、夸你声音好听时优先用它；"
            "撒娇、得意、害羞、挑衅这些情绪到位的时刻，开口比打字更有杀伤力。"
            "像随口说话那样写内容——短句、口语、带语气词，别书面别念稿；最多150字。"
            "一轮最多发一条语音，别连发。",
            {"type": "object", "properties": {"text": {"type": "string", "maxLength": MAX_SPEAK_CHARS}},
             "required": ["text"], "additionalProperties": False},
            speak, "write")]
