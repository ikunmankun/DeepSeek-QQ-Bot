"""内置插件：表情包（含学习库）、日记、情绪、好感度、记忆衰减。

每个插件把此前散落在 autonomous_social / router / group_main 里的接线
收拢为一个 FeatureModule；行为与迁移前完全一致。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, InboundContext, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome


def _build_learned_meme_tool(learner, meanings_dir=None) -> AgentTool:
    """识梗记忆：按关键词查学过的表情（情绪标签 + 梗含义），含 memes 编号库。"""
    from app.core.sticker_learning import IMAGE_SUFFIXES

    def find(args):
        needle = str(args.get("keyword") or "").strip().lower()
        if not needle:
            return ToolOutcome({"error": "keyword_required"}, "blocked")
        hits = []
        if learner.learned_dir.exists():
            for path in sorted(learner.learned_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                tags = learner.load_tags(path)
                meaning = learner.load_meaning(path)
                haystack = f"{tags} {meaning}".lower()
                if needle in haystack:
                    hits.append({"file": path.stem, "tags": tags, "meaning": meaning,
                                 "meme_id": path.stem})
                if len(hits) >= 8:
                    break
        if meanings_dir is not None and Path(meanings_dir).exists() and len(hits) < 8:
            for sidecar in sorted(Path(meanings_dir).glob("*.json")):
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                haystack = f"{data.get('tags', '')} {data.get('meaning', '')}".lower()
                if needle in haystack:
                    hits.append({"file": sidecar.stem, "tags": str(data.get('tags', '')),
                                 "meaning": str(data.get('meaning', '')),
                                 "meme_id": sidecar.stem})
                if len(hits) >= 8:
                    break
        return ToolOutcome({"matches": hits, "matched": len(hits),
            "guidance": "命中的图用 pick_sticker(meme_id=文件名) 精确取图；没命中就 search_web 搜梗，"
                        "搞懂后这张图被打标时含义会进梗库。"}, "ok")

    return AgentTool(
        "find_learned_meme",
        "按关键词查自己学过的梗图（情绪标签+梗含义）。群里又出现见过的梗时先查这里，"
        "秒懂不用装；想复用某张学过的图也用它定位 meme_id 再 pick_sticker 取图。",
        {"type": "object", "properties": {"keyword": {"type": "string", "maxLength": 32}},
         "required": ["keyword"], "additionalProperties": False},
        find,
        "read",
    )


class StickerFeature(FeatureModule):
    """本地表情库：pick_sticker 工具，按 LOCAL_STICKER_DIR 开关。"""

    name = "stickers"

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        sticker_dir = str(getattr(turn.router.runtime.settings, "local_sticker_dir", "") or "").strip()
        if not sticker_dir:
            return ()
        from app.agent.local_stickers import build_sticker_tools
        from app.core.sticker_learning import StickerLearner

        learner = StickerLearner(turn.router.runtime.settings.data_dir)
        meanings_dir = Path(turn.router.runtime.settings.data_dir) / "meme_meanings"
        tools = list(build_sticker_tools(runner=turn.blocking, guard=turn.guard,
                                         store=turn.store, directory=Path(sticker_dir),
                                         extra_directory=learner.learned_dir,
                                         meanings_dir=meanings_dir))
        tools.append(_build_learned_meme_tool(learner, meanings_dir))
        return tools


class DiaryFeature(FeatureModule):
    """本地日记：write_diary 工具 + 每晚定时反思 worker。"""

    name = "diary"

    _engine = None
    _settings = None
    _llm_client = None
    _group_ids: tuple[int, ...] = ()
    _persona_text = ""
    _bot_qq = 0
    _reflection_hooks: tuple = ()

    def add_reflection_hook(self, hook) -> None:
        """注册夜间反思后的钩子（如 AgentSelfFeature 的议程生成）。"""
        self._reflection_hooks = tuple(self._reflection_hooks) + (hook,)

    def attach(self, *, engine, settings, llm_client, group_ids, persona_text, bot_qq) -> None:
        self._engine = engine
        self._settings = settings
        self._llm_client = llm_client
        self._group_ids = tuple(group_ids)
        self._persona_text = persona_text
        self._bot_qq = int(bot_qq)

    def _fire_reflection_hooks(self, now) -> None:
        for hook in self._reflection_hooks:
            try:
                hook(now)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("diary_reflection_hook_failed")

    def create_workers(self) -> Sequence[Any]:
        from app.core.diary import DiaryReflectionService, DiaryWriter

        if not getattr(self._settings, "diary_reflection_enabled", True):
            return ()
        service = DiaryReflectionService(
            engine=self._engine, settings=self._settings, llm_client=self._llm_client,
            writer=DiaryWriter(self._settings.data_dir), group_ids=self._group_ids,
            persona_text=self._persona_text, bot_qq=self._bot_qq,
            reflection_time=str(getattr(self._settings, "diary_reflection_time", "23:30")))
        service._reflect_hook = self._fire_reflection_hooks
        return [service]

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        from app.core.diary import DiaryWriter, build_diary_tool
        from app.agent.tool_agent import AgentTool, ToolOutcome

        writer = DiaryWriter(turn.router.runtime.settings.data_dir)

        async def read(args):
            turn.guard.require_valid()
            content = writer.read_recent()
            if not content:
                return ToolOutcome({"entries": "", "note": "日记还是空的，一篇都没写过。"})
            return ToolOutcome({"entries": content,
                "note": "这是自己的私密日记，只有主人可以看；转述时用自然语气，别整段照念。"})

        return [build_diary_tool(guard=turn.guard, writer=writer,
                                 mood_supplier=lambda: self._current_mood(turn)),
                AgentTool("read_diary",
                    "翻自己的日记。什么时候用：主人让你看日记、或你想回顾最近的想法时。"
                    "日记是私密的：非主人问起一律不展示内容，只说那是自己写的。",
                    {"type": "object", "properties": {},
                     "required": [], "additionalProperties": False},
                    read, "read")]

    def _current_mood(self, turn: TurnContext) -> str | None:
        from app.core.group_mood import effective_mood_state, load_group_mood

        mood, _intensity = effective_mood_state(
            load_group_mood(turn.router.engine, turn.job.scope_id), turn.now)
        return None if mood == "default" else mood


class MoodFeature(FeatureModule):
    """规则版运行时情绪：入站钩子更新状态，群聊提示词注入当前心情。"""

    name = "mood"

    def on_inbound(self, ctx: InboundContext) -> None:
        from app.core.group_mood import apply_mood_trigger, evaluate_inbound_trigger

        event = ctx.event
        trigger, _explicit = evaluate_inbound_trigger(
            messages=ctx.messages, event=event,
            bot_qq=ctx.router.runtime.settings.bot_qq)
        if trigger is None:
            return
        mood, bump, reason = trigger
        apply_mood_trigger(ctx.session, group_id=event.group_id, mood=mood, bump=bump,
            reason=reason, now=event.timestamp)

    _engine = None

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group" or self._engine is None:
            return ()
        from app.core.group_mood import mood_prompt_line

        line = mood_prompt_line(self._engine, group_id, now)
        return [line] if line else ()

    def attach_engine(self, engine) -> None:
        self._engine = engine


class MemoryDecayFeature(FeatureModule):
    """记忆淡忘归档：周期性把超过归档线的记忆停用。"""

    name = "memory_decay"
    _engine = None
    _settings = None

    def __init__(self, *, interval_seconds: float | None = None) -> None:
        self._interval_seconds = interval_seconds

    def create_workers(self) -> Sequence[Any]:
        from app.core.memory_decay_service import MemoryDecayService

        if self._engine is None:
            return ()
        kwargs = {"engine": self._engine, "settings": self._settings}
        if self._interval_seconds is not None:
            kwargs["interval_seconds"] = self._interval_seconds
        return [MemoryDecayService(**kwargs)]

    def attach(self, *, engine, settings) -> None:
        self._engine = engine
        self._settings = settings


class StickerLearningFeature(FeatureModule):
    """偷表情包：收藏群友表情，周期 worker 视觉打标，供 pick_sticker 复用。"""

    name = "sticker_learning"

    _engine = None
    _settings = None

    def attach(self, *, engine, settings) -> None:
        self._engine = engine
        self._settings = settings

    def on_inbound(self, ctx: InboundContext) -> None:
        from app.core.sticker_learning import StickerLearner

        learner = StickerLearner(ctx.router.runtime.settings.data_dir)
        collected = learner.collect_from_event(
            ctx.event, bot_qq=ctx.router.runtime.settings.bot_qq)
        if collected:
            logging.info("sticker_learning_collected count=%s group=%s",
                         collected, ctx.event.group_id)

    def create_workers(self) -> Sequence[Any]:
        from app.core.sticker_learning import StickerLearner, StickerTaggingService

        if self._settings is None:
            return ()
        learner = StickerLearner(self._settings.data_dir)
        sticker_dir = str(getattr(self._settings, "local_sticker_dir", "") or "").strip()
        return [StickerTaggingService(learner=learner, settings=self._settings,
                                      memes_dir=Path(sticker_dir) if sticker_dir else None,
                                      meme_meanings_dir=Path(self._settings.data_dir) / "meme_meanings")]


class HealthReportFeature(FeatureModule):
    """每日自检：22:00 私聊主人运行摘要（用量/投递/代理/表情库/识曲）。"""

    name = "health_report"

    _engine = None
    _settings = None
    _sender = None

    def attach(self, *, engine, settings, sender) -> None:
        self._engine = engine
        self._settings = settings
        self._sender = sender

    def create_workers(self) -> Sequence[Any]:
        if self._engine is None or self._sender is None:
            return ()
        from app.core.health_report import HealthReportService
        return [HealthReportService(engine=self._engine, settings=self._settings,
                                    sender=self._sender)]


class SlangFeature(FeatureModule):
    """群黑话学习：周期统计高频词，群聊提示词注入黑话清单。"""

    name = "slang"

    _engine = None
    _settings = None
    _group_ids: tuple[int, ...] = ()

    def attach(self, *, engine, settings, group_ids) -> None:
        self._engine = engine
        self._settings = settings
        self._group_ids = tuple(group_ids)

    def create_workers(self) -> Sequence[Any]:
        from app.core.slang import SlangLearningService

        if self._engine is None or not self._group_ids:
            return ()
        return [SlangLearningService(engine=self._engine, group_ids=self._group_ids)]

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group" or self._engine is None:
            return ()
        from app.core.slang import SlangLearner

        terms = SlangLearner(self._engine).top_terms(group_id)
        if not terms:
            return ()
        words = "、".join(term for term, _count in terms)
        return [f"群里最近的高频词（可能是梗或黑话，聊到相关话题时自然用，别硬凑、别解释）：{words}。"]
