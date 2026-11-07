"""本地日记：给机器人一个不发群里的表达出口。

两层：
* write_diary 工具——巡检/闲聊中"想说但不发群里"的话，随手写一条；
* 每日反思（DiaryReflectionService）——每晚固定时刻雷打不动的复盘：
  把当天的素材（群聊摘要、心情变化、和谁的互动、自己说过的话）摆在一起，
  让模型自己决定写什么，写进同一天的日记文件。群友看不到。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.time_utils import shanghai_now_naive

MAX_DIARY_CHARS = 2000


class DiaryWriter:
    """按天追加的 markdown 日记本。"""

    def __init__(self, data_dir: Path) -> None:
        self.directory = Path(data_dir) / "diary"

    def already_reflected_today(self) -> bool:
        """今天的日记里是否已有睡前反思（重启后不重写）。"""

        return self._reflection_marker_in(today_file=True)

    def _reflection_marker_in(self, *, today_file: bool) -> bool:
        day = shanghai_now_naive()
        path = self.directory / f"{day:%Y-%m-%d}.md"
        if not path.exists():
            return False
        return "睡前记" in path.read_text(encoding="utf-8")

    def read_recent(self, *, max_chars: int = 1500, max_days: int = 7) -> str:
        """从今天往回读最近几天的日记，拼到 max_chars 为止；没有日记返回空串。"""

        self.directory.mkdir(parents=True, exist_ok=True)
        today = shanghai_now_naive().date()
        parts: list[str] = []
        used = 0
        for offset in range(max_days):
            path = self.directory / f"{today - timedelta(days=offset):%Y-%m-%d}.md"
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            header = f"【{(today - timedelta(days=offset)):%m-%d}】" if offset else "【今天】"
            block = f"{header}\n{content}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def write(self, *, text: str, now: datetime, mood: str | None = None,
              tag: str = "") -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        day = shanghai_now_naive() if now is None else now
        path = self.directory / f"{day:%Y-%m-%d}.md"
        clock = f"{day:%H:%M}"
        header = f"## {clock}"
        if mood:
            header += f" · 心情：{mood}"
        if tag:
            header += f" · {tag}"
        entry = f"{header}\n\n{text.strip()}\n\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(entry)
        return path


def build_diary_tool(*, guard, writer: DiaryWriter,
                     mood_supplier: Callable[[], str | None] | None = None) -> AgentTool:
    """write_diary：本地写入，不产生任何对外效果，因此不受发送额度约束。"""

    async def write(args):
        guard.require_valid()
        text = str(args.get("text") or "").strip()
        if not text:
            return ToolOutcome({"error": "empty_diary_text",
                "guidance": "日记内容不能为空；没话说就不用调用这个工具。"}, "blocked")
        if len(text) > MAX_DIARY_CHARS:
            return ToolOutcome({"error": "diary_too_long",
                "guidance": f"单篇日记不超过 {MAX_DIARY_CHARS} 字；写当天的想法，不是写作文。"}, "blocked")
        now = shanghai_now_naive()
        mood = None
        if mood_supplier is not None:
            try:
                mood = mood_supplier()
            except Exception:
                mood = None
        path = writer.write(text=text, now=now, mood=mood)
        return ToolOutcome({"saved": True, "path": str(path), "chars": len(text),
            "guidance": "日记已写入本地文件，群友看不到；不需要也不可能把它发出去。"})

    return AgentTool(
        "write_diary",
        "把自己的想法写进本地日记（markdown 文件，群友看不到、也不会发到群里）。"
        "什么时候该用：巡检时有想说但不适合公开发的话、对群聊氛围的观察、"
        "记挂某个人、或者单纯想碎碎念。日记是私下表达，语气可以比群里更真。"
        "一次写一段话，不写流水账。",
        {"type": "object", "properties": {"text": {"type": "string", "maxLength": MAX_DIARY_CHARS}},
            "required": ["text"], "additionalProperties": False},
        write,
        "read",
    )


# ---------------------------------------------------------------- 每日反思

MAX_MATERIAL_CHARS = 1200

_WEEKDAY_NAMES = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}


def collect_daily_material(engine, *, group_ids, bot_qq: int, now: datetime) -> str:
    """把当天的素材摆在一起，供睡前反思。每块独立容错，缺啥跳啥。"""

    from sqlalchemy import func, select

    from app.storage.db import session_scope
    from app.storage.models import Message, Summary, UserAffinityState
    from app.storage.repositories import SummaryRepository

    blocks: list[str] = []
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1) 各群今天的聊天摘要（记忆系统已生成）
    try:
        with session_scope(engine) as session:
            repo = SummaryRepository(session)
            for group_id in list(group_ids)[:4]:
                for item in repo.list_group_summaries(
                        scope_id=str(group_id), limit=4,
                        start_at=day_start, end_at=now):
                    snippet = str(item.content).strip()[:160]
                    if snippet:
                        blocks.append(f"[群{group_id}] {snippet}")
    except Exception:
        pass

    # 2) 心情状态（非 default 的才值得记）
    try:
        with session_scope(engine) as session:
            rows = list(session.scalars(select(GroupMoodState)))
            mood_bits = [f"群{r.group_id}:{r.mood}"
                         for r in rows if str(r.mood or "") not in {"", "default"}]
            if mood_bits:
                blocks.append("今天的心情：" + "；".join(mood_bits))
    except Exception:
        pass

    # 3) 互动最多的人（好感度前 3）
    try:
        with session_scope(engine) as session:
            rows = session.execute(
                select(UserAffinityState)
                .order_by(func.abs(UserAffinityState.score).desc())
                .limit(3)
            ).scalars().all()
            bits = [f"用户{r.user_id}(互动{int(r.interactions or 0)}次,好感{float(r.score):.1f})"
                    for r in rows if int(r.interactions or 0) > 0]
            if bits:
                blocks.append("今天互动最多的人：" + "；".join(bits))
    except Exception:
        pass

    # 4) 自己今天说了多少话、最后几句是什么（帮助回忆"我今天经历了什么"）
    try:
        with session_scope(engine) as session:
            count = int(session.execute(
                select(func.count()).select_from(Message).where(
                    Message.user_id == bot_qq,
                    Message.timestamp >= day_start,
                )).scalar_one())
            recent = list(session.execute(
                select(Message.plain_text).where(
                    Message.user_id == bot_qq,
                    Message.timestamp >= day_start,
                    Message.msg_type == "text",
                ).order_by(Message.id.desc()).limit(3)
            ).scalars())
        if count:
            blocks.append(f"今天你自己发了{count}条消息" +
                          ("，最后几句：" + " / ".join(str(x)[:40] for x in reversed(recent))
                           if recent else ""))
    except Exception:
        pass

    return "\n".join(blocks)[:MAX_MATERIAL_CHARS]


def build_reflection_prompt(*, persona_text: str, material: str, now: datetime) -> list[str]:
    weekday = _WEEKDAY_NAMES.get(now.weekday(), "")
    lines = ["System persona: " + persona_text.strip(),
        f"Group policy: 现在是{now:%m月%d日}（{weekday}）深夜，睡前写今天的日记。"
        "日记是私密的，没人会看到，写你真正想记的——对某个人今天的看法、一件小事、"
        "一句没说出口的话，或者今天就是平平无奇的一天也照实写。"
        "不是汇报也不是总结陈词，别复述素材本身；一两句到两百字都行。"
        "子句之间用逗号或连写，不用空格；不写颜文字和emoji。"]
    if material.strip():
        lines.append("今天的素材（只是参考，不是必答题）：\n" + material.strip())
    lines.append("直接输出日记正文：不要日期抬头，不要解释，不要汇报腔。")
    return lines



class DiaryReflectionService:
    """每晚固定时刻的日记反思：收集当天素材 → 人设现写一篇 → 落盘。

    这不是聊天回合，是机器人自己的睡前仪式；写出来的东西只进日记文件。
    """

    def __init__(self, *, engine, settings, llm_client, writer: DiaryWriter,
                 group_ids: Sequence[int], persona_text: str, bot_qq: int,
                 reflection_time: str = "23:30") -> None:
        self.engine = engine
        self.settings = settings
        self.llm_client = llm_client
        self.writer = writer
        self.group_ids = tuple(group_ids)
        self.persona_text = persona_text
        self.bot_qq = int(bot_qq)
        hour, minute = str(reflection_time or "23:30").split(":")[:2]
        self._hour, self._minute = int(hour), int(minute)
        self._reflect_hook = None  # 由插件注入：写完日记后生成议程/推进项目

    def next_run(self, now: datetime) -> datetime:
        candidate = now.replace(hour=self._hour, minute=self._minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    def _reflect(self) -> Path | None:
        now = shanghai_now_naive()
        if self.writer.already_reflected_today():
            return None  # 重启后不重写当天的睡前反思
        material = collect_daily_material(self.engine, group_ids=self.group_ids,
                                          bot_qq=self.bot_qq, now=now)
        prompt = build_reflection_prompt(persona_text=self.persona_text,
                                         material=material, now=now)
        reply = self.llm_client.generate_text(
            prompt, conversation_key=f"diary:{now:%Y-%m-%d}")
        text = str(reply or "").strip()
        if not text:
            return None
        path = self.writer.write(text=text[:400], now=now, mood=None, tag="睡前记")
        if self._reflect_hook is not None:
            try:
                self._reflect_hook(now)
            except Exception:
                logging.getLogger(__name__).exception("diary_reflect_hook_failed")
        return path

    async def run(self) -> None:
        import asyncio

        while True:
            now = shanghai_now_naive()
            nxt = self.next_run(now)
            await asyncio.sleep(max(1.0, (nxt - now).total_seconds()))
            try:
                path = await asyncio.to_thread(self._reflect)
                if path is not None:
                    logging.getLogger(__name__).info(
                        "diary_reflection_written path=%s", path)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.getLogger(__name__).exception("diary_reflection_failed")
