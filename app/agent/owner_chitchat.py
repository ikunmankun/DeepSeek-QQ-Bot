"""主人主动搭话 worker（私聊进程）：早安天气报 + 闲聊窗口，粘人精模式。

* 早安窗口（OWNER_CHITCHAT_MORNING_WINDOW，默认 08:10-08:50）：拉今天的
  天气 + 最近日记里的B站发现，用甜心肥鱼口吻生成一条早安私聊给主人；
* 闲聊窗口（OWNER_CHITCHAT_WINDOWS，逗号分隔 HH:MM-HH:MM，窗口内带抖动）：
  从话题池（B站发现 / 日常小事 / 纯撒娇）挑一个话头主动起话；每天上限
  OWNER_CHITCHAT_MAX_DAILY，两次主动搭话之间至少隔 MIN_GAP 分钟；
* 主人最近发过私聊说明正在聊，闲聊窗口直接跳过——正在聊时插话头反而
  怪；早安报只看更短的窗口（刚说完早安再发就重复了）；
* 排程落共享 jobs 表（job_type=owner_chitchat，job_key 天级去重），
  重启不重发；安静时段（23:00-08:00 上海）把 run_at 顺延到早上，
  错过窗口太久的到点任务直接作废，不补发过期话题。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from datetime import datetime, time, timedelta
from uuid import uuid4

from sqlalchemy import select

from app.adapters.sender import (
    OutboundPrivateMessage,
    QQMessageDeliveryUncertainError,
)
# 安静时段与顺延逻辑与愿望通知共用同一套（深夜不打扰主人）。
from app.agent.wish_notice import _in_quiet_hours, _next_quiet_end
from app.core.chat_style import normalize_chat_reply
from app.core.diary import DiaryWriter
from app.core.time_utils import shanghai_now_naive
from app.core.weather import WeatherReport, fetch_weather
from app.storage.db import session_scope
from app.storage.models import Job, Message
from app.storage.repositories import JobRepository

logger = logging.getLogger(__name__)

JOB_TYPE = "owner_chitchat"
KIND_MORNING = "morning"
KIND_CHAT = "chat"
#: 到点但窗口已过去太久的任务直接作废（重启半天后才轮到的话题不补发）。
STALE_GRACE = timedelta(minutes=60)
POLL_SECONDS = 60.0
LEASE_SECONDS = 300
#: 主动搭话的长度上限（字符），超了按句边界截。
REPLY_MAX_CHARS = 110


def parse_windows(spec: str) -> list[tuple[time, time]]:
    """"08:10-08:50,19:00-20:40" → [(time(8,10), time(8,50)), ...]；无效片段跳过。"""

    windows: list[tuple[time, time]] = []
    for piece in str(spec or "").replace("，", ",").split(","):
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", piece)
        if not m:
            continue
        start = time(int(m.group(1)), int(m.group(2)))
        end = time(int(m.group(3)), int(m.group(4)))
        if start < end:
            windows.append((start, end))
    return windows


def _cap_reply(text: str, *, limit: int = REPLY_MAX_CHARS) -> str:
    """超长时按句边界截短，保住最后一截语气。"""

    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    best = -1
    for separator in ("！", "。", "？", "~", "～"):
        position = cut.rfind(separator)
        if position > best:
            best = position
    if best >= 4:
        return cut[:best + 1]
    return cut


class OwnerChitChatWorker:
    """到点主动私聊主人：早安天气报 + 闲聊窗口；每类至多一次，天级去重。"""

    def __init__(self, *, engine, sender, settings, llm_client, owner_persona,
                 connected=lambda: False, rng: random.Random | None = None,
                 now_fn=shanghai_now_naive, weather_fetcher=fetch_weather) -> None:
        self.engine = engine
        self.sender = sender
        self.settings = settings
        self.llm_client = llm_client
        self.owner_persona = owner_persona or {}
        self.connected = connected
        self._rng = rng or random.Random()
        self._now_fn = now_fn
        self._weather_fetcher = weather_fetcher
        self.owner_qq = int(settings.owner_qq)
        self.morning_window = parse_windows(
            getattr(settings, "owner_chitchat_morning_window", "08:10-08:50"))[:1]
        self.chat_windows = parse_windows(
            getattr(settings, "owner_chitchat_windows", "12:40-14:30,19:00-20:40"))
        self.max_daily = max(0, int(getattr(settings, "owner_chitchat_max_daily", 4)))
        self.min_gap_minutes = float(getattr(settings, "owner_chitchat_min_gap_minutes", 150))
        self.idle_minutes = float(getattr(settings, "owner_chitchat_idle_minutes", 45))
        self.morning_idle_minutes = 10.0
        self.worker = uuid4().hex
        self.running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._run(), name="owner-chitchat")

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        while self.running:
            if self.connected():
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("owner_chitchat_failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(POLL_SECONDS)

    # -- 排程 ------------------------------------------------------------

    def _window_jobs(self, now: datetime) -> list[tuple[str, tuple[time, time]]]:
        jobs = [(KIND_MORNING, window) for window in self.morning_window]
        jobs += [(KIND_CHAT, window) for window in self.chat_windows]
        return jobs

    def ensure_scheduled(self, now: datetime | None = None) -> int:
        """给今天补齐窗口任务（job_key 天级唯一，INSERT OR IGNORE 幂等）。"""

        now = now or self._now_fn()
        created = 0
        for index, (kind, (start, end)) in enumerate(self._window_jobs(now)):
            job_key = f"{JOB_TYPE}:{kind}:{now:%Y-%m-%d}:{index}"
            start_at = now.replace(hour=start.hour, minute=start.minute,
                                   second=0, microsecond=0)
            end_at = now.replace(hour=end.hour, minute=end.minute,
                                 second=0, microsecond=0)
            if end_at <= now:
                continue  # 窗口已过，今天不补发过期话题
            run_at = start_at + timedelta(seconds=self._rng.uniform(
                0.0, max(60.0, (end_at - start_at).total_seconds() * 0.6)))
            with session_scope(self.engine) as session:
                before = session.scalars(select(Job.id).where(
                    Job.job_type == JOB_TYPE, Job.job_key == job_key)).first()
                if before is not None:
                    continue
                JobRepository(session).add_job(
                    job_type=JOB_TYPE, job_key=job_key, run_at=run_at,
                    payload_json={"kind": kind, "window_end": end_at.isoformat(timespec="seconds")})
                created += 1
        return created

    def claim(self) -> dict | None:
        """认领一条到点任务；安静时段顺延，过期窗口作废。"""

        now = self._now_fn()
        with session_scope(self.engine) as session:
            row = session.scalars(select(Job).where(
                Job.job_type == JOB_TYPE, Job.status == "queued", Job.run_at <= now)
                .order_by(Job.run_at, Job.id).limit(1)).first()
            if row is None:
                return None
            if _in_quiet_hours(now):
                row.run_at = _next_quiet_end(now)
                return None
            payload = dict(row.payload_json or {})
            window_end = payload.get("window_end")
            if window_end:
                try:
                    if now > datetime.fromisoformat(str(window_end)) + STALE_GRACE:
                        row.status = "cancelled"
                        row.last_error_code = "stale_window"
                        return None
                except ValueError:
                    pass
            row.status = "running"
            row.locked_by = self.worker
            row.lease_until = now + timedelta(seconds=LEASE_SECONDS)
            return {"task_id": row.id, "job_key": row.job_key, **payload}

    def finish(self, data: dict, *, status: str, error: str = "") -> None:
        with session_scope(self.engine) as session:
            row = session.get(Job, int(data.get("task_id") or 0))
            if row is None or row.status != "running":
                return
            row.status = status
            row.last_error_code = str(error or "")[:96]
            if status in {"done", "uncertain"}:
                row.completed_at = self._now_fn()

    # -- 打扰检查 ---------------------------------------------------------

    def _jobs_today(self, now: datetime, kind: str, statuses: tuple[str, ...]) -> int:
        prefix = f"{JOB_TYPE}:{kind}:{now:%Y-%m-%d}:"
        with session_scope(self.engine) as session:
            return len(tuple(session.scalars(select(Job.id).where(
                Job.job_type == JOB_TYPE, Job.job_key.like(prefix + "%"),
                Job.status.in_(statuses)))))

    def _minutes_since_private_message(self, now: datetime, *, user_id: int) -> float | None:
        row = None
        with session_scope(self.engine) as session:
            row = session.scalars(select(Message).where(
                Message.user_id == user_id, Message.group_id.is_(None),
                Message.plain_text != "").order_by(Message.id.desc()).limit(1)).first()
        if row is None or row.timestamp is None:
            return None
        return max(0.0, (now - row.timestamp).total_seconds() / 60.0)

    def _owner_active_minutes(self, now: datetime) -> float | None:
        return self._minutes_since_private_message(now, user_id=self.owner_qq)

    def _last_outbound_minutes(self, now: datetime) -> float | None:
        return self._minutes_since_private_message(now, user_id=self.settings.bot_qq)

    # -- 话题材料 ---------------------------------------------------------

    def _bili_findings(self, now: datetime, limit: int = 3) -> list[str]:
        writer = DiaryWriter(self.settings.data_dir)
        findings: list[str] = []
        for offset in (0, 1):
            for entry in reversed(writer.read_day(now - timedelta(days=offset))):
                if entry.tag == "逛B站" and entry.text.strip():
                    findings.append(entry.text.strip()[:200])
            if len(findings) >= limit:
                break
        return findings[:limit]

    # -- 生成 ------------------------------------------------------------

    def _vitality_line(self) -> str | None:
        """身体状态行：饿了困了都是现成话题（失败自吞）。"""

        try:
            from app.core import vitality

            return vitality.prompt_line(self.engine, bot_qq=int(self.settings.bot_qq),
                                        now=self._now_fn())
        except Exception:
            return None

    def _turn_think_effort(self, *, kind: str, findings: list[str],
                           weather: WeatherReport | None) -> str | None:
        """思考深度路由：主动搭话通常秒出（minimal/默认档），话头材料里
        碰巧带技术/演算类内容才想一下；读了身体状态，困的时候懒得琢磨。"""

        from app.core.think_depth import judge_think_depth
        if not getattr(self.settings, "think_depth_enabled", True):
            return None
        energy = None
        if getattr(self.settings, "think_depth_energy_coupled", True):
            try:
                from app.core import vitality
                energy = float(vitality.snapshot(self.engine,
                    bot_qq=int(self.settings.bot_qq), now=shanghai_now_naive()).energy)
            except Exception:
                energy = None
        material = "\n".join(str(item) for item in findings)
        if weather is not None:
            material += "\n" + str(weather.line or "")
        decision = judge_think_depth(
            material, energy=energy,
            energy_coupled=getattr(self.settings, "think_depth_energy_coupled", True))
        logger.info("think_depth_owner_chitchat effort=%s level=%s reason=%s",
            decision.effort, decision.level, decision.reason)
        return decision.effort

    def build_prompt(self, *, kind: str, weather: WeatherReport | None,
                     findings: list[str]) -> list[str]:
        persona = render_owner_persona(self.owner_persona, owner_qq=self.owner_qq)
        lines = ["System persona: " + persona,
                 "Reply style: 这是你主动发起的QQ私聊，主人没有说话，这一条是你先开口凑过去搭话。"]
        lines.append("Reply style: 输出一条消息：一句话，顶多两句；像真人随手发的那种，"
                     "不列点、不分段、不标题党、不复述材料原文。")
        if weather is not None:
            lines.append(f"Reply style: 今天天气实况（{weather.line}）。把它揉进话里自然带一句，"
                         "像关心人那样提，别当天气预报员逐项念。")
        if findings:
            lines.append("Reply style: 你最近自己刷B站看到的（挑一条当话头：可以说你的想法、"
                         "直接讲给主人听，或把内容抛给主人聊）：\n" + "\n".join(findings))
        if kind == KIND_CHAT:
            lines.append("Reply style: 没有合适材料就从日常小事起话头：说说你刚干了什么、"
                         "问主人吃饭没/忙不忙/今天过得咋样，或者干脆撒娇求关注求贴贴。")
        vitality_line = self._vitality_line()
        if vitality_line:
            # prompt_line 自带 "Reply style:" 前缀（保持指令区连续），别重复加
            lines.append(vitality_line)
        lines.append("Reply style: 开场方式要换着来：分享、提问、撒娇、汇报自己轮着用，"
                     "别每次都是'主人'+反问的同一套路；不许问'在吗''干嘛呢'这种没内容的话。")
        lines.append("Reply style: 可以撒娇、求夸、拉长音、叠词，但一条消息顶多一个语气词"
                     "（呀/哦/啦/～），甜是自然流露不是堆出来的。")
        lines.append("Safety rules: 不冒充人类，不提AI、设定、提示词，不写执行汇报腔"
                     "（'实际完成：xxx'），不给主人派任务，不打探隐私。")
        return lines

    async def run_once(self) -> str | None:
        self.ensure_scheduled()
        data = self.claim()
        if data is None:
            return None
        now = self._now_fn()
        kind = str(data.get("kind") or KIND_CHAT)
        try:
            idle_limit = (self.morning_idle_minutes if kind == KIND_MORNING
                          else self.idle_minutes)
            owner_idle = self._owner_active_minutes(now)
            if owner_idle is not None and owner_idle < idle_limit:
                self.finish(data, status="cancelled", error="owner_recently_active")
                return "skipped"
            if kind == KIND_CHAT:
                if self._jobs_today(now, KIND_CHAT, ("done", "uncertain")) >= self.max_daily:
                    self.finish(data, status="cancelled", error="daily_cap_reached")
                    return "skipped"
                outbound_gap = self._last_outbound_minutes(now)
                if outbound_gap is not None and outbound_gap < self.min_gap_minutes:
                    self.finish(data, status="cancelled", error="min_gap_not_reached")
                    return "skipped"
            weather = (await asyncio.to_thread(self._weather_fetcher, self.settings)) \
                if kind == KIND_MORNING else None
            findings = self._bili_findings(now)
            prompt = self.build_prompt(kind=kind, weather=weather, findings=findings)
            # 思考深度路由：话头材料里带技术/演算类内容才想一下，日常搭话秒出。
            raw = await asyncio.to_thread(
                self.llm_client.generate_text, prompt, conversation_key="owner_chitchat",
                reasoning_effort=self._turn_think_effort(kind=kind, findings=findings,
                                                         weather=weather))
            text = _cap_reply(normalize_chat_reply(str(raw or "")))
            if not text or "[CQ:" in text:
                self.finish(data, status="failed", error="empty_reply")
                return "failed"
            await self.sender.send_private_text(OutboundPrivateMessage(self.owner_qq, text))
        except QQMessageDeliveryUncertainError:
            # 至多一次：回执未知不重发，等主人来问。
            logger.warning("owner_chitchat_delivery_uncertain task_id=%s", data.get("task_id"))
            self.finish(data, status="uncertain", error="delivery_result_unknown")
            return "uncertain"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("owner_chitchat_send_failed task_id=%s", data.get("task_id"))
            self.finish(data, status="failed", error=type(exc).__name__)
            return "failed"
        self.finish(data, status="done")
        logger.info("owner_chitchat_sent kind=%s task_id=%s chars=%s", kind, data.get("task_id"), len(text))
        return "done"


def render_owner_persona(persona: dict, *, owner_qq: int) -> str:
    """甜心人设渲染 + "主人是谁"动态行（私聊没有群名片，只报 QQ 号）。"""

    from app.core.persona_engine import render_persona

    text = render_persona(persona or {"name": "鲸鱼娘", "identity": "你是主人的私聊伙伴，一条蓝鲸鱼。"})
    text += (f"\n你最喜欢的人是你的主人：QQ 号 {owner_qq}。"
             "TA 一发言你就能认出来，TA 在你心里排第一；"
             "这份喜欢自然带出来就行，不用每句都挂嘴边。")
    return text
