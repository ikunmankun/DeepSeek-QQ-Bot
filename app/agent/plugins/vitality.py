"""vitality 插件：身体状态模拟（饥饿/口渴/精力 + 一日三餐 + 概率夜宵）。

群进程是身体模拟的宿主：
* ``prompt_lines``    —— 每个群聊回合注入一行当前身体状态（直接回复和
                          观察轮都注入，饿了困了在回复里自然带出来）；
* ``build_tools``     —— self_care 工具：模型自己饿了就去干饭、渴了喝水、
                          困了眯一会儿，与排程餐共用同一入口和冷却；
* ``create_workers``  —— 三餐/喝水/夜宵的排程 worker：jobs 表天级去重，
                          吃饭写日记（tag=干饭），夜宵是概率事件且要真饿。
私聊进程只读状态（prompt 行 + chitchat 话题），不重复跑 worker。
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, time, timedelta
from typing import Any, Sequence

from sqlalchemy import select

from app.agent.plugins.base import FeatureModule
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core import vitality
from app.core.diary import DiaryWriter
from app.core.time_utils import shanghai_now_naive
from app.storage.db import session_scope
from app.storage.models import Job
from app.storage.repositories import JobRepository

logger = logging.getLogger(__name__)

JOB_TYPE = "vitality_meal"
#: 到点但窗口已过去太久的餐不补吃（错过就错过，作息才真实）。
STALE_GRACE = timedelta(minutes=90)
LEASE_SECONDS = 300
POLL_SECONDS = 120.0


class VitalityMealWorker:
    """三餐/喝水/夜宵排程：jobs 表去重，吃饭写日记，夜宵看概率和饿度。"""

    WINDOWS = {
        "breakfast": (time(7, 40), time(8, 30)),
        "water1": (time(10, 0), time(11, 0)),
        "lunch": (time(11, 40), time(12, 50)),
        "water2": (time(14, 30), time(15, 30)),
        "dinner": (time(18, 10), time(19, 20)),
        "supper": (time(22, 0), time(23, 40)),
    }
    #: 一日三餐是作息，到点就吃（刚吃过会被冷却挡掉）；夜宵是欲望事件。
    MEAL_KINDS = ("breakfast", "lunch", "dinner")

    def __init__(self, *, engine, settings, rng: random.Random | None = None,
                 now_fn=shanghai_now_naive) -> None:
        self.engine = engine
        self.settings = settings
        self.bot_qq = int(settings.bot_qq)
        self.data_dir = settings.data_dir
        self.supper_probability = float(getattr(settings, "vitality_supper_probability", 0.35))
        self.supper_min_hunger = 40.0
        self._rng = rng or random.Random()
        self._now_fn = now_fn
        self.running = False
        self._task: Any = None

    async def run(self) -> None:
        self.running = True
        try:
            while self.running:
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("vitality_worker_failed error_type=%s", type(exc).__name__)
                await asyncio.sleep(POLL_SECONDS)
        finally:
            self.running = False

    async def run_once(self) -> str | None:
        self.ensure_scheduled()
        data = self.claim()
        if data is None:
            return None
        now = self._now_fn()
        kind = str(data.get("kind") or "")
        try:
            if kind.startswith("water"):
                result = vitality.perform(self.engine, bot_qq=self.bot_qq, now=now,
                                          action="drink", rng=self._rng)
                if result.get("ok"):
                    self.finish(data, status="done")
                else:
                    self.finish(data, status="cancelled", error=str(result.get("reason") or "skipped"))
                return "done" if result.get("ok") else "skipped"
            if kind == "supper":
                # 夜宵是欲望：概率 + 真饿才吃，不是每天都吃。
                if self._rng.random() >= self.supper_probability:
                    self.finish(data, status="cancelled", error="no_craving_tonight")
                    return "skipped"
                snap = vitality.snapshot(self.engine, bot_qq=self.bot_qq, now=now)
                if snap.hunger < self.supper_min_hunger:
                    self.finish(data, status="cancelled", error="not_hungry_enough")
                    return "skipped"
                kind_for_meal = "supper"
            elif kind in self.MEAL_KINDS:
                kind_for_meal = kind
            else:
                self.finish(data, status="cancelled", error="unknown_kind")
                return "skipped"
            menu = self._rng.choice(vitality.MEAL_MENUS.get(kind_for_meal, ("随手垫了点东西",)))
            result = vitality.perform(self.engine, bot_qq=self.bot_qq, now=now,
                                      action="eat", kind=kind_for_meal, menu=menu,
                                      rng=self._rng)
            if not result.get("ok"):
                # 刚吃过（工具里吃过）就不重复吃；作息让位于真实的肚子。
                self.finish(data, status="cancelled", error=str(result.get("reason") or "skipped"))
                return "skipped"
            DiaryWriter(self.data_dir).write(
                text=f"吃了{menu}。", now=now,
                mood=None, tag="夜宵" if kind_for_meal == "supper" else "干饭")
            self.finish(data, status="done")
            logger.info("vitality_meal_done kind=%s menu_chars=%s", kind_for_meal, len(menu))
            return "done"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("vitality_meal_failed kind=%s error_type=%s", kind, type(exc).__name__)
            self.finish(data, status="failed", error=type(exc).__name__)
            return "failed"

    # -- 排程 ------------------------------------------------------------

    def ensure_scheduled(self, now: datetime | None = None) -> int:
        now = now or self._now_fn()
        created = 0
        for index, (kind, (start, end)) in enumerate(sorted(self.WINDOWS.items(),
                                                            key=lambda item: item[1][0])):
            job_key = f"{JOB_TYPE}:{kind}:{now:%Y-%m-%d}"
            start_at = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
            end_at = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
            if end_at <= now:
                continue
            run_at = start_at + timedelta(seconds=self._rng.uniform(
                0.0, max(60.0, (end_at - start_at).total_seconds() * 0.5)))
            with session_scope(self.engine) as session:
                exists = session.scalars(select(Job.id).where(
                    Job.job_type == JOB_TYPE, Job.job_key == job_key)).first()
                if exists is not None:
                    continue
                JobRepository(session).add_job(
                    job_type=JOB_TYPE, job_key=job_key, run_at=run_at,
                    payload_json={"kind": kind, "window_end": end_at.isoformat(timespec="seconds")})
                created += 1
        return created

    def claim(self) -> dict | None:
        now = self._now_fn()
        with session_scope(self.engine) as session:
            row = session.scalars(select(Job).where(
                Job.job_type == JOB_TYPE, Job.status == "queued", Job.run_at <= now)
                .order_by(Job.run_at, Job.id).limit(1)).first()
            if row is None:
                return None
            payload = dict(row.payload_json or {})
            window_end = payload.get("window_end")
            if window_end:
                try:
                    if now > datetime.fromisoformat(str(window_end)) + STALE_GRACE:
                        row.status, row.last_error_code = "cancelled", "stale_window"
                        return None
                except ValueError:
                    pass
            row.status = "running"
            row.locked_by = "vitality"
            row.lease_until = now + timedelta(seconds=LEASE_SECONDS)
            return {"task_id": row.id, "job_key": row.job_key, **payload}

    def finish(self, data: dict, *, status: str, error: str = "") -> None:
        with session_scope(self.engine) as session:
            row = session.get(Job, int(data.get("task_id") or 0))
            if row is None or row.status != "running":
                return
            row.status = status
            row.last_error_code = str(error or "")[:96]


class VitalityFeature(FeatureModule):
    name = "vitality"

    def __init__(self) -> None:
        self.engine = None
        self.settings = None
        self.bot_qq = 0
        self.enabled = True
        self._rng = random.Random()

    def attach(self, *, engine, settings) -> None:
        self.engine = engine
        self.settings = settings
        self.bot_qq = int(settings.bot_qq)
        self.enabled = bool(getattr(settings, "vitality_enabled", True))

    # -- 提示行 -----------------------------------------------------------

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if not self.enabled or kind != "group" or self.engine is None:
            return ()
        line = vitality.prompt_line(self.engine, bot_qq=self.bot_qq, now=now)
        return (line,) if line else ()

    # -- 工具 -------------------------------------------------------------

    def build_tools(self, turn) -> Sequence[AgentTool]:
        if not self.enabled or self.engine is None:
            return ()

        async def self_care(args):
            turn.guard.require_valid()
            action = str(args.get("action") or "").strip().lower()
            result = await turn.blocking.run(
                vitality.perform, self.engine, bot_qq=self.bot_qq,
                now=shanghai_now_naive(), action=action, rng=self._rng)
            if not result.get("ok"):
                guidance = {"cooldown": "刚吃过/喝过/歇过，别重复。",
                            "not_tired": "现在精神着呢，不用歇。",
                            "invalid_action": "action 只能是 eat/drink/rest。"}.get(
                    str(result.get("reason")), "身体状态这会儿没记上，如实说就行。")
                return ToolOutcome({**result, "guidance": guidance}, "blocked")
            return ToolOutcome({**result,
                "guidance": "身体状态已更新。聊天里自然带出来就行：饿了说去干饭、"
                    "吃完说饱了，困了说眯一会儿；别报数值，别提工具。"}, "ok")

        return [AgentTool(
            "self_care",
            "照顾自己的身体：action=eat 干饭（饿了就去吃，别等排餐）、action=drink 喝水、"
            "action=rest 眯一会儿。跟排程的三餐共用状态：刚吃过会被挡。聊天里想说自己"
            "要去吃饭/喝水/休息时，先调这个把状态记上，再正常说话。",
            {"type": "object",
             "properties": {"action": {"type": "string", "enum": ["eat", "drink", "rest"]}},
             "required": ["action"], "additionalProperties": False},
            self_care, "read")]

    # -- worker -----------------------------------------------------------

    def create_workers(self) -> Sequence[Any]:
        if not self.enabled or self.engine is None:
            return ()
        return [VitalityMealWorker(engine=self.engine, settings=self.settings,
                                   rng=self._rng)]
