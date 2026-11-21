"""vitality 插件：三餐/喝水/夜宵排程 worker、提示行注入、self_care 工具。"""
import random
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.agent.plugins.vitality import JOB_TYPE, VitalityFeature, VitalityMealWorker
from app.core import vitality
from app.core.diary import DiaryWriter
from sqlalchemy import select

from app.storage.db import session_scope
from app.storage.models import Job

BOT = 2900000001
BASE = datetime(2026, 11, 19, 12, 0)


def _settings(tmp_path, **overrides):
    base = dict(bot_qq=BOT, data_dir=tmp_path, vitality_enabled=True,
                vitality_supper_probability=0.35)
    base.update(overrides)
    return SimpleNamespace(**base)


class _Blocking:
    async def run(self, function, *args, **kwargs):
        return function(*args, **kwargs)


class _Guard:
    def require_valid(self):
        return True


def _worker(sqlite_engine, tmp_path, *, rng=None, now=BASE, schedule=True, **overrides):
    """schedule=False 清空窗口：行为测试里 ensure_scheduled 不造任务，
    只有测试自己种的种子任务会到期（避免自动排程抢跑）。"""
    holder = {"value": now}
    worker = VitalityMealWorker(
        engine=sqlite_engine, settings=_settings(tmp_path, **overrides),
        rng=rng if rng is not None else random.Random(7),
        now_fn=lambda: holder["value"])
    if not schedule:
        worker.WINDOWS = {}
    return worker, holder


def _seed_job(engine, *, kind, run_at, window_end, job_key=None):
    with session_scope(engine) as session:
        from app.storage.repositories import JobRepository

        row = JobRepository(session).add_job(
            job_type=JOB_TYPE,
            job_key=job_key or f"{JOB_TYPE}:{kind}:seed:{random.randint(0, 10**9)}",
            run_at=run_at,
            payload_json={"kind": kind,
                          "window_end": window_end.isoformat(timespec="seconds")})
        session.flush()
        return row.id


def _job_status(engine, job_id):
    with session_scope(engine) as session:
        row = session.get(Job, job_id)
        return row.status, row.last_error_code


def test_ensure_scheduled_creates_six_daily_jobs_once(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path,
                        now=datetime(2026, 11, 19, 7, 0))
    assert worker.ensure_scheduled() == 6
    assert worker.ensure_scheduled() == 0
    with session_scope(sqlite_engine) as session:
        kinds = sorted(row.payload_json["kind"] for row in
                       session.scalars(select(Job).where(Job.job_type == JOB_TYPE)))
    assert kinds == ["breakfast", "dinner", "lunch", "supper", "water1", "water2"]


def test_ensure_scheduled_skips_passed_windows(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path, now=BASE)  # 12:00：早班窗口已过
    created = worker.ensure_scheduled()
    with session_scope(sqlite_engine) as session:
        kinds = {row.payload_json["kind"] for row in
                 session.scalars(select(Job).where(Job.job_type == JOB_TYPE))}
    assert "breakfast" not in kinds
    assert {"lunch", "dinner", "supper"} <= kinds


async def test_meal_job_eats_and_writes_diary(sqlite_engine, tmp_path):
    worker, holder = _worker(sqlite_engine, tmp_path, schedule=False)
    job_id = _seed_job(sqlite_engine, kind="lunch", run_at=BASE - timedelta(minutes=2),
                       window_end=BASE + timedelta(minutes=30))
    assert await worker.run_once() == "done"
    assert _job_status(sqlite_engine, job_id) == ("done", "")
    snap = vitality.snapshot(sqlite_engine, bot_qq=BOT, now=BASE)
    assert snap.last_meal_kind == "lunch" and snap.last_meal_menu
    entries = DiaryWriter(tmp_path).read_day(BASE)
    assert any(entry.tag == "干饭" and entry.text.startswith("吃了") for entry in entries)


async def test_supper_probability_and_hunger_gate(sqlite_engine, tmp_path):
    # 概率为 0：今晚没馋
    worker, _ = _worker(sqlite_engine, tmp_path, rng=random.Random(1), schedule=False,
                        vitality_supper_probability=0.0)
    job_id = _seed_job(sqlite_engine, kind="supper", run_at=BASE - timedelta(minutes=1),
                       window_end=BASE + timedelta(minutes=30))
    assert await worker.run_once() == "skipped"
    assert _job_status(sqlite_engine, job_id) == ("cancelled", "no_craving_tonight")

    # 概率为 1 但刚吃过午饭（饿度低）：不吃
    worker2, _ = _worker(sqlite_engine, tmp_path, rng=random.Random(1), schedule=False,
                         vitality_supper_probability=1.0)
    vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE - timedelta(minutes=40),
                     action="eat", kind="lunch", rng=random.Random(2))
    job_id2 = _seed_job(sqlite_engine, kind="supper", run_at=BASE - timedelta(seconds=1),
                        window_end=BASE + timedelta(minutes=30))
    assert await worker2.run_once() == "skipped"
    assert _job_status(sqlite_engine, job_id2) == ("cancelled", "not_hungry_enough")


async def test_scheduled_meal_skipped_when_tool_just_ate(sqlite_engine, tmp_path):
    # 模型刚通过 self_care 吃过 → 排程午饥被冷却挡掉，不重复干饭
    vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE - timedelta(minutes=15),
                     action="eat", kind="snack", rng=random.Random(3))
    worker, _ = _worker(sqlite_engine, tmp_path, schedule=False)
    job_id = _seed_job(sqlite_engine, kind="lunch", run_at=BASE - timedelta(seconds=1),
                       window_end=BASE + timedelta(minutes=30))
    assert await worker.run_once() == "skipped"
    assert _job_status(sqlite_engine, job_id)[1] == "cooldown"


async def test_water_job_drinks(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path, schedule=False)
    _seed_job(sqlite_engine, kind="water1", run_at=BASE - timedelta(seconds=1),
              window_end=BASE + timedelta(minutes=30))
    assert await worker.run_once() == "done"
    snap = vitality.snapshot(sqlite_engine, bot_qq=BOT, now=BASE)
    assert snap.thirst < 30.0 and snap.last_drink_at is not None


def test_claim_cancels_stale_meal(sqlite_engine, tmp_path):
    worker, _ = _worker(sqlite_engine, tmp_path, now=BASE)
    job_id = _seed_job(sqlite_engine, kind="dinner", run_at=BASE - timedelta(minutes=5),
                       window_end=BASE - timedelta(hours=12))
    assert worker.claim() is None
    assert _job_status(sqlite_engine, job_id) == ("cancelled", "stale_window")


async def test_feature_prompt_line_and_tool(sqlite_engine, tmp_path):
    feature = VitalityFeature()
    feature.attach(engine=sqlite_engine, settings=_settings(tmp_path))
    line = feature.prompt_lines(kind="group", group_id=123, now=BASE, observing=False)
    assert len(line) == 1
    assert "身体状态" in line[0]
    assert not re.search(r"\d+\.\d", line[0])
    assert feature.prompt_lines(kind="private", group_id=0, now=BASE,
                                observing=False) == ()

    tools = feature.build_tools(SimpleNamespace(
        router=None, job=None, guard=_Guard(), store=None,
        blocking=_Blocking(), interactive=True, now=BASE))
    assert [tool.name for tool in tools] == ["self_care"]
    outcome = await tools[0].handler({"action": "eat"})
    assert outcome.status == "ok"
    blocked = await tools[0].handler({"action": "eat"})
    assert blocked.status == "blocked"  # 冷却
    bad = await tools[0].handler({"action": "swim"})
    assert bad.status == "blocked"


def test_feature_disabled_has_no_tools_or_workers(sqlite_engine, tmp_path):
    feature = VitalityFeature()
    feature.attach(engine=sqlite_engine, settings=_settings(tmp_path, vitality_enabled=False))
    assert feature.prompt_lines(kind="group", group_id=1, now=BASE, observing=False) == ()
    assert feature.build_tools(None) == ()
    assert feature.create_workers() == ()
