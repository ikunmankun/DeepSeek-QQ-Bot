"""身体状态模拟核心：衰减推进、吃/喝/歇动作、冷却、提示词行。"""
import random
from datetime import datetime, timedelta

import pytest

from app.core import vitality
from app.core.vitality import VitalitySnapshot, describe_energy, prompt_line, snapshot

BOT = 2900000001
BASE = datetime(2026, 11, 19, 12, 0)  # 午间，非睡眠时段


def _snap(engine, now=BASE, **kwargs):
    return snapshot(engine, bot_qq=BOT, now=now, **kwargs)


def test_decay_rises_hunger_and_thirst_faster_than_energy_drains(sqlite_engine):
    start = _snap(sqlite_engine, now=BASE)
    later = _snap(sqlite_engine, now=BASE + timedelta(hours=3))
    assert later.hunger > start.hunger
    assert later.thirst > start.thirst
    assert later.energy < start.energy
    # 口渴涨得比饿快（人不喝水比不吃饭垮得快）
    assert (later.thirst - start.thirst) > (later.hunger - start.hunger)


def test_sleep_window_restores_energy(sqlite_engine):
    # 深夜 00:30 起睡 7 小时，跨过整个睡眠窗口（01:00-07:30）
    night = _snap(sqlite_engine, now=datetime(2026, 11, 19, 0, 30))
    morning = _snap(sqlite_engine, now=datetime(2026, 11, 19, 7, 40))
    assert morning.energy > night.energy


def test_long_downtime_is_clamped(sqlite_engine):
    # 十天不启动，补算封顶两天：数值顶格但不溢出
    _snap(sqlite_engine, now=BASE)
    stale = _snap(sqlite_engine, now=BASE + timedelta(days=10))
    assert stale.hunger == pytest.approx(100.0)
    assert stale.thirst == pytest.approx(100.0)
    assert 0.0 <= stale.energy <= 100.0


def test_eat_resets_hunger_and_records_meal(sqlite_engine):
    _snap(sqlite_engine, now=BASE)  # 初始化并推进
    result = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=30),
                              action="eat", kind="lunch", rng=random.Random(1))
    assert result["ok"] is True
    assert result["hunger"] < 20.0
    after = _snap(sqlite_engine, now=BASE + timedelta(minutes=31))
    assert after.last_meal_kind == "lunch"
    assert after.last_meal_menu
    assert after.hunger == pytest.approx(result["hunger"], abs=0.5)


def test_eat_cooldown_blocks_second_meal(sqlite_engine):
    first = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE, action="eat",
                             kind="lunch", rng=random.Random(2))
    assert first["ok"] is True
    second = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=20),
                              action="eat", kind="snack", rng=random.Random(3))
    assert second["ok"] is False
    assert second["reason"] == "cooldown"
    # 冷却过后可以再吃
    third = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=70),
                             action="eat", kind="snack", rng=random.Random(4))
    assert third["ok"] is True


def test_supper_only_half_relieves(sqlite_engine):
    _snap(sqlite_engine, now=BASE)
    vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=5),
                     action="eat", kind="lunch", rng=random.Random(5))
    # 午饭 3 小时后饿度涨上来，吃夜宵只垫到中等水平
    late = BASE + timedelta(hours=4)
    result = vitality.perform(sqlite_engine, bot_qq=BOT, now=late, action="eat",
                              kind="supper", rng=random.Random(6))
    assert result["ok"] is True
    assert 20.0 <= result["hunger"] <= 50.0


def test_drink_resets_thirst_with_own_cooldown(sqlite_engine):
    _snap(sqlite_engine, now=BASE)
    result = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=10),
                              action="drink", rng=random.Random(7))
    assert result["ok"] is True and result["thirst"] < 20.0
    again = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=15),
                             action="drink", rng=random.Random(8))
    assert again["ok"] is False and again["reason"] == "cooldown"


def test_rest_blocked_when_energetic_works_when_tired(sqlite_engine):
    # 从凌晨 3 点起推到中午：睡眠回满 + 白天小耗，精力仍在高位
    _snap(sqlite_engine, now=datetime(2026, 11, 19, 3, 0))
    blocked = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=2),
                               action="rest", rng=random.Random(9))
    assert blocked["ok"] is False and blocked["reason"] == "not_tired"
    # 再耗到晚上九点，精力一般了才歇得动
    tired_time = BASE + timedelta(hours=9)
    result = vitality.perform(sqlite_engine, bot_qq=BOT, now=tired_time,
                              action="rest", rng=random.Random(10))
    assert result["ok"] is True
    assert result["energy"] > 30.0


def test_invalid_action_rejected(sqlite_engine):
    result = vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE, action="swim")
    assert result["ok"] is False and result["reason"] == "invalid_action"


def test_prompt_line_describes_state_without_numbers(sqlite_engine):
    _snap(sqlite_engine, now=BASE)
    vitality.perform(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=3),
                     action="eat", kind="lunch", rng=random.Random(11))
    line = prompt_line(sqlite_engine, bot_qq=BOT, now=BASE + timedelta(minutes=4))
    assert line is not None
    assert "身体状态" in line and "别报数值" in line
    assert "午饭" in line  # 上一餐信息带出来
    import re
    assert not re.search(r"\d+\.\d", line)  # 状态数值不进提示词


def test_describe_bands():
    assert describe_energy(90) == "精力满格，脑子转得飞快"
    assert describe_energy(60) == "精力还行，正常在线"
    assert describe_energy(40) == "有点乏，节奏慢下来了"
    assert describe_energy(15) == "很困，眼皮在打架"
    assert vitality.describe_hunger(90) == "饿得发慌，看什么都像白饭"
    assert vitality.describe_hunger(50) == "有点饿了，开始惦记吃的"
    assert vitality.describe_thirst(10) == "刚喝过水，嗓子润着"
    assert vitality.describe_thirst(70) == "口渴，想灌水"


def test_combined_vitality_note_ladder():
    # 组合状态：又饿又困 > 单维上头 > 状态好 > 无话可说。
    assert "又困又饿" in vitality.combined_vitality_note(70, 30, 15)
    assert vitality.combined_vitality_note(80, 20, 70) == "饿意上头，满脑子饭。"
    assert "状态很好" in vitality.combined_vitality_note(20, 20, 85)
    assert vitality.combined_vitality_note(20, 20, 60) == ""


def test_snapshot_never_raises(sqlite_engine):
    # 即使引擎坏了也返回兜底值（提示注入不能炸聊天）
    class Broken:
        def connect(self):
            raise RuntimeError("boom")

    result = snapshot(Broken(), bot_qq=BOT, now=BASE)
    assert isinstance(result, VitalitySnapshot)
