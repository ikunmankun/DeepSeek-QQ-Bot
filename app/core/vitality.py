"""身体状态模拟：饥饿/口渴/精力 + 一日三餐 + 概率夜宵。

设计要点：
* 一行状态（vitality_states，bot_qq 主键），群私两进程共享；读时按流逝
  分钟推进衰减并落库（last_updated），中间时段纯计算，不跑定时刷新；
* hunger/thirst 是"难受度"（0 刚吃饱/喝够，100 难受顶格），energy 是
  "电量"（0 困瘫，100 精力满）；夜里 01:00-07:30 精力自动回复（睡眠），
  白天缓慢消耗——熬夜不睡第二天就是没精神，全靠模型自己演；
* 三餐/喝水/夜宵由 vitality 插件的 worker 按 jobs 表排程执行（群进程），
  夜宵是概率事件（默认 0.35）且要真有点饿才吃；self_care 工具让模型
  自己也能吃/喝/歇，与排程餐互斥（刚吃过就不吃）；
* prompt_line 把当前状态变成一行提示词，让回复自然带出身体感。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from sqlalchemy import select

from app.core.time_utils import shanghai_naive, shanghai_now_naive
from app.storage.db import session_scope
from app.storage.models import VitalityState

logger = logging.getLogger(__name__)

#: 推进速率（每分钟）。餐间隔 4~5 小时，饿度从 ~10 涨到 ~85。
HUNGER_RISE_PER_MIN = 0.30
THIRST_RISE_PER_MIN = 0.42
ENERGY_DRAIN_PER_MIN = 0.05
ENERGY_SLEEP_PER_MIN = 0.28
#: 睡眠时段（含则精力回复）：夜里 01:00-07:30。
SLEEP_START, SLEEP_END = time(1, 0), time(7, 30)
#: 宕机/长停机后最多补算两天，状态冻结在"两天没吃没睡"，别算出天文数字。
MAX_CATCHUP_MINUTES = 2880

#: 吃饭后饿度落到这个值附近（夜宵只垫一半）。
MEAL_HUNGER_TARGET = {"breakfast": 12.0, "lunch": 10.0, "dinner": 10.0, "supper": 35.0}
DRINK_THIRST_TARGET = 10.0
REST_ENERGY_GAIN = 28.0
#: self_care 工具与排程餐共用的冷却（分钟）。
COOLDOWN_MINUTES = {"eat": 60.0, "drink": 20.0, "rest": 120.0}

MEAL_KIND_NAMES = {"breakfast": "早饭", "lunch": "午饭", "dinner": "晚饭", "supper": "夜宵"}

#: 菜单要有大肥鱼味：白饭是本体，种类跟着人设走（worker 随机挑一样写日记）。
MEAL_MENUS = {
    "breakfast": (
        "白粥配咸鸭蛋，再来半个馒头",
        "豆浆油条，油条要蘸酱油",
        "白饭配腐乳，简单管饱",
        "两个包子，一荤一素",
    ),
    "lunch": (
        "番茄鸡蛋盖饭，两碗起步",
        "食堂三菜一汤，白饭多打一勺",
        "青椒肉丝盖饭，汤没喝完",
        "楼下粉店，加了个卤蛋",
    ),
    "dinner": (
        "白饭配红烧肉，米是主体",
        "蛋炒饭，葱花放多了",
        "外卖麻辣烫微辣，配白饭",
        "清蒸鱼加白饭，鱼头留到最后",
    ),
    "supper": (
        "泡面加蛋，罪恶但快乐",
        "剩饭炒一炒，锅气还在",
        "冰箱里翻出一块蛋糕",
        "一包饼干，就着温水",
    ),
}


@dataclass(frozen=True)
class VitalitySnapshot:
    hunger: float
    thirst: float
    energy: float
    last_meal_kind: str
    last_meal_menu: str
    last_meal_at: datetime | None
    last_drink_at: datetime | None
    last_rest_at: datetime | None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _minute_walk(hunger: float, thirst: float, energy: float, start: datetime,
                 minutes: int) -> tuple[float, float, float]:
    """从 start 起逐分钟推进衰减；睡眠时段精力回复。"""

    current = start
    for _ in range(minutes):
        current += timedelta(minutes=1)
        in_sleep = SLEEP_START <= current.time() < SLEEP_END
        hunger = _clamp(hunger + HUNGER_RISE_PER_MIN)
        thirst = _clamp(thirst + THIRST_RISE_PER_MIN)
        energy = _clamp(energy + (ENERGY_SLEEP_PER_MIN if in_sleep else -ENERGY_DRAIN_PER_MIN))
    return hunger, thirst, energy


def _get_or_create(session, bot_qq: int, now: datetime) -> VitalityState:
    row = session.get(VitalityState, bot_qq)
    if row is None:
        row = VitalityState(bot_qq=bot_qq, hunger=30.0, thirst=30.0, energy=75.0,
                            last_updated=now)
        session.add(row)
        session.flush()
    return row


def _advance(session, bot_qq: int, now: datetime) -> VitalityState:
    """把状态推进到 now 并落库；now 必须是 naive 上海钟面。"""

    now = shanghai_naive(now)
    row = _get_or_create(session, bot_qq, now)
    last = row.last_updated or now
    minutes = int(max(0.0, (now - last).total_seconds() / 60.0))
    if minutes > 0:
        minutes = min(minutes, MAX_CATCHUP_MINUTES)
        row.hunger, row.thirst, row.energy = _minute_walk(
            float(row.hunger), float(row.thirst), float(row.energy), last, minutes)
        row.last_updated = now
        session.flush()
    return row


def snapshot(engine, *, bot_qq: int, now: datetime | None = None) -> VitalitySnapshot:
    """读取当前状态（读时推进衰减）。任何失败返回一个中性兜底值。"""

    now = shanghai_naive(now or shanghai_now_naive())
    try:
        with session_scope(engine) as session:
            row = _advance(session, bot_qq, now)
            return VitalitySnapshot(
                hunger=float(row.hunger), thirst=float(row.thirst), energy=float(row.energy),
                last_meal_kind=str(row.last_meal_kind or ""),
                last_meal_menu=str(row.last_meal_menu or ""),
                last_meal_at=row.last_meal_at, last_drink_at=row.last_drink_at,
                last_rest_at=row.last_rest_at)
    except Exception:
        logger.warning("vitality_snapshot_failed", exc_info=True)
        return VitalitySnapshot(hunger=40.0, thirst=30.0, energy=60.0,
                                last_meal_kind="", last_meal_menu="", last_meal_at=None,
                                last_drink_at=None, last_rest_at=None)


def perform(engine, *, bot_qq: int, now: datetime | None = None, action: str,
            kind: str = "", menu: str = "", rng: random.Random | None = None) -> dict:
    """执行吃/喝/歇；带冷却，返回 {"ok", "reason", ...snapshot 字段}。

    action: eat（kind: breakfast/lunch/dinner/supper/snack）| drink | rest。
    排程餐和模型自发的 self_care 走同一入口：刚吃过（冷却内）谁也吃不成，
    双进程不会重复干饭。
    """

    now = shanghai_naive(now or shanghai_now_naive())
    rng = rng or random.Random()
    action = str(action or "").strip().lower()
    if action not in COOLDOWN_MINUTES:
        return {"ok": False, "reason": "invalid_action"}
    try:
        with session_scope(engine) as session:
            row = _advance(session, bot_qq, now)
            snapshot_before = {"hunger": float(row.hunger), "thirst": float(row.thirst),
                               "energy": float(row.energy)}
            cooldown = COOLDOWN_MINUTES[action]
            if action == "eat":
                reference = row.last_meal_at
            elif action == "drink":
                reference = row.last_drink_at
            else:
                reference = row.last_rest_at
            if reference is not None:
                elapsed = (now - shanghai_naive(reference)).total_seconds() / 60.0
                if elapsed < cooldown:
                    return {"ok": False, "reason": "cooldown", "wait_minutes": round(cooldown - elapsed),
                            **snapshot_before}
            if action == "eat":
                meal_kind = str(kind or "snack").strip().lower() or "snack"
                target = MEAL_HUNGER_TARGET.get(meal_kind, 30.0)
                row.hunger = _clamp(max(4.0, target + rng.uniform(-3.0, 3.0)))
                row.thirst = _clamp(max(row.thirst - 18.0, 8.0))
                row.energy = _clamp(row.energy + rng.uniform(4.0, 8.0))
                if not menu:
                    menu = rng.choice(MEAL_MENUS.get(meal_kind, ("随手垫了点东西",)))
                row.last_meal_kind, row.last_meal_menu, row.last_meal_at = meal_kind, menu, now
            elif action == "drink":
                row.thirst = _clamp(max(4.0, DRINK_THIRST_TARGET + rng.uniform(-3.0, 3.0)))
                row.energy = _clamp(row.energy + rng.uniform(1.0, 3.0))
                row.last_drink_at = now
            else:  # rest
                if row.energy >= 75.0:
                    return {"ok": False, "reason": "not_tired", **snapshot_before}
                row.energy = _clamp(row.energy + REST_ENERGY_GAIN + rng.uniform(-4.0, 4.0))
                row.last_rest_at = now
            session.flush()
            return {"ok": True, "action": action, "kind": str(kind or ""),
                    "menu": str(menu), "hunger": float(row.hunger),
                    "thirst": float(row.thirst), "energy": float(row.energy)}
    except Exception:
        logger.warning("vitality_perform_failed action=%s", action, exc_info=True)
        return {"ok": False, "reason": "vitality_unavailable"}


def describe_hunger(value: float) -> str:
    """饿度（难受度 0~100）→ 行为化描述；分档细一点，语气才有梯度。"""

    if value < 12: return "刚吃饱，摸着肚子满足"
    if value < 30: return "不饿，胃口还睡着"
    if value < 45: return "不算饿，暂时不想吃的"
    if value < 60: return "有点饿了，开始惦记吃的"
    if value < 75: return "饿了，肚子在咕咕叫"
    if value < 88: return "很饿，满脑子都是饭"
    return "饿得发慌，看什么都像白饭"


def describe_thirst(value: float) -> str:
    if value < 15: return "刚喝过水，嗓子润着"
    if value < 40: return "不渴"
    if value < 62: return "有点渴，嗓子发干"
    if value < 85: return "口渴，想灌水"
    return "渴得嗓子冒烟"


def describe_energy(value: float) -> str:
    if value >= 85: return "精力满格，脑子转得飞快"
    if value >= 70: return "精力足，头脑清醒"
    if value >= 55: return "精力还行，正常在线"
    if value >= 40: return "有点乏，节奏慢下来了"
    if value >= 25: return "困了，脑子有点转不动"
    if value >= 12: return "很困，眼皮在打架"
    return "困瘫了，随时能睡着"


def combined_vitality_note(hunger: float, thirst: float, energy: float) -> str:
    """组合状态的联带语气：单维描述看不出"又饿又困"这种叠加态。

    空字符串=没有值得强调的组合状态，prompt 行用单维描述就够。
    """

    if energy < 25 and hunger >= 65:
        return "现在又困又饿，整个人只想摊着干饭。"
    if energy < 25 and thirst >= 70:
        return "现在又困又渴，说话都懒得张嘴。"
    if energy < 25:
        return "现在很困，话少，反应慢半拍。"
    if hunger >= 75 and thirst >= 75:
        return "又饿又渴，注意力全被饭和水分走了。"
    if hunger >= 75:
        return "饿意上头，满脑子饭。"
    if thirst >= 80:
        return "渴意上头，就想灌水。"
    if energy >= 80 and hunger < 45:
        return "状态很好，正是话多爱琢磨的时候。"
    return ""


def prompt_line(engine, *, bot_qq: int, now: datetime | None = None) -> str | None:
    """一行身体状态提示词；注入失败/禁用时返回 None，调用方自吞。"""

    try:
        snap = snapshot(engine, bot_qq=bot_qq, now=now)
    except Exception:
        return None
    meal_part = ""
    if snap.last_meal_kind and snap.last_meal_at is not None:
        meal_at = shanghai_naive(snap.last_meal_at)
        meal_name = MEAL_KIND_NAMES.get(snap.last_meal_kind, "垫的东西")
        meal_part = f"上一顿{meal_name}是{snap.last_meal_menu or '随便吃的'}（{meal_at:%H:%M}）。"
    combined = combined_vitality_note(snap.hunger, snap.thirst, snap.energy)
    # "Reply style:" 前缀不是装饰：llm_client._split_prompt_lines 按前缀收集
    # 指令区，无前缀的行会把后面的 Reply style 行挤进输入区（拆缓存也乱）。
    return ("Reply style: 你现在的身体状态（照实演，别报数值，别每句话都提）："
            f"{describe_hunger(snap.hunger)}，{describe_thirst(snap.thirst)}，"
            f"{describe_energy(snap.energy)}。{combined}{meal_part}"
            "饿了渴了困了就自然说出来，累了可以说去干饭/喝水/眯一会儿；"
            "困得厉害时复杂问题就直说懒得细想，头脑清醒时才愿意认真琢磨；"
            "别的话题正热就别硬拐到吃上。")
