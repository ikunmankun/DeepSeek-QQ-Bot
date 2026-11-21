"""好感度（规则版）：麦麦式关系系统的轻量实现。

人跟人的熟络是攒出来的：日常互动小幅加分，被夸、被求助时加得多，
被怼小扣分（斗嘴也是相处），长期不联系慢慢淡。分数只做定性描述
（陌生 / 认识 / 熟人 / 老朋友），不搞数值化的亲密表演。

衰减模型：距上次互动超过 AFFINITY_HALF_LIFE_DAYS 后，分数按半衰期
向 0 回落——人对人的印象也是不联系就慢慢淡，但攒出来的基础不会清零。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.core.time_utils import shanghai_now_naive, stored_as_utc
from app.storage.models import UserAffinityState

AFFINITY_HALF_LIFE_DAYS = 14.0
AFFINITY_MIN = -10.0
AFFINITY_MAX = 10.0

#: 各类事件的分值。斗嘴小扣——相处本来就有摩擦，别把损人当真敌意。
EVENT_DELTAS: dict[str, float] = {
    "smug": 1.0,       # 被夸
    "serious": 0.6,    # 被求助/对方需要
    "annoyed": -0.3,   # 被怼
    "chat": 0.2,       # 普通当面互动
}

#: 分数 → 关系定性（与提示词口吻一致）。四档→六档：顶档"死党"撑起
#: 长期陪伴的老群友，底档"疏远"接住持续扣分的关系，中间档才有梯度。
LEVELS: tuple[tuple[float, str, str], ...] = (
    (8.0, "死党", "最铁的那一档，什么梗都接得住，可以主动想起对方、没话找话"),
    (6.0, "老朋友", "很随意，互相损也没事，偶尔主动提起对方"),
    (3.0, "熟人", "熟络，开玩笑没顾忌"),
    (0.5, "认识", "打过交道，语气可以放松但不越界"),
    (-5.0, "有点相处不快", "收敛损的力度，正常礼貌回应"),
    (-999.0, "疏远", "相处不愉快积累多了，公事公办地回，不主动接对方的话"),
)


def clamp_score(score: float) -> float:
    return max(AFFINITY_MIN, min(AFFINITY_MAX, float(score)))


def apply_affinity_event(session, *, group_id: int, user_id: int,
                         delta: float, now: datetime) -> None:
    """累加一次互动的好感变化；分数限幅，互动计数单调增。"""

    row = session.get(UserAffinityState, (int(group_id), int(user_id)))
    if row is None:
        row = UserAffinityState(group_id=int(group_id), user_id=int(user_id),
                                score=0.0, interactions=0)
        session.add(row)
    row.score = clamp_score(float(row.score or 0.0) + float(delta))
    row.interactions = int(row.interactions or 0) + 1
    row.last_event_at = now
    row.updated_at = now


def load_affinity(engine, *, group_id: int, user_id: int) -> UserAffinityState | None:
    from app.storage.db import session_scope

    with session_scope(engine) as session:
        row = session.get(UserAffinityState, (int(group_id), int(user_id)))
        if row is None:
            return None
        session.expunge(row)
        return row


def decayed_score(row: UserAffinityState, now: datetime) -> float:
    """距上次互动按半衰期衰减后的当前好感。"""

    reference = row.last_event_at or row.updated_at
    if reference is None:
        return float(row.score or 0.0)
    try:
        elapsed_days = (stored_as_utc(now) - stored_as_utc(reference)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return float(row.score or 0.0)
    if elapsed_days <= 0:
        return float(row.score or 0.0)
    return float(row.score or 0.0) * 2.0 ** (-elapsed_days / AFFINITY_HALF_LIFE_DAYS)


def affinity_level(score: float) -> str:
    for threshold, label, _hint in LEVELS:
        if score >= threshold:
            return label
    return LEVELS[-1][1]


def affinity_prompt_line(engine, *, group_id: int, user_id: int, now: datetime) -> str | None:
    """直接互动对象的当前关系行；没有记录或对方是默认生人时不注入。"""

    if not user_id:
        return None
    try:
        row = load_affinity(engine, group_id=group_id, user_id=user_id)
    except Exception:
        return None
    if row is None or int(row.interactions or 0) < 2:
        return None
    score = decayed_score(row, now)
    level = affinity_level(score)
    hint = next(h for t, l, h in LEVELS if l == level)
    return (f"当前关系：你和这位是{level}（打过 {row.interactions} 次交道）。{hint}；"
            "关系只是语气基准，不要念出来。")


#: 插话判定用的关系偏置：跟谁说话决定插话的分寸——同样一句话，
#: 老朋友听是斗嘴，陌生人听是找事。
AFFINITY_JUDGE_HINTS: dict[str, str] = {
    "死党": "最铁的关系，插话最放得开，损得重也没事，可以没话找话。",
    "老朋友": "很熟，插话可以更放得开，损得重一点也没事。",
    "熟人": "熟络，正常接梗即可。",
    "认识": "打过几次交道，插话收敛些、轻些。",
    "有点相处不快": "之前的互动有摩擦，默认别主动接这个人的话。",
    "疏远": "相处的摩擦攒多了，这人不主动找你就别插话。",
}


def affinity_judge_hint(engine, *, group_id: int, user_id: int, now: datetime) -> str | None:
    """给插话判定的目标关系偏置行；没有互动记录或读取失败时返回 None。"""

    if not user_id:
        return None
    try:
        row = load_affinity(engine, group_id=group_id, user_id=user_id)
        if row is None or int(row.interactions or 0) < 2:
            return None
        level = affinity_level(decayed_score(row, now))
    except Exception:
        return None
    directive = AFFINITY_JUDGE_HINTS.get(level)
    if not directive:
        return None
    return f"正在说话的人和你关系：{level}。{directive}"
