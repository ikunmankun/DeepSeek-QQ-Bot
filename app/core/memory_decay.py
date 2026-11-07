"""人脑式记忆衰减：强度、半衰期、淡忘与归档判定。

真人不是过目不忘的：经常提起的记忆被强化，长期不提的慢慢淡忘，
深刻的事记得久。本模块用三个已有字段模拟这条规律——

* ``last_seen_at``：每次被提起都会刷新（upsert_canonical_memory 已实现）；
* ``mention_count``：被提起的次数，间隔效应——每多提一次，留存时间翻倍式延长；
* ``importance``：提取阶段的"深刻程度"，直接乘进半衰期。

半衰期（天）= 基础半衰期（按 memory_kind）× importance × (1 + log2(mention_count))

生命周期三段（模仿"想不起来 → 被一提醒就想起来 → 彻底忘了"）：

* 鲜活：未过半衰期，正常参与自动回忆；
* 淡忘：超过半衰期——排序沉底、不进参与提示，但显式问起（查询词命中）
  仍能检索到，被重新提起时现有链路自动刷新 last_seen_at / mention_count 复活；
* 归档：淡忘超过 ARCHIVE_HALF_LIVES 个半衰期，由后台任务停用
  （走既有 deactivate_memory_items 审计链，不物理删除）。

豁免类永不衰减：taboo（禁忌）、relationship、profile，以及 preference 里的
称呼规则（predicate 以 ``addressing rule`` 开头）——忘了这些不是真人感，是事故。

晋升为身份级记忆（证据制，对照"特质是长出来的"）：一条记忆若被反复提起
（mention_count ≥ PROMOTION_MIN_MENTIONS 次）且这些提起跨越至少
PROMOTION_MIN_SPAN_DAYS 天（last_seen_at - valid_from），说明它已经从
"一件发生过的事"沉淀成"对一个人的稳定认知"——自动晋升，与豁免类同样
永不淡忘。mention_count 只增不减，所以晋升是单向稳定的。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from app.core.time_utils import stored_as_utc

#: 按记忆类型的基础半衰期（天）。current（正在做什么）本来就带 expires_at，
#: 这里给一个兜底值以防提取时漏写有效期。
BASE_HALF_LIFE_DAYS: dict[str, float] = {
    "current": 3.0,
    "plan": 14.0,
    "event": 60.0,
    "fact": 90.0,
    "running_joke": 120.0,
    "preference": 180.0,
    "decision": 365.0,
}
DEFAULT_HALF_LIFE_DAYS = 90.0

#: 永不淡忘的记忆类型。
NEVER_FORGET_KINDS = frozenset({"taboo", "relationship", "profile"})

#: 称呼规则（"叫我XX"）存在 preference 里，靠 predicate 前缀识别。
ADDRESSING_RULE_PREFIX = "addressing rule"

#: 晋升为身份级记忆的证据线：至少被提起 3 次，且首次与最近一次提起
#: 相隔至少 2 天——"反复、跨天出现"才是稳定认知，一次刷屏不算。
PROMOTION_MIN_MENTIONS = 3
PROMOTION_MIN_SPAN_DAYS = 2.0

#: 只有偏"长期认知"的类型才参与晋升：current（正在做什么）/ plan（打算）
#: 天生短暂，提起再多也是过眼云烟，不该变成永久印象。
PROMOTABLE_KINDS = frozenset({"fact", "event", "preference", "running_joke", "decision"})

#: 超过一个半衰期（强度 < 50%）即视为淡忘。
#: 归档线：淡忘后再过 ARCHIVE_HALF_LIVES 个半衰期（强度 < ~1.6%）。
ARCHIVE_HALF_LIVES = 6.0

#: 每轮归档最多处理的条数，防止单次 tick 做太多写。
ARCHIVE_BATCH_LIMIT = 500


def _memory_attr(memory: Any, name: str):
    return getattr(memory, name, None)


def is_promoted(memory: Any) -> bool:
    """证据制晋升：反复、跨天被提起的长期认知，升级为身份级记忆。

    没有提及元数据的旧记忆（mention_count/时间戳缺失）不参与晋升——
    宁可让它按类型基础半衰期正常淡忘，也不凭空豁免。
    """

    kind = str(_memory_attr(memory, "memory_kind") or "").strip()
    if kind not in PROMOTABLE_KINDS:
        return False
    try:
        mention_count = int(_memory_attr(memory, "mention_count") or 0)
    except (TypeError, ValueError):
        return False
    if mention_count < PROMOTION_MIN_MENTIONS:
        return False
    first = _memory_attr(memory, "valid_from")
    last = _memory_attr(memory, "last_seen_at")
    if first is None or last is None:
        return False
    try:
        span_days = (stored_as_utc(last) - stored_as_utc(first)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return False
    return span_days >= PROMOTION_MIN_SPAN_DAYS


def is_immutable(memory: Any) -> bool:
    """豁免类：禁忌、关系、画像、称呼规则，以及已晋升的身份级记忆，永不衰减。"""

    kind = str(_memory_attr(memory, "memory_kind") or "").strip()
    if kind in NEVER_FORGET_KINDS:
        return True
    predicate = str(_memory_attr(memory, "predicate") or "").strip()
    if predicate.lower().startswith(ADDRESSING_RULE_PREFIX):
        return True
    return is_promoted(memory)


def half_life_days(memory: Any) -> float:
    """单条记忆的半衰期（天）。

    缺时间戳/计数字段时按最小值处理（importance=1、mention_count=1），
    因此没有衰减元数据的旧记忆也遵循类型基础半衰期，不会被误判为新鲜。
    """

    kind = str(_memory_attr(memory, "memory_kind") or "").strip()
    base = BASE_HALF_LIFE_DAYS.get(kind, DEFAULT_HALF_LIFE_DAYS)
    try:
        importance = max(1, min(5, int(_memory_attr(memory, "importance") or 1)))
    except (TypeError, ValueError):
        importance = 1
    try:
        mention_count = max(1, int(_memory_attr(memory, "mention_count") or 1))
    except (TypeError, ValueError):
        mention_count = 1
    # 间隔效应：提 1 次不强化，2 次 ×2，4 次 ×4，8 次 ×5……
    reinforcement = 1.0 + math.log2(max(1, mention_count))
    return base * importance * reinforcement


def memory_age_days(memory: Any, now: datetime) -> float | None:
    """距最近一次被提起的天数；完全没有时间戳时返回 None（视为新鲜）。"""

    reference = _memory_attr(memory, "last_seen_at") or _memory_attr(memory, "valid_from")
    if reference is None:
        return None
    try:
        delta = stored_as_utc(now) - stored_as_utc(reference)
    except (TypeError, ValueError):
        return None
    return delta.total_seconds() / 86400.0


def strength(memory: Any, now: datetime) -> float:
    """记忆强度，1.0 = 刚提起，0.5 = 一个半衰期，趋近 0 = 遗忘。

    无时间戳的记忆无法判定年龄，按强度 1.0 处理（宁可记得，不可误删）。
    """

    age_days = memory_age_days(memory, now)
    if age_days is None:
        return 1.0
    if age_days <= 0:
        return 1.0
    return 2.0 ** (-age_days / half_life_days(memory))


def is_faded(memory: Any, now: datetime) -> bool:
    """淡忘：强度跌破 50%。豁免类与无时间戳的记忆永远不淡忘。"""

    if is_immutable(memory):
        return False
    age_days = memory_age_days(memory, now)
    return age_days is not None and age_days > half_life_days(memory)


def is_archivable(memory: Any, now: datetime) -> bool:
    """归档：淡忘后再拖过 ARCHIVE_HALF_LIVES 个半衰期仍无人提起。

    只对会淡忘的类型生效；豁免类（禁忌/关系/画像/称呼规则）永远返回 False。
    """

    if is_immutable(memory):
        return False
    age_days = memory_age_days(memory, now)
    if age_days is None:
        return False
    return age_days > half_life_days(memory) * ARCHIVE_HALF_LIVES
