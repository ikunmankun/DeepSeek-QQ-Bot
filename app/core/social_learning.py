"""沉默学习：把"什么时候插话没人接"沉淀成数据。

学一个人怎么说话，一半是学他什么时候不说话。此前机器人只有"被骂了才
安静一会儿"的硬负反馈（group_participation_feedback），骂完过期即忘——
它从不长记性。这里补上另一半：

* 记录——每次主动插话（judge 批准的 proactive 回复）落一行 pending；
* 回填——观察窗过后由周期 worker 看结果：有人接 responded、被怼
  negative、没人理 ignored、压根没发出去 not_sent；
* 反哺——按群 × 时段聚合最近两周的成绩喂给插话判定当参考；某时段
  反复没人接（样本足够、接话率极低）就直接硬闸掉该时段的主动插话。

"有人接话"的判定故意偏严：要么直接回应机器人（回复/@），要么观察窗内
群里至少还有两条人声（话题还活着）。插进一段正在热聊的对话不算本事，
让对话冷场才是要学出来避开的事。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from app.core.time_utils import (
    shanghai_aware,
    shanghai_naive,
    shanghai_now_naive,
    stored_as_utc,
)
from app.storage.models import GroupInterjectionOutcome

logger = logging.getLogger(__name__)

#: 插话后观察多久：窗口内的群里动静才算这次插话的后续。
FOLLOWUP_WINDOW_MINUTES = 10.0
#: 时段桶宽度（小时）：0-2 点、2-4 点……按粗时段学"什么时候该闭嘴"。
BUCKET_HOURS = 2
#: 硬闸证据线：某时段最近两周至少这么多条已回填记录才参与判定。
GATE_MIN_SAMPLES = 5
#: 某时段接话率低于该值（且样本足够）时，硬闸掉该时段的主动插话。
GATE_MAX_RESPONSE_RATE = 0.2
#: 成绩统计窗口。
STATS_WINDOW_DAYS = 14.0
#: 单轮回填的行数上限。
EVALUATE_BATCH_LIMIT = 200
#: 提示词成绩线的最少样本：太少的样本不构成"成绩"。
STATS_MIN_SAMPLES = 3

OUTCOME_NOT_SENT = "not_sent"
OUTCOME_IGNORED = "ignored"
OUTCOME_RESPONDED = "responded"
OUTCOME_NEGATIVE = "negative"

#: 被怼识别：和 participation_feedback 的负反馈同一族词，独立维护，
#: 避免为一条正则把两个模块耦起来。
_NEGATIVE = re.compile(r"烦|吵|闭嘴|闭麦|安静|别插|别说话|没人问|没人理你|刷屏|消失|滚")


def hour_bucket(at: datetime) -> int:
    """把时刻折进 BUCKET_HOURS 宽的时段桶（0-11）。"""

    return shanghai_aware(at).hour // BUCKET_HOURS


def bucket_label(bucket: int) -> str:
    start = bucket * BUCKET_HOURS
    return f"{start}-{start + BUCKET_HOURS} 点"


def record_interjection(engine, *, group_id: int, bot_msg_id: str, sent_at: datetime) -> bool:
    """登记一次主动插话（pending）；同一回复只登记一次。失败自吞。"""

    try:
        from app.storage.db import session_scope

        with session_scope(engine) as session:
            exists = session.scalars(
                select(GroupInterjectionOutcome.id)
                .where(GroupInterjectionOutcome.bot_msg_id == str(bot_msg_id))
                .limit(1)
            ).first()
            if exists is not None:
                return False
            session.add(GroupInterjectionOutcome(
                group_id=int(group_id),
                bot_msg_id=str(bot_msg_id),
                sent_at=sent_at,
                hour_bucket=hour_bucket(sent_at),
            ))
        return True
    except Exception:
        logger.exception("interjection_record_failed group_id=%s", group_id)
        return False


def evaluate_pending(engine, *, now: datetime | None = None,
                     batch_limit: int = EVALUATE_BATCH_LIMIT) -> int:
    """回填观察窗已过的 pending 插话结果；返回回填条数（worker 用）。"""

    now = now or shanghai_now_naive()
    from app.storage.db import session_scope
    from app.storage.repositories import MessageRepository

    evaluated = 0
    with session_scope(engine) as session:
        rows = list(
            session.scalars(
                select(GroupInterjectionOutcome)
                .where(GroupInterjectionOutcome.outcome.is_(None))
                .order_by(GroupInterjectionOutcome.sent_at.asc())
                .limit(max(1, batch_limit))
            )
        )
        repo = MessageRepository(session)
        for row in rows:
            age = stored_as_utc(now) - stored_as_utc(row.sent_at)
            if age < timedelta(minutes=FOLLOWUP_WINDOW_MINUTES):
                continue
            bot_message = repo.get_by_platform_msg_id(row.bot_msg_id)
            if bot_message is None or not _outbound_delivered(bot_message):
                # 回复没真正发出去（被风控/撤销/还在预留态），不计入任何成绩。
                row.outcome = OUTCOME_NOT_SENT
            else:
                # SQL 里的时间比较统一用 naive 上海钟面（SQLite 按字符串比，
                # 混进 UTC aware 会整个窗口失效）。
                sent_at_naive = shanghai_naive(row.sent_at)
                window_end = sent_at_naive + timedelta(minutes=FOLLOWUP_WINDOW_MINUTES)
                followups = repo.list_group_messages_between(
                    group_id=row.group_id,
                    since=sent_at_naive,
                    until=window_end,
                    bot_user_id=None,
                )
                others = [
                    message
                    for message in followups
                    if message.user_id is not None
                    and message.user_id != _bot_user_id(bot_message)
                ]
                row.outcome = _classify(row.bot_msg_id, others)
            row.evaluated_at = now
            session.add(row)
            evaluated += 1
    return evaluated


def _bot_user_id(bot_message) -> int | None:
    return getattr(bot_message, "user_id", None)


def _outbound_delivered(bot_message) -> bool:
    """机器人那条回复真的发出去了吗（预留/被风控/被撤回都算没发出去）。"""

    raw = getattr(bot_message, "raw_json", None)
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = None
    if not isinstance(raw, dict):
        return False
    return str(raw.get("delivery_state") or "") in ("sent", "uncertain")


def _classify(bot_msg_id: str, followups: list) -> str:
    """有人怼 → negative；直接回应/@ 或话题仍活着（≥2 条人声）→ responded。"""

    for message in followups:
        text = str(getattr(message, "plain_text", "") or "")
        if text and _NEGATIVE.search(text):
            return OUTCOME_NEGATIVE
    for message in followups:
        if bool(getattr(message, "mentioned_bot", False)):
            return OUTCOME_RESPONDED
        if str(getattr(message, "reply_to_msg_id", "") or "") == str(bot_msg_id):
            return OUTCOME_RESPONDED
    if len(followups) >= 2:
        return OUTCOME_RESPONDED
    return OUTCOME_IGNORED


@dataclass(frozen=True)
class SilenceGate:
    """插话前的沉默闸门：是否放行 + 给判定看的成绩线。"""

    allowed: bool
    stats_line: str | None = None
    reason: str = ""


def silence_gate(engine, *, group_id: int, now: datetime | None = None) -> SilenceGate:
    """该群此刻是否还该主动插话：读最近两周成绩，冷场时段硬闸。

    任何读取/计算失败都放行（fail-open）——沉默学习永远不能卡死发言。
    """

    try:
        now = now or shanghai_now_naive()
        if now is None:
            return SilenceGate(True)
        return _silence_gate_inner(engine, group_id=group_id, now=now)
    except Exception:
        logger.exception("silence_gate_failed group_id=%s", group_id)
        return SilenceGate(True)


def _silence_gate_inner(engine, *, group_id: int, now: datetime) -> SilenceGate:
    from app.storage.db import session_scope

    # SQL 过滤用 naive 上海钟面（SQLite 字符串比较，混 UTC 会失准）。
    cutoff = shanghai_naive(now) - timedelta(days=STATS_WINDOW_DAYS)
    with session_scope(engine) as session:
        rows = list(
            session.scalars(
                select(GroupInterjectionOutcome)
                .where(
                    GroupInterjectionOutcome.group_id == int(group_id),
                    GroupInterjectionOutcome.outcome.is_not(None),
                    GroupInterjectionOutcome.sent_at >= cutoff,
                )
            )
        )
    scored = [row for row in rows if row.outcome in (OUTCOME_RESPONDED, OUTCOME_IGNORED, OUTCOME_NEGATIVE)]
    if not scored:
        return SilenceGate(True)

    def is_success(row) -> bool:
        return row.outcome == OUTCOME_RESPONDED

    current_bucket = hour_bucket(now)
    bucket_rows = [row for row in scored if int(row.hour_bucket) == current_bucket]
    if (
        len(bucket_rows) >= GATE_MIN_SAMPLES
        and sum(1 for row in bucket_rows if is_success(row)) / len(bucket_rows)
        <= GATE_MAX_RESPONSE_RATE
    ):
        return SilenceGate(
            False,
            reason=f"bucket_silence({bucket_label(current_bucket)},"
                   f"{len(bucket_rows)}次几乎没人接)",
        )

    total = len(scored)
    if total < STATS_MIN_SAMPLES:
        return SilenceGate(True)
    responded = sum(1 for row in scored if is_success(row))
    negative = sum(1 for row in scored if row.outcome == OUTCOME_NEGATIVE)
    ignored = total - responded - negative
    line = (f"你最近{STATS_WINDOW_DAYS:.0f}天主动插话 {total} 次："
            f"{responded} 次有人接，{negative} 次被怼，{ignored} 次没人理。")
    # 找出"最冷场"的时段提醒判定方；比硬闸宽一条线，靠模型自己掂量。
    worst_bucket, worst_rate, worst_count = None, 1.0, 0
    by_bucket: dict[int, list] = {}
    for row in scored:
        by_bucket.setdefault(int(row.hour_bucket), []).append(row)
    for bucket, bucket_group in by_bucket.items():
        if len(bucket_group) < GATE_MIN_SAMPLES:
            continue
        rate = sum(1 for row in bucket_group if is_success(row)) / len(bucket_group)
        if rate < worst_rate:
            worst_bucket, worst_rate, worst_count = bucket, rate, len(bucket_group)
    if worst_bucket is not None and worst_rate <= 0.4:
        line += (f"另外{bucket_label(worst_bucket)}的插话 {worst_count} 次里"
                 f"只有 {round(worst_rate * worst_count)} 次有人接，这种时段少开口。")
    return SilenceGate(True, stats_line=line)
