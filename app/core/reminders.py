"""提醒/闹钟：自然语言时间解析、存取、到点发送与贪睡式催促。

大肥鱼不是手机闹钟——它碰不到系统闹钟，也没有语音通话 API。它能做的
是 QQ 生态内最接近的形态：到点在群里 @你（或私聊你），配上语音条
（alarms 目录里你放的音频，随机抽一条）和表情包式催促；你没回应就隔
几分钟再催，最多 max_nags 次——贪睡功能的精神续作。

时间解析走确定性规则（相对：半小时后/N分钟/N小时；绝对：明早8点/
下午3点半/21:30；重复：每天8点），不用 LLM——设提醒这件事 100% 要说
中，宁可让用户按格式再说一遍也不猜错时间。
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from app.adapters.sender import OutboundPrivateMessage
from app.core.time_utils import shanghai_now_naive, stored_as_utc
from app.storage.models import Message, Reminder

logger = logging.getLogger(__name__)

#: 到点没回应，隔多久催下一次。
NAG_INTERVAL_MINUTES = 5.0
#: 默认最多催几次（首次发送之后再催的次数）。
DEFAULT_MAX_NAGS = 2
#: 每个人最多同时挂几条待响提醒（防刷屏）。
MAX_PENDING_PER_USER = 10
#: 语音素材目录里认的后缀。
VOICE_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg", ".silk", ".amr"})

_REPEAT_ONCE = "once"
_REPEAT_DAILY = "daily"

# ---- 时间解析 -----------------------------------------------------------

_PERIOD_WORDS = ("早上", "上午", "中午", "下午", "晚上", "傍晚", "凌晨", "半夜")
_PERIOD_PATTERN = r"(?:早上|上午|中午|下午|晚上|傍晚|凌晨|半夜)?"
_DAY_WORDS = ("明天", "后天", "大后天")

_VERB_PREFIX = r"(?:帮我)?(?:提醒|叫|喊)我"
#: 是"设提醒"意图的最低门槛：动词带"我"、或提到闹钟、或每天+点。
_SET_INTENT = re.compile(rf"{_VERB_PREFIX}|闹钟|每天")
_CANCEL_PATTERN = re.compile(r"^(?:取消|删掉|删除|停止|别)\s*(?:我的)?\s*(?:提醒|闹钟|叫早)\s*(\d+)?\s*$")
_LIST_PATTERN = re.compile(r"^(?:我的|查一下?|看看)?\s*(?:提醒|闹钟)(?:列表|清单)?\s*$")
_VERB_SEARCH_RE = re.compile(r"\s*" + _VERB_PREFIX + r"[一下]*\s*[:：,，]?")
#: before 片段像不像一个时间表达（时间在动词前的语序用）
_TIME_LIKE = re.compile(r"^(?:每\s*天|明天|明早|明晚|今早|今晚|今午|今天|后天|大后天|半|[\d零一二两三四五六七八九十])")

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(text: str) -> int | None:
    """中文数字 0-99（一/两/十/十五/二十一半外的直接失败）。"""
    text = str(text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return _CN_DIGITS.get(text)
_LEADING_FILLER = re.compile(r"^(?:大肥鱼|鲸鱼娘|小町|哦鲸鲸|帮个忙|麻烦|请|你|给我|给我个)\s*")

_NUM = r"(?:\d{1,3}|[零一二两三四五六七八九十]{1,3})"
_RELATIVE_PATTERNS: tuple[tuple[re.Pattern[str], object], ...] = (
    (re.compile(r"^半\s*个?小时\s*(?:之|以)?后"), lambda m: timedelta(minutes=30)),
    (re.compile(rf"^({_NUM})\s*分钟\s*(?:之|以)?后"), lambda m: timedelta(minutes=_cn_to_int(m.group(1)))),
    (re.compile(rf"^({_NUM})\s*个半\s*小时\s*(?:之|以)?后"), lambda m: timedelta(hours=_cn_to_int(m.group(1)) + 0.5)),
    (re.compile(rf"^({_NUM})\s*小时\s*(?:之|以)?后"), lambda m: timedelta(hours=_cn_to_int(m.group(1)))),
    (re.compile(rf"^({_NUM})\s*天\s*(?:之|以)?后"), lambda m: timedelta(days=_cn_to_int(m.group(1)))),
)


@dataclass(frozen=True)
class ParsedReminder:
    due_at: datetime
    repeat: str
    content: str


@dataclass(frozen=True)
class ReminderCommand:
    action: str  # set | cancel | list | error
    parsed: ParsedReminder | None = None
    target_id: int | None = None
    error: str | None = None


class ReminderParseError(ValueError):
    """是提醒意图但时间没说清；message 面向用户。"""


def _resolve_clock(hour: int, minute: int, period: str | None) -> int:
    """把时段词折进 24 小时制；返回小时。"""
    hour = int(hour)
    if period in ("下午", "晚上", "傍晚") and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12
    elif period == "半夜" and hour == 12:
        hour = 0
    return hour % 24


def _parse_clock_fragment(fragment: str, *, now: datetime, daily: bool) -> tuple[datetime, str] | None:
    """解析"明天早上8点半 / 今晚11点 / 下午3点 / 21:30"这类钟点片段。

    返回 (到期时间, 剩余文本)；不是钟点片段返回 None。
    """
    fragment = str(fragment or "").strip()
    if not fragment:
        return None
    day_offset = 0
    forced_period = None
    day_match = re.match(r"^(明天|明早|明午|明晚|今早|今午|今晚|今天|后天|大后天)\s*", fragment)
    if day_match:
        word = day_match.group(1)
        day_offset, forced_period = {
            "明天": (1, None), "后天": (2, None), "大后天": (3, None),
            "今天": (0, None), "明早": (1, "早上"), "明午": (1, "中午"),
            "明晚": (1, "晚上"), "今早": (0, "早上"), "今午": (0, "中午"),
            "今晚": (0, "晚上"),
        }[word]
        fragment = fragment[day_match.end():]
    match = re.match(
        rf"^({_PERIOD_PATTERN})\s*(\d{{1,2}})\s*[点时:：]\s*(半|整|\d{{1,2}}\s*分?)?\s*",
        fragment,
    )
    if match is None:
        colon = re.match(r"^(\d{1,2})[:：](\d{2})\s*", fragment)
        if colon is None:
            return None
        hour, minute = int(colon.group(1)), int(colon.group(2))
        rest = fragment[colon.end():]
        period = forced_period
    else:
        period = match.group(1) or forced_period
        hour = _resolve_clock(int(match.group(2)), 0, period)
        tail = (match.group(3) or "整").strip()
        if tail in ("半", "整"):
            minute = 30 if tail == "半" else 0
        else:
            minute = int(re.sub(r"\D", "", tail) or 0)
        rest = fragment[match.end():]
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_offset:
        due += timedelta(days=day_offset)
    elif daily:
        if due <= now:
            due += timedelta(days=1)
    elif due <= now:
        # 没带"明天"但时间已过：默认算明天（晚上10点说"8点叫我"→明早8点）。
        due += timedelta(days=1)
    return due, rest.strip()


def _parse_relative_fragment(fragment: str, *, now: datetime) -> tuple[datetime, str] | None:
    fragment = str(fragment or "").strip()
    for pattern, to_delta in _RELATIVE_PATTERNS:
        match = pattern.match(fragment)
        if match is None:
            continue
        delta = to_delta(match)
        if delta <= timedelta(0):
            return None
        return now + delta, fragment[match.end():].strip()
    return None


def match_reminder_command(text: str) -> ReminderCommand | None:
    """兼容别名：等价于 parse_reminder_command（tests 与旧调用点使用旧名）。"""
    return parse_reminder_command(text)


def _clean_content(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[，,。.、!！?？\s]+", "", text)
    text = re.sub(r"[，,。.\s]+$", "", text)
    return text[:200]


def parse_reminder_command(text: str, *, now: datetime | None = None) -> ReminderCommand | None:
    """识别一条群消息是不是提醒指令；不是返回 None。

    支持"提醒我 明早8点 起床"和"半小时后提醒我喝水"两种语序，
    以及"每天8点叫我起床""取消提醒/取消提醒 3""我的提醒"。
    """
    now = now or shanghai_now_naive()
    normalized = _LEADING_FILLER.sub("", str(text or "").strip())
    normalized = normalized.lstrip("？?！!，, ")
    cancel = _CANCEL_PATTERN.match(normalized)
    if cancel is not None:
        target = cancel.group(1)
        return ReminderCommand("cancel", target_id=int(target) if target else None)
    if _LIST_PATTERN.match(normalized) is not None:
        return ReminderCommand("list")
    if _SET_INTENT.search(normalized) is None:
        return None

    match_a = _VERB_SEARCH_RE.search(normalized)
    if match_a is None:
        return ReminderCommand("error", error="想提醒你，但没看懂——说个时间呀，比如「半小时后提醒我喝水」")
    before = normalized[:match_a.start()].strip()
    after = normalized[match_a.end():].strip()
    if before and _TIME_LIKE.match(before):
        # 时间在动词前："半小时后提醒我喝水" / "每天早上8点叫我起床"
        when, what = before, after
    elif not before:
        # 动词在前："提醒我 明早8点 起床"——时间（可能连着内容）都在后面
        when, what = after, ""
    else:
        # 动词前是一段闲聊（"你怎么不提醒我"）——不是指令
        return None

    daily = bool(re.match(r"^每\s*天", when.strip()))
    if daily:
        when_body = re.sub(r"^每\s*天\s*", "", when.strip())
        if not re.search(r"\d|点|半", when_body):
            when_body = "早上8点"  # "每天叫我起床"——默认早八
        parsed = _parse_clock_fragment(when_body, now=now, daily=True)
        if parsed is None:
            return ReminderCommand("error", error="每天的提醒要带钟点呀，比如「每天早上8点叫我起床」")
        due, rest = parsed
    else:
        parsed = _parse_relative_fragment(when, now=now)
        if parsed is None:
            clock = _parse_clock_fragment(when, now=now, daily=False)
            if clock is None:
                return ReminderCommand(
                    "error",
                    error="这个时间我没看懂……说成「N分钟后」「明天早上8点」「每天晚上11点」这种就行",
                )
            due, rest = clock
        else:
            due, rest = parsed
        if rest and not what:
            what = rest
    content = _clean_content(what)
    if not content:
        content = "时间到啦"
    return ReminderCommand("set", parsed=ParsedReminder(due_at=due, repeat=_REPEAT_DAILY if daily else _REPEAT_ONCE, content=content))


# ---- 存取 ---------------------------------------------------------------

def create_reminder(session, *, group_id: int | None, user_id: int, channel: str,
                    parsed: ParsedReminder, now: datetime, max_nags: int = DEFAULT_MAX_NAGS) -> Reminder:
    pending = session.scalars(
        select(Reminder).where(
            Reminder.user_id == int(user_id),
            Reminder.status.in_(("pending", "nagging")),
        )
    ).all()
    if len(pending) >= MAX_PENDING_PER_USER:
        raise ValueError(f"你已经有 {MAX_PENDING_PER_USER} 条没响的提醒啦，先取消几条再来")
    row = Reminder(
        group_id=int(group_id) if group_id is not None else None,
        user_id=int(user_id),
        channel=channel,
        content=parsed.content,
        due_at=parsed.due_at,
        repeat=parsed.repeat,
        status="pending",
        nag_count=0,
        max_nags=max_nags,
        created_at=now,
    )
    session.add(row)
    session.flush()
    return row


def list_user_reminders(session, *, user_id: int) -> list[Reminder]:
    return list(
        session.scalars(
            select(Reminder)
            .where(
                Reminder.user_id == int(user_id),
                Reminder.status.in_(("pending", "nagging")),
            )
            .order_by(Reminder.due_at.asc())
        )
    )


def cancel_user_reminder(session, *, user_id: int, reminder_id: int | None) -> int:
    """取消本人的一条（指定 id）或全部待响提醒；返回取消条数。"""

    query = select(Reminder).where(
        Reminder.user_id == int(user_id),
        Reminder.status.in_(("pending", "nagging")),
    )
    if reminder_id is not None:
        query = query.where(Reminder.id == int(reminder_id))
    rows = list(session.scalars(query))
    for row in rows:
        row.status = "cancelled"
        session.add(row)
    return len(rows)


# ---- 后台服务 -----------------------------------------------------------

_NAG_LINES = ("到点啦，别装没看见——{content}", "{content}！我在说真的呀", "最后催一次啦：{content}！！")


@dataclass(frozen=True)
class _DueSnapshot:
    id: int
    group_id: int | None
    user_id: int
    channel: str
    content: str
    repeat: str
    status: str
    nag_count: int
    max_nags: int
    next_nag_at: datetime | None
    last_sent_at: datetime | None
    due_at: datetime


def pick_alarm_voice(alarms_dir: Path | None) -> Path | None:
    """alarms 目录里随机抽一条语音素材；目录为空返回 None（纯文字提醒）。"""

    if alarms_dir is None:
        return None
    try:
        candidates = [
            path
            for path in Path(alarms_dir).iterdir()
            if path.suffix.lower() in VOICE_SUFFIXES and path.stat().st_size > 0
        ]
    except OSError:
        return None
    return random.choice(candidates) if candidates else None


class ReminderService:
    """周期 worker：扫到期提醒 → 发送 → 贪睡催促 → 日常顺延。"""

    def __init__(self, *, engine, sender, alarms_dir: Path | None = None,
                 interval_seconds: float = 30.0,
                 nag_interval_minutes: float = NAG_INTERVAL_MINUTES,
                 max_nags: int = DEFAULT_MAX_NAGS, now_fn=None) -> None:
        self._engine = engine
        self._sender = sender
        self._alarms_dir = Path(alarms_dir) if alarms_dir else None
        self.interval_seconds = float(interval_seconds)
        self._nag_interval = timedelta(minutes=float(nag_interval_minutes))
        self._max_nags = int(max_nags)
        self._now_fn = now_fn or shanghai_now_naive

    def _now(self) -> datetime:
        return self._now_fn()

    async def run(self) -> None:
        import asyncio

        from app.core.worker_status import write_worker_status

        while True:
            try:
                fired = await asyncio.to_thread(self._collect_due)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("reminder_collect_failed")
                fired = []
            for snapshot in fired:
                try:
                    await self._process(snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("reminder_process_failed id=%s", snapshot.id)
            if fired:
                logger.info("reminder_fired count=%s", len(fired))
            await asyncio.sleep(self.interval_seconds)

    def _collect_due(self) -> list[_DueSnapshot]:
        now = self._now()
        from app.storage.db import session_scope

        with session_scope(self._engine) as session:
            rows = list(
                session.scalars(
                    select(Reminder).where(
                        Reminder.status.in_(("pending", "nagging")),
                        Reminder.due_at <= now,
                    )
                )
            )
            due = [
                row for row in rows
                if row.status == "pending"
                or (row.next_nag_at is not None and row.next_nag_at <= now)
            ]
            return [
                _DueSnapshot(
                    id=int(row.id), group_id=row.group_id, user_id=int(row.user_id),
                    channel=str(row.channel), content=str(row.content or ""),
                    repeat=str(row.repeat), status=str(row.status),
                    nag_count=int(row.nag_count or 0), max_nags=int(row.max_nags or 0),
                    next_nag_at=row.next_nag_at, last_sent_at=row.last_sent_at,
                    due_at=row.due_at,
                )
                for row in due
            ]

    async def _process(self, snapshot: _DueSnapshot) -> None:
        if snapshot.status == "pending":
            await self._fire(snapshot)
        else:
            await self._nag(snapshot)

    def _pick_voice(self) -> Path | None:
        return pick_alarm_voice(self._alarms_dir)

    async def _fire(self, snapshot: _DueSnapshot) -> None:
        now = self._now()
        voice = self._pick_voice()
        line = f"提醒：{snapshot.content}"
        if snapshot.channel == "group" and snapshot.group_id is not None:
            if voice is not None:
                await self._sender.send_group_voice(
                    group_id=snapshot.group_id, voice_file=str(voice)
                )
            await self._sender.send_group_at_text(
                group_id=snapshot.group_id, user_id=snapshot.user_id, text=line
            )
        else:
            if voice is not None:
                await self._sender.send_private_voice(
                    user_id=snapshot.user_id, voice_file=str(voice)
                )
            await self._sender.send_private_text(
                OutboundPrivateMessage(user_id=snapshot.user_id, text=line)
            )
        self._after_send(snapshot, now=now)

    async def _nag(self, snapshot: _DueSnapshot) -> None:
        now = self._now()
        since = snapshot.last_sent_at
        if since is not None and self._user_responded(snapshot, since=since):
            self._settle(snapshot, now=now)
            return
        if snapshot.nag_count >= snapshot.max_nags:
            self._settle(snapshot, now=now)
            return
        line = _NAG_LINES[min(snapshot.nag_count, len(_NAG_LINES) - 1)].format(
            content=snapshot.content
        )
        if snapshot.channel == "group" and snapshot.group_id is not None:
            await self._sender.send_group_at_text(
                group_id=snapshot.group_id, user_id=snapshot.user_id, text=line
            )
        else:
            await self._sender.send_private_text(
                OutboundPrivateMessage(user_id=snapshot.user_id, text=line)
            )
        self._after_nag(snapshot, now=now)

    def _user_responded(self, snapshot: _DueSnapshot, *, since: datetime) -> bool:
        from app.storage.db import session_scope
        from app.storage.repositories import MessageRepository

        with session_scope(self._engine) as session:
            if snapshot.channel == "group" and snapshot.group_id is not None:
                return MessageRepository(session).has_group_messages_from(
                    group_id=snapshot.group_id, user_id=snapshot.user_id, since=since
                )
            return MessageRepository(session).has_private_messages_from(
                user_id=snapshot.user_id, since=since
            )

    def _after_send(self, snapshot: _DueSnapshot, *, now: datetime) -> None:
        from app.storage.db import session_scope

        with session_scope(self._engine) as session:
            row = session.get(Reminder, snapshot.id)
            if row is None:
                return
            row.last_sent_at = now
            row.status = "nagging"
            row.nag_count = 0
            row.next_nag_at = now + self._nag_interval
            session.add(row)

    def _after_nag(self, snapshot: _DueSnapshot, *, now: datetime) -> None:
        from app.storage.db import session_scope

        with session_scope(self._engine) as session:
            row = session.get(Reminder, snapshot.id)
            if row is None:
                return
            row.last_sent_at = now
            row.nag_count = int(row.nag_count or 0) + 1
            row.next_nag_at = now + self._nag_interval
            if int(row.nag_count) >= int(row.max_nags or 0):
                self._expire_or_reschedule(row, now=now)
            session.add(row)

    def _settle(self, snapshot: _DueSnapshot, *, now: datetime) -> None:
        """有人应了或催满次数：一次性提醒收尾，每日提醒顺延到明天。"""

        from app.storage.db import session_scope

        with session_scope(self._engine) as session:
            row = session.get(Reminder, snapshot.id)
            if row is None:
                return
            self._expire_or_reschedule(row, now=now)
            session.add(row)

    @staticmethod
    def _expire_or_reschedule(row: Reminder, *, now: datetime) -> None:
        if str(row.repeat) == _REPEAT_DAILY:
            row.due_at = row.due_at + timedelta(days=1)
            row.status = "pending"
            row.nag_count = 0
            row.next_nag_at = None
            row.last_sent_at = None
        else:
            row.status = "done"
