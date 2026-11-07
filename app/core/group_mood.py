"""规则版运行时情绪：每群一份会流动、会累积、会消退的当前心情。

静态人设每轮都从零开始"演"心情；这里把情绪变成真实状态：

* 事件抬升——群里有人跟机器人说话时按规则分类：被夸 → smug（得意）、
  被怼/被调戏 → annoyed（恼火）、求助或对方难受 → serious（收起损认真帮忙）；
  同一种情绪反复触发会累积（真人会被连续招惹惹毛）。
* 时间消退——intensity 按各情绪自己的半衰期惰性衰减，跌破 FLOOR 回到 ambient。
  不需要后台任务，读取时现算。
* 环境底色——没有事件情绪时，凌晨 0-7 点自动进入 sleepy（犯困）。
* 优先级——事件情绪永远压过环境底色（半夜被怼也是恼火优先于犯困）。

语气本身不在这里写：注入提示词时只报当前状态名/强度/起因，
具体怎么说话由 persona.yaml 的 ``mood`` 段（"你在不同状态下的样子"）提供。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.core.time_utils import stored_as_utc
from app.storage.models import GroupMoodState

#: 情绪标签与 persona.yaml ``mood`` 段的键保持一致，模型对着人设取语气。
MOOD_LABELS: dict[str, str] = {
    "smug": "得意",
    "annoyed": "恼火",
    "serious": "认真",
    "sleepy": "犯困",
    "default": "",
}

#: 事件情绪的半衰期（分钟）：恼火来得快去得快，得意飘得久一点，
#: 认真只在求助进行中保持，一旦没人继续问很快回底。
HALF_LIFE_MINUTES: dict[str, float] = {
    "smug": 60.0,
    "annoyed": 45.0,
    "serious": 15.0,
}

#: 强度低于此值视为回落到 ambient。
FLOOR = 0.25
#: 单次事件抬升的上限（连续招惹也封顶，防止失控）。
CAP = 2.0
#: 凌晨犯困窗口 [0, 7)。
SLEEPY_WINDOW = (0, 7)

#: 接话斗嘴窗口：机器人最近 N 分钟内在本群说过话，后面的话默认接着聊。
#: 群友斗嘴从来不 @，全发生在接话里；不开这个口子情绪系统永远睡着。
ONGOING_EXCHANGE_MINUTES = 10
_SECOND_PERSON = re.compile(r"你|您")

#: 情绪上头时顺手提示"可以甩表情"的状态：被怼到不想打字、被夸到
#: 得意，都是表情比文字自然的时刻。permission 式提示，发不发仍由
#: 人设的频率约束（平均三五轮一张）决定。
STICKER_HINT_MOODS = frozenset({"annoyed", "smug"})
STICKER_HINT = "被怼或对方甩图斗图时，优先甩一张表情包（pick_sticker 挑，send_group 发），文字只当配菜。"

#: 起因摘录里去掉会剧透具体聊天的长句，只留短因。
_REASON_MAX_CHARS = 40

_PRAISE_PATTERN = re.compile(
    r"厉害|牛[比逼bB]|nb|666|太强|真棒|好棒|好强|好聪明|聪明|可爱|学到了|谢谢|感谢|辛苦|夸|真行|优秀"
)
_TEASE_PATTERN = re.compile(
    r"笨|傻|蠢|杂鱼|垃圾|废物|废柴|真菜|好菜|菜狗|滚|闭嘴|讨人厌|烦人|欺负|调戏|戏弄|怼|嘲笑|看不起|看扁|就这|急了|破防|便宜|抠门|小气|弄你|打你|踹"
)
_SERIOUS_PATTERN = re.compile(
    r"帮我|求助|怎么办|咋办|咋弄|怎么弄|救命|难受|难过|伤心|不开心|不高兴|emo|想哭|好哭|崩了|崩溃|出事|急死|焦虑|失眠|生病|发烧|头疼|胃疼|难受死了"
)


def should_evaluate_mood(*, text: str, mentioned_bot: bool = False, reply_to_bot: bool = False,
                         called_by_name: bool = False, other_addressee: bool = False,
                         bot_spoke_recently: bool = False) -> bool:
    """方向门控：一条入站消息是否参与情绪判定。

    * 点名（@/回复/叫名字）永远算当面互动；
    * 接话场景——机器人最近说过话、这条又没明显对别人说——也算当面；
    * 但接话只认短打（≤6 字，如"杂鱼""就这"）或带"你"的话，避免把群友
      聊游戏时的"这游戏好菜"错记成怼机器人。
    """

    if mentioned_bot or reply_to_bot or called_by_name:
        return True
    if not bot_spoke_recently or other_addressee:
        return False
    normalized = str(text or "").strip()
    return len(normalized) <= 6 or bool(_SECOND_PERSON.search(normalized))


def evaluate_inbound_trigger(*, messages: Any, event: Any, bot_qq: int) -> tuple[tuple[str, float, str] | None, bool]:
    """入站情绪判定的组合入口：方向门控 + 模式分类。

    返回 ``(trigger, explicit)``：trigger 为 None 表示没有情绪事件；
    explicit 表示是点名互动（@/回复/叫名字），供好感度等其它插件复用，
    避免同一套门控逻辑被抄两份。
    """

    if event.user_id == bot_qq:
        return None, False
    quoted = (messages.get_by_platform_msg_id(event.reply_to_msg_id)
              if event.reply_to_msg_id else None)
    reply_to_bot = quoted is not None and quoted.user_id == bot_qq
    from app.core.message_mentions import addressing_evidence

    evidence = addressing_evidence(event.raw_payload, bot_id=bot_qq,
        text=event.plain_text,
        reply_author_id=quoted.user_id if quoted is not None else None,
        author_id=event.user_id)
    explicit = bool(event.mentioned_bot) or reply_to_bot or bool(evidence["bot_explicitly_called"])
    bot_spoke_recently = False
    if not explicit:
        from datetime import timedelta

        cutoff = event.timestamp - timedelta(minutes=ONGOING_EXCHANGE_MINUTES)
        bot_spoke_recently = messages.has_group_messages_from(
            group_id=event.group_id, user_id=bot_qq, since=cutoff)
    if not should_evaluate_mood(text=event.plain_text,
            mentioned_bot=bool(event.mentioned_bot), reply_to_bot=reply_to_bot,
            called_by_name=bool(evidence["bot_explicitly_called"]),
            other_addressee=bool(evidence["other_addressee"]),
            bot_spoke_recently=bot_spoke_recently):
        return None, False
    return classify_mood_trigger(event.plain_text), explicit


def classify_mood_trigger(text: str) -> tuple[str, float, str] | None:
    """把一条已判定为"当面互动"的消息分类成情绪事件；无关内容返回 None。

    顺序即优先级：对方真的在求助/难受时，即使话里带刺也先收起损（serious
    优先于 annoyed）。识别不到任何模式不产生事件，避免把普通聊天演成情绪戏。
    """

    normalized = str(text or "").strip()
    if not normalized:
        return None
    if _SERIOUS_PATTERN.search(normalized):
        return ("serious", 0.9, _reason_snippet(normalized))
    if _TEASE_PATTERN.search(normalized):
        return ("annoyed", 0.6, _reason_snippet(normalized))
    if _PRAISE_PATTERN.search(normalized):
        return ("smug", 0.5, _reason_snippet(normalized))
    return None


def _reason_snippet(text: str) -> str:
    snippet = re.sub(r"\s+", " ", str(text or "")).strip()
    return snippet[:_REASON_MAX_CHARS]


def apply_mood_trigger(session, *, group_id: int, mood: str, bump: float, reason: str, now: datetime) -> None:
    """把一次情绪事件写进群状态；同一情绪累积、不同情绪直接切换。"""

    row = session.get(GroupMoodState, int(group_id))
    if row is None:
        row = GroupMoodState(group_id=int(group_id), mood=mood, intensity=0.0, updated_at=now)
        session.add(row)
    effective = _decayed_intensity(row, now) if row.mood == mood else 0.0
    row.mood = mood
    row.intensity = min(CAP, max(float(bump), effective + float(bump)))
    row.reason = str(reason or "")[:160]
    row.updated_at = now


def _decayed_intensity(row: GroupMoodState, now: datetime) -> float:
    half_life_minutes = HALF_LIFE_MINUTES.get(str(row.mood), 45.0)
    try:
        elapsed = (stored_as_utc(now) - stored_as_utc(row.updated_at)).total_seconds() / 60.0
    except (TypeError, ValueError):
        return 0.0
    if elapsed <= 0:
        return float(row.intensity or 0.0)
    return float(row.intensity or 0.0) * 2.0 ** (-elapsed / half_life_minutes)


def load_group_mood(engine, group_id: int) -> GroupMoodState | None:
    from app.storage.db import session_scope

    with session_scope(engine) as session:
        row = session.get(GroupMoodState, int(group_id))
        if row is None:
            return None
        session.expunge(row)
        return row


def effective_mood_state(row: GroupMoodState | None, now: datetime) -> tuple[str, float]:
    """当前生效的情绪与强度；事件情绪衰减见底后回落到 ambient。"""

    ambient = _ambient_mood(now)
    if row is None:
        return ambient, 0.0
    mood = str(row.mood or "default")
    if mood not in HALF_LIFE_MINUTES:
        return ambient, 0.0
    intensity = _decayed_intensity(row, now)
    if intensity < FLOOR:
        return ambient, 0.0
    return mood, intensity


def _ambient_mood(now: datetime) -> str:
    try:
        hour = _shanghai_hour(now)
    except (TypeError, ValueError):
        return "default"
    return "sleepy" if SLEEPY_WINDOW[0] <= hour < SLEEPY_WINDOW[1] else "default"


def _shanghai_hour(now: datetime) -> int:
    from app.core.time_utils import shanghai_aware

    return shanghai_aware(now).hour


def mood_prompt_line(engine, group_id: int, now: datetime) -> str | None:
    """给 agent 提示词的当前情绪行；平静时返回 None（不打扰人设）。"""

    try:
        row = load_group_mood(engine, group_id)
    except Exception:
        return None
    mood, intensity = effective_mood_state(row, now)
    if mood == "default":
        return None
    label = MOOD_LABELS.get(mood, mood)
    level = "有点" if intensity < 1.0 else "相当"
    origin = f"（起因：{row.reason}）" if row is not None and str(row.reason or "").strip() else ""
    line = (
        f"当前情绪状态：{mood}/{label}，{level}{label}{origin}。"
        "这是你此刻真实的心理状态，不是让你表演或说出来的东西——"
        "语气按人设里对应状态的样子自然带出来，不要解释自己在闹情绪。"
    )
    if mood in STICKER_HINT_MOODS:
        line += STICKER_HINT
    return line


#: 插话判定用的情绪偏置：跟提示词的情绪行不同，这不是语气描述，而是告诉
#: 判定方"此刻的你到底想不想说话"——情绪该影响要不要开口，不只是怎么开口。
MOOD_JUDGE_HINTS: dict[str, str] = {
    "annoyed": "你现在正恼火（刚被怼/被招惹）。默认不要插话；除非话题真值得回一句，"
               "而且只朝惹你的人去，别殃及其他人。",
    "serious": "你正在认真帮人处理事情。别用斗嘴打断正经对话，除非确实能帮上忙。",
    "smug": "你正有点得意。遇到能调侃的机会可以更积极一些，但别凑热闹凑到烦人。",
    "sleepy": "现在是凌晨，你正犯困，基本不想说话。除非有人点名找你，默认不插话。",
}


def mood_judge_hint(engine, group_id: int, now: datetime) -> str | None:
    """给插话判定的当前情绪偏置行；平静/读取失败时返回 None。"""

    try:
        row = load_group_mood(engine, group_id)
        mood, _intensity = effective_mood_state(row, now)
    except Exception:
        return None
    return MOOD_JUDGE_HINTS.get(mood)
