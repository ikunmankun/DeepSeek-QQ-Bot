"""思考深度路由：按消息复杂度动态选 reasoning effort，模拟人脑的
"简单问题秒答、难题认真想"。

设计要点：
* 规则式判定，不打 LLM——消息文本一进来就能定档，零成本零延迟，
  与 search_policy 的规则函数同构；
* 档位语义：high=深思（数学/代码/多问句长消息/明确要求认真想），
  medium=想一下（解释类问题/求助决策/带引用的长讨论），minimal=秒答
  （语气词/表情包式回应），None=普通闲聊（大多数消息），沿用 client
  固化档位不动——老配置一个都不用改；
* 精力联动：energy 低时难题也逐级降档（high→medium→默认），演
  "困了懒得深想"；被点名/主人直接求助时豁免——主人叫了必须认真答；
* 回合档位存在 contextvar：打字延迟（chat_style.typing_delay_seconds）
  读它把"想了一会儿才回"演出来，发送端不用穿参数。contextvar 在
  同一 asyncio 任务内有效，每回合 set/reset，不会串到别的回合。
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass


#: 明确要求认真思考的说法（直接顶到 high）。
SERIOUS_THINK_HINTS = (
    "认真想想", "仔细想想", "好好想想", "认真分析", "仔细分析",
    "认真回答", "帮我想想", "帮我想个", "动动脑子", "想清楚",
    "展开讲讲", "细说", "论证一下",
)

#: 数学/演算触发：数字带运算符、解方程、统计口径等。
MATH_PATTERN = re.compile(
    r"\d+\s*[+\-*/×÷^=]\s*\d+"
    r"|帮我算|帮忙算|算一下|算算|等于多少|解方程|因式分解|"
    r"概率是多少|百分比|几分之几|平均数|方差|开根号|乘方|阶乘"
)

#: 代码/技术触发：报错排查、写代码、命令行。
CODE_HINTS = (
    "```", "Traceback", "traceback", "报错", "报异常", "抛异常",
    "python", "Python", "java", "Java", "c++", "C++", "js ", "TS ",
    "bug", "Bug", "debug", "代码", "脚本", "正则", "regex",
    "SQL", "sql", "docker", "Docker", "命令行", "编译", "运行不起来",
)

#: 解释类问题（想一下就行，不用顶到 high）。
EXPLAIN_HINTS = (
    "为什么", "为啥", "怎么回事", "什么原理", "什么意思", "怎么理解",
    "怎么实现的", "是如何", "怎么才能", "怎么做", "怎么弄", "怎么用",
    "啥意思", "讲讲", "介绍一下", "科普一下",
)

#: 求助决策类（比较、选择、建议）。
DECIDE_HINTS = (
    "帮我选", "帮我挑", "哪个好", "值不值得", "值得吗", "该不该",
    "要不要", "给个建议", "有什么建议", "怎么办", "帮我参谋", "推荐哪个",
)

#: 极简回应的字集：真人看到这种就是秒回。去标点空白后全由这些字
#: 组成且不超过 8 字就算极简。
TRIVIAL_CHARS = set(
    "哦噢喔嗯呃哈呵嘿哟欸好行啊吧呀哇噻嘞咯哎唉咩嘛呗啦"
    "牛赞绷乐急泪草麻赢寄睡了想你贴听看说dnbOKok六九嗯"
    "0123456789"
)

_LEVEL_EFFORT = {"high": "high", "medium": "medium", "minimal": "minimal", "none": None}

#: 档位 → 打字延迟系数：秒答的更干脆，深想的"想了一会儿才回"。
TYPING_DELAY_FACTORS = {"minimal": 0.7, "low": 1.0, "medium": 1.15, "high": 1.5}

#: energy 低于该值视为"困了"，难题降档（数值口径同 vitality：0~100）。
SLEEPY_ENERGY_THRESHOLD = 30.0

_turn_effort: contextvars.ContextVar[str] = contextvars.ContextVar(
    "think_depth_turn_effort", default=""
)


@dataclass(frozen=True)
class ThinkDepthDecision:
    """一次思考深度判定。

    effort: None=不覆盖（沿用 client 固化档位）；""=这次关闭思考；
    minimal/medium/high=按该档位覆盖。
    level: none|minimal|medium|high，打字延迟联动按这个取系数。
    """

    effort: str | None
    level: str
    reason: str


def _is_trivial_reply(text: str) -> bool:
    stripped = re.sub(r"[\s，,。.!！?？~～、…]+", "", text)
    if len(stripped) > 8:
        return False
    # 剩下空串（纯标点/纯"？"）也算极简回应——那本来就是秒回的场景。
    return all(char in TRIVIAL_CHARS for char in stripped)


def _complexity_level(text: str, *, has_quote: bool, has_image: bool) -> tuple[str, str]:
    if any(hint in text for hint in SERIOUS_THINK_HINTS):
        return "high", "serious_request"
    if MATH_PATTERN.search(text):
        return "high", "math"
    if any(hint in text for hint in CODE_HINTS):
        return "high", "code"
    question_marks = text.count("？") + text.count("?")
    if question_marks >= 3:
        return "high", "multi_question"
    if len(text) >= 150 and question_marks >= 1:
        return "high", "long_multi_part"
    if any(hint in text for hint in EXPLAIN_HINTS):
        return "medium", "explain"
    if any(hint in text for hint in DECIDE_HINTS):
        return "medium", "decision"
    if has_quote and len(text) >= 40:
        return "medium", "quoted_discussion"
    if has_image and question_marks >= 1:
        return "medium", "image_question"
    if len(text) >= 150:
        return "medium", "long_message"
    return "none", "casual_chat"


def judge_think_depth(
    text: str,
    *,
    has_quote: bool = False,
    has_image: bool = False,
    explicit_request: bool = False,
    energy: float | None = None,
    energy_coupled: bool = True,
) -> ThinkDepthDecision:
    """按消息内容判定本回合思考深度。

    explicit_request：对方点名/主人直接求助——只在精力联动里豁免降档，
    不影响复杂度本身（简单消息该秒答还是秒答）。
    """

    normalized = str(text or "").strip()
    if not normalized:
        return ThinkDepthDecision(None, "none", "empty")
    if _is_trivial_reply(normalized):
        return ThinkDepthDecision("minimal", "minimal", "trivial_reply")
    level, reason = _complexity_level(normalized, has_quote=has_quote, has_image=has_image)
    if level == "none":
        return ThinkDepthDecision(None, "none", reason)
    if (
        energy_coupled
        and energy is not None
        and float(energy) < SLEEPY_ENERGY_THRESHOLD
        and not explicit_request
    ):
        downgraded = {"high": "medium", "medium": "none"}[level]
        if downgraded == "none":
            return ThinkDepthDecision(None, "none", reason + "+sleepy_dropped")
        return ThinkDepthDecision(
            _LEVEL_EFFORT[downgraded], downgraded, reason + "+sleepy_downgraded"
        )
    return ThinkDepthDecision(_LEVEL_EFFORT[level], level, reason)


def set_turn_effort(effort: str | None):
    """记录本回合思考档位；返回 token 供 reset（同任务内 turn 边界防串）。"""

    return _turn_effort.set(str(effort or ""))


def reset_turn_effort(token) -> None:
    _turn_effort.reset(token)


def current_turn_effort() -> str:
    return _turn_effort.get()


def typing_delay_factor(effort: str | None = None) -> float:
    """打字延迟系数：不传 effort 时读当前回合档位（contextvar）。"""

    resolved = current_turn_effort() if effort is None else effort
    return TYPING_DELAY_FACTORS.get(resolved, 1.0)
