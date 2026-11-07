"""发送前的风格审查：客服腔 / Markdown / 说教检测，翻车就带批评重写一次。

AI 腔的根源是大模型太爱总结、爱分点、爱在结尾祝人顺利——"人味不在优点
里，在缺陷里"。此前的防线是 prompt 规则 + 正则裁剪（chat_style.normalize_*），
坏回复只会被截断清洗，不会被识别后重写。这里补上识别与重写：

* 确定性快检（quick_style_faults）——零成本抓硬伤：markdown、列表、
  客服腔起手式、"不是A而是B"句式；
* LLM 复审（judge_reply_style）——只有配置了轻量 judge client 才启用，
  抓语气层面的说教腔；任何失败降级为通过（宁可放过，不可不发）；
* 重写（rewrite）——把失败原因原样喂回去重新生成一次，重写仍不合格
  就按原样发送：风格审查永远不能拦住回复本身。

采样策略在调用方（router）：主动插话全量审，普通回复按比例抽样。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StyleJudgeResult:
    passed: bool
    issues: list[str]

    @property
    def issue_text(self) -> str:
        return "；".join(self.issues)


#: 硬伤类：markdown / 列表符号。群聊里出现这些基本就是 AI 腔穿帮。
_MARKDOWN_BOLD = re.compile(r"\*\*|__")
_MARKDOWN_HEADER = re.compile(r"^\s*#{1,6}\s", re.MULTILINE)
_BULLET_LINE = re.compile(r"^\s*[-*•·]\s+", re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*\d+[.、)]\s*", re.MULTILINE)

#: 客服腔 / 总结腔：出现即算翻车，这些词真人打字几乎不会用。
_CUSTOMER_SERVICE = re.compile(
    r"建议你|建议您|值得注意的是|总而言之|综上所述|总的来说|"
    r"希望这能帮到你|希望对你有帮助|如果还有(其他|什么)问题|欢迎随时|"
    r"很抱歉给你带来|为您(服务|解答|提供)|作为一名|首先.{0,40}其次"
)

#: "不是A而是B"对照句式：最典型的机器痕迹之一（真人偶尔用，但机器人
#: 一用就整段都在用，先按硬伤拦）。
_CONTRAST_PATTERN = re.compile(r"不是[^，。？！\n]{1,16}[，,]?而是")


def quick_style_faults(text: str) -> list[str]:
    """确定性风格快检：返回问题清单，空列表 = 无硬伤。零成本，先跑这个。"""

    normalized = str(text or "").strip()
    if not normalized:
        return []
    faults: list[str] = []
    if _MARKDOWN_BOLD.search(normalized):
        faults.append("用了 markdown 加粗符号")
    if _MARKDOWN_HEADER.search(normalized):
        faults.append("用了 markdown 标题")
    if _BULLET_LINE.search(normalized):
        faults.append("用了列表符号分点")
    if _NUMBERED_LINE.search(normalized):
        faults.append("用了编号列表分点")
    match = _CUSTOMER_SERVICE.search(normalized)
    if match:
        faults.append(f"客服腔（\"{match.group(0)}\"）")
    if _CONTRAST_PATTERN.search(normalized):
        faults.append("\"不是A而是B\"的书面对照句式")
    return faults


def build_style_judge_prompt(*, bot_name: str, reply: str, proactive: bool) -> list[str]:
    kind = (
        "这是一条准备主动插进群聊的短消息"
        if proactive
        else "这是一条准备发到 QQ 群里的聊天回复"
    )
    return [
        f"System persona: You are a style gatekeeper for the QQ chat bot {bot_name}.",
        "Reply with exactly two lines in this grammar:\nSTYLE: pass|fail\nISSUES: <one short line, empty if pass>\nDo not add anything else.",
        "Fail only for clear AI tells: customer-service tone ( polite suggestions, well-wishing closers, lecturing ), markdown or bullet lists, hedging like '不是A而是B', or sounding like a written essay instead of casual chat. Do NOT fail for personality quirks, teasing, typos, slang, or short/curt replies — those are desired.",
        f"Message kind: {kind}.",
        "Reply text:",
        reply,
    ]


_STYLE_PATTERN = re.compile(r"STYLE\s*:\s*(pass|fail)", re.IGNORECASE)
_ISSUES_PATTERN = re.compile(r"ISSUES\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def parse_style_judge(text: str | None) -> StyleJudgeResult:
    if not text:
        return StyleJudgeResult(True, [])
    style_match = _STYLE_PATTERN.search(text)
    if style_match is None:
        return StyleJudgeResult(True, [])
    issues_match = _ISSUES_PATTERN.search(text)
    issues = [issues_match.group(1).strip()] if issues_match else []
    issues = [issue for issue in issues if issue]
    return StyleJudgeResult(style_match.group(1).lower() == "pass", issues)


def judge_reply_style(*, client, bot_name: str, reply: str, proactive: bool) -> StyleJudgeResult:
    """LLM 风格复审；任何传输/解析失败都降级为通过。"""

    try:
        raw = client.generate_text(
            build_style_judge_prompt(bot_name=bot_name, reply=reply, proactive=proactive)
        )
    except Exception:  # noqa: BLE001 - judge 挂了不能拦住回复
        return StyleJudgeResult(True, [])
    return parse_style_judge(raw)


def build_style_rewrite_prompt(
    *, prompt_lines: list[str], reply: str, issues: list[str], proactive: bool
) -> list[str]:
    issue_text = "；".join(issues) or "语气太像客服"
    shape = "一两句、短" if proactive else "像平时群聊那样"
    return [
        *prompt_lines,
        "----",
        f"你刚才按上面的要求写了一条回复：\n{reply}\n"
        f"但这条回复有 AI 腔，被风格检查拦下了：{issue_text}。",
        f"重写这条回复：意思和立场不变，{shape}，像真人在群里打字——"
        "可以不完整、可以没头没尾，但绝不能分点、不能用书面句式、不能总结或建议。"
        "只输出重写后的回复本身，不要解释。",
    ]
