"""戳一戳反应与新人欢迎：notice 事件驱动的真人感互动。

设计原则：
* 反应用一次轻量 LLM 调用按人设现说，不预置台词池（每句都不一样才像人）；
* 冷却与每日上限管住频率——被连环戳最多一天回应十次、每群每分钟一次；
* 戳回来是随机彩蛋（30%），不是固定动作；
* 欢迎新人每次进群都欢迎，但一句话就走，不追着新人聊。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

DAILY_POKE_REACTION_CAP = 10
PER_GROUP_POKE_COOLDOWN_SECONDS = 60.0
POKE_BACK_PROBABILITY = 0.3
WELCOME_COOLDOWN_SECONDS = 30.0


class InteractionState:
    """进程内冷却状态（重启清零可接受：冷却本来就是软约束）。"""

    def __init__(self) -> None:
        self.last_poke_react: dict[int, datetime] = {}
        self.poke_react_day: dict[int, tuple[str, int]] = {}
        self.last_welcome: dict[int, datetime] = {}

    def poke_allowed(self, group_id: int, now: datetime) -> bool:
        last = self.last_poke_react.get(group_id)
        if last is not None and (now - last).total_seconds() < PER_GROUP_POKE_COOLDOWN_SECONDS:
            return False
        today = f"{now:%Y-%m-%d}"
        _day, count = self.poke_react_day.get(group_id, (today, 0))
        if _day != today:
            count = 0
        if count >= DAILY_POKE_REACTION_CAP:
            return False
        return True

    def record_poke_react(self, group_id: int, now: datetime) -> None:
        self.last_poke_react[group_id] = now
        today = f"{now:%Y-%m-%d}"
        _day, count = self.poke_react_day.get(group_id, (today, 0))
        self.poke_react_day[group_id] = (today, count + 1 if _day == today else 1)

    def welcome_allowed(self, group_id: int, now: datetime) -> bool:
        last = self.last_welcome.get(group_id)
        return last is None or (now - last).total_seconds() >= WELCOME_COOLDOWN_SECONDS

    def record_welcome(self, group_id: int, now: datetime) -> None:
        self.last_welcome[group_id] = now


def poke_payload_info(payload: dict, *, bot_qq: int) -> tuple[int, int] | None:
    """从 notice payload 里认出"有人戳了bot"：返回 (group_id, poker_user_id)。"""

    if str(payload.get("post_type", "")) != "notice" or str(payload.get("notice_type", "")) != "notify":
        return None
    if str(payload.get("sub_type", "")) != "poke":
        return None
    group_id = int(payload.get("group_id") or 0)
    target_id = int(payload.get("target_id") or 0)
    user_id = int(payload.get("user_id") or 0)
    if not group_id or not user_id or target_id != bot_qq or user_id == bot_qq:
        return None
    return group_id, user_id


def group_increase_info(payload: dict, *, bot_qq: int) -> tuple[int, int] | None:
    """从 notice payload 里认出"有新人进群"：返回 (group_id, new_user_id)。"""

    if str(payload.get("post_type", "")) != "notice":
        return None
    if str(payload.get("notice_type", "")) != "group_increase":
        return None
    group_id = int(payload.get("group_id") or 0)
    user_id = int(payload.get("user_id") or 0)
    if not group_id or not user_id or user_id == bot_qq:
        return None
    sub_type = str(payload.get("sub_type", "approve"))
    if sub_type not in {"approve", "invite"}:
        return None
    return group_id, user_id


def build_poke_prompt(*, persona_text: str, poker_name: str | None = None,
                      relation_hint: str | None = None,
                      recent_lines: Sequence[str] = ()) -> list[str]:
    lines = ["System persona: " + persona_text.strip(),
        "Group policy: 有人在QQ群里戳了戳你（戳一戳，想引起你注意，不是消息）。"
        "用你的语气回一句话，最多二十个字：可以怼、可以好奇、可以装没感觉。"
        "不要描述动作，不要说'你戳我'之外的旁白，直接说话。"
        "回复里绝对不要出现QQ号码。"]
    if poker_name:
        lines.append(f"戳你的人：{poker_name}。")
    else:
        lines.append("戳你的人：一位群友（查不到名片，不知道是谁，别编）。")
    if relation_hint:
        lines.append(f"你们的关系：{relation_hint}。")
    if recent_lines:
        lines.append("最近的群聊（只作背景，不用逐条回应）：\n" + "\n".join(recent_lines[-3:]))
    return lines


def build_welcome_prompt(*, persona_text: str) -> list[str]:
    return ["System persona: " + persona_text.strip(),
        "Group policy: 有个新人刚进群。用你的语气说一句话欢迎：热情、带点损、"
        "可以问一句从哪来的。最多二十五字，像老群友随口招呼，不要像迎宾机器人。"
        "不要描写动作，回复里不要出现QQ号。"]


def should_poke_back(now_seed: float) -> bool:
    import random

    return random.random() < POKE_BACK_PROBABILITY
