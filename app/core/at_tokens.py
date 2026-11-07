"""[@名字]/[@qq:号] 记号 ↔ 真正的 at 段。

两层设计（与小黄脸 token 同一套思路）：
* ``resolve_at_names``——解析层：``[@名字]`` 用群成员身份库解析成
  ``[@qq:QQ号]``；解析不了（陌生/歧义）就降级成纯名字文本；
* ``split_at_segments``——传输层：``[@qq:N]`` 拆成 OneBot 的 at 段与文本段
  交替。Sender 的 ``_message_field`` 调用它，同时负责把 CQ 控制段字符
  全角化（防注入：模型的 at 只能走这条 token 通道）。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from app.core.member_identity import resolve_group_member_reference

_AT_QQ_TOKEN = re.compile(r"\[@qq:(\d{5,20})\]")
_AT_NAME_TOKEN = re.compile(r"\[@([^\[\]@:]{1,32})\]")


def split_at_segments(text: str) -> list[dict[str, Any]] | None:
    """把 ``[@qq:N]`` 记号拆成 OneBot 段（at + 文本交替）。

    没有记号时返回 None，调用方按原样发送，行为不变。
    """

    if not _AT_QQ_TOKEN.search(text):
        return None
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in _AT_QQ_TOKEN.finditer(text):
        prefix = text[cursor:match.start()]
        if prefix:
            segments.append({"type": "text", "data": {"text": prefix}})
        segments.append({"type": "at", "data": {"qq": match.group(1)}})
        cursor = match.end()
    tail = text[cursor:]
    if tail:
        segments.append({"type": "text", "data": {"text": tail}})
    return segments


def resolve_at_names(text: str, *, members: Sequence[Any]) -> str:
    """把 ``[@名字]`` 解析成 ``[@qq:N]``。

    解析不了（没这个人 / 多个人重名歧义）就降级为纯名字文本——宁可丢一个
    @，也不能把记号原样发出去或 @ 错人。
    """

    if "[@" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if not name:
            return match.group(0)
        resolved = resolve_group_member_reference(name, members, match_mode="contained")
        if resolved is not None:
            return f"[@qq:{resolved.user_id}]"
        return name

    return _AT_NAME_TOKEN.sub(replace, text)


def strip_at_tokens(text: str) -> str:
    """降级用：把所有 at 记号还原成纯文本（丢 @ 保正文）。"""

    text = _AT_NAME_TOKEN.sub(r"\1", text)
    return _AT_QQ_TOKEN.sub("", text)
