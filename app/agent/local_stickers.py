"""Local sticker library: expose a mounted image directory as sendable artifacts.

The web image pipeline only accepts public URLs (see app/core/url_policy.py), so a
local meme folder can never enter through list_web_images/download_web_image.
This module provides the missing bridge: pick_sticker copies one file from the
sticker directory into the turn's artifact store, so the model can hand the
resulting artifact_id to send_group/send_private like any other image.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from uuid import uuid4

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.message_content import ImageAttachment

_IMAGE_SUFFIXES = frozenset({".webp", ".png", ".jpg", ".jpeg", ".gif"})
_MAX_STICKER_BYTES = 8 * 1024 * 1024


def _catalog(directory: Path) -> list[Path]:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    return sorted(
        entry for entry in entries
        if entry.suffix.lower() in _IMAGE_SUFFIXES
        and not entry.name.startswith(".")
        and entry.is_file()
    )


def _match_tags(path: Path, needle: str) -> bool:
    """学习库图的关键词在 <hash>.json 的 tags 或 meaning 里（编号库的标签在文件名里）。"""

    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return False
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        tags = str(data.get("tags", ""))
        meaning = str(data.get("meaning", "") or "")
    except (OSError, ValueError):
        return False
    return needle in tags.lower() or needle in meaning.lower()


def build_sticker_tools(*, runner, guard, store, directory: Path,
                        extra_directory: Path | None = None):
    directory = Path(directory)
    extra_directory = Path(extra_directory) if extra_directory else None

    def schema(properties, required):
        return {"type": "object", "properties": properties, "required": required,
                "additionalProperties": False}

    async def pick(args):
        guard.require_valid()
        files = _catalog(directory) + (_catalog(extra_directory) if extra_directory else [])
        if not files:
            return ToolOutcome(
                {"error": "sticker_library_unavailable",
                 "detail": "local sticker directory is missing or has no readable image"},
                "blocked")
        needle = (str(args.get("meme_id") or "").strip() or str(args.get("keyword") or "").strip()).lower()
        if needle:
            pool = [path for path in files
                    if needle in path.stem.lower() or _match_tags(path, needle)]
        else:
            pool = files
        if not pool:
            pool = files
        choice = random.choice(pool)

        def save() -> str:
            raw = choice.read_bytes()
            if not raw or len(raw) > _MAX_STICKER_BYTES:
                raise ValueError("sticker byte budget exceeded")
            store.root.mkdir(parents=True, exist_ok=True)
            staging = store.root / (".sticker-" + uuid4().hex + choice.suffix.lower())
            staging.write_bytes(raw)
            try:
                return store.add(staging, "local_sticker_library")
            finally:
                staging.unlink(missing_ok=True)

        try:
            artifact_id = await runner.run(save)
        except ValueError as exc:
            return ToolOutcome({"error": "sticker_not_sendable", "detail": str(exc)[:120]}, "blocked")
        guard.require_valid()
        path = await runner.run(store.get, artifact_id)
        images = ((ImageAttachment(url="", local_path=str(path), reference_subject=choice.stem),)
                  if path is not None else ())
        return ToolOutcome({
            "artifact_id": artifact_id,
            "meme_id": choice.stem,
            "library_size": len(files),
            "matched": len(pool),
            "origin": "local_sticker_library",
            "guidance": ("本地表情库中的真实图片。用 send_group/send_private 的 artifact_ids 发送这个 "
                         "artifact_id；一次一张，除非用户明确要求多发。发送成功以回执为准。"),
        }, images=images)

    return (
        AgentTool(
            "pick_sticker",
            "从本地表情库挑一张表情发出去。她是个表情包用得很大方的群友：心情到位、斗图、"
            "无语、得意、看戏、懒得打字，都可以直接甩图，不必等特殊时刻；对方发图过来时，"
            "首选回击一张（先 pick_sticker 取图再 send_group），纯文字接图是下策。"
            "唯一的约束是别一口气连发两张以上（斗图回合除外）。"
            "取到 artifact_id 后必须立刻用 send_group 的 artifact_ids 把它发出去，"
            "否则对方什么也看不到，只返回 artifact_id 等于没发。"
            "meme_id 按编号精确选，keyword 按文件名里的情绪标签筛选（如'无语''看戏''吃瓜'），都不传就随机一张。",
            schema({
                "meme_id": {"type": "string", "maxLength": 64},
                "keyword": {"type": "string", "maxLength": 32},
            }, []),
            pick,
            "image",
        ),
    )
