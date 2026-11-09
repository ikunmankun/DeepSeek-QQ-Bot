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


def _match_tags(path: Path, needle: str, meanings_dir: Path | None = None) -> bool:
    """学习库图的关键词在 <hash>.json 的 tags/meaning 里；memes 编号库的关键词
    在文件名 + data/meme_meanings/<编号>.json 的 tags/meaning 里。"""
    for sidecar in (path.with_suffix(".json"),
                    None if meanings_dir is None else meanings_dir / (path.stem + ".json")):
        if sidecar is None or not sidecar.exists():
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            tags = str(data.get("tags", ""))
            meaning = str(data.get("meaning", "") or "")
        except (OSError, ValueError):
            continue
        if needle in tags.lower() or needle in meaning.lower():
            return True
    return False


def build_sticker_tools(*, runner, guard, store, directory: Path,
                        extra_directory: Path | None = None,
                        meanings_dir: Path | None = None):
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
                    if needle in path.stem.lower() or _match_tags(path, needle, meanings_dir)]
            if not pool:
                # 本机补丁：匹配不到不再静默随机兜底——模型以为发了对的表情，
                # 实际发的是无关图（"用错表情包"的直接病灶）。报错让它换词或明说随机。
                return ToolOutcome({"error": "keyword_no_match", "keyword": needle,
                    "guidance": "表情库没有匹配这个关键词的图：换个情绪词/梗名再试（标签和梗含义都参与匹配），"
                                "或明确改用不传参数的随机取图。"}, "blocked")
        else:
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
            "回应纯表情时可以直接只发图、不配文字。"
            "唯一的约束是别一口气连发两张以上（斗图回合除外）。"
            "取到 artifact_id 后必须立刻用 send_group 的 artifact_ids 把它发出去，"
            "否则对方什么也看不到，只返回 artifact_id 等于没发。"
            "meme_id 按编号精确选；keyword 按文件名情绪标签和梗含义筛选（如'无语''看戏''吃瓜'），"
            "匹配不到会报 keyword_no_match（换词重试或干脆不传参数随机取一张）。",
            schema({
                "meme_id": {"type": "string", "maxLength": 64},
                "keyword": {"type": "string", "maxLength": 32},
            }, []),
            pick,
            "image",
        ),
    )
