"""字幕路线：让机器人"看懂"B站视频的内容。

B站大量视频有 CC 字幕或 AI 字幕。路线：yt-dlp（带登录 cookie）拉字幕
→ 解析成纯文本 → 喂给模型，它就真的"看过"这个视频了。

cookie 来源：扫码登录后，浏览器 context 的 bilibili cookies 被导出为
data/bili_cookies.json；本模块按需转成 yt-dlp 的 Netscape 格式。
没有 cookie 时字幕拉取返回 None，并提示需要登录。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

logger = __import__("logging").getLogger(__name__)

COOKIE_FILE_NAME = "bili_cookies.json"
SUB_LANGS = "zh-CN,zh-Hans,zh"
MAX_SUBTITLE_CHARS = 6000


def cookie_json_path(settings) -> Path:
    return Path(settings.data_dir) / COOKIE_FILE_NAME


def cookies_available(settings) -> bool:
    path = cookie_json_path(settings)
    return path.exists() and path.stat().st_size > 10


def save_cookies_from_browser(cookies: list[dict], settings) -> int:
    """把 playwright context.cookies() 里的 bilibili cookie 存盘（登录后调用）。"""

    keep = [c for c in cookies if "bilibili" in str(c.get("domain", ""))]
    if not keep:
        return 0
    path = cookie_json_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    return len(keep)


def _netscape_cookie_file(settings, workdir: Path) -> Path | None:
    """把 JSON cookie 转成 yt-dlp 需要的 Netscape 格式文件。"""

    path = cookie_json_path(settings)
    if not cookies_available(settings):
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    lines = ["# Netscape HTTP Cookie File"]
    for c in raw:
        domain = str(c.get("domain", "")).lstrip(".")
        expiry = int(float(c.get("expires", 0) or 0)) or 2147483647
        secure = "TRUE" if c.get("secure") else "FALSE"
        lines.append("\t".join([
            domain if domain.startswith(".") else f".{domain}",
            "TRUE", str(c.get("path", "/")), secure, str(expiry),
            str(c.get("name", "")), str(c.get("value", "")),
        ]))
    out = workdir / "bili_cookies.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _subtitle_text_from_file(path: Path) -> str:
    """json3 / vtt 字幕文件 → 纯文本（去时间轴，按顺序拼接）。"""

    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json3" or raw.lstrip().startswith("{"):
        try:
            payload = json.loads(raw)
            lines = []
            for event in payload.get("body", []):
                line = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
                if line.strip():
                    lines.append(line.strip())
            return "\n".join(lines)
        except ValueError:
            return ""
    # vtt 兜底：跳过时间轴行
    lines = []
    for line in raw.splitlines():
        if "-->" in line or line.strip() in {"", "WEBVTT"} or line.startswith(("NOTE", "Kind:", "Language:")):
            continue
        if line.strip().isdigit():
            continue
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def fetch_subtitle_text(url: str, *, settings) -> tuple[str, str] | None:
    """拉一个视频的字幕文本。返回 (文本, 来源说明)；没有字幕/无登录返回 None。"""

    if not cookies_available(settings):
        return None
    workdir = Path(settings.data_dir) / "bili_subtitle_work"
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.iterdir():
        old.unlink(missing_ok=True)
    cookie_file = _netscape_cookie_file(settings, workdir)
    if cookie_file is None:
        return None
    cmd = [
        "yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
        "--sub-langs", SUB_LANGS, "--sub-format", "json3/vtt",
        "--no-playlist", "--cookies", str(cookie_file),
        "-o", str(workdir / "%(id)s"), url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("subtitle_ytdlp_failed error=%s", type(exc).__name__)
        return None
    subtitle_files = [p for p in sorted(workdir.iterdir())
                      if p.suffix in {".json3", ".vtt"}]
    for path in subtitle_files:
        text = _subtitle_text_from_file(path)
        if text.strip():
            return text[:MAX_SUBTITLE_CHARS], f"字幕文件 {path.name}"
    return None
