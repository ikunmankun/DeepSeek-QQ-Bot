"""听歌识曲：音频指纹识别（主）+ B站字幕歌词兜底。

路线A（主）：视频/音频 → ffmpeg 从中段切单声道片段 → audD 指纹 API → 歌名/歌手/链接。
路线C（兜底）：B站 AI 字幕常把唱词转成文字 → LLM 挑出像唱词的短句 → 网页搜歌词 → 歌名。

识别源是外部限额服务：未配置 MUSIC_ID_API_TOKEN 时上层会把整个功能隐藏
（工具不注册、服务不建），与 voice.py 的 TTS 开关约定一致。
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

AUDIO_CLIP_SECONDS = 15.0
_MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024
DEFAULT_AUDD_URL = "https://api.audd.io/"


@dataclass(slots=True)
class MusicMatch:
    title: str
    artist: str = ""
    album: str = ""
    link: str = ""
    #: fingerprint=指纹实锤；lyrics=字幕歌词推断（低置信度）。
    source: str = "fingerprint"
    timecode: str = ""


def extract_audio_clip(source: Path, workdir: Path,
                       *, seconds: float = AUDIO_CLIP_SECONDS) -> Path:
    """ffmpeg 从音频中段切一个降码率单声道片段（指纹识别够用且省流量）。"""
    workdir.mkdir(parents=True, exist_ok=True)
    duration = 0.0
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(source)],
            capture_output=True, text=True, timeout=30, check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            duration = float(json.loads(probe.stdout)["format"]["duration"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        duration = 0.0
    offset = max(0.0, (duration - seconds) / 2.0) if duration > seconds else 0.0
    clip = workdir / f"clip-{source.stem}-{int(seconds)}.mp3"
    run = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.2f}", "-t", f"{seconds:.2f}",
         "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
         "-acodec", "libmp3lame", "-b:a", "96k", str(clip)],
        capture_output=True, text=True, timeout=60, check=False)
    if run.returncode != 0 or not clip.exists() or clip.stat().st_size == 0:
        raise ValueError("audio_clip_failed")
    return clip


def download_audio(url: str, workdir: Path, *, settings, timeout: int = 150) -> Path:
    """yt-dlp 取音轨；B站链接自动带登录 cookie。返回本地文件。"""
    from app.core.bili_subtitle import _netscape_cookie_file
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "-f", "bestaudio/best", "--no-playlist",
           "--max-filesize", str(_MAX_DOWNLOAD_BYTES),
           "-o", str(workdir / "music-src.%(ext)s")]
    cookie = _netscape_cookie_file(settings, workdir)
    if cookie is not None:
        cmd += ["--cookies", str(cookie)]
    cmd.append(url)
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    produced = sorted(p for p in workdir.iterdir()
                      if p.name.startswith("music-src.") and p.is_file())
    if run.returncode != 0 or not produced:
        logger.info("music_download_failed url=%s rc=%s", url[:120], run.returncode)
        raise ValueError("audio_download_failed")
    return produced[0]


def identify_by_fingerprint(audio: Path, *, api_token: str,
                            api_url: str = DEFAULT_AUDD_URL,
                            timeout: float = 40.0) -> MusicMatch | None:
    """audD 指纹识别。无匹配返回 None；鉴权/接口错误抛 OSError。"""
    with audio.open("rb") as stream:
        response = httpx.post(
            api_url,
            data={"api_token": api_token, "return": "timecode"},
            files={"file": (audio.name, stream, "audio/mpeg")},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("status")) != "success":
        err = payload.get("error") or {}
        raise OSError(f"audd_error:{err.get('error_message') or payload.get('status')}"[:200])
    result = payload.get("result")
    if not isinstance(result, dict) or not str(result.get("title") or "").strip():
        return None
    return MusicMatch(
        title=str(result.get("title") or "").strip(),
        artist=str(result.get("artist") or "").strip(),
        album=str(result.get("album") or "").strip(),
        link=str(result.get("link") or "").strip(),
        timecode=str(result.get("timecode") or "").strip(),
        source="fingerprint",
    )


_LYRIC_PROMPT = (
    "下面是一段视频的字幕文本。只挑出【像歌词】的句子（人物说话的口播不算），"
    "每行一句原样摘出，最多5行，不要翻译不要解释。一句都不像就只输出：无"
)


def guess_from_lyrics(subtitle_text: str, *, llm, web_search_client) -> MusicMatch | None:
    """字幕歌词兜底：LLM 挑唱词 → 搜歌词 → 取最可信命中。低置信度来源。"""
    lines = [ln.strip() for ln in subtitle_text.splitlines() if len(ln.strip()) >= 4]
    if not lines:
        return None
    try:
        raw = str(llm.generate_text([_LYRIC_PROMPT, "\n".join(lines[:80])]) or "").strip()
    except Exception as exc:
        logger.info("music_lyric_extract_failed error=%s", type(exc).__name__)
        return None
    candidates = [ln.strip("-•\t 、") for ln in raw.splitlines() if 4 <= len(ln.strip()) <= 40][:2]
    if not candidates:
        return None
    try:
        hits = web_search_client.search("歌词 " + " ".join(candidates), max_results=5)
    except Exception as exc:
        logger.info("music_lyric_search_failed error=%s", type(exc).__name__)
        return None
    for hit in hits or ():
        title = str(getattr(hit, "title", "") or "").strip()
        url = str(getattr(hit, "url", "") or "").strip()
        if not title or not url:
            continue
        lowered = url.lower()
        if any(site in lowered for site in ("music.163.com", "y.qq.com", "kugou.com",
                                            "kuwo.cn", "bilibili.com", "moegirl.org.cn",
                                            "baike.baidu.com", "wikipedia.org")):
            clean = title.split("_")[0].split("-")[0].strip()
            if clean:
                return MusicMatch(title=clean, link=url, source="lyrics")
    return None
