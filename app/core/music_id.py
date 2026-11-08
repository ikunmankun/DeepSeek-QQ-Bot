"""听歌识曲：音频指纹识别（主）+ B站字幕歌词兜底。

路线A（主）：视频/音频 → ffmpeg 从中段切单声道片段 → 指纹 API → 歌名/歌手。
    识别源两个可选（MUSIC_ID_PROVIDER）：
    * audd（默认）：https://api.audd.io/，单 token 鉴权，HTTP multipart。
    * xfyun：讯飞「歌曲识别ACRCloud」WebAPI，appid/api_key/api_secret HMAC 签名，
      mode=music 支持原声 BGM（原生"歌曲识别"只有哼唱 afs，别拿它识别原声）。
路线C（兜底）：B站 AI 字幕常把唱词转成文字 → LLM 挑出像唱词的短句 → 网页搜歌词 → 歌名。

识别源是外部限额服务：凭证未配置时上层会把整个功能隐藏
（工具不注册、服务不建），与 voice.py 的 TTS 开关约定一致。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import subprocess
from dataclasses import dataclass
from email.utils import formatdate
from pathlib import Path
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

AUDIO_CLIP_SECONDS = 15.0
_MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024
DEFAULT_AUDD_URL = "https://api.audd.io/"
#: 讯飞「歌曲识别ACRCloud」音乐识别（原声）端点；哼唱版是 s9884ba49。
XFYUN_MUSIC_URL = "https://cn-east-1.api.xf-yun.com/v1/private/s29ebee0d"


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
    """yt-dlp 取音轨；B站链接自动带登录 cookie。返回本地文件。

    --proxy "" 显式禁代理：容器 HTTP_PROXY 指向的 mihomo 一旦挂掉，
    国内站（B站）也会被它截胡到超时（2026-11-08 实测），直连才是活的。
    """
    from app.core.bili_subtitle import _netscape_cookie_file
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "-f", "bestaudio/best", "--no-playlist",
           "--proxy", "",
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
    """audD 指纹识别。无匹配返回 None；鉴权/接口错误抛 OSError。

    trust_env=False 直连：容器代理环境变量对 audd.io 是死路（2026-11-08 实测
    直连 2s 成功、走代理 15s 超时），与 bili 登录"国内站绕代理"同一课。
    """
    with audio.open("rb") as stream:
        response = httpx.post(
            api_url,
            data={"api_token": api_token, "return": "timecode"},
            files={"file": (audio.name, stream, "audio/mpeg")},
            timeout=timeout,
            trust_env=False,
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


def xfyun_authorization(api_key: str, api_secret: str, url: str, date: str) -> str:
    """讯飞 WebAPI 鉴权头：HMAC-SHA256 签名后整体再 base64（官方拼接格式）。"""
    parsed = urlsplit(url)
    origin = f"host: {parsed.netloc}\ndate: {date}\nPOST {parsed.path} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode("utf-8"), origin.encode("utf-8"), hashlib.sha256).digest()
    ).decode("ascii")
    auth_origin = (f'api_key="{api_key}",algorithm="hmac-sha256",'
                   f'headers="host date request-line",signature="{signature}"')
    return base64.b64encode(auth_origin.encode("utf-8")).decode("ascii")


def identify_by_fingerprint_xfyun(audio: Path, *, app_id: str, api_key: str,
                                  api_secret: str, api_url: str = XFYUN_MUSIC_URL,
                                  timeout: float = 40.0) -> MusicMatch | None:
    """讯飞音乐识别（ACRCloud 版，mode=music）。无匹配返回 None；鉴权/接口错误抛 OSError。"""
    date = formatdate(usegmt=True)
    headers = {
        "Authorization": xfyun_authorization(api_key, api_secret, api_url, date),
        "Date": date,
        "Host": urlsplit(api_url).netloc,
        "Content-Type": "application/json",
    }
    request = {
        "header": {"app_id": app_id, "status": 3},
        "parameter": {"acr_music": {"mode": "music",
                                    "output_text": {"encoding": "utf8", "compress": "raw",
                                                    "format": "json"}}},
        "payload": {"data": {"encoding": "lame", "sample_rate": 16000, "channels": 1,
                             "bit_depth": 16, "status": 3,
                             "audio": base64.b64encode(audio.read_bytes()).decode("ascii"),
                             "frame_size": 0}},
    }
    response = httpx.post(api_url, headers=headers, json=request, timeout=timeout,
                          trust_env=False)
    response.raise_for_status()
    payload = response.json()
    header = payload.get("header") or {}
    if int(header.get("code", -1)) != 0:
        raise OSError(f"xfyun_error:{header.get('message')}"[:200])
    text = str(((payload.get("payload") or {}).get("output_text") or {}).get("text") or "")
    if not text:
        return None
    decoded = json.loads(base64.b64decode(text).decode("utf-8"))
    items = (decoded.get("metadata") or {}).get("music") or []
    if not items:
        return None
    items = sorted(items, key=lambda item: float(item.get("score") or 0), reverse=True)
    top = items[0]
    artists = "、".join(str(a.get("name") or "").strip()
                        for a in (top.get("artists") or []) if a.get("name"))
    return MusicMatch(
        title=str(top.get("title") or "").strip(),
        artist=artists,
        album=str((top.get("album") or {}).get("name") or "").strip(),
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
