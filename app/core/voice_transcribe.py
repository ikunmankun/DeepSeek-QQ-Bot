"""语音转文字（ASR）：QQ 语音消息 → 文本，支持方言。

链路：
1. 取音频：语音消息 record segment 的 silk 原件在 napcat 容器里，
   ffmpeg 不认 SILK_V3，所以经 OneBot ``get_record``（out_format=mp3）
   让 napcat 自己转码，返回的路径在本容器可见（compose 把 napcat 的
   ntqq 数据卷和 cache 卷只读挂给了 bot，目标路径保持一致）。
2. 归一化：ffmpeg → 16k 单声道 wav。
3. 识别：讯飞语音听写（WebSocket 流式），``accent`` 支持方言
   （如 cantonese/henanese 等，见讯飞方言清单）；普通话默认。

凭证未配置时上层把整个功能隐藏（工具不注册），与识曲/TTS 同一约定。
接口错误抛 OSError，无语音返回空文本。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import formatdate
from pathlib import Path
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

#: 讯飞听写单次最大音频秒数（超过会被服务端拒绝，QQ 群语音上限 60s 覆盖得住）
MAX_AUDIO_SECONDS = 60
#: 每帧音频字节数（1280B @ 16k16bit ≈ 40ms，讯飞建议值）
FRAME_BYTES = 1280


@dataclass(slots=True)
class VoiceTranscript:
    text: str
    audio_seconds: float
    accent: str


# ---------- 音频取材与归一化 ----------


def extract_record_segment(raw_json: dict) -> dict | None:
    """从消息事件里取出第一个 record segment 的 data。"""
    for segment in (raw_json.get("message") or []):
        if isinstance(segment, dict) and segment.get("type") == "record":
            data = segment.get("data")
            return data if isinstance(data, dict) else None
    return None


async def fetch_record_mp3(sender, *, file_id: str, out_format: str = "mp3") -> Path:
    """经 OneBot get_record 让 napcat 把 silk 转成 mp3，返回容器内可见路径。"""
    result = await sender.call_api("get_record", {"file": file_id, "out_format": out_format})
    status = str(result.get("status") or "")
    if status != "ok":
        raise OSError(f"get_record failed: {result.get('message') or status}")
    file_value = str(result.get("data", {}).get("file") or "")
    if not file_value:
        raise OSError("get_record returned no file")
    if file_value.startswith("base64://"):
        payload = base64.b64decode(file_value.split("base64://", 1)[1])
        target = Path("/tmp") / f"voice-{abs(hash(file_value)) % 10**10}.{out_format}"
        target.write_bytes(payload)
        return target
    path = file_value.removeprefix("file://")
    resolved = Path(path)
    if not resolved.is_file():
        raise OSError(f"get_record file not visible in this container: {path}")
    return resolved


def to_wav16k_mono(source: Path, workdir: Path) -> Path:
    """ffmpeg 归一化成 16k 单声道 wav（讯飞要求格式）。"""
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / f"asr-{source.stem[:40]}-{int(workdir.stat().st_mtime)}.wav"
    run = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source),
         "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(target)],
        capture_output=True, text=True, timeout=60, check=False)
    if run.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        raise OSError("voice convert failed: " + run.stderr[-160:])
    return target


def wav_duration_seconds(wav: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav)], capture_output=True, text=True, timeout=30, check=False)
    try:
        return float(probe.stdout.strip() or 0)
    except ValueError:
        return 0.0


# ---------- 讯飞语音听写（WebSocket） ----------


def build_iat_ws_url(*, api_key: str, api_secret: str,
                     host: str = "iat-api.xfyun.cn", path: str = "/v2/iat") -> str:
    """讯飞 WS 鉴权：HMAC-SHA256 后 base64，拼成 query 参数（同 music_id 家族）。"""
    import hashlib
    import hmac

    date = formatdate(usegmt=True)
    origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode("utf-8"), origin.encode("utf-8"),
                 hashlib.sha256).digest()).decode("ascii")
    authorization = base64.b64encode(
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'.encode("ascii")
    ).decode("ascii")
    query = urlencode({"authorization": authorization, "date": date, "host": host})
    return f"wss://{host}{path}?{query}"


async def transcribe_iat(wav: Path, *, app_id: str, api_key: str, api_secret: str,
                         accent: str = "mandarin", language: str = "zh_cn",
                         timeout_seconds: float = 30.0) -> str:
    """讯飞语音听写：流式推 16k wav，收齐文本。"""
    import websockets

    url = build_iat_ws_url(api_key=api_key, api_secret=api_secret)
    audio = wav.read_bytes()
    frames = [audio[i:i + FRAME_BYTES] for i in range(0, len(audio), FRAME_BYTES)] or [b""]
    collected: list[str] = []
    async with websockets.connect(url, open_timeout=10) as ws:
        for index, frame in enumerate(frames):
            status = 0 if index == 0 else (2 if index == len(frames) - 1 else 1)
            await ws.send(json.dumps({
                "common": {"app_id": app_id},
                "business": {"language": language, "domain": "iat", "accent": accent,
                             "vad_eos": 10000},
                "data": {"status": status, "format": "audio/L16;rate=16000",
                         "encoding": "raw", "audio": base64.b64encode(frame).decode("ascii")},
            }, ensure_ascii=False))
            await asyncio.sleep(0.04)
        deadline = asyncio.get_running_loop().time() + max(5.0, timeout_seconds)
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            if int(reply.get("code") or 0) != 0:
                raise OSError(f"iat error {reply.get('code')}: {reply.get('message')}")
            data = reply.get("data") or {}
            result = data.get("result") or {}
            for ws_item in result.get("ws") or []:
                for cw in ws_item.get("cw") or []:
                    text = str(cw.get("w") or "")
                    if text:
                        collected.append(text)
            if int(data.get("status") or 0) == 2:
                return "".join(collected).strip()
    return "".join(collected).strip()


# ---------- 对外入口 ----------


def transcribe_sync(wav: Path, *, app_id: str, api_key: str, api_secret: str,
                    accent: str, language: str, timeout_seconds: float) -> VoiceTranscript:
    """阻塞版识别入口（调用方放 to_thread）。错误抛 OSError，无语音返回空文本。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    seconds = wav_duration_seconds(wav)
    if seconds <= 0:
        return VoiceTranscript(text="", audio_seconds=0.0, accent=accent)
    if seconds > MAX_AUDIO_SECONDS:
        raise OSError(f"voice too long: {seconds:.0f}s > {MAX_AUDIO_SECONDS}s")
    if loop is not None and loop.is_running():
        raise RuntimeError("transcribe_sync must not run inside a running loop")
    text = asyncio.run(transcribe_iat(
        wav, app_id=app_id, api_key=api_key, api_secret=api_secret,
        accent=accent, language=language, timeout_seconds=timeout_seconds))
    return VoiceTranscript(text=text, audio_seconds=seconds, accent=accent)


def format_for_chat(transcript: VoiceTranscript, speaker: str) -> str:
    """给 agent 看的转写结果格式。"""
    label = {"mandarin": "普通话"}.get(transcript.accent, transcript.accent)
    head = f"{speaker}的语音（约{transcript.audio_seconds:.0f}秒，{label}转写）：" if speaker else "语音转写："
    text = transcript.text or "（没有听出内容）"
    return head + text
