"""语音（TTS）：把模型想说的话变成 QQ 语音条。

提供商可插拔（TTS_PROVIDER，默认 fishaudio）：generate_voice(text) -> mp3
文件路径。未配置 TTS_API_KEY 时整个功能自动隐藏（工具不注册、worker 不建）。

发送端早已就绪：Sender.send_group_voice / send_private_voice（record 段，
NapCat 负责转码 silk）。配额：TTS_DAILY_LIMIT（默认 20 条/天）防刷爆账单。
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 300


class TtsError(RuntimeError):
    pass


class FishAudioTts:
    """FishAudio REST TTS：POST /v1/tts 返回 mp3 流。"""

    def __init__(self, *, api_key: str, api_url: str = "https://api.fish.audio",
                 voice_id: str = "") -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.voice_id = voice_id

    def synthesize(self, text: str, workdir: Path) -> Path:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            response = client.post(
                f"{self.api_url}/v1/tts",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"text": text, "format": "mp3",
                      **({"reference_id": self.voice_id} if self.voice_id else {})},
            )
            response.raise_for_status()
            out = workdir / f"tts_{abs(hash(text)) % 10**10}.mp3"
            out.write_bytes(response.content)
            if out.stat().st_size < 1000:
                raise TtsError("tts_audio_too_small")
            return out


class MiMoTts:
    """小米 MiMo TTS：OpenAI 兼容协议（chat completions 出音频）。

    user 消息控制语气风格，assistant 消息是要合成的正文；音频以 base64
    放在 choices[0].message.audio.data。
    """

    def __init__(self, *, api_key: str,
                 api_url: str = "https://api.xiaomimimo.com",
                 model: str = "mimo-v2.6-tts", voice: str = "mimo_default",
                 style: str = "活泼俏皮的少女语气，语速稍快",
                 speed: float = 1.05) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.model = model or "mimo-v2.6-tts"
        self.voice = voice or "mimo_default"
        self.style = style
        self.speed = max(0.5, min(2.0, float(speed or 1.0)))

    def synthesize(self, text: str, workdir: Path) -> Path:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            response = client.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": self.style},
                        {"role": "assistant", "content": text},
                    ],
                    "audio": {"format": "mp3", "voice": self.voice,
                              "speed": self.speed},
                },
            )
            response.raise_for_status()
            payload = response.json()
            message = ((payload.get("choices") or [{}])[0].get("message") or {})
            audio_b64 = str((message.get("audio") or {}).get("data", ""))
            if not audio_b64:
                raise TtsError("mimo_empty_audio")
            out = workdir / f"tts_{abs(hash(text)) % 10**10}.mp3"
            workdir.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(audio_b64))
            if out.stat().st_size < 1000:
                raise TtsError("tts_audio_too_small")
            return out


class MiniMaxTts:
    """MiniMax T2A v2：POST /v1/t2a_v2?GroupId=...，返回 hex 编码的 mp3。

    需要 TTS_API_KEY（Bearer 密钥）和 TTS_GROUP_ID（账户 GroupId，
    在 MiniMax 开放平台密钥管理页能看到）。
    """

    def __init__(self, *, api_key: str, group_id: str,
                 api_url: str = "https://api.minimax.chat",
                 voice_id: str = "female-shaonv", model: str = "speech-02-turbo",
                 speed: float = 1.0) -> None:
        self.api_key = api_key
        self.group_id = group_id
        self.api_url = api_url.rstrip("/")
        self.voice_id = voice_id or "female-shaonv"
        self.model = model or "speech-02-turbo"
        self.speed = max(0.5, min(2.0, float(speed or 1.0)))

    def list_voices(self) -> list[dict]:
        """拉全量系统音色列表（挑音色用）。"""
        with httpx.Client(timeout=15.0, trust_env=False) as client:
            response = client.get(
                f"{self.api_url}/v1/get_voice",
                params={"GroupId": self.group_id},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            if int((payload.get("base_resp") or {}).get("status_code", 0)) != 0:
                return []
            voices = []
            for item in payload.get("system_voice") or []:
                voices.append({"voice_id": item.get("voice_id"),
                               "name": item.get("voice_name"),
                               "desc": (item.get("description") or "")[:60]})
            return voices

    def synthesize(self, text: str, workdir: Path) -> Path:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            response = client.post(
                f"{self.api_url}/v1/t2a_v2",
                params={"GroupId": self.group_id},
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "text": text,
                    "stream": False,
                    "voice_setting": {"voice_id": self.voice_id,
                                       "speed": self.speed,
                                       "vol": 1.0, "pitch": 0},
                    "audio_setting": {"sample_rate": 32000, "bitrate": 128000,
                                      "format": "mp3", "channel": 1},
                },
            )
            response.raise_for_status()
            payload = response.json()
            base = payload.get("base_resp") or {}
            if int(base.get("status_code", 0)) != 0:
                raise TtsError(f"minimax_error: {base.get('status_msg', 'unknown')}")
            audio_hex = str((payload.get("data") or {}).get("audio", ""))
            if not audio_hex:
                raise TtsError("minimax_empty_audio")
            out = workdir / f"tts_{abs(hash(text)) % 10**10}.mp3"
            out.write_bytes(bytes.fromhex(audio_hex))
            if out.stat().st_size < 1000:
                raise TtsError("tts_audio_too_small")
            return out


def build_tts(settings) -> Any | None:
    """按配置构造 TTS 适配器；未配置返回 None（功能自动隐藏）。"""
    api_key = str(getattr(settings, "tts_api_key", "") or "").strip()
    if not api_key:
        return None
    provider = str(getattr(settings, "tts_provider", "fishaudio") or "").strip().lower()
    if provider == "mimo":
        api_url = str(getattr(settings, "tts_api_url", "") or "").strip()
        if not api_url or "fish" in api_url:
            api_url = "https://api.xiaomimimo.com"
        return MiMoTts(
            api_key=api_key, api_url=api_url,
            model=str(getattr(settings, "tts_model", "") or "").strip() or "mimo-v2.6-tts",
            voice=str(getattr(settings, "tts_voice", "") or "").strip() or "mimo_default",
            style=str(getattr(settings, "tts_style", "") or "").strip(),
            speed=float(getattr(settings, "tts_speed", 1.05) or 1.05),
        )
    if provider == "fishaudio":
        return FishAudioTts(
            api_key=api_key,
            api_url=str(getattr(settings, "tts_api_url", "https://api.fish.audio") or ""),
            voice_id=str(getattr(settings, "tts_voice", "") or "").strip(),
        )
    if provider == "minimax":
        return MiniMaxTts(
            api_key=api_key,
            group_id=str(getattr(settings, "tts_group_id", "") or "").strip(),
            api_url=str(getattr(settings, "tts_api_url", "https://api.minimax.chat") or ""),
            voice_id=str(getattr(settings, "tts_voice", "") or "").strip(),
            model=str(getattr(settings, "tts_model", "speech-02-turbo") or "").strip(),
            speed=float(getattr(settings, "tts_speed", 1.0) or 1.0),
        )
    logger.warning("unknown_tts_provider provider=%s", provider)
    return None
