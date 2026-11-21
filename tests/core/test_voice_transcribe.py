"""语音转文字单元：record 解析、音频归一化、讯飞 WS 鉴权与流式协议、错误约定。"""
import base64
import hashlib
import hmac
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from app.core.voice_transcribe import (
    extract_record_segment, build_iat_ws_url, fetch_record_mp3,
    to_wav16k_mono, transcribe_sync,
)


def test_extract_record_segment():
    event = {"message": [
        {"type": "at", "data": {"qq": "123"}},
        {"type": "record", "data": {"file": "abc-uuid", "path": "/app/.config/QQ/x"}},
        {"type": "text", "data": {"text": "嗯"}},
    ]}
    segment = extract_record_segment(event)
    assert segment["file"] == "abc-uuid"
    assert extract_record_segment({"message": [{"type": "text", "data": {"text": "没语音"}}]}) is None


def test_build_iat_ws_url_signature():
    url = build_iat_ws_url(api_key="key-123", api_secret="secret-456")
    parsed = urlsplit(url)
    assert parsed.scheme == "wss" and parsed.hostname == "iat-api.xfyun.cn" and parsed.path == "/v2/iat"
    query = parse_qs(parsed.query)
    date = query["date"][0]
    host = query["host"][0]
    # 重算签名并比对，确保 HMAC 拼接格式与讯飞一致
    origin = f"host: {host}\ndate: {date}\nGET /v2/iat HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(b"secret-456", origin.encode(), hashlib.sha256).digest()).decode()
    assert query["authorization"][0] == base64.b64encode(
        f'api_key="key-123", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'.encode()
    ).decode()


def test_fetch_record_mp3_decodes_base64(tmp_path):
    class FakeSender:
        def __init__(self, payload):
            self.payload = payload
            self.action = None

        async def call_api(self, action, params):
            self.action = action
            return self.payload

    sender = FakeSender({"status": "ok", "data": {"file": "base64://SGVsbG8="}})
    target = Path(asyncio_run_fetch(sender))
    assert target.read_bytes() == b"Hello"


def asyncio_run_fetch(sender):
    import asyncio
    return asyncio.run(fetch_record_mp3(sender, file_id="uuid-1", out_format="mp3"))


def test_fetch_record_mp3_error_raises(tmp_path):
    import asyncio

    from app.core.voice_transcribe import fetch_record_mp3

    class FakeSender:
        async def call_api(self, action, params):
            return {"status": "failed", "message": "file not found"}

    with pytest.raises(OSError):
        asyncio.run(fetch_record_mp3(FakeSender(), file_id="gone", out_format="mp3"))


def test_transcribe_sync_short_circuits_without_network(tmp_path, monkeypatch):
    import app.core.voice_transcribe as vt

    # 空音频（0 秒）：直接返回空文本，不发起任何网络请求
    silent = tmp_path / "silent.wav"
    silent.write_bytes(b"")
    monkeypatch.setattr(vt, "wav_duration_seconds", lambda p: 0.0)
    transcript = vt.transcribe_sync(silent, app_id="a", api_key="k", api_secret="s",
                                    accent="mandarin", language="zh_cn", timeout_seconds=5)
    assert transcript.text == "" and transcript.audio_seconds == 0.0

    # 超长语音：抛 OSError
    monkeypatch.setattr(vt, "wav_duration_seconds", lambda p: 90.0)
    with pytest.raises(OSError):
        vt.transcribe_sync(silent, app_id="a", api_key="k", api_secret="s",
                           accent="mandarin", language="zh_cn", timeout_seconds=5)

    # 正常：走（被替换掉的）iat 客户端
    monkeypatch.setattr(vt, "wav_duration_seconds", lambda p: 12.0)
    async def fake_iat(wav, **kwargs):
        return "转出来的话"
    monkeypatch.setattr(vt, "transcribe_iat", fake_iat)
    transcript = vt.transcribe_sync(silent, app_id="a", api_key="k", api_secret="s",
                                    accent="cantonese", language="zh_cn", timeout_seconds=5)
    assert transcript.text == "转出来的话" and transcript.accent == "cantonese"


def test_format_for_chat():
    from app.core.voice_transcribe import VoiceTranscript, format_for_chat

    transcript = VoiceTranscript(text="说了句方言", audio_seconds=23.0, accent="cantonese")
    text = format_for_chat(transcript, speaker="群友甲")
    assert "群友甲" in text and "方言" in text and "说了句方言" in text
