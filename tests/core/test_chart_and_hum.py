"""图表精读与哼唱识曲单元：载荷结构、通道配置、识别链顺序。"""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.chart_vision import analyze_chart_image
from app.core.music_id import identify_music_from_voice
from app.runtime.providers import build_chart_vision_settings, build_voice_asr_settings


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


def test_chart_prompt_and_payload(tmp_path, monkeypatch):
    import app.core.chart_vision as cv
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["model"] = json["model"]
        content = json["messages"][0]["content"]
        captured["text"] = content[0]["text"]
        captured["image_uri"] = content[1]["image_url"]["url"]
        return _FakeResponse({"choices": [{"message": {"content": "折线图，整体上升"}}]})

    monkeypatch.setattr(cv.httpx, "post", fake_post)
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG fake")
    text = analyze_chart_image(b"\x89PNG fake", "image/png",
                               base_url="https://api.siliconflow.cn/v1",
                               api_key="sk-x", model="Qwen/Qwen2.5-VL-72B-Instruct")
    assert text == "折线图，整体上升"
    assert captured["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["model"] == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert captured["image_uri"].startswith("data:image/png;base64,")
    assert "坐标轴" in captured["text"] and "看不清" in captured["text"]


def test_chart_empty_analysis_raises(tmp_path, monkeypatch):
    import app.core.chart_vision as cv
    monkeypatch.setattr(cv.httpx, "post",
                        lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": ""}}]}))
    with pytest.raises(OSError):
        analyze_chart_image(b"x", "image/jpeg", base_url="https://x/v1",
                            api_key="k", model="m")


def test_vision_and_asr_channel_gating():
    empty = SimpleNamespace(chart_vision_base_url="", chart_vision_api_key="", chart_vision_model="m",
                            chart_vision_timeout_seconds=60,
                            voice_asr_provider="xfyun", voice_asr_app_id="", voice_asr_api_key="",
                            voice_asr_api_secret="", voice_asr_accent="mandarin",
                            voice_asr_language="zh_cn", voice_asr_timeout_seconds=30)
    assert build_chart_vision_settings(settings=empty) is None
    assert build_voice_asr_settings(settings=empty) is None
    full = SimpleNamespace(**{**empty.__dict__,
        "chart_vision_base_url": "https://api.siliconflow.cn/v1",
        "chart_vision_api_key": "sk", "voice_asr_app_id": "app", "voice_asr_api_key": "k",
        "voice_asr_api_secret": "s"})
    assert build_chart_vision_settings(settings=full) == (
        "https://api.siliconflow.cn/v1", "sk", "m", 60.0)
    assert build_voice_asr_settings(settings=full)[0] == "app"


def _write_mp3(tmp_path):
    audio = tmp_path / "hum.mp3"
    audio.write_bytes(b"fake-mp3")
    return audio


def test_voice_music_chain_audd_hits_first(tmp_path, monkeypatch):
    import app.core.music_id as mid
    from app.core.music_id import MusicMatch

    calls = []
    monkeypatch.setattr(mid, "identify_by_fingerprint",
                        lambda audio, api_token: calls.append("audd") or MusicMatch(
                            title="晴天", artist="周杰伦", source="fingerprint"))

    def fail_hum(*args, **kwargs):
        raise AssertionError("audd 命中后不应再走哼唱")

    monkeypatch.setattr(mid, "identify_by_humming_xfyun", fail_hum)
    match = identify_music_from_voice(_write_mp3(tmp_path), audd_api_token="tok",
                                      xfyun_creds=("a", "k", "s"))
    assert match is not None and match.title == "晴天"
    assert calls == ["audd"]


def test_voice_music_chain_falls_to_humming(tmp_path, monkeypatch):
    import app.core.music_id as mid
    from app.core.music_id import MusicMatch

    monkeypatch.setattr(mid, "identify_by_fingerprint", lambda audio, api_token: None)
    captured = {}
    monkeypatch.setattr(mid, "identify_by_humming_xfyun",
                        lambda audio, app_id, api_key, api_secret: captured.update(
                            {"mode_ok": True}) or MusicMatch(
                            title="七里香", artist="周杰伦", source="humming"))
    match = identify_music_from_voice(_write_mp3(tmp_path), audd_api_token="tok",
                                      xfyun_creds=("a", "k", "s"))
    assert match.source == "humming"
    assert captured["mode_ok"]


def test_voice_music_chain_all_miss_returns_none(tmp_path, monkeypatch):
    import app.core.music_id as mid

    monkeypatch.setattr(mid, "identify_by_fingerprint", lambda audio, api_token: None)
    monkeypatch.setattr(mid, "identify_by_humming_xfyun", lambda *a, **k: None)
    assert identify_music_from_voice(_write_mp3(tmp_path), audd_api_token="tok",
                                     xfyun_creds=("a", "k", "s")) is None


def test_voice_music_without_xfyun_creds_skips_humming(tmp_path, monkeypatch):
    import app.core.music_id as mid

    monkeypatch.setattr(mid, "identify_by_fingerprint", lambda audio, api_token: None)

    def fail_hum(*args, **kwargs):
        raise AssertionError("无讯飞凭证不应调哼唱")

    monkeypatch.setattr(mid, "identify_by_humming_xfyun", fail_hum)
    assert identify_music_from_voice(_write_mp3(tmp_path), audd_api_token="tok",
                                     xfyun_creds=None) is None


def test_humming_payload_uses_afs_mode(tmp_path, monkeypatch):
    import app.core.music_id as mid

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["mode"] = json["parameter"]["acr_music"]["mode"]
        captured["audio_b64"] = json["payload"]["data"]["audio"]
        return _FakeResponse({"header": {"code": 0},
                              "payload": {"output_text": {"text": ""}}})  # 无匹配 → None

    monkeypatch.setattr(mid.httpx, "post", fake_post)
    audio = tmp_path / "hum.mp3"
    audio.write_bytes(b"\xff\xfb fake mp3")
    assert mid.identify_by_humming_xfyun(audio, app_id="app", api_key="k",
                                         api_secret="s") is None
    assert captured["mode"] == "afs"
    assert base64.b64decode(captured["audio_b64"]) == b"\xff\xfb fake mp3"
