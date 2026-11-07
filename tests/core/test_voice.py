"""语音插件：配额、禁用隐藏、工具行为（TTS 用假实现，不发网络）。"""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from app.core.time_utils import shanghai_now_naive

from app.agent.plugins.voice import VoiceFeature
from app.core.voice import FishAudioTts


class Guard:
    def require_valid(self):
        return None


class FakeTts:
    def __init__(self):
        self.texts = []

    def synthesize(self, text, workdir):
        self.texts.append(text)
        out = Path(workdir) / "tts_test.mp3"
        out.write_bytes(b"ID3fake-mp3-data" * 100)
        return out


class FakeSender:
    def __init__(self):
        self.voices = []

    async def send_group_voice(self, *, group_id, voice_file):
        self.voices.append((group_id, voice_file))


def _feature_with_key(tmp_path, *, quota_spent=0):
    settings = SimpleNamespace(
        data_dir=tmp_path, tts_api_key="test-key", tts_provider="fishaudio",
        tts_api_url="http://localhost:1", tts_voice="",
        tts_daily_limit=20)
    feature = VoiceFeature()
    feature.attach(settings=settings)
    feature._tts = FakeTts()
    feature._usage_day = f"{shanghai_now_naive():%Y-%m-%d}"
    feature._usage_count = quota_spent
    return feature, settings


def _turn():
    return SimpleNamespace(
        router=SimpleNamespace(sender=FakeSender()),
        job=SimpleNamespace(kind="group", scope_id=900000001),
        guard=Guard(), store=None, blocking=None, interactive=True, now=None)


def test_no_api_key_hides_tools(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, tts_api_key="")
    feature = VoiceFeature()
    feature.attach(settings=settings)
    assert feature.available() is False
    turn = _turn()
    assert feature.build_tools(turn) == ()


def test_speak_synthesizes_and_sends(tmp_path):
    import asyncio

    feature, settings = _feature_with_key(tmp_path)
    turn = _turn()
    tool = {t.name: t for t in feature.build_tools(turn)}["speak"]
    result = asyncio.run(tool.handler({"text": "今天天气不错，出去走走？"}))
    assert result.status == "ok" and result.data["spoken"] is True
    assert len(feature._tts.texts) == 1
    group, voice_file = turn.router.sender.voices[0]
    assert group == 900000001
    assert Path(voice_file).exists()


def test_daily_quota_blocks(tmp_path):
    import asyncio

    feature, settings = _feature_with_key(tmp_path, quota_spent=20)
    turn = _turn()
    tool = {t.name: t for t in feature.build_tools(turn)}["speak"]
    result = asyncio.run(tool.handler({"text": "还想说话"}))
    assert result.status == "blocked" and result.data["error"] == "voice_daily_limit"


def test_oversize_text_blocked(tmp_path):
    import asyncio

    feature, settings = _feature_with_key(tmp_path)
    turn = _turn()
    tool = {t.name: t for t in feature.build_tools(turn)}["speak"]
    result = asyncio.run(tool.handler({"text": "长" * 151}))
    assert result.status == "blocked" and result.data["error"] == "text_too_long"
