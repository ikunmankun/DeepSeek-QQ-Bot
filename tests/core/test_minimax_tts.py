"""MiniMax TTS 适配器：hex 音频解码、GroupId 参数、错误抛出。"""

from app.core.voice import MiniMaxTts, TtsError


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.captured = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.captured = {"url": url, "params": kwargs.get("params"),
                         "json": kwargs.get("json")}
        return FakeResp(self.payload)


def test_minimax_synthesize_decodes_hex_audio(tmp_path, monkeypatch):
    audio_hex = bytes(range(256)).hex() * 4
    fake = FakeClient({"data": {"audio": audio_hex},
                       "base_resp": {"status_code": 0, "status_msg": "success"}})
    monkeypatch.setattr("app.core.voice.httpx.Client", lambda **kw: fake)

    tts = MiniMaxTts(api_key="k", group_id="12345", voice_id="female-shaonv",
                     model="speech-02-turbo", speed=1.1)
    out = tts.synthesize("测试语音", tmp_path)

    assert out.exists() and out.stat().st_size == len(audio_hex) // 2  # hex → bytes
    assert fake.captured["params"]["GroupId"] == "12345"
    body = fake.captured["json"]
    assert body["model"] == "speech-02-turbo"
    assert body["voice_setting"]["voice_id"] == "female-shaonv"
    assert body["voice_setting"]["speed"] == 1.1
    assert body["text"] == "测试语音"
    assert body["audio_setting"]["format"] == "mp3"


def test_minimax_error_raises_tts_error(tmp_path, monkeypatch):
    fake = FakeClient({"data": {"audio": ""},
                       "base_resp": {"status_code": 1004,
                                     "status_msg": "invalid api key"}})
    monkeypatch.setattr("app.core.voice.httpx.Client", lambda **kw: fake)

    tts = MiniMaxTts(api_key="bad", group_id="12345")
    try:
        tts.synthesize("测试", tmp_path)
        raise AssertionError("应当抛 TtsError")
    except TtsError as exc:
        assert "invalid api key" in str(exc)
