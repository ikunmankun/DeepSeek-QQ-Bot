"""听歌识曲单元：ffmpeg 切片、audD 解析、字幕歌词兜底。"""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.music_id import (
    MusicMatch, extract_audio_clip, guess_from_lyrics, identify_by_fingerprint,
)


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe unavailable")


@pytest.fixture(scope="module")
def tone_wav(tmp_path_factory):
    """ffmpeg 生成 30 秒正弦波音频作为识别源。"""
    work = tmp_path_factory.mktemp("music-src")
    src = work / "tone.wav"
    import subprocess
    run = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=30", "-ac", "1", str(src)],
        capture_output=True, text=True, timeout=60, check=False)
    if run.returncode != 0:
        pytest.skip("ffmpeg sine generation failed")
    return src


def test_extract_audio_clip_takes_middle_and_matches_duration(tone_wav, tmp_path):
    clip = extract_audio_clip(tone_wav, tmp_path, seconds=15.0)
    assert clip.exists() and clip.stat().st_size > 1000
    import json as _json
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(clip)],
        capture_output=True, text=True, timeout=30, check=False)
    duration = float(_json.loads(probe.stdout)["format"]["duration"])
    assert 14.0 <= duration <= 16.0


def test_extract_audio_clip_short_source_clamps_offset(tone_wav, tmp_path):
    clip = extract_audio_clip(tone_wav, tmp_path / "short", seconds=45.0)
    assert clip.exists()


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


def test_fingerprint_parses_match(tmp_path, monkeypatch):
    import app.core.music_id as mid

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake")
    captured = {}

    def fake_post(url, data=None, files=None, timeout=None):
        captured["url"] = url
        captured["token"] = data["api_token"]
        captured["file"] = files["file"][0]
        return _FakeResponse({"status": "success", "result": {
            "title": "Track A", "artist": "歌手B", "album": "专辑C",
            "link": "https://music.apple.com/x", "timecode": "0:12"}})

    monkeypatch.setattr(mid.httpx, "post", fake_post)
    match = identify_by_fingerprint(audio, api_token="tok-123")
    assert match == MusicMatch(title="Track A", artist="歌手B", album="专辑C",
                               link="https://music.apple.com/x", source="fingerprint",
                               timecode="0:12")
    assert captured["token"] == "tok-123" and captured["file"] == "clip.mp3"


def test_fingerprint_no_match_returns_none(tmp_path, monkeypatch):
    import app.core.music_id as mid

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(mid.httpx, "post",
                        lambda *a, **k: _FakeResponse({"status": "success", "result": None}))
    assert identify_by_fingerprint(audio, api_token="t") is None


def test_fingerprint_error_raises_oserror(tmp_path, monkeypatch):
    import app.core.music_id as mid

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(mid.httpx, "post",
                        lambda *a, **k: _FakeResponse({"status": "error", "error": {
                            "error_message": "invalid token"}}))
    with pytest.raises(OSError, match="invalid token"):
        identify_by_fingerprint(audio, api_token="bad")


class _Hit:
    def __init__(self, title, url):
        self.title = title
        self.url = url


class _FakeSearch:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, max_results=3):
        self.queries.append(query)
        return self.hits


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate_text(self, prompt_lines, **kwargs):
        return self.reply


def test_guess_from_lyrics_extracts_and_searches(monkeypatch):
    search = _FakeSearch([_Hit("歌曲X_歌词", "https://music.163.com/song?id=1"),
                          _Hit("无关", "https://example.com/a")])
    llm = _FakeLLM("像唱词的句子一\n像唱词的句子二")
    match = guess_from_lyrics("字幕第一行\n像唱词的句子一\n说话内容", llm=llm, web_search_client=search)
    assert match is not None and match.title == "歌曲X" and match.source == "lyrics"
    assert search.queries and "像唱词的句子一" in search.queries[0]


def test_guess_from_lyrics_no_candidates_returns_none():
    llm = _FakeLLM("无")
    match = guess_from_lyrics("只有说话内容", llm=llm, web_search_client=_FakeSearch([]))
    assert match is None
