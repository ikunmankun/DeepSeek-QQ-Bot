"""识曲后台服务：排队、指纹命中投递、字幕兜底、无 token 拒收。"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.core.music_identification as mi
from app.core.music_identification import MusicIdentificationRequest, MusicIdentificationService


def _settings(tmp_path, token="tok"):
    return SimpleNamespace(data_dir=tmp_path, music_id_api_token=token,
                           music_id_api_url="", music_id_queue_capacity=2,
                           music_id_timeout_seconds=60)


class _FakeSender:
    def __init__(self):
        self.groups = []
        self.privates = []

    async def send_group_text(self, outbound):
        self.groups.append((outbound.group_id, outbound.text))
        return ("r",)

    async def send_private_text(self, outbound):
        self.privates.append((outbound.user_id, outbound.text))
        return ("r",)


def _request(group_id=123):
    return MusicIdentificationRequest(source_url="https://b23.tv/x", group_id=group_id,
                                      trigger_message_id="m-1", requester_user_id=9)


def _service(tmp_path, sender, **kwargs):
    return MusicIdentificationService(settings=_settings(tmp_path, **kwargs),
                                      llm_client=SimpleNamespace(), sender=sender,
                                      web_search_client=SimpleNamespace())


def _fake_media(tmp_path, monkeypatch, source_name="music-src.mp4"):
    """跳过真实下载/切片：伪造一个本地音轨文件并固定切片输出。"""
    media = tmp_path / "fake-src"
    media.mkdir(parents=True, exist_ok=True)
    audio = media / source_name
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(mi, "download_audio",
                        lambda url, workdir, settings=None: audio)
    monkeypatch.setattr(mi, "extract_audio_clip",
                        lambda source, workdir, seconds=15.0: source)
    return audio


def test_enqueue_rejects_without_token(tmp_path):
    service = _service(tmp_path, _FakeSender(), token="")
    assert service.enqueue(_request()) is None


def test_enqueue_rejects_when_full(tmp_path):
    service = _service(tmp_path, _FakeSender())
    service.queue_capacity = 1
    assert service.enqueue(_request()) == 1
    assert service.enqueue(_request()) is None


def test_identified_match_delivered_to_group(tmp_path, monkeypatch):
    sender = _FakeSender()
    service = _service(tmp_path, sender)
    _fake_media(tmp_path, monkeypatch)
    monkeypatch.setattr(mi, "identify_by_fingerprint",
                        lambda audio, api_token, api_url=None, timeout=None:
                        mi.MusicMatch(title="Track A", artist="歌手B",
                                      link="https://music.apple.com/x"))
    state = mi.MusicJobState(request=_request())
    asyncio.run(service._process(state))
    assert state.status == "delivered" and state.match.source == "fingerprint"
    assert sender.groups and sender.groups[0][0] == 123
    assert "「Track A」-歌手B" in sender.groups[0][1]
    assert "music.apple.com" in sender.groups[0][1]
    assert not list((tmp_path / "music_id_work").rglob("music-src*"))


def test_miss_falls_back_to_lyrics_then_delivers_miss(tmp_path, monkeypatch):
    sender = _FakeSender()
    service = _service(tmp_path, sender)
    _fake_media(tmp_path, monkeypatch)
    monkeypatch.setattr(mi, "identify_by_fingerprint",
                        lambda audio, api_token, api_url=None, timeout=None: None)

    def fake_subtitles(url, settings):
        return ("像唱词的句子一\n说话内容", "字幕文件 x.json3")

    import app.core.bili_subtitle as bili
    monkeypatch.setattr(bili, "fetch_subtitle_text", fake_subtitles)
    monkeypatch.setattr(mi, "guess_from_lyrics",
                        lambda text, llm, web_search_client:
                        mi.MusicMatch(title="歌曲X", source="lyrics"))
    state = mi.MusicJobState(request=_request())
    asyncio.run(service._process(state))
    assert state.status == "delivered" and state.match.source == "lyrics"
    assert "应该是「歌曲X」" in sender.groups[0][1]


def test_total_miss_delivers_honest_no_match(tmp_path, monkeypatch):
    sender = _FakeSender()
    service = _service(tmp_path, sender)
    _fake_media(tmp_path, monkeypatch)
    monkeypatch.setattr(mi, "identify_by_fingerprint",
                        lambda audio, api_token, api_url=None, timeout=None: None)
    import app.core.bili_subtitle as bili
    monkeypatch.setattr(bili, "fetch_subtitle_text", lambda url, settings: None)
    state = mi.MusicJobState(request=_request())
    asyncio.run(service._process(state))
    assert state.status == "delivered_miss"
    assert sender.groups and "没识别出来" in sender.groups[0][1]


def test_worker_loop_runs_enqueue_to_delivery(tmp_path, monkeypatch):
    sender = _FakeSender()
    service = _service(tmp_path, sender)
    _fake_media(tmp_path, monkeypatch)
    monkeypatch.setattr(mi, "identify_by_fingerprint",
                        lambda audio, api_token, api_url=None, timeout=None:
                        mi.MusicMatch(title="Loop Song"))

    async def scenario():
        await service.start()
        assert service.enqueue(_request()) == 1
        for _ in range(100):
            if sender.groups:
                break
            await asyncio.sleep(0.02)
        await service.stop()

    asyncio.run(scenario())
    assert sender.groups and "「Loop Song」" in sender.groups[0][1]


def test_private_delivery_targets_user(tmp_path, monkeypatch):
    sender = _FakeSender()
    service = _service(tmp_path, sender)
    _fake_media(tmp_path, monkeypatch)
    monkeypatch.setattr(mi, "identify_by_fingerprint",
                        lambda audio, api_token, api_url=None, timeout=None:
                        mi.MusicMatch(title="私聊歌"))
    state = mi.MusicJobState(request=MusicIdentificationRequest(
        source_url="https://example.com/v", private_user_id=777))
    asyncio.run(service._process(state))
    assert sender.privates and sender.privates[0][0] == 777


def test_builder_hides_feature_without_token(tmp_path):
    from app.runtime.providers import build_music_identification_service

    assert build_music_identification_service(
        settings=_settings(tmp_path, token=""), llm_client=None, sender=None) is None
    service = build_music_identification_service(
        settings=_settings(tmp_path), llm_client=SimpleNamespace(), sender=_FakeSender())
    assert service is not None and service.api_token == "tok"
