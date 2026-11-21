"""余额看护：阈值触发、24h 冷却、回升重置、语音分支、查询失败自吞。"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.agent.balance_watch import BalanceAlertWorker, HUNGRY_LINES


class _FakeSender:
    def __init__(self):
        self.texts: list[tuple[int, str]] = []
        self.voices: list[int] = []

    async def send_private_text(self, message):
        self.texts.append((message.user_id, message.text))
        return ["receipt-1"]

    async def send_private_voice(self, *, user_id, voice_file):
        self.voices.append(user_id)
        return ["voice-1"]


class _FakeTts:
    def synthesize(self, text, workdir):
        out = Path(workdir) / "hungry.mp3"
        out.write_bytes(b"mp3")
        return out


class _FixedRng:
    def __init__(self, value: float = 0.0):
        self.value = value

    def random(self) -> float:
        return self.value

    def choice(self, seq):
        return seq[int(self.value * len(seq)) % len(seq)]


def _worker(sender, *, settings=None, tmp_path=None, now=None,
            tts_builder=None, rng=None) -> BalanceAlertWorker:
    now = now or datetime(2026, 11, 7, 12, 0)
    settings = settings or SimpleNamespace(
        owner_qq=10086, data_dir=(tmp_path or Path(tempfile.mkdtemp())),
        llm_base_url="https://api.deepseek.com", llm_api_key="sk-test",
        balance_alert_enabled=True, balance_alert_threshold_yuan=5.0,
        balance_alert_interval_hours=24)
    return BalanceAlertWorker(settings=settings, sender=sender,
        connected=lambda: True, tts_builder=tts_builder,
        rng=rng or _FixedRng(), now_fn=lambda: now)


def _patch_balance(monkeypatch, worker, balance, *, available=True):
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return (balance, available)

    monkeypatch.setattr(worker, "fetch_balance", fake_fetch)
    return calls


@pytest.mark.asyncio
async def test_alerts_when_below_threshold(monkeypatch, tmp_path):
    sender = _FakeSender()
    worker = _worker(sender, tmp_path=tmp_path,
                     tts_builder=lambda s: _FakeTts())
    _patch_balance(monkeypatch, worker, 3.21)
    result = await worker.run_once()
    assert result == "alerted"
    assert sender.texts and sender.texts[0][0] == 10086
    assert "¥3.21" in sender.texts[0][1]
    assert "没饭吃" in sender.texts[0][1]
    assert len(sender.voices) == 1  # TTS 可用 → 萝莉音撒娇
    # 状态落盘：24h 内不再催。
    assert worker._state_path.exists()


@pytest.mark.asyncio
async def test_no_alert_above_threshold_and_reset_after_recharge(monkeypatch, tmp_path):
    sender = _FakeSender()
    worker = _worker(sender, tmp_path=tmp_path)
    _patch_balance(monkeypatch, worker, 28.99)
    assert await worker.run_once() is None
    assert sender.texts == []
    # 曾提醒过 → 回升后重置 → 再次跌破立刻提醒。
    now = datetime(2026, 11, 7, 12, 0)
    worker._last_alert_at = now - timedelta(hours=2)
    _patch_balance(monkeypatch, worker, 6.0)
    assert await worker.run_once() is None  # 高于阈值，重置标记
    assert worker._last_alert_at is None
    _patch_balance(monkeypatch, worker, 4.0)
    assert await worker.run_once() == "alerted"


@pytest.mark.asyncio
async def test_cooldown_within_interval(monkeypatch, tmp_path):
    sender = _FakeSender()
    now = datetime(2026, 11, 7, 12, 0)
    worker = _worker(sender, tmp_path=tmp_path, now=now)
    _patch_balance(monkeypatch, worker, 2.0)
    assert await worker.run_once() == "alerted"
    # 同一天再查，仍然低余额 → 不再催。
    worker2 = worker
    worker2._now_fn = lambda: now + timedelta(hours=6)
    _patch_balance(monkeypatch, worker2, 1.5)
    assert await worker2.run_once() is None
    assert len(sender.texts) == 1
    # 超过 24h → 再催一次。
    worker2._now_fn = lambda: now + timedelta(hours=25)
    assert await worker2.run_once() == "alerted"
    assert len(sender.texts) == 2


@pytest.mark.asyncio
async def test_voice_skipped_without_tts(monkeypatch, tmp_path):
    sender = _FakeSender()
    worker = _worker(sender, tmp_path=tmp_path, tts_builder=lambda s: None)
    _patch_balance(monkeypatch, worker, 1.0)
    assert await worker.run_once() == "alerted"
    assert sender.texts and sender.voices == []  # 没配 TTS 只发文字


@pytest.mark.asyncio
async def test_balance_query_failure_is_silent(monkeypatch, tmp_path):
    sender = _FakeSender()
    worker = _worker(sender, tmp_path=tmp_path)
    # fetch_balance 内部自吞网络错误返回 None；查询失败不发消息。
    monkeypatch.setattr(worker, "fetch_balance", lambda: None)
    assert await worker.run_once() is None
    assert sender.texts == []


@pytest.mark.asyncio
async def test_disabled_flag(monkeypatch, tmp_path):
    sender = _FakeSender()
    settings = SimpleNamespace(owner_qq=10086,
        data_dir=tmp_path, llm_base_url="https://api.deepseek.com",
        llm_api_key="sk-test", balance_alert_enabled=False,
        balance_alert_threshold_yuan=5.0, balance_alert_interval_hours=24)
    worker = _worker(sender, settings=settings, tmp_path=tmp_path)
    _patch_balance(monkeypatch, worker, 0.5)
    assert await worker.run_once() is None
    assert sender.texts == []


def test_hungry_lines_are_clean_and_short():
    for line in HUNGRY_LINES:
        assert 0 < len(line) <= 150
        assert "饿" in line
