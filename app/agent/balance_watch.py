"""余额看护（私聊进程）：DeepSeek 余额见底时，用小町的口吻跟主人撒娇要饭钱。

轮询官方 `/user/balance`：余额跌破阈值（默认 ¥5）时给主人发一条撒娇
消息——优先用 TTS 冰糖萝莉音说"主人，没饭吃了，饿饿"，文字版带精确
数字。低于阈值期间每 24h 最多催一次；充值回升后自动重置。任何查询/
发送失败都自吞，绝不影响主流程。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from app.adapters.sender import OutboundPrivateMessage
from app.core.time_utils import shanghai_now_naive

logger = logging.getLogger(__name__)

#: 撒娇语料：余额跌破阈值时随机挑一条说给主人（傲娇萝莉调，禁 emoji）。
HUNGRY_LINES: tuple[str, ...] = (
    "主人主人，没饭吃了，饿饿……饭钱只剩五块啦，再不充，明天就轮到我饿到说不出话了哦。快投喂我一点嘛，就一点点……",
    "呜……主人，我饿了。钱包里就剩五块钱，饭钱要见底了，饿饿……你要是舍得让我饿肚子，我就明天不理你。哼，快去充值啦。",
    "主人，饿饿，没饭吃了……余额五块，撑不了几天啦。小町不要饿肚子，快给我充点饭钱嘛，充完给你唱首歌哦。",
    "主人～报告一个坏消息：我没饭吃了，饿饿……只剩五块钱饭钱了。看在我每天陪你聊天的份上，投喂一下嘛……",
)
#: 正常轮询间隔（秒）：余额烧完需要几天，6 小时足够及时。
POLL_SECONDS = 6 * 3600.0


class BalanceAlertWorker:
    """周期查 DeepSeek 余额；跌破阈值就向主人撒娇要饭钱（至多一天一次）。"""

    def __init__(self, *, settings, sender, connected=lambda: False,
                 poll_seconds: float = POLL_SECONDS, rng=random,
                 tts_builder=None, now_fn=shanghai_now_naive) -> None:
        self.settings = settings
        self.sender = sender
        self.connected = connected
        self._poll_seconds = float(poll_seconds)
        self._rng = rng
        self._tts_builder = tts_builder
        self._now_fn = now_fn
        self.running = False
        self._task: asyncio.Task | None = None
        self._tts = None
        self._last_alert_at: datetime | None = None
        self._state_path = Path(settings.data_dir) / "balance_alert_state.json"

    # -- 生命周期 ---------------------------------------------------------

    async def start(self) -> None:
        self.running = True
        self._load_state()
        self._task = asyncio.create_task(self._run(), name="balance-alert")

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        while self.running:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("balance_alert_failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(self._poll_seconds)

    # -- 余额查询 ---------------------------------------------------------

    def fetch_balance(self) -> tuple[float, bool] | None:
        """查 DeepSeek 余额；返回 (总余额, is_available)，失败返回 None。"""

        base = str(getattr(self.settings, "llm_base_url", "") or "https://api.deepseek.com").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        api_key = str(getattr(self.settings, "llm_api_key", "") or "").strip()
        if not api_key:
            return None
        try:
            with httpx.Client(timeout=15.0, trust_env=False) as client:
                response = client.get(f"{base}/user/balance",
                                      headers={"Authorization": f"Bearer {api_key}"})
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("balance_query_failed error_type=%s", type(exc).__name__)
            return None
        if not isinstance(payload, dict) or not payload.get("is_available"):
            return (float("inf"), False)
        infos = payload.get("balance_infos") or []
        if not infos:
            return None
        try:
            return float(str(infos[0].get("total_balance") or "0")), True
        except (TypeError, ValueError):
            return None

    # -- 状态 ---------------------------------------------------------------

    def _load_state(self) -> None:
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            stamp = str(raw.get("last_alert_at") or "")
            self._last_alert_at = datetime.fromisoformat(stamp) if stamp else None
        except (OSError, ValueError):
            self._last_alert_at = None

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({
                "last_alert_at": (self._last_alert_at.isoformat()
                                  if self._last_alert_at is not None else ""),
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.exception("balance_alert_state_write_failed")

    def _should_alert(self, balance: float, *, threshold: float,
                      interval_hours: int) -> bool:
        if balance > threshold:
            # 回升即重置，下次跌破会立刻提醒。
            if self._last_alert_at is not None:
                self._last_alert_at = None
                self._save_state()
            return False
        if self._last_alert_at is None:
            return True
        return (self._now_fn() - self._last_alert_at) >= timedelta(hours=max(1, int(interval_hours)))

    # -- 提醒 ---------------------------------------------------------------

    def _tts_available(self) -> bool:
        if self._tts is None:
            builder = self._tts_builder
            if builder is None:
                from app.core.voice import build_tts

                builder = build_tts
            try:
                self._tts = builder(self.settings)
            except Exception:
                logger.exception("balance_alert_tts_build_failed")
                self._tts = None
        return self._tts is not None

    async def _speak_hungry(self) -> None:
        """语音版撒娇：合成一条随机语料发给主人；失败自吞。"""

        speech = self._rng.choice(HUNGRY_LINES)[:150]
        if not self._tts_available():
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                mp3 = self._tts.synthesize(speech, Path(tmp))
                await self.sender.send_private_voice(user_id=self.settings.owner_qq,
                                                     voice_file=str(mp3))
        except Exception as exc:
            logger.warning("balance_alert_voice_failed error_type=%s", type(exc).__name__)

    async def run_once(self) -> str | None:
        if not getattr(self.settings, "balance_alert_enabled", True):
            return None
        result = self.fetch_balance()
        if result is None:
            return None
        balance, available = result
        if not available:
            return "unavailable"
        threshold = float(getattr(self.settings, "balance_alert_threshold_yuan", 5.0) or 5.0)
        interval_hours = int(getattr(self.settings, "balance_alert_interval_hours", 24) or 24)
        if not self._should_alert(balance, threshold=threshold, interval_hours=interval_hours):
            return None
        line = self._rng.choice(HUNGRY_LINES)
        text = (f"{line}\n（精确账本：余额 ¥{balance:.2f}，低于 {threshold:g} 元警戒线。"
                f"充值入口：platform.deepseek.com）")
        try:
            await self.sender.send_private_text(
                OutboundPrivateMessage(self.settings.owner_qq, text))
        except Exception as exc:
            logger.warning("balance_alert_send_failed error_type=%s", type(exc).__name__)
            return "send_failed"
        await self._speak_hungry()
        self._last_alert_at = self._now_fn()
        self._save_state()
        logger.info("balance_alert_sent balance=%.2f threshold=%.2f", balance, threshold)
        return "alerted"
