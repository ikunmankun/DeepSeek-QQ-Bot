"""听歌识曲后台服务：排队 → 取音轨 → 指纹/字幕兜底 → 把结果发回原会话。

结构对齐 GroupImageGenerationService 的内存队列模式：工具线程只 enqueue，
worker 串行处理，结果由服务直接投递（带请求者称呼），状态字段记录投递回执。
主人的"刷视频听到BGM想单曲循环"和群友发视频问歌都走这一条路。
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from app.adapters.sender import (
    OutboundMessage, OutboundPrivateMessage, QQMessageDeliveryUncertainError,
)
from app.core.music_id import (
    DEFAULT_AUDD_URL, MusicMatch, download_audio, extract_audio_clip,
    guess_from_lyrics, identify_by_fingerprint,
)

logger = logging.getLogger(__name__)

MUSIC_WORK_DIR_NAME = "music_id_work"


@dataclass(slots=True)
class MusicIdentificationRequest:
    source_url: str
    #: 已下载到本地的媒体直接给路径，跳过 yt-dlp。
    source_path: Path | None = None
    #: 群聊填 group_id，私聊填 private_user_id；两者必居其一。
    group_id: int | None = None
    private_user_id: int | None = None
    trigger_message_id: str = ""
    requester_user_id: int = 0
    requested_text: str = ""


@dataclass(slots=True)
class MusicJobState:
    request: MusicIdentificationRequest
    status: str = "queued"
    detail: str = ""
    match: MusicMatch | None = None


class MusicIdentificationService:
    """内存队列 + 单 worker；进程重启丢队列（识别是即时需求，丢了就重发）。"""

    def __init__(self, *, settings, llm_client, sender, web_search_client=None,
                 queue_capacity: int = 4, timeout_seconds: float = 240.0) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.sender = sender
        self.web_search_client = web_search_client
        self.queue_capacity = int(queue_capacity)
        self.timeout_seconds = float(timeout_seconds)
        self.api_token = str(getattr(settings, "music_id_api_token", "") or "").strip()
        self.api_url = str(getattr(settings, "music_id_api_url", "") or "").strip() or None
        self._queue: deque[MusicJobState] = deque()
        self._worker: asyncio.Task | None = None
        self._wakeup = asyncio.Event()

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._wakeup.clear()
            self._worker = asyncio.create_task(self._worker_loop(), name="music-id-worker")

    async def stop(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        self._worker = None

    # ---- enqueue ----

    def enqueue(self, request: MusicIdentificationRequest) -> int | None:
        """入队；队列满返回 None（调用方报 music_queue_full）。"""
        if not self.api_token:
            return None
        if len(self._queue) >= self.queue_capacity:
            return None
        state = MusicJobState(request=request)
        self._queue.append(state)
        position = len(self._queue)
        self._wakeup.set()
        return position

    # ---- worker ----

    async def _worker_loop(self) -> None:
        while True:
            if not self._queue:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            state = self._queue.popleft()
            try:
                await self._process(state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.status = "failed"
                state.detail = type(exc).__name__
                logger.exception("music_id_job_failed url=%s", state.request.source_url[:120])
                await self._deliver_failure(state)

    async def _process(self, state: MusicJobState) -> None:
        request = state.request
        state.status = "working"
        workdir = Path(self.settings.data_dir) / MUSIC_WORK_DIR_NAME / str(int(time.time() * 1000))
        match = await asyncio.to_thread(self._identify_sync, request, workdir, state)
        state.match = match
        if match is None:
            state.status = "no_match"
            state.detail = "fingerprint_and_lyrics_missed"
        else:
            state.status = "identified"
        self._cleanup_dir(workdir)
        await self._deliver(state)

    def _identify_sync(self, request: MusicIdentificationRequest, workdir: Path,
                       state: MusicJobState) -> MusicMatch | None:
        source = request.source_url
        downloaded: Path | None = None
        if request.source_path is not None and Path(request.source_path).is_file():
            downloaded = Path(request.source_path)
        else:
            try:
                downloaded = download_audio(source, workdir, settings=self.settings)
            except Exception as exc:
                logger.info("music_download_miss url=%s error=%s", source[:120], type(exc).__name__)
                state.detail = "audio_download_failed"
                return None
        try:
            clip = extract_audio_clip(downloaded, workdir)
            match = identify_by_fingerprint(clip, api_token=self.api_token,
                                            api_url=self.api_url or DEFAULT_AUDD_URL,
                                            timeout=min(self.timeout_seconds, 60.0))
            if match is not None:
                return match
        except Exception as exc:
            logger.info("music_fingerprint_miss url=%s error=%s", source[:120], type(exc).__name__)
        # 指纹无匹配或异常，都落到兜底路线C：B站字幕歌词（仅B站链接有字幕）。
        try:
            from app.core.bili_subtitle import fetch_subtitle_text
            fetched = fetch_subtitle_text(source, settings=self.settings)
            if fetched is None:
                return None
            subtitle_text, _origin = fetched
            if self.web_search_client is None:
                return None
            return guess_from_lyrics(subtitle_text, llm=self.llm_client,
                                     web_search_client=self.web_search_client)
        except Exception as exc:
            logger.info("music_lyrics_fallback_miss error=%s", type(exc).__name__)
            return None

    @staticmethod
    def _cleanup_dir(workdir: Path) -> list[Path]:
        removed = []
        if workdir.exists():
            for child in sorted(workdir.iterdir()):
                try:
                    child.unlink(missing_ok=True)
                    removed.append(child)
                except OSError:
                    pass
            try:
                workdir.rmdir()
            except OSError:
                pass
        return removed

    # ---- delivery ----

    def _match_text(self, state: MusicJobState) -> str:
        match = state.match
        assert match is not None
        head = f"「{match.title}」"
        if match.artist:
            head += f"-{match.artist}"
        link = f" {match.link}" if match.link else ""
        if match.source == "lyrics":
            return f"查了字幕，应该是{head}，不一定准{link}"
        return f"{head}，就是这首{link}"

    async def _deliver(self, state: MusicJobState) -> None:
        request = state.request
        text = self._match_text(state) if state.match else (
            "这首没识别出来，太小众了，你自己知道名字的话告诉我一声，我记下来")
        try:
            if request.group_id:
                await self.sender.send_group_text(OutboundMessage(group_id=request.group_id, text=text))
            elif request.private_user_id:
                await self.sender.send_private_text(OutboundPrivateMessage(user_id=request.private_user_id, text=text))
            state.status = "delivered" if state.match else "delivered_miss"
        except QQMessageDeliveryUncertainError:
            state.status = "uncertain"
            logger.warning("music_id_delivery_uncertain msg_id=%s", request.trigger_message_id)
        except Exception:
            state.status = "delivery_failed"
            logger.exception("music_id_delivery_failed msg_id=%s", request.trigger_message_id)

    async def _deliver_failure(self, state: MusicJobState) -> None:
        if state.status == "delivered" or state.status.startswith("delivered"):
            return
        try:
            request = state.request
            if request.group_id:
                await self.sender.send_group_text(OutboundMessage(
                    group_id=request.group_id, text="识别跑到一半炸了，再发一次链接我重试"))
            elif request.private_user_id:
                await self.sender.send_private_text(OutboundPrivateMessage(
                    user_id=request.private_user_id, text="识别跑到一半炸了，再发一次链接我重试"))
        except Exception:
            logger.exception("music_id_failure_notice_failed msg_id=%s", state.request.trigger_message_id)
