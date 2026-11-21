"""B 站浏览器插件：定时窗口自主刷视频，产出进日记与观察轮。"""

from __future__ import annotations

import time
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule
from app.core.bili_browser import BiliBrowserService

FINDINGS_TTL_SECONDS = 8 * 3600


class BilibiliBrowserFeature(FeatureModule):
    name = "bili_browser"

    _service: BiliBrowserService | None = None
    _attached = False

    def attach(self, *, settings, sender, llm_client, persona_text) -> None:
        self._settings = settings
        self._sender = sender
        self._llm_client = llm_client
        self._persona_text = persona_text

    def create_workers(self) -> Sequence[Any]:
        if not getattr(self._settings, "bili_browser_enabled", True):
            return ()
        from pathlib import Path

        chrome_path = str(getattr(self._settings, "bili_chrome_path", ""))
        if not chrome_path or not Path(chrome_path).exists():
            import logging

            logging.getLogger(__name__).warning(
                "bili_browser_chrome_missing path=%s", chrome_path)
            return ()
        self._service = BiliBrowserService(
            settings=self._settings,
            sender=self._sender,
            llm_client=self._llm_client,
            writer=self._writer(),
            owner_qq=int(self._settings.owner_qq),
            persona_text=self._persona_text)
        return [self._service]

    def _writer(self):
        from app.core.diary import DiaryWriter

        return DiaryWriter(self._settings.data_dir)

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group" or not observing or self._service is None:
            return ()
        now_mono = time.monotonic()
        fresh = [text for stamp, text in self._service.latest_findings
                 if now_mono - stamp < FINDINGS_TTL_SECONDS]
        if not fresh:
            return ()
        # 2026-11-18：观察轮从"被动可选"改成"鼓励主动挑话头"——这是机器人
        # 自己刷到的内容，主动分享比接话更像真人；仍然保留不聊的自由。
        return ["这是你自己刷B站看到的内容。群聊冷场、没人接话，或者你本来就想说点什么时，"
                "主动把它抛出来当话题：贴链接、说一句你自己的看法都行；有人正聊得热闹就别打断，"
                "不想聊也可以当没看见：\n" + "\n".join(fresh)]
