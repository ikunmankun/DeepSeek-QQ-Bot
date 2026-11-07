"""B 站插件：链接解析入上下文、刷B站工具、观察轮热榜注入。

模仿流行的 B 站插件能力（链接解析 / bilichat 总结 / 热榜快报）的组合版：
* 群里有人甩 BV/av/短链 → 自动取视频元数据注入本轮提示词，回复有据可依；
* 观察轮注入 B 站热门榜，机器人可以自然地跟群友聊视频或写进日记；
* browse_bilibili 工具让它随时能"刷B站"（热榜 / 查某个视频）。
全部只读公开 API，不登录不带 cookie。
"""

from __future__ import annotations

import logging
import asyncio
import time
from typing import Any, Sequence

import httpx

from app.agent.plugins.base import FeatureModule, InboundContext, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.bilibili_patrol import (
    BiliInfoCache,
    VideoInfo,
    bump_group_interest,
    extract_bilibili_refs,
    fetch_popular,
    fetch_video_info,
    hot_prompt_line,
    interest_prompt_line,
    share_prompt_line,
)


class BilibiliFeature(FeatureModule):
    name = "bilibili"

    def __init__(self, *, hot_interval_seconds: float = 6 * 3600) -> None:
        self.cache = BiliInfoCache()
        self._hot: list[VideoInfo] = []
        self._hot_at = 0.0
        self._hot_interval = float(hot_interval_seconds)
        # 分享视频的字幕预取缓存：group_id -> (mono_ts, 字幕前段, 标题)
        self._subtitle_cache: dict[int, tuple[float, str, str]] = {}
        self._subtitle_prefetched: dict[int, str] = {}

    # -- helpers ---------------------------------------------------------

    def _client(self) -> httpx.Client:
        # trust_env=False：B站是国内直连站点，不吃容器的代理 env
        #（否则热榜/链接解析会被代理劫持而全挂）。
        return httpx.Client(trust_env=False)

    def _info_for(self, ref: str) -> VideoInfo | None:
        now = time.monotonic()
        cached = self.cache.get_info(ref, now=now)
        if cached is not None:
            return cached
        with self._client() as client:
            info = fetch_video_info(ref, client=client)
        if info is not None:
            self.cache.put_info(ref, info, now=now)
        return info

    def _refresh_hot(self) -> list[VideoInfo]:
        now = time.monotonic()
        if self._hot and now - self._hot_at < self._hot_interval:
            return self._hot
        with self._client() as client:
            self._hot = fetch_popular(client=client)
        self._hot_at = now
        return self._hot

    # -- hooks -----------------------------------------------------------

    def on_inbound(self, ctx: InboundContext) -> None:
        refs = extract_bilibili_refs(ctx.event.plain_text)
        if not refs:
            return
        for ref in refs[:2]:
            info = self._info_for(ref)
            if info is not None:
                self.cache.put_share(ctx.event.group_id, info, now=time.monotonic())
                self._remember_interest(ctx.event.group_id, info, now=ctx.event.timestamp)
                self._maybe_prefetch_subtitle(ctx, info)
                break

    def _maybe_prefetch_subtitle(self, ctx: InboundContext, info) -> None:
        """分享的视频有登录态时，后台预取字幕；同一视频只取一次。"""
        import asyncio

        from app.core.bili_subtitle import cookies_available, fetch_subtitle_text

        settings = ctx.router.runtime.settings
        if not cookies_available(settings):
            return
        group_id = ctx.event.group_id
        if self._subtitle_prefetched.get(group_id) == info.bvid:
            return
        self._subtitle_prefetched[group_id] = info.bvid

        async def work():
            result = await asyncio.to_thread(
                fetch_subtitle_text, info.url, settings=settings)
            if result:
                text, _source = result
                self._subtitle_cache[group_id] = (
                    time.monotonic(), text[:1200], info.title)

        asyncio.get_running_loop().create_task(work())

    def _remember_interest(self, group_id: int, info: VideoInfo, *, now) -> None:
        """群友分享过什么，机器人就记住这个群在看什么（失败自吞）。"""

        if not info.zone:
            return
        try:
            from app.storage.db import session_scope

            with session_scope(self._engine) as session:
                bump_group_interest(session, group_id=group_id, zone=info.zone, now=now)
        except Exception:
            logging.getLogger(__name__).debug(
                "bilibili_interest_bump_failed group=%s", group_id, exc_info=True
            )

    _engine = None

    def attach_engine(self, engine) -> None:
        self._engine = engine

    def prompt_lines(self, *, kind: str, group_id: int, now, observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group":
            return ()
        now_mono = time.monotonic()
        lines: list[str] = []
        shared = self.cache.recent_share(group_id, now=now_mono)
        if shared is not None:
            lines.append(share_prompt_line(shared))
        sub = self._subtitle_cache.get(group_id)
        if sub and now_mono - sub[0] < 1200:
            sub_line = ("对方刚分享的视频《" + sub[2] + "》的字幕前段（=视频说了什么，接话用）：\n"
                        + sub[1])
            lines.append(sub_line)
        if observing:
            if self._engine is not None:
                interest = interest_prompt_line(self._engine, group_id=group_id, now=now)
                if interest:
                    lines.append(interest)
            hot = self._hot or self._refresh_hot()
            if hot:
                lines.append(hot_prompt_line(hot))
        return lines

    def create_workers(self) -> Sequence[Any]:
        async def refresh_hot_loop():
            import asyncio

            while True:
                try:
                    await asyncio.to_thread(self._refresh_hot)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception("bilibili_hot_refresh_failed")
                await asyncio.sleep(self._hot_interval)

        return [type("_Worker", (), {"run": staticmethod(refresh_hot_loop)})()]

    # -- tool ------------------------------------------------------------

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        async def browse(args):
            turn.guard.require_valid()
            action = str(args.get("action") or "hot").strip()
            query = str(args.get("query") or "").strip()
            if action == "hot":
                hot = self._refresh_hot()
                if not hot:
                    return ToolOutcome({"error": "bilibili_hot_unavailable"}, "blocked")
                # 群兴趣排序：群友常看的分区靠前（稳定排序，不丢其他结果）。
                if self._engine is not None:
                    try:
                        from app.core.bilibili_patrol import decayed_interests

                        zones = set(decayed_interests(
                            self._engine, group_id=turn.job.scope_id, now=turn.now))
                        hot = sorted(
                            hot,
                            key=lambda video: 0 if video.zone in zones else 1,
                        )
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "bilibili_interest_sort_failed", exc_info=True
                        )
            elif action == "video":
                refs = extract_bilibili_refs(query)
                if not refs:
                    return ToolOutcome({"error": "invalid_bilibili_ref",
                        "guidance": "传 BV 号、av 号或 b23.tv 链接。"}, "blocked")
                info = self._info_for(refs[0])
                if info is None:
                    return ToolOutcome({"error": "bilibili_video_unavailable",
                        "guidance": "视频元数据取不到（可能下架或风控），如实说明即可。"}, "blocked")
                hot = [info]
            elif action == "subtitle":
                from app.core.bili_subtitle import cookies_available, fetch_subtitle_text

                if not cookies_available(turn.router.runtime.settings):
                    return ToolOutcome({"error": "login_required",
                        "guidance": "读字幕需要B站登录态。主人说'扫码登录B站'完成登录后再试。"}, "blocked")
                refs = extract_bilibili_refs(query)
                if not refs:
                    return ToolOutcome({"error": "invalid_bilibili_ref",
                        "guidance": "传 BV 号、av 号或 b23.tv 链接。"}, "blocked")
                info = self._info_for(refs[0])
                if info is None:
                    return ToolOutcome({"error": "bilibili_video_unavailable",
                        "guidance": "视频元数据取不到，如实说明即可。"}, "blocked")
                result = fetch_subtitle_text(info.url, settings=turn.router.runtime.settings)
                if result is None:
                    return ToolOutcome({"error": "subtitle_unavailable",
                        "note": f"《{info.title}》没有可下载的字幕（可能没有CC/AI字幕）。基于标题和简介聊，别编情节。"})
                text, source = result
                return ToolOutcome({"title": info.title, "subtitle_chars": len(text),
                    "subtitle": text[:4000],
                    "guidance": "这是视频的字幕文本（=视频说了什么）。基于它接话或点评；"
                        "字幕没覆盖的画面细节不要编。"})

            else:
                return ToolOutcome({"error": "invalid_action",
                    "guidance": "action 只能是 hot、video 或 subtitle。"}, "blocked")
            return ToolOutcome({"videos": [
                {"title": v.title, "author": v.author, "summary": v.summary,
                 "zone": v.zone, "url": v.url}
                for v in hot[:6]],
                "guidance": "只有标题/UP主/简介等公开元数据，没看过视频内容本身；别编造画面和情节。"
                    "觉得哪个有意思，回复时把它的 url 直接贴给群友（QQ会渲染成视频卡片）。"})

        async def login_qr(args):
            import app.core.bili_browser as bb

            turn.guard.require_valid()
            if bb.login_in_progress():
                return ToolOutcome({"error": "login_in_progress",
                    "note": "已有一轮登录在跑，登录码稍后会再发；等它完成。"}, "blocked")
            settings = turn.router.runtime.settings
            sender = turn.router.sender
            owner_qq = int(settings.owner_qq)
            bb._LOGIN_TASK = asyncio.get_running_loop().create_task(
                bb.qr_login_flow(settings=settings, sender=sender, owner_qq=owner_qq))
            return ToolOutcome({"sent": True,
                "note": "登录二维码已发到主人私聊，5分钟内扫码有效（过期自动换码，最多3张）；"
                        "扫码完成后会话自动写入，不需要再做别的。"})

        return [AgentTool(
            "bili_login_qr",
            "主人让你登录B站、发登录码、扫码登录B站时调用：把登录二维码发到主人私聊，"
            "主人扫码后会话自动写入浏览器。仅主人可用；一轮登录最多15分钟。",
            {"type": "object", "properties": {}, "required": [],
             "additionalProperties": False},
            login_qr, "write"),
            AgentTool(
            "browse_bilibili",
            "刷B站：action=hot 看当前热门榜（群里常看的分区会排在前面），"
            "action=video+query(BV号/av号/链接) 查某个视频的标题、UP主、分区和简介，"
            "action=subtitle+query 读取视频的字幕文本（=视频说了什么，需要登录态）。"
            "想跟群友聊视频、回应对方分享的B站链接、或观察轮里找话题时用。"
            "只能拿到公开元数据，看不到视频画面和字幕，别假装看过。",
            {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["hot", "video", "subtitle"]},
                "query": {"type": "string", "maxLength": 200},
            }, "required": ["action"], "additionalProperties": False},
            browse, "read")]
