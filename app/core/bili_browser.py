"""B 站浏览器：playwright 驱动镜像内现成 Chrome，像人一样定期逛B站。

设计要点：
* 复用镜像里的 chrome-for-testing（/opt/xhs/chrome/chrome），零浏览器下载；
* persistent profile 存登录态：首次观察轮检测未登录时，把登录二维码发给
  主人私聊，扫码一次会话常驻；
* 每个窗口（默认每天 12:00）：热门页滚动 → 按兴趣挑 1~2 个视频 → 停留
  评论区截图 → 视觉模型解读 → 写进日记并存为观察轮的分享候选；
* 窗口结束即关浏览器；全程只读，不做任何写操作。
"""

from __future__ import annotations

import asyncio
import logging
import random
import io
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from app.core.time_utils import shanghai_now_naive

logger = logging.getLogger(__name__)

CHROME_DEFAULT_PATH = "/opt/xhs/chrome/chrome"

# profile 是排他资源：定时窗口与主人触发的登录流程互斥；同一时间只允许
# 一轮登录流程在跑。
_PROFILE_LOCK = asyncio.Lock()
_LOGIN_TASK: asyncio.Task | None = None


def login_in_progress() -> bool:
    return _LOGIN_TASK is not None and not _LOGIN_TASK.done()


async def qr_login_flow(*, settings, sender, owner_qq: int) -> str:
    """主人触发的扫码登录：清残留 → 起浏览器 → 发码 → 轮询 → 会话落盘。

    返回给工具的结论文本；整个流程作为后台任务运行（最长约 15 分钟）。
    """

    import subprocess

    from playwright.async_api import async_playwright

    profile = Path(settings.data_dir) / "bili_browser_profile"
    chrome = str(getattr(settings, "bili_chrome_path", CHROME_DEFAULT_PATH))
    # 清掉持有本 profile 的残留浏览器（上次中断的登录/窗口）
    subprocess.run(["pkill", "-f", "bili_browser_profile"], check=False)

    async with _PROFILE_LOCK:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as http:
            async with async_playwright() as pw:
                context = await pw.chromium.launch_persistent_context(
                    user_data_dir=str(profile), executable_path=chrome, headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--no-proxy-server"],
                    viewport={"width": 1366, "height": 900}, locale="zh-CN")
                try:
                    page = context.pages[0] if context.pages else await context.new_page()
                    await page.goto("https://www.bilibili.com", timeout=30000)
                    await page.wait_for_timeout(2000)
                    if await page.evaluate(_LOGIN_JS):
                        return "已经是登录状态，无需扫码。"

                    result = "未完成"
                    for round_index in range(3):
                        resp = (await http.get(
                            "https://passport.bilibili.com/x/passport-login/web/qrcode/generate",
                            headers=_HEADERS)).json()
                        data = resp.get("data") or {}
                        qrcode_key, login_url = str(data.get("qrcode_key") or ""), str(data.get("url") or "")
                        if not qrcode_key:
                            return "二维码生成失败，稍后再试。"
                        buffer = io.BytesIO()
                        qrcode.make(login_url).resize((360, 360)).save(buffer, format="PNG")
                        png = Path(__import__("tempfile").mkdtemp()) / "bili_qr.png"
                        png.write_bytes(buffer.getvalue())
                        await sender.send_private_media(
                            user_id=owner_qq,
                            text=(f"B站登录确认（5分钟内有效，小号！）。第{round_index + 1}张。"
                                  "手机上点这个链接拉起App确认，或用小号App扫码：")
                            if round_index == 0 else
                            f"上一张过期了，这是新码（5分钟内有效）。手机点链接拉起App确认，或扫码：",
                            image_files=(str(png),))

                        deadline = asyncio.get_running_loop().time() + 290
                        refreshed = False
                        while asyncio.get_running_loop().time() < deadline:
                            poll = (await http.get(
                                "https://passport.bilibili.com/x/passport-login/web/qrcode/poll",
                                params={"qrcode_key": qrcode_key}, headers=_HEADERS)).json()
                            code = int((poll.get("data") or {}).get("code") or -1)
                            if code == 0:
                                success_url = str((poll.get("data") or {}).get("url") or "")
                                if success_url:
                                    await page.goto(success_url, timeout=20000)
                                    await page.wait_for_timeout(2000)
                                logged = await page.evaluate(_LOGIN_JS)
                                from app.core.bili_subtitle import save_cookies_from_browser

                                save_cookies_from_browser(await context.cookies(), settings)
                                result = ("登录成功，会话已写入，之后每天中午的B站窗口就是登录态了。"
                                          if logged else "扫码成功但会话校验未通过，重发一次码试试。")
                                break
                            if code == 86038:
                                refreshed = True
                                break
                            await asyncio.sleep(2)
                        if result != "未完成":
                            break
                        if not refreshed:
                            result = "没等到扫码，码已过期。要登录就再说一次扫码登录B站。"
                            break
                    return result
                finally:
                    await context.close()
MAX_WINDOW_SECONDS = 480
QR_WAIT_SECONDS = 180
FINDINGS_TTL_SECONDS = 8 * 3600

_ENTRIES_JS = """
() => {
  const out = [];
  const seen = new Set();
  for (const a of document.querySelectorAll('a[href*="/video/BV"]')) {
    const href = a.href.split('?')[0];
    if (seen.has(href)) continue;
    const title = (a.getAttribute('title') || a.textContent || '').trim().slice(0, 120);
    if (!title) continue;
    seen.add(href);
    out.push({title, href});
    if (out.length >= 24) break;
  }
  return out;
}
"""

_LOGIN_JS = """
async () => {
  try {
    const r = await fetch('https://api.bilibili.com/x/web-interface/nav', {credentials: 'include'});
    const j = await r.json();
    return !!(j.data && j.data.isLogin);
  } catch (e) { return null; }
}
"""


def parse_windows(spec: str) -> list[tuple[int, int]]:
    """"12:00,18:00" -> [(12,0),(18,0)]；无效片段跳过。"""

    windows: list[tuple[int, int]] = []
    for piece in str(spec or "").split(","):
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", piece)
        if not m:
            continue
        hour, minute = int(m.group(1)), int(m.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            windows.append((hour, minute))
    return sorted(windows) or [(12, 0)]


def next_window(now: datetime, windows: Sequence[tuple[int, int]]) -> datetime:
    candidates = []
    for offset in (0, 1):
        for hour, minute in windows:
            candidate = (now + timedelta(days=offset)).replace(
                hour=hour, minute=minute, second=0, microsecond=0)
            if candidate > now:
                candidates.append(candidate)
    return min(candidates)


def pick_entries(entries: Sequence[dict], interests: Sequence[str], *,
                 rng: random.Random, limit: int = 2) -> list[dict]:
    """按兴趣关键词优先挑选；没有命中就随机。"""

    scored = []
    for entry in entries:
        title = str(entry.get("title", ""))
        interest_hits = sum(1 for keyword in interests if keyword and keyword in title)
        scored.append((interest_hits, rng.random(), entry))
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = [item[2] for item in scored[:limit]]
    return picked


def format_findings(video_title: str, reading: str, url: str = "") -> str:
    out = f"刷B站看到《{video_title}》：{reading.strip()[:200]}"
    if url:
        out += f" {url}"
    return out


class BiliBrowserService:
    """定时窗口驱动浏览器逛B站：登录、浏览、解读、写日记。"""

    def __init__(self, *, settings, sender, llm_client, writer, owner_qq: int,
                 persona_text: str) -> None:
        self.settings = settings
        self.sender = sender
        self.llm_client = llm_client
        self.writer = writer
        self.owner_qq = int(owner_qq)
        self.persona_text = persona_text
        self.windows = parse_windows(getattr(settings, "bili_browser_windows", "12:00"))
        self.interests = [s.strip() for s in
                          str(getattr(settings, "bili_browser_interests", "")).split(",")
                          if s.strip()]
        self.max_stay = int(getattr(settings, "bili_browser_max_stay_seconds", 480))
        self.chrome_path = str(getattr(settings, "bili_chrome_path", CHROME_DEFAULT_PATH))
        self.profile_dir = Path(settings.data_dir) / "bili_browser_profile"
        self.latest_findings: list[tuple[float, str]] = []
        self._login_reminded = False

    # -- 生命周期 --------------------------------------------------------

    async def run(self) -> None:
        while True:
            now = shanghai_now_naive()
            nxt = next_window(now, self.windows)
            wait = (nxt - now).total_seconds()
            logger.info("bili_browser_next_window at=%s", nxt.isoformat(timespec="seconds"))
            await asyncio.sleep(max(1.0, wait))
            try:
                await asyncio.wait_for(self._browse_window(), timeout=self.max_stay + 90)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("bili_browser_window_failed")

    # -- 浏览窗口 --------------------------------------------------------

    async def _browse_window(self) -> None:
        # profile 排他：与主人触发的扫码登录流程互斥。
        async with _PROFILE_LOCK:
            await self._browse_window_locked()

    async def _browse_window_locked(self) -> None:
        from playwright.async_api import async_playwright

        started = datetime.now()
        async with async_playwright() as pw:
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                executable_path=self.chrome_path,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--no-proxy-server"],
                viewport={"width": 1366, "height": 900},
                locale="zh-CN",
            )
            try:
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto("https://www.bilibili.com", timeout=30000)
                await page.wait_for_timeout(2000)
                logged = await page.evaluate(_LOGIN_JS)
                from app.core.bili_subtitle import save_cookies_from_browser

                saved = save_cookies_from_browser(await context.cookies(), self.settings)
                if saved:
                    logger.info("bili_cookies_saved count=%s", saved)
                if logged is False and not self._login_reminded:
                    await self._qr_login(page, context)
                    logged = await page.evaluate(_LOGIN_JS)
                if logged is False:
                    logger.info("bili_browser_anonymous_window login=false")
                await self._browse_feed(page)
            finally:
                await context.close()

    async def _browse_feed(self, page) -> None:
        rng = random.Random()
        await page.goto("https://www.bilibili.com/v/popular/all", timeout=30000)
        for _ in range(3):
            await page.mouse.wheel(0, 1400)
            await page.wait_for_timeout(rng.randint(800, 2200))
        entries = await page.evaluate(_ENTRIES_JS)
        picked = pick_entries(entries, self.interests, rng=rng, limit=2)
        if not picked:
            logger.info("bili_browser_no_entries")
            return
        for entry in picked:
            reading = await self._read_video(page, entry)
            if reading:
                finding = format_findings(entry["title"], reading, url=entry.get("href", ""))
                self.latest_findings.append((datetime.now().timestamp(), finding))
                self.latest_findings = self.latest_findings[-4:]
                self.writer.write(
                    text=finding, now=shanghai_now_naive(), mood=None)

    async def _read_video(self, page, entry: dict) -> str:
        """打开一个视频页，滚到评论区截图，视觉模型解读；返回要点文本。"""
        import tempfile

        await page.goto(entry["href"], timeout=30000)
        await page.wait_for_timeout(rng_wait())
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.55)")
        await page.wait_for_timeout(2500)
        screenshot = Path(tempfile.mkdtemp()) / "bili_comments.png"
        await page.screenshot(path=str(screenshot), full_page=False)
        from app.core.message_content import ImageAttachment

        reading = self.llm_client.generate_text(
            ["System persona: 你是群聊AI，刚在B站一个视频页停留，看了标题和评论区截图。",
             "Group policy: 只描述截图里能看到的：标题、评论区的氛围和争论点。看不清就说看不清，"
             "禁止编造视频内容和情节。两三句话。"],
            images=[ImageAttachment(url="", local_path=str(screenshot))],
            conversation_key="bili_browser",
        )
        return str(reading or "").strip()

    # -- 登录 ------------------------------------------------------------

    async def _qr_login(self, page, context) -> None:
        """二维码发到主人私聊；扫码成功后浏览器导航写入会话。"""

        import httpx

        owner = self.owner_qq
        with httpx.Client(timeout=15.0, trust_env=False) as client:
            resp = client.get(
                "https://passport.bilibili.com/x/passport-login/web/qrcode/generate")
            payload = resp.json().get("data") or {}
            qrcode_key = str(payload.get("qrcode_key") or "")
            login_url = str(payload.get("url") or "")
            if not qrcode_key or not login_url:
                logger.warning("bili_qr_generate_failed")
                return
            import io
            import qrcode

            png = Path(tempfile.mkdtemp()) / "bili_qr.png"
            qrcode.make(login_url).resize((360, 360)).save(
                png, format="PNG")
            try:
                await self.sender.send_private_media(
                    user_id=owner,
                    text=f"B站登录确认（5分钟内有效，小号！）。手机上点这个链接拉起App确认：{login_url}"
                         "，或扫这张二维码",
                    image_files=(str(png),))
            except Exception:
                logger.exception("bili_qr_send_failed")

            deadline = time.monotonic() + QR_WAIT_SECONDS
            while time.monotonic() < deadline:
                poll = client.get(
                    "https://passport.bilibili.com/x/passport-login/web/qrcode/poll",
                    params={"qrcode_key": qrcode_key}).json()
                code = int((poll.get("data") or {}).get("code") or -1)
                if code == 0:
                    success_url = str((poll.get("data") or {}).get("url") or "")
                    if success_url:
                        await page.goto(success_url, timeout=20000)
                    self._login_reminded = False
                    logger.info("bili_qr_login_success")
                    return
                if code == 86038:  # 二维码过期
                    break
                await asyncio.sleep(2)
            self._login_reminded = True
            logger.info("bili_qr_login_not_completed")


def rng_wait() -> int:
    return random.randint(1500, 3500)
