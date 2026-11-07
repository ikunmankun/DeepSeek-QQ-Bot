"""Computer-use：playwright 驱动的通用网页操作能力。

agent 的"身体"：能打开任意网页、点击、输入、滚动、截图并理解页面。
安全边界：
* 敏感域名黑名单（银行/支付/政务）直接拒绝；
* 每轮会话最多 12 个动作、单页 45 秒超时；
* 独立 profile（data/web_hands_profile），不与用户账号混用；
* 所有动作只读公开页面内容，不提交表单到登录态站点（v1 不做登录）。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ACTIONS_PER_SESSION = 12
PAGE_TIMEOUT_MS = 45000
MAX_TEXT_CHARS = 2000

# 敏感站点：禁止 agent 操作（银行/支付/政务/账号安全页）
_BLOCKED_HOSTS = re.compile(
    r"(bank|pay|alipay|tenpay|unionpay|gov\.cn|12306|appleid|account\."
    r"|icbc|abchina|ccb\.com|boc\.cn|cmbchina|spdb|cebbank|cib\.com\.cn|psbc|pingan)", re.I)

HEADLESS_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--no-proxy-server"]

_NAV_TEXT_JS = """
() => {
  const title = document.title || '';
  const body = (document.body && document.body.innerText) || '';
  const links = [];
  for (const a of document.querySelectorAll('a[href]')) {
    const t = (a.textContent || '').trim().slice(0, 60);
    if (t && a.href.startsWith('http')) links.push(t + ' -> ' + a.href);
    if (links.length >= 12) break;
  }
  return {title, text: body.replace(/\\n{3,}/g, '\\n\\n'), links};
}
"""


def is_blocked_host(url: str) -> bool:
    host = re.sub(r"^https?://", "", str(url or "")).split("/")[0]
    return bool(_BLOCKED_HOSTS.search(host))


class WebHands:
    """共享的浏览器会话管理：惰性启动、动作预算、安全边界。"""

    def __init__(self, *, settings, chrome_path: str = "") -> None:
        self.settings = settings
        self.profile_dir = Path(settings.data_dir) / "web_hands_profile"
        self.chrome_path = chrome_path or "/opt/xhs/chrome/chrome"
        self._context = None
        self._page = None
        self._actions_left = MAX_ACTIONS_PER_SESSION

    def _budget_ok(self) -> bool:
        return self._actions_left > 0

    def _spend(self) -> None:
        self._actions_left -= 1

    async def _ensure(self):
        if self._context is not None:
            return self._page
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir), executable_path=self.chrome_path,
            headless=True, args=HEADLESS_ARGS,
            viewport={"width": 1366, "height": 900}, locale="zh-CN")
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        return self._page

    async def close(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
        self._context = self._page = None
        self._actions_left = MAX_ACTIONS_PER_SESSION

    # -- 动作 ------------------------------------------------------------

    async def navigate(self, url: str) -> dict:
        if is_blocked_host(url):
            return {"error": "blocked_host", "url": url,
                    "note": "这个域名在禁止操作名单里（银行/支付/政务）。"}
        if not self._budget_ok():
            return {"error": "action_budget_exhausted"}
        page = await self._ensure()
        self._spend()
        try:
            await page.goto(url, timeout=PAGE_TIMEOUT_MS)
            await page.wait_for_timeout(1500)
            data = await page.evaluate(_NAV_TEXT_JS)
            return {"url": page.url, "title": data["title"],
                    "text": data["text"][:MAX_TEXT_CHARS], "links": data["links"]}
        except Exception as exc:
            return {"error": f"navigate_failed: {type(exc).__name__}"}

    async def click(self, target: str) -> dict:
        if self._budget_ok() is False:
            return {"error": "action_budget_exhausted"}
        page = await self._ensure()
        self._spend()
        try:
            # 优先按可见文本点（更像人），失败退回 CSS 选择器
            try:
                await page.get_by_text(target, exact=False).first.click(timeout=8000)
            except Exception:
                await page.click(target, timeout=8000)
            await page.wait_for_timeout(1500)
            data = await page.evaluate(_NAV_TEXT_JS)
            return {"url": page.url, "title": data["title"], "text": data["text"][:MAX_TEXT_CHARS]}
        except Exception as exc:
            return {"error": f"click_failed: {type(exc).__name__}"}

    async def type_text(self, target: str, text: str) -> dict:
        if not self._budget_ok():
            return {"error": "action_budget_exhausted"}
        page = await self._ensure()
        self._spend()
        try:
            try:
                await page.get_by_placeholder(target).first.fill(text, timeout=8000)
            except Exception:
                await page.fill(target, text, timeout=8000)
            return {"typed": True, "url": page.url}
        except Exception as exc:
            return {"error": f"type_failed: {type(exc).__name__}"}

    async def scroll(self, amount: int = 1200) -> dict:
        if not self._budget_ok():
            return {"error": "action_budget_exhausted"}
        page = await self._ensure()
        self._spend()
        await page.mouse.wheel(0, int(amount))
        await page.wait_for_timeout(1200)
        data = await page.evaluate(_NAV_TEXT_JS)
        return {"url": page.url, "text": data["text"][:MAX_TEXT_CHARS]}

    async def screenshot_analyze(self, *, question: str, llm_client, persona_hint: str = "") -> dict:
        """截图当前页面 → 视觉模型按问题解读。"""
        if not self._budget_ok():
            return {"error": "action_budget_exhausted"}
        page = await self._ensure()
        self._spend()
        import tempfile

        shot = Path(tempfile.mkdtemp()) / "web_shot.png"
        await page.screenshot(path=str(shot), full_page=False)
        from app.core.message_content import ImageAttachment

        reading = llm_client.generate_text(
            ["System persona: 你是群聊AI，刚用浏览器打开了一个网页并截了图。",
             "Group policy: 只描述截图里能看到的内容，回答问题；看不清/没找到就如实说。"
             + (f"人设提示：{persona_hint}" if persona_hint else "")],
            images=[ImageAttachment(url="", local_path=str(shot))],
            conversation_key="web_hands",
        )
        return {"url": page.url, "title": await page.title(), "reading": str(reading or "").strip()[:800]}
