"""一次性触发B站扫码登录：二维码发到主人私聊，扫码后会话写入浏览器 profile。

用法（容器内）：PYTHONPATH=/workspace python scripts/bili_login_now.py
独立于常驻机器人进程运行；profile 与 BiliBrowserService 共用，登录结果互认。
自动换码：每张码约 5 分钟，最多换 3 张（总窗口约 15 分钟）。
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import httpx
import qrcode

from app.adapters.napcat_ws import NapCatGateway
from app.config import AppSettings
from app.core.bili_browser import CHROME_DEFAULT_PATH

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Referer": "https://passport.bilibili.com/"}

_LOGIN_JS = """
async () => {
  try {
    const r = await fetch('https://api.bilibili.com/x/web-interface/nav', {credentials: 'include'});
    const j = await r.json();
    return !!(j.data && j.data.isLogin);
  } catch (e) { return null; }
}
"""

ROUNDS = 3
ROUND_SECONDS = 290


async def send_qr_to_owner(gateway: NapCatGateway, owner: int, login_url: str) -> None:
    buffer = io.BytesIO()
    qrcode.make(login_url).resize((360, 360)).save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    await gateway.call_api("send_private_msg", {
        "user_id": owner,
        "message": [
            {"type": "text", "data": {"text":
                "B站登录确认（5分钟内有效，小号！）。两种方式任选：\n"
                "① 在装了小号App的手机上点这个链接，会拉起App确认：\n"
                f"{login_url}\n"
                "② 或用小号App扫下面这张二维码"}},
            {"type": "image", "data": {"file": f"base64://{image_b64}"}},
        ],
    })


async def _noop_payload(payload) -> None:
    return None


async def main() -> int:
    from playwright.async_api import async_playwright

    settings = AppSettings()
    owner = int(settings.owner_qq)
    profile = Path(settings.data_dir) / "bili_browser_profile"
    chrome = str(getattr(settings, "bili_chrome_path", CHROME_DEFAULT_PATH))
    gateway = NapCatGateway(ws_url=settings.napcat_ws_url, reconnect_forever=False,
                            access_token=getattr(settings, "napcat_access_token", None) or None)

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
                    print("已经是登录状态，无需扫码。")
                    return 0

                gateway_task = asyncio.create_task(
                    gateway.connect_and_consume(_noop_payload))
                for _ in range(50):
                    if gateway.websocket is not None:
                        break
                    await asyncio.sleep(0.1)

                result = "未完成"
                for round_index in range(ROUNDS):
                    resp = (await http.get(
                        "https://passport.bilibili.com/x/passport-login/web/qrcode/generate",
                        headers=_HEADERS)).json()
                    data = resp.get("data") or {}
                    qrcode_key, login_url = str(data.get("qrcode_key") or ""), str(data.get("url") or "")
                    if not qrcode_key:
                        print("二维码生成失败")
                        break
                    await send_qr_to_owner(gateway, owner, login_url)
                    print(f"第{round_index + 1}张码已发到主人私聊，等待扫码（5分钟）……")

                    deadline = asyncio.get_running_loop().time() + ROUND_SECONDS
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
                            result = ("登录成功，会话已写入 profile" if logged
                                      else "扫码成功但会话校验未通过")
                            break
                        if code == 86038:  # 本张码过期 → 自动换新码
                            refreshed = True
                            print("本轮码过期，自动换新码……")
                            break
                        await asyncio.sleep(2)
                    if result != "未完成":
                        break
                    if not refreshed:
                        result = "未完成（超时或未扫）"
                        break
                    print("自动换新码……")
                print(result)
                gateway_task.cancel()
                return 0 if "成功" in result else 1
            finally:
                await context.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
