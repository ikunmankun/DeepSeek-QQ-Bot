"""Computer-use：域名黑名单、动作预算、工具注册。"""

from types import SimpleNamespace

from app.core.web_hands import WebHands, is_blocked_host


def test_blocked_hosts():
    for url in ("https://www.icbc.com.cn/x", "https://pay.example.com/t",
                "https://alipay.com", "http://www.gov.cn/page"):
        assert is_blocked_host(url), url
    for url in ("https://www.bilibili.com/video/BV1", "https://www.zhihu.com/question",
                "https://example.com"):
        assert not is_blocked_host(url), url


def test_action_budget_exhausts(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, bili_chrome_path="")
    hands = WebHands(settings=settings, chrome_path="/nonexistent")
    hands._actions_left = 0  # 预算在启动浏览器之前就拦截
    result = asyncio_run(hands.navigate("https://example.com"))
    assert result == {"error": "action_budget_exhausted"}


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
