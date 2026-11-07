"""Computer-use 插件：把浏览器操作能力暴露为 agent 工具。"""

from __future__ import annotations

from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule, TurnContext
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.web_hands import WebHands


class WebHandsFeature(FeatureModule):
    """agent 的"身体"：navigate/click/type/scroll/screenshot 五个动作工具。

    每轮唤醒共享一个 WebHands 会话（动作预算 12 个动作）；敏感域名黑名单、
    动作预算与超时都在 WebHands 里强制执行。
    """

    name = "web_hands"

    _settings = None
    _hands: WebHands | None = None

    def attach(self, *, settings, persona_text: str = "") -> None:
        self._settings = settings
        self._persona_text = persona_text

    def _hands_for(self, turn: TurnContext) -> WebHands:
        if self._hands is None:
            chrome = str(getattr(self._settings, "bili_chrome_path", ""))
            self._hands = WebHands(settings=self._settings, chrome_path=chrome)
        return self._hands

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        hands = self._hands_for(turn)
        persona = self._persona_text

        async def navigate(args):
            turn.guard.require_valid()
            url = str(args.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                return ToolOutcome({"error": "invalid_url",
                    "guidance": "url 要以 http:// 或 https:// 开头。"}, "blocked")
            result = await hands.navigate(url)
            if "error" in result:
                return ToolOutcome(result, "blocked")
            return ToolOutcome(result)

        async def click(args):
            turn.guard.require_valid()
            result = await hands.click(str(args.get("target") or ""))
            if "error" in result:
                return ToolOutcome(result, "blocked")
            return ToolOutcome(result)

        async def type_text(args):
            turn.guard.require_valid()
            result = await hands.type_text(str(args.get("target") or ""),
                                           str(args.get("text") or ""))
            if "error" in result:
                return ToolOutcome(result, "blocked")
            return ToolOutcome(result)

        async def scroll(args):
            turn.guard.require_valid()
            result = await hands.scroll(int(args.get("amount") or 1200))
            if "error" in result:
                return ToolOutcome(result, "blocked")
            return ToolOutcome(result)

        async def screenshot(args):
            turn.guard.require_valid()
            llm = turn.router.llm_client
            if llm is None:
                return ToolOutcome({"error": "llm_unavailable"}, "blocked")
            result = await hands.screenshot_analyze(
                question=str(args.get("question") or "这个页面上有什么？"), llm_client=llm,
                persona_hint=persona)
            if "error" in result:
                return ToolOutcome(result, "blocked")
            return ToolOutcome(result)

        common = {"type": "object", "additionalProperties": False}
        return [
            AgentTool("web_navigate", "用浏览器打开一个网页，返回页面标题、可见文本和链接。"
                "url 必须完整（http/https 开头）。银行/支付/政务类站点会被拒绝。",
                {"type": "object", "properties": {"url": {"type": "string", "maxLength": 500}},
                 "required": ["url"], **common}, navigate, "read"),
            AgentTool("web_click", "在当前页面点击一个元素：target 写可见文字（如'登录'）或 CSS 选择器。",
                {"type": "object", "properties": {"target": {"type": "string", "maxLength": 200}},
                 "required": ["target"], **common}, click, "read"),
            AgentTool("web_type", "在当前页面的输入框里输入文字：target 写 placeholder 或选择器。",
                {"type": "object", "properties": {
                    "target": {"type": "string", "maxLength": 200},
                    "text": {"type": "string", "maxLength": 500}},
                 "required": ["target", "text"], **common}, type_text, "read"),
            AgentTool("web_scroll", "向下滚动当前页面，返回滚动后的可见文本。",
                {"type": "object", "properties": {"amount": {"type": "integer"}},
                 "required": [], **common}, scroll, "read"),
            AgentTool("web_screenshot_analyze", "给当前页面截图并按你的问题解读画面内容。"
                "想'看'页面的布局、图片、验证码样式时用；别假装能看视频。",
                {"type": "object", "properties": {"question": {"type": "string", "maxLength": 300}},
                 "required": ["question"], **common}, screenshot, "read"),
        ]
