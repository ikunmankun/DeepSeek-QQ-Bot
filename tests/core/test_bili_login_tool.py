"""bili_login_qr 主人专属工具：发码、防并发、owner 门。"""

import asyncio
from types import SimpleNamespace

import app.core.bili_browser as bb
from app.agent.plugins.bilibili import BilibiliFeature
from app.agent.tool_agent import ToolOutcome
from app.core.member_identity import GroupMemberIdentity


class Guard:
    def require_valid(self):
        return None


def _feature_turn():
    feature = BilibiliFeature()
    turn = SimpleNamespace(
        router=SimpleNamespace(runtime=SimpleNamespace(settings=SimpleNamespace(
            bot_qq=900000101, owner_qq=900000102, data_dir="/tmp/x",
            bili_chrome_path="/nonexistent/chrome")),
            sender=None),
        job=None, guard=Guard(), store=None, blocking=None,
        interactive=True, now=None)
    return feature, turn


def test_login_tool_exists_and_owner_gated():
    assert "bili_login_qr" in __import__("app.agent.owner_actions",
                                         fromlist=["OWNER_CONTROL_TOOLS"]).OWNER_CONTROL_TOOLS
    feature, turn = _feature_turn()
    tools = {t.name: t for t in feature.build_tools(turn)}
    assert "bili_login_qr" in tools and "browse_bilibili" in tools


def test_login_tool_reports_in_progress(monkeypatch):
    feature, turn = _feature_turn()
    tools = {t.name: t for t in feature.build_tools(turn)}

    class NeverDone:
        def done(self):
            return False

    monkeypatch.setattr(bb, "_LOGIN_TASK", NeverDone())
    result = asyncio.run(tools["bili_login_qr"].handler({}))
    assert result.status == "blocked" and result.data["error"] == "login_in_progress"


def test_login_tool_spawns_background_flow(monkeypatch):
    feature, turn = _feature_turn()
    tools = {t.name: t for t in feature.build_tools(turn)}
    monkeypatch.setattr(bb, "_LOGIN_TASK", None)

    spawned = {}

    async def fake_flow(**kwargs):
        spawned.update(kwargs)
        return "登录成功"

    monkeypatch.setattr(bb, "qr_login_flow", fake_flow)
    result = asyncio.run(tools["bili_login_qr"].handler({}))
    assert result.status == "ok" and result.data["sent"] is True
    assert bb._LOGIN_TASK is not None
    task = bb._LOGIN_TASK
    assert task.get_coro().__name__ != ""  # 任务已创建
    task.cancel()  # 清理（fake_flow 立即返回）
    monkeypatch.setattr(bb, "_LOGIN_TASK", None)


def test_restrict_denies_login_for_nonowner():
    from app.agent.owner_actions import restrict_directed_tools
    from app.adapters.sender import OutboundMessage

    async def unused(args):
        return None

    tools = [__import__("app.agent.tool_agent", fromlist=["AgentTool"]).AgentTool(
        "bili_login_qr", "d", {}, unused, "write")]
    restricted = restrict_directed_tools(tools, owner=False, allow_group_images=False)
    denied = restricted[0]
    assert denied.authorization is not None and denied.authorization() is False
