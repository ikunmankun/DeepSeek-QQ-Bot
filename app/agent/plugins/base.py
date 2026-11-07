"""FeatureModule 插件协议与注册表（Phase 3，参照 AstrBot star 协议）。

一个功能 = 一个插件类，可选提供四种能力：

* ``build_tools(turn)``          —— 往 agent 工具清单里注册工具；
* ``prompt_lines(...)``          —— 往群聊提示词注入动态行（如当前情绪）；
* ``on_inbound(...)``            —— 入站消息钩子（在消息持久化事务内调用，
                                    异常必须自吞，不得影响入库）；
* ``create_workers()``           —— 周期性后台服务（带 ``run()`` 协程）。

注册表（FeatureRegistry）是唯一的接线点：group_main 装配插件并启动 worker，
autonomous_social 收集工具与提示行，router 分发入站钩子。新功能落成插件后
不再触碰核心文件。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from app.agent.tool_agent import AgentTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnContext:
    """一轮 agent 唤醒里插件可用的东西；由 autonomous_social 每轮构造。"""

    router: Any
    job: Any
    guard: Any
    store: Any
    blocking: Any
    interactive: bool
    now: datetime


@dataclass(frozen=True)
class NoticeContext:
    """一条 OneBot notice 事件（戳一戳、进群、退群等）的上下文。"""

    router: Any
    payload: dict
    now: datetime


@dataclass(frozen=True)
class InboundContext:
    """一条入站群消息的持久化事务上下文。

    ``session``/``messages`` 由 router 的入库事务提供；钩子在事务内运行，
    只做轻量写（如情绪状态），失败自吞。
    """

    router: Any
    session: Any
    messages: Any
    event: Any
    now: datetime


class FeatureModule:
    """插件基类：全部钩子都有安全默认实现，子类按需覆盖。"""

    name: str = "feature"

    def build_tools(self, turn: TurnContext) -> Sequence[AgentTool]:
        return ()

    def prompt_lines(self, *, kind: str, group_id: int, now: datetime,
                     observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        return ()

    def on_inbound(self, ctx: InboundContext) -> None:
        return None

    def on_notice(self, ctx: "NoticeContext") -> None:
        return None

    def create_workers(self) -> Sequence[Any]:
        return ()


class FeatureRegistry:
    """插件集合：接线点唯一，核心文件只跟注册表对话。"""

    def __init__(self, features: Sequence[FeatureModule]) -> None:
        self._features = list(features)
        self._worker_tasks: list[Any] = []

    @property
    def names(self) -> list[str]:
        return [f.name for f in self._features]

    def build_tools(self, turn: TurnContext) -> list[AgentTool]:
        tools: list[AgentTool] = []
        for feature in self._features:
            try:
                tools.extend(feature.build_tools(turn))
            except Exception:
                logger.exception("plugin_build_tools_failed feature=%s", feature.name)
        return tools

    def prompt_lines(self, *, kind: str, group_id: int, now: datetime,
                     observing: bool,
                     target_user_id: int | None = None) -> list[str]:
        lines: list[str] = []
        for feature in self._features:
            try:
                lines.extend(feature.prompt_lines(kind=kind, group_id=group_id,
                                                  now=now, observing=observing,
                                                  target_user_id=target_user_id))
            except Exception:
                logger.exception("plugin_prompt_lines_failed feature=%s", feature.name)
        return lines

    def dispatch_notice(self, ctx: "NoticeContext") -> None:
        for feature in self._features:
            try:
                feature.on_notice(ctx)
            except Exception:
                logger.exception("plugin_on_notice_failed feature=%s", feature.name)

    def dispatch_inbound(self, ctx: InboundContext) -> None:
        for feature in self._features:
            try:
                feature.on_inbound(ctx)
            except Exception:
                logger.exception("plugin_on_inbound_failed feature=%s", feature.name)

    async def start_workers(self) -> list[Callable[[], Any]]:
        """创建插件 worker 任务，返回任务列表交给宿主管理生命周期。"""
        import asyncio

        tasks = []
        for feature in self._features:
            try:
                for worker in feature.create_workers():
                    tasks.append(asyncio.create_task(worker.run()))
            except Exception:
                logger.exception("plugin_create_workers_failed feature=%s", feature.name)
        self._worker_tasks.extend(tasks)
        return tasks

    async def stop_workers(self) -> None:
        import asyncio

        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
