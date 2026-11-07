"""特质进化插件：证据制人格生长，确认特质注入人格提示词。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from app.agent.plugins.base import FeatureModule


class TraitEvolutionFeature(FeatureModule):
    """从日记/行为记忆里归纳特质，凑够证据后进入人格提示词。"""

    name = "trait_evolution"

    _engine = None
    _settings = None
    _bot_name = ""

    def attach(self, *, engine, settings, bot_name: str) -> None:
        self._engine = engine
        self._settings = settings
        self._bot_name = bot_name

    def create_workers(self) -> Sequence[Any]:
        from app.core.trait_evolution import TraitEvolutionService

        if self._engine is None or self._settings is None:
            return ()
        if not getattr(self._settings, "trait_evolution_enabled", True):
            return ()
        return [TraitEvolutionService(
            engine=self._engine,
            settings=self._settings,
            bot_name=self._bot_name,
        )]

    def prompt_lines(self, *, kind: str, group_id: int, now: datetime,
                     observing: bool,
                     target_user_id: int | None = None) -> Sequence[str]:
        if kind != "group" or self._engine is None:
            return ()
        from app.core.trait_evolution import trait_prompt_line

        line = trait_prompt_line(self._engine)
        return (line,) if line else ()
