"""群黑话学习（麦麦式"黑话挖掘"的轻量版）。

周期 worker 扫最近几天的群消息，统计高频特征词（CJK 2-4 字片段 + 3 字符
以上英文串），过滤通用停用词后落表；prompt_lines 把当前的黑话清单注入
群聊提示词，让模型聊到相关话题时自然使用——用词越来越像本群人。
纯数据库+规则，不调 LLM。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select

from app.agent.plugins.base import FeatureModule
from app.core.time_utils import shanghai_now_naive, stored_as_utc

logger = logging.getLogger(__name__)

WINDOW_DAYS = 3
SCAN_LIMIT = 2000
MIN_COUNT = 6
MAX_TERMS = 12
TERMS_IN_PROMPT = 8

_STOPWORDS = frozenset({
    "的", "了", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "哈", "是", "我", "你",
    "他", "她", "它", "这", "那", "就", "都", "也", "还", "不", "没", "有", "在",
    "好", "说", "去", "来", "个", "们", "什么", "怎么", "这个", "那个", "不是",
    "自己", "因为", "所以", "但是", "可以", "现在", "一下", "知道", "感觉",
})

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,4}")
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class SlangLearner:
    """统计窗口内的群高频词，落 group_slang_terms 表。"""

    def __init__(self, engine) -> None:
        self.engine = engine

    def learn(self, *, group_ids: Sequence[int], now: datetime) -> int:
        from app.storage.db import session_scope
        from app.storage.models import Message, GroupSlangTerm

        learned = 0
        since = (now - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
        for group_id in group_ids:
            with session_scope(self.engine) as session:
                rows = session.execute(
                    select(Message.plain_text).where(
                        Message.group_id == group_id,
                        Message.timestamp >= since,
                        Message.msg_type == "text",
                    ).limit(SCAN_LIMIT)
                ).scalars().all()
                counts: dict[str, int] = {}
                for text in rows:
                    normalized = " ".join(str(text or "").split())
                    if not normalized:
                        continue
                    for run in _CJK_RUN.findall(normalized):
                        if run in _STOPWORDS:
                            continue
                        counts[run] = counts.get(run, 0) + 1
                    for word in _ASCII_WORD.findall(normalized):
                        key = word.lower()
                        if key in _STOPWORDS:
                            continue
                        counts[key] = counts.get(key, 0) + 1
                for term, count in sorted(counts.items(), key=lambda kv: -kv[1])[:MAX_TERMS]:
                    if count < MIN_COUNT:
                        break
                    row = session.get(GroupSlangTerm, (group_id, term))
                    if row is None:
                        row = GroupSlangTerm(group_id=group_id, term=term, count=count)
                        session.add(row)
                    else:
                        row.count = count
                    row.last_seen = now
                    learned += 1
        return learned

    def top_terms(self, group_id: int, *, limit: int = TERMS_IN_PROMPT) -> list[tuple[str, int]]:
        from app.storage.db import session_scope
        from app.storage.models import GroupSlangTerm

        with session_scope(self.engine) as session:
            rows = session.execute(
                select(GroupSlangTerm.term, GroupSlangTerm.count)
                .where(GroupSlangTerm.group_id == group_id)
                .order_by(GroupSlangTerm.count.desc())
                .limit(limit)
            ).all()
            return [(str(term), int(count)) for term, count in rows]


class SlangLearningService:
    """周期 worker：全量刷新群黑话表。"""

    def __init__(self, *, engine, group_ids: Sequence[int],
                 interval_seconds: float = 6 * 3600) -> None:
        self.learner = SlangLearner(engine)
        self.group_ids = tuple(group_ids)
        self.interval_seconds = float(interval_seconds)

    async def run(self) -> None:
        import asyncio

        while True:
            try:
                learned = await asyncio.to_thread(
                    self.learner.learn, group_ids=self.group_ids,
                    now=shanghai_now_naive())
                if learned:
                    logger.info("slang_learned entries=%s groups=%s",
                                learned, len(self.group_ids))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("slang_learning_tick_failed")
            await asyncio.sleep(self.interval_seconds)
