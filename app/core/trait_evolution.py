"""证据制人格进化：特质从行为记录里"长"出来，不是设定出来的。

参照 dsh-identity 的核心思路——性格不是一段写死的人设，而是"哪些行为
反复出现"的沉淀：一条候选特质必须被证据命中至少 PROMOTION_MENTIONS 次、
且证据跨越至少 PROMOTION_SPAN_DAYS 天，才从 candidate 晋升为 confirmed，
进入人格提示词。证据来源是机器人自己的日记和关于它行为的记忆，由周期
worker 用 LLM 归纳成短特质句。

与静态人设的关系：persona.yaml 里的核心人格（语气、禁忌、例句）永远
以静态文件为准，本模块只补一段有界的"衍生特质"——人格可以生长，
但不能被一次幻觉改写。已确认特质不会自动删除；要清理得人工改表
（status 置 dismissed）。
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select

from app.core.time_utils import shanghai_naive, shanghai_now_naive, stored_as_utc
from app.storage.models import MemoryItem, PersonaTraitCandidate

logger = logging.getLogger(__name__)

#: 晋升证据线：与记忆晋升同一族规则——反复、跨天出现才算稳定特质。
PROMOTION_MENTIONS = 3
PROMOTION_SPAN_DAYS = 2.0
#: 提示词最多带几条已确认特质；人格是主人设+有界生长，不能喧宾夺主。
MAX_PROMPT_TRAITS = 5
#: 全库确认特质上限：超出的旧特质按 last_seen 淘汰出提示词（仍留库）。
MAX_CONFIRMED_TRAITS = 12
#: 单条特质文本长度上限——特质是短句，不是小作文。
MAX_TRAIT_CHARS = 24
#: 保留的证据摘录条数。
MAX_EVIDENCE_SNIPPETS = 5
#: 相似度合并阈值：不同措辞描述同一特质时合并计数。
SIMILARITY_THRESHOLD = 0.82
#: 每次 tick 归纳的候选上限（宁缺毋滥，LLM 幻觉出来的特质攒不齐证据）。
MAX_CANDIDATES_PER_TICK = 3

_STATUS_CANDIDATE = "candidate"
_STATUS_CONFIRMED = "confirmed"
_STATUS_DISMISSED = "dismissed"


def _trait_key(text: str) -> str:
    normalized = re.sub(r"[\s，。！？、,.!?；;：:'\"「」（）()~*]+", "", str(text or "")).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= SIMILARITY_THRESHOLD


@dataclass(frozen=True)
class TraitEvidence:
    """一条原始证据（日记片段或记忆内容）。"""

    text: str
    origin: str  # diary | memory


def collect_evidence(*, data_dir, bot_name: str, engine, now: datetime) -> list[TraitEvidence]:
    """汇总可用于归纳特质的原始证据：最近几天日记 + 关于 bot 的记忆。"""

    evidence: list[TraitEvidence] = []
    try:
        from app.core.diary import DiaryWriter

        diary_text = DiaryWriter(data_dir).read_recent(max_chars=2500, max_days=3)
        if diary_text.strip():
            evidence.append(TraitEvidence(text=diary_text[:2500], origin="diary"))
    except Exception:
        logger.exception("trait_evidence_diary_failed")
    try:
        from app.storage.db import session_scope

        # SQL 过滤用 naive 上海钟面（SQLite 字符串比较，混 UTC 会失准）。
        cutoff = shanghai_naive(now) - timedelta(days=30)
        with session_scope(engine) as session:
            rows = list(
                session.scalars(
                    select(MemoryItem)
                    .where(
                        MemoryItem.status == "active",
                        MemoryItem.memory_kind.in_(["decision", "fact", "event"]),
                        MemoryItem.last_seen_at >= cutoff,
                        MemoryItem.content.like(f"%{bot_name}%"),
                    )
                    .order_by(MemoryItem.last_seen_at.desc())
                    .limit(30)
                )
            )
        evidence.extend(
            TraitEvidence(text=str(row.content or "")[:200], origin="memory")
            for row in rows
            if str(row.content or "").strip()
        )
    except Exception:
        logger.exception("trait_evidence_memory_failed")
    return evidence


TRAIT_EXTRACT_PROMPT = (
    "你负责维护一个 QQ 群聊机器人的性格成长。下面是它最近的日记和行为记录。"
    "请从这些记录里归纳出它身上\"反复表现出来的\"性格或说话特质，"
    f"最多 {MAX_CANDIDATES_PER_TICK} 条；每条不超过 {MAX_TRAIT_CHARS} 个字，"
    "必须是记录里真的反复出现的行为（例如\"嘴上嫌弃还是会把事办了\"），"
    "一次性的想法、单次事件、泛泛的形容词都不要。没有够格的特质就返回空数组。\n"
    '只输出 JSON 数组，格式：[{"trait": "特质短句", "evidence": "支持它的原文摘录（30字内）"}]'
)


def extract_trait_candidates(*, client, evidence: list[TraitEvidence]) -> list[dict]:
    """调 LLM 把原始证据归纳成候选特质；返回 [{trait, evidence}]，失败返回 []。"""

    if not evidence:
        return []
    joined = "\n\n".join(f"[{item.origin}]\n{item.text}" for item in evidence)
    try:
        raw = client.generate_text([
            TRAIT_EXTRACT_PROMPT,
            "以下是行为记录：",
            joined[:6000],
        ])
    except Exception:
        logger.exception("trait_extract_llm_failed")
        return []
    return parse_trait_candidates(raw)


def parse_trait_candidates(raw: str | None) -> list[dict]:
    """解析 LLM 输出的候选特质；宽松处理，解析不出就当这轮没有候选。"""

    if not raw:
        return []
    text = str(raw).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    parsed: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        trait = re.sub(r"\s+", " ", str(item.get("trait") or "")).strip()[:MAX_TRAIT_CHARS]
        evidence = re.sub(r"\s+", " ", str(item.get("evidence") or "")).strip()[:60]
        if trait:
            parsed.append({"trait": trait, "evidence": evidence})
    return parsed[:MAX_CANDIDATES_PER_TICK]


def merge_trait_candidates(session, *, candidates: list[dict], origin: str,
                           now: datetime) -> int:
    """把候选特质合并进证据表：新/相似措辞都累加证据并刷新时间戳。

    返回本次真正新增/加的证据条数。晋升在读取时按证据现算（见
    confirmed_traits），与记忆晋升同款：mention 只增不减，晋升单向稳定。
    """

    merged = 0
    existing = list(
        session.scalars(
            select(PersonaTraitCandidate).where(
                PersonaTraitCandidate.status != _STATUS_DISMISSED
            )
        )
    )
    for candidate in candidates:
        trait = str(candidate.get("trait") or "").strip()[:MAX_TRAIT_CHARS]
        snippet = str(candidate.get("evidence") or "").strip()[:60]
        if not trait:
            continue
        target = None
        key = _trait_key(trait)
        for row in existing:
            if row.trait_key == key or _similar(row.display_text, trait):
                target = row
                break
        if target is None:
            target = PersonaTraitCandidate(
                trait_key=key,
                display_text=trait,
                origin=origin,
                evidence_count=0,
                evidence_json=[],
                status=_STATUS_CANDIDATE,
                first_seen_at=now,
                last_seen_at=now,
            )
            session.add(target)
            existing.append(target)
        target.evidence_count = int(target.evidence_count or 0) + 1
        target.last_seen_at = now
        if snippet:
            snippets = list(target.evidence_json or [])
            snippets.append(snippet)
            target.evidence_json = snippets[-MAX_EVIDENCE_SNIPPETS:]
        merged += 1
    return merged


def confirmed_traits(engine, *, limit: int = MAX_PROMPT_TRAITS) -> list[str]:
    """当前已确认的特质（按最近出现排序，最多 limit 条）。"""

    try:
        from app.storage.db import session_scope

        with session_scope(engine) as session:
            rows = list(
                session.scalars(
                    select(PersonaTraitCandidate)
                    .where(PersonaTraitCandidate.status == _STATUS_CONFIRMED)
                    .order_by(PersonaTraitCandidate.last_seen_at.desc())
                    .limit(MAX_CONFIRMED_TRAITS)
                )
            )
            return [str(row.display_text) for row in rows[: max(1, limit)]]
    except Exception:
        logger.exception("trait_confirmed_read_failed")
        return []


def promote_due_traits(session, *, now: datetime) -> int:
    """把证据凑够（次数 + 跨天）的候选特质晋升为 confirmed；返回晋升数。"""

    rows = list(
        session.scalars(
            select(PersonaTraitCandidate).where(
                PersonaTraitCandidate.status == _STATUS_CANDIDATE
            )
        )
    )
    promoted = 0
    for row in rows:
        if int(row.evidence_count or 0) < PROMOTION_MENTIONS:
            continue
        span = stored_as_utc(now) - stored_as_utc(row.first_seen_at)
        if span < timedelta(days=PROMOTION_SPAN_DAYS):
            continue
        row.status = _STATUS_CONFIRMED
        row.confirmed_at = now
        session.add(row)
        promoted += 1
    return promoted


def trait_prompt_line(engine) -> str | None:
    """人格提示词的衍生特质行；还没有确认特质时返回 None。"""

    traits = confirmed_traits(engine)
    if not traits:
        return None
    return (
        "这段时间的相处里，你在自己身上稳定表现出的特点（自然带出来就行，"
        "别总结自己、别全部用上）：" + "；".join(traits) + "。"
    )


class TraitEvolutionService:
    """周期 worker：归纳证据 → 合并候选 → 晋升已凑够证据的特质。"""

    def __init__(self, *, engine, settings, bot_name: str,
                 interval_seconds: float = 6 * 3600.0) -> None:
        self._engine = engine
        self._settings = settings
        self._bot_name = bot_name
        self.interval_seconds = float(interval_seconds)

    async def run(self) -> None:
        import asyncio

        from app.core.worker_status import write_worker_status

        while True:
            write_worker_status(
                self._settings.log_dir, "trait_evolution", "running", self.interval_seconds
            )
            try:
                merged = await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("trait_evolution_tick_failed")
                write_worker_status(
                    self._settings.log_dir, "trait_evolution", "error", self.interval_seconds
                )
            else:
                write_worker_status(
                    self._settings.log_dir, "trait_evolution", "idle", self.interval_seconds
                )
                if merged:
                    logger.info("trait_evolution_merged count=%s", merged)
            await asyncio.sleep(self.interval_seconds)

    def _tick(self) -> int:
        from app.storage.db import session_scope

        now = shanghai_now_naive()
        evidence = collect_evidence(
            data_dir=self._settings.data_dir,
            bot_name=self._bot_name,
            engine=self._engine,
            now=now,
        )
        if not evidence:
            return 0
        candidates = extract_trait_candidates(
            client=self._build_client(), evidence=evidence
        )
        if not candidates:
            return 0
        with session_scope(self._engine) as session:
            merged = merge_trait_candidates(
                session, candidates=candidates, origin=evidence[0].origin, now=now
            )
            promoted = promote_due_traits(session, now=now)
        if promoted:
            logger.info("trait_evolution_promoted count=%s", promoted)
        return merged

    def _build_client(self):
        from app.providers.llm_client import LlmClient

        return LlmClient(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            model=self._settings.llm_model,
            fallback_model="",
            responses_only=True,
            responses_model=self._settings.llm_model,
            max_output_tokens=2048,
            timeout_seconds=90.0,
            # 辅助小任务用 low：medium 的推理 token 会吃光输出预算，
            # 连正文都产不出（部署后实测 deepseek-flash 仍截断无文本）。
            reasoning_effort="low",
            request_max_attempts=1,
        )
