"""模糊记忆检索：用户只记得特征、想不起名字时的"话到嘴边"救援。

链路：LLM 扩词（关键词/别名 + 最可能的候选名）→ 多路混合检索
（记忆事实 FTS+向量、检索文档 FTS+向量、群聊原消息 LIKE、私聊原消息 LIKE、人生笔记 FTS+向量）
→ RRF 合并去重 → 带来源/时间的候选列表。记忆全空时把 LLM 猜的候选名作为 web_hints
交回调用方，由 agent 拿现成 web_search 自证后再答——记忆管"聊过/记过"，
网页管"从没聊过但见过"。

同步调用（DB 查询 + 一次 LLM 扩词，秒级），调用方用 blocking runner 包裹。
群聊面只搜本群；私聊面搜全部群 + 本人私聊历史——大肥鱼是一个人，不是群权限墙。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import or_, select

from app.storage.db import session_scope
from app.storage.models import Group, LifeNote, Message, RetrievalDocument
from app.storage.repositories.memory import MemoryRepository
from app.storage.repositories.messages import MessageRepository
from app.storage.repositories.retrieval import RetrievalDocumentRepository

logger = logging.getLogger(__name__)

RECALL_LIMIT_MAX = 12
_SNIPPET_CHARS = 200
_RRF_K = 60
#: 通道权重：提取过的事实记忆 > 原话 ≈ 文档 > 私聊
_CHANNEL_WEIGHTS = {
    "memory_fts": 1.2, "memory_vec": 1.0, "doc_fts": 1.0, "doc_vec": 1.0,
    "group_msg": 1.1, "private_msg": 0.9, "life": 1.0,
}

_RECALL_PROMPT = (
    "用户想找回一个想不起名字的东西（游戏/动画/漫画/主播/梗/事件都可能是），只记得一些特征。"
    "下面是用户的描述。请输出站内检索用的关键词，每行一个，最多6行："
    "可能是的名字（中文/英文/日文/简称/别名都算）、题材或玩法词、画面或角色特征词。"
    "不要解释，不要编号。实在写不出就只输出：无\n"
    "最后一行额外给最可能的候选名，格式固定为：猜:名字1、名字2（0-3个，没有把握就写 猜:无）"
)


@dataclass(slots=True)
class RecallHit:
    source: str           # memory / document / group_message / private_message / life_note
    label: str            # 展示用：记忆/摘要/记录/群聊/私聊/笔记
    group_id: int | None
    when: str             # YYYY-MM-DD，未知为 ""
    speaker: str
    content: str
    source_id: str


@dataclass(slots=True)
class FuzzyRecallResult:
    hits: list[RecallHit] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    web_hints: list[str] = field(default_factory=list)


def expand_recall_queries(description: str, *, llm) -> tuple[list[str], list[str]]:
    """LLM 扩词。失败/无 LLM 时容忍降级为空（主描述兜底）。"""
    if llm is None or not description.strip():
        return [], []
    try:
        raw = str(llm.generate_text([_RECALL_PROMPT, description.strip()[:800]]) or "").strip()
    except Exception as exc:
        logger.info("fuzzy_recall_expand_failed error=%s", type(exc).__name__)
        return [], []
    keywords: list[str] = []
    web_hints: list[str] = []
    for line in raw.splitlines():
        line = line.strip().strip("-•\t ")
        if not line or line in ("无", "猜:无", "猜：无"):
            continue
        if line.startswith(("猜:", "猜：")):
            names = re.split(r"[、，,;；/]", line[2:].strip())
            web_hints.extend(name.strip() for name in names if 1 <= len(name.strip()) <= 40)
        elif 2 <= len(line) <= 30:
            keywords.append(line)
    return keywords[:5], web_hints[:3]


def _snippet(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:_SNIPPET_CHARS]


def _day(value) -> str:
    try:
        return value.strftime("%Y-%m-%d") if value else ""
    except AttributeError:
        return ""


def _message_speaker(message: Message) -> str:
    payload = message.raw_json if isinstance(message.raw_json, dict) else {}
    sender = payload.get("sender") or {}
    return str(sender.get("card") or sender.get("nickname") or message.user_id or "")


class FuzzyRecallService:
    """无状态检索门面：引擎 + 配置（embedding 懒建单例）+ 可选 LLM。"""

    def __init__(self, *, engine, settings, llm_client=None):
        self._engine = engine
        self._settings = settings
        self._llm = llm_client

    def recall(self, description: str, *, group_ids: tuple[int, ...] | None = (),
               private_user_id: int | None = None, limit: int = 8) -> dict:
        """group_ids=() 表示不搜群聊；None 表示搜全部群（私聊面用）。"""
        provider = get_shared_embedding_provider(self._settings)
        result = fuzzy_recall(
            self._engine, description=description,
            group_ids=self._resolve_group_ids(group_ids),
            private_user_id=private_user_id,
            embedding_provider=provider, llm=self._llm, limit=limit)
        return {
            "hits": [
                {"source": hit.source, "label": hit.label, "group_id": hit.group_id,
                 "when": hit.when, "speaker": hit.speaker, "content": hit.content,
                 "source_id": hit.source_id}
                for hit in result.hits],
            "keywords": result.keywords, "web_hints": result.web_hints,
        }

    def _resolve_group_ids(self, group_ids) -> tuple[int, ...]:
        if group_ids is not None:
            return tuple(int(gid) for gid in group_ids if int(gid) > 0)
        try:
            with session_scope(self._engine) as session:
                return tuple(int(gid) for gid in session.scalars(select(Group.group_id)) if int(gid) > 0)
        except Exception:
            logger.exception("fuzzy_recall_group_enumeration_failed")
            return ()


def fuzzy_recall(engine, *, description: str, group_ids: tuple[int, ...] = (),
                 private_user_id: int | None = None, embedding_provider=None, llm=None,
                 limit: int = 8) -> FuzzyRecallResult:
    description = str(description or "").strip()
    limit = max(1, min(RECALL_LIMIT_MAX, int(limit)))
    if not description:
        return FuzzyRecallResult()
    keywords, web_hints = expand_recall_queries(description, llm=llm)
    # LIKE 通道的关键词：LLM 扩词优先；没 LLM 时按标点把描述切小块
    # （"白毛 角色 画风好"这类特征词才 LIKE 得中存过的原话）。
    chunks = [chunk.strip() for chunk in
              re.split(r"[\s，。,.;；!！?？、\n（）()\"']+", description) if len(chunk.strip()) >= 2]
    like_terms = list(dict.fromkeys(
        keywords[:3] + ([description] if len(description) <= 24 else []) + chunks))[:5]
    scores: dict[tuple[str, str], float] = {}
    payloads: dict[tuple[str, str], RecallHit] = {}

    def add(channel: str, key: tuple[str, str], hit: RecallHit, rank: int = 0) -> None:
        scores[key] = scores.get(key, 0.0) + _CHANNEL_WEIGHTS[channel] / (_RRF_K + rank + 1)
        existing = payloads.get(key)
        if existing is None or len(hit.content) > len(existing.content):
            payloads[key] = hit

    with session_scope(engine) as session:
        for group_id in group_ids:
            # 1) 提取过的记忆事实：FTS + 向量（记忆向量自带 hash 查询向量，无需 provider）
            memory_repo = MemoryRepository(session)
            memory_hit = _memory_hit_factory(group_id)
            for rank, item in enumerate(memory_repo.search_group_memories_fts(
                    scope_id=str(group_id), query=description, limit=5)):
                add("memory_fts", ("memory", f"{item.id}"), memory_hit(item), rank)
            for rank, item in enumerate(memory_repo.search_group_memories_vector(
                    scope_id=str(group_id), query=description, limit=5)):
                add("memory_vec", ("memory", f"{item.id}"), memory_hit(item), rank)
            for keyword in keywords[:3]:
                for rank, item in enumerate(memory_repo.search_group_memories_fts(
                        scope_id=str(group_id), query=keyword, limit=3)):
                    add("memory_fts", ("memory", f"{item.id}"), memory_hit(item), rank)
            # 2) 检索文档层（摘要/事实/原始消息，取决于索引开关）：FTS + 向量
            doc_repo = RetrievalDocumentRepository(session)
            for rank, doc in enumerate(doc_repo.search_group_documents_fts(
                    group_id=group_id, query=description, limit=5)):
                add("doc_fts", ("document", f"{doc.id}"), _document_hit(doc, group_id), rank)
            if keywords:
                for rank, doc in enumerate(doc_repo.search_group_documents_fts(
                        group_id=group_id, query=keywords[0], limit=3)):
                    add("doc_fts", ("document", f"{doc.id}"), _document_hit(doc, group_id), rank)
            _document_vector_add(add, session, group_id=group_id, description=description,
                                 embedding_provider=embedding_provider)
            # 3) 群聊原消息 LIKE（raw_v3 未开时的原话主通道）
            message_repo = MessageRepository(session)
            if like_terms:
                for rank, message in enumerate(message_repo.list_group_messages_matching_terms(
                        group_id=group_id, terms=like_terms[:4], exclude_platform_msg_ids=set(),
                        limit=6)):
                    add("group_msg", ("group_message", f"{message.id}"), RecallHit(
                        source="group_message", label="群聊", group_id=group_id,
                        when=_day(message.timestamp), speaker=_message_speaker(message),
                        content=_snippet(message.plain_text), source_id=str(message.id)), rank)
    # 4) 私聊原消息（只搜本人）
    if private_user_id:
        _private_message_add(add, engine, user_id=int(private_user_id), terms=like_terms)
    # 5) 人生笔记：自带 FTS+向量混合
    try:
        from app.core.life_notes import search_life_notes
        for rank, note in enumerate(search_life_notes(
                engine, query=description, limit=3, embedding_provider=embedding_provider)):
            add("life", ("life_note", f"{note.id}"), RecallHit(
                source="life_note", label="笔记", group_id=None,
                when=_day(getattr(note, "updated_at", None) or getattr(note, "created_at", None)),
                speaker=str(getattr(note, "period", "") or ""),
                content=_snippet(f"{getattr(note, 'title', '')} {getattr(note, 'content', '')}"),
                source_id=str(note.id)), rank)
    except Exception:
        logger.exception("fuzzy_recall_life_notes_failed")

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    hits = [payloads[key] for key, _ in ranked if key in payloads]
    return FuzzyRecallResult(hits=hits, keywords=keywords, web_hints=web_hints)


def _memory_hit_factory(group_id: int):
    def build(item: MemoryItem) -> RecallHit:
        return RecallHit(
            source="memory", label="记忆", group_id=group_id,
            when=_day(item.last_seen_at), speaker=str(item.subject_id or ""),
            content=_snippet(item.content), source_id=str(item.id))
    return build


def _document_hit(doc: RetrievalDocument, group_id: int) -> RecallHit:
    kind_label = "摘要" if "summary" in str(doc.document_kind or "") else "记录"
    return RecallHit(
        source="document", label=kind_label, group_id=group_id,
        when=_day(doc.start_at), speaker="",
        content=_snippet(doc.content), source_id=str(doc.id))


def _document_vector_add(add, session, *, group_id: int, description: str,
                         embedding_provider) -> None:
    """文档向量通道：provider 不可用/索引缺失/维度不匹配都直接跳过，不拖垮其它通道。"""
    if embedding_provider is None or not getattr(embedding_provider, "available", False):
        return
    try:
        vector = embedding_provider.embed_query(description)
        if not vector:
            return
        identity = embedding_provider.identity
        doc_repo = RetrievalDocumentRepository(session)
        for hit_row in doc_repo.search_group_documents_vector_hits(
                group_id=group_id, embedding=vector,
                provider=identity.provider, model=identity.model,
                dimensions=identity.dimensions, version=identity.version,
                generation=None, limit=5):
            document = session.get(RetrievalDocument, int(hit_row.document_id))
            if document is None or not str(document.content or "").strip():
                continue
            add("doc_vec", ("document", f"{hit_row.document_id}"), RecallHit(
                source="document", label="记录", group_id=group_id,
                when=_day(hit_row.start_at), speaker="", content=_snippet(document.content),
                source_id=str(hit_row.document_id)))
    except Exception as exc:
        logger.info("fuzzy_recall_doc_vector_skipped error=%s", type(exc).__name__)


def _private_message_add(add, engine, *, user_id: int, terms: list[str]) -> None:
    if not terms:
        return
    try:
        with session_scope(engine) as session:
            stmt = (select(Message)
                    .where(Message.group_id.is_(None), Message.user_id == user_id,
                           Message.plain_text.is_not(None),
                           or_(*(Message.plain_text.ilike(f"%{term}%") for term in terms)))
                    .order_by(Message.timestamp.desc(), Message.id.desc())
                    .limit(8))
            for rank, message in enumerate(session.scalars(stmt)):
                add("private_msg", ("private_message", f"{message.id}"), RecallHit(
                    source="private_message", label="私聊", group_id=None,
                    when=_day(message.timestamp), speaker="用户",
                    content=_snippet(message.plain_text), source_id=str(message.id)), rank)
    except Exception:
        logger.exception("fuzzy_recall_private_failed")


#: ---- 共享 embedding provider（懒建单例，避免每个面各加载一份 ONNX 模型）----

_EMBEDDING_PROVIDER = None


def get_shared_embedding_provider(settings):
    global _EMBEDDING_PROVIDER
    if _EMBEDDING_PROVIDER is not None:
        return _EMBEDDING_PROVIDER
    from app.providers.semantic_embeddings import (
        DisabledEmbeddingProvider, build_embedding_provider,
    )
    try:
        provider = build_embedding_provider(
            provider=settings.memory_embedding_provider,
            device=settings.memory_embedding_device,
            model=settings.memory_embedding_model,
            dimensions=settings.memory_embedding_dimensions,
            cache_dir=settings.memory_embedding_cache_dir,
            local_files_only=settings.memory_embedding_local_files_only,
            version=settings.memory_embedding_version,
            base_url=settings.memory_embedding_base_url,
            api_key=settings.memory_embedding_api_key,
            timeout_seconds=settings.memory_embedding_timeout_seconds,
        )
    except Exception:
        # 构建失败不缓存，下次调用重试；本次退化为纯 FTS/LIKE
        logger.exception("fuzzy_recall_embedding_provider_build_failed")
        return DisabledEmbeddingProvider(
            dimensions=int(getattr(settings, "memory_embedding_dimensions", 512) or 512))
    _EMBEDDING_PROVIDER = provider
    return _EMBEDDING_PROVIDER
