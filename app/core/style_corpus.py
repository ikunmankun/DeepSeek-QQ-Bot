"""真人对话语料向量库（PersonalDialog 蒸馏）：按当前消息检索"相似情境下
真人的接法"，作为风格参考注入回复 prompt。

设计要点：
* 存储：元数据表 corpus_style_examples（常规 Base 表，启动自建）+ vec0
  虚拟表 corpus_style_vec（SQL 侧 KNN，与 retrieval_documents_vec_* 同一套
  sqlite-vec 模式）。两表靠 hit_id 关联，重建=drop+重导；
* 向量 L2 归一化后入库 → vec0 默认 L2 距离可精确换算 cosine（cos = 1 - d²/2），
  不依赖 sqlite-vec 版本的 metric 选项；
* kind 分区：0=全量 casual 池，1=女性声子集（私聊甜心面用）——vec0 PARTITION
  KEY 查询期过滤，不用全表扫；
* 全链 fail-soft：provider 不可用/表不存在/任何异常一律返回空，绝不影响回复
  主链路（与思考深度路由同一原则）；
* 注入只给"长度、节奏、接法"参考，渲染前缀明确要求内容别照抄（微博语料
  不能原样进人设）。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path

from sqlalchemy import text

from app.storage.models import CorpusStyleExample

logger = logging.getLogger(__name__)

CORPUS_VEC_TABLE = "corpus_style_vec"
#: 语料句长度口径（与 scripts/analyze_personal_dialog.py 的样本过滤一致）。
REPLY_MIN_CHARS = 3
REPLY_MAX_CHARS = 40
CONTEXT_MIN_CHARS = 5
CONTEXT_MAX_CHARS = 160
#: 同首二字多样性封顶：防饭圈刷屏式重复（"易烊千玺"×N）占满库。
PER_PREFIX_CAP_DEFAULT = 60

_EMOTICON = re.compile(r"\[[^\[\]]{1,10}\]")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_MENTION = re.compile(r"@[^\s@]{1,20}")
_ASCII_RUN = re.compile(r"[0-9A-Za-z]+(?:['\-][0-9A-Za-z]+)*")


def detokenize(text: str) -> str:
    """还原 PersonalDialog 的分词空格：ASCII 词间留一个空格，其余直接拼接。"""

    tokens = text.split()
    if not tokens:
        return ""
    parts = [tokens[0]]
    for token in tokens[1:]:
        prev = parts[-1]
        if prev and _is_ascii_alnum(prev[-1]) and token and _is_ascii_alnum(token[0]):
            parts.append(" " + token)
        else:
            parts.append(token)
    return "".join(parts)


def _is_ascii_alnum(char: str) -> bool:
    return char.isascii() and char.isalnum()


def is_quality_reply(text: str, *, min_chars: int = REPLY_MIN_CHARS,
                     max_chars: int = REPLY_MAX_CHARS) -> bool:
    """蒸馏入库的回句质量门：短句优先、无链接提及、表情占比低。"""

    if not (min_chars <= len(text) <= max_chars):
        return False
    if _URL.search(text) or _MENTION.search(text):
        return False
    if len(_EMOTICON.findall(text)) > 1:
        return False
    return True


def is_quality_context(text: str) -> bool:
    return CONTEXT_MIN_CHARS <= len(text) <= CONTEXT_MAX_CHARS


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def distance_to_cosine(distance: float) -> float:
    """L2 归一化向量间的欧氏距离 → 余弦相似度（cos = 1 - d²/2）。"""

    return max(0.0, min(1.0, 1.0 - (float(distance) ** 2) / 2.0))


def ensure_corpus_vec_table(engine, dimensions: int) -> bool:
    """确保 vec0 虚拟表存在；返回是否存在（或已成功创建）。"""

    try:
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
                {"name": CORPUS_VEC_TABLE},
            ).fetchone()
            if row is not None:
                return True
            connection.execute(
                text(
                    f"CREATE VIRTUAL TABLE {CORPUS_VEC_TABLE} USING vec0("
                    "hit_id INTEGER PRIMARY KEY, "
                    "kind INTEGER PARTITION KEY, "
                    f"embedding float[{int(dimensions)}])"
                )
            )
            connection.commit()
            return True
    except Exception:
        logger.warning("corpus_vec_table_ensure_failed", exc_info=True)
        return False


def corpus_vec_row_count(engine) -> int:
    try:
        with engine.connect() as connection:
            row = connection.execute(text(f"SELECT COUNT(*) FROM {CORPUS_VEC_TABLE}")).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def insert_corpus_rows(engine, provider, rows: list[dict], *, batch_size: int = 256) -> int:
    """批量嵌入并双写元数据表 + vec0 表；返回成功写入的行数。

    rows 每项：{text, context_before: [str...], kind: int, gender: str, prev_len: int}。
    嵌入失败/维度不符的行跳过（fail-soft，不中断整批）。
    """

    if not rows:
        return 0
    dimensions = int(getattr(provider.identity, "dimensions", 0) or 0)
    if dimensions <= 0 or not ensure_corpus_vec_table(engine, dimensions):
        logger.warning("corpus_insert_unavailable dims=%s", dimensions)
        return 0
    identity = provider.identity
    # hit_id 从现有最大值续起：重复运行（非 rebuild）不与已有行撞主键。
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT COALESCE(MAX(hit_id), 0) FROM corpus_style_examples")
            ).fetchone()
            next_hit_id = int(row[0]) + 1 if row else 1
    except Exception:
        next_hit_id = 1
    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            vectors = provider.embed_documents([row["text"] for row in batch])
        except Exception:
            logger.warning("corpus_embed_batch_failed batch=%s", start // batch_size, exc_info=True)
            continue
        meta_rows: list[CorpusStyleExample] = []
        vec_rows: list[dict] = []
        for row, vector in zip(batch, vectors):
            if not vector:
                continue
            if len(vector) != dimensions:
                continue
            normalized = normalize_vector([float(value) for value in vector])
            hit_id = next_hit_id + len(vec_rows)
            meta_rows.append(CorpusStyleExample(
                hit_id=hit_id,
                text=row["text"],
                context_before=[str(item) for item in (row.get("context_before") or [])],
                kind=int(row.get("kind") or 0),
                gender=str(row.get("gender") or "")[:8],
                prev_len=int(row.get("prev_len") or 0),
                provider=str(getattr(identity, "provider", "")),
                model=str(getattr(identity, "model", "")),
            ))
            vec_rows.append({
                "hit_id": hit_id,
                "kind": int(row.get("kind") or 0),
                "embedding": json.dumps(normalized, separators=(",", ":")),
            })
        if not meta_rows:
            continue
        try:
            from app.storage.db import session_scope

            with session_scope(engine) as session:
                session.add_all(meta_rows)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"INSERT INTO {CORPUS_VEC_TABLE}(hit_id, kind, embedding) "
                        "VALUES (:hit_id, :kind, :embedding)"
                    ),
                    vec_rows,
                )
            inserted += len(meta_rows)
            next_hit_id += len(meta_rows)
        except Exception:
            logger.warning("corpus_insert_batch_failed batch=%s", start // batch_size, exc_info=True)
    return inserted


def retrieve_similar(
    engine,
    settings,
    query_text: str,
    *,
    kind: int = 0,
    limit: int = 2,
    min_cosine: float = 0.60,
    provider=None,
) -> list[dict]:
    """按当前消息语义检索最相似的真人接法；任何失败返回 []（fail-soft）。

    provider 可注入（测试/脚本复用）；缺省走 get_shared_embedding_provider
    进程级单例（bge-small-zh，模型加载 ~秒级只发生一次）。
    """

    started = time.monotonic()
    try:
        if not str(query_text or "").strip():
            return []
        if provider is None:
            from app.core.fuzzy_recall import get_shared_embedding_provider

            provider = get_shared_embedding_provider(settings)
        if provider is None or not getattr(provider, "available", False):
            logger.debug("style_corpus_provider_unavailable")
            return []
        vector = provider.embed_query(str(query_text))
        if not vector:
            return []
        normalized = normalize_vector([float(value) for value in vector])
        fetch_limit = max(16, int(limit) * 16)
        with engine.connect() as connection:
            ranked = connection.execute(
                text(
                    f"SELECT v.hit_id, v.distance FROM {CORPUS_VEC_TABLE} AS v "
                    "WHERE v.embedding MATCH :embedding AND v.kind = :kind AND v.k = :k"
                ),
                {
                    "embedding": json.dumps(normalized, separators=(",", ":")),
                    "kind": int(kind),
                    "k": fetch_limit,
                },
            ).fetchall()
        hits: list[dict] = []
        seen_prefixes: set[str] = set()
        for hit_id, distance in ranked:
            cosine = distance_to_cosine(distance)
            if cosine < float(min_cosine):
                break  # vec0 按距离升序返回，后面只会更差
            hits.append({"hit_id": int(hit_id), "similarity": round(cosine, 3)})
        if not hits:
            return []
        with engine.connect() as connection:
            id_list = ",".join(str(hit["hit_id"]) for hit in hits)
            rows = connection.execute(
                text(
                    "SELECT hit_id, text, context_before, gender FROM corpus_style_examples "
                    f"WHERE hit_id IN ({id_list})"
                )
            ).fetchall()
        by_id = {int(row[0]): row for row in rows}
        query_compact = re.sub(r"\s+", "", str(query_text))
        results: list[dict] = []
        for hit in hits:
            row = by_id.get(hit["hit_id"])
            if row is None:
                continue
            reply = str(row[1] or "")
            # 与当前消息几乎相同的语料句没有参考价值
            if reply and re.sub(r"\s+", "", reply) == query_compact:
                continue
            prefix = reply[:2]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            context = []
            try:
                context = [str(item) for item in json.loads(row[2] or "[]")]
            except (TypeError, ValueError):
                context = []
            results.append({
                "text": reply,
                "context": context[-1:] if context else [],
                "similarity": hit["similarity"],
            })
            if len(results) >= int(limit):
                break
        elapsed = time.monotonic() - started
        logger.info(
            "style_corpus kind=%s query_len=%s hits=%s top_sim=%s elapsed_ms=%s",
            kind, len(str(query_text)), len(results),
            results[0]["similarity"] if results else "-",
            round(elapsed * 1000),
        )
        return results
    except Exception:
        logger.warning("style_corpus_retrieve_failed", exc_info=True)
        return []


def render_corpus_block(hits: list[dict], *, max_chars: int = 300) -> str | None:
    """渲染注入块；空结果返回 None。前缀沿用 'Reply style:' 指令区口径。"""

    if not hits:
        return None
    parts: list[str] = []
    used = 0
    for hit in hits:
        reply = str(hit.get("text") or "").strip()
        if not reply:
            continue
        context = (hit.get("context") or [""])[0]
        fragment = f"情境「{context}」→ 真人接「{reply}」" if context else f"真人接「{reply}」"
        if used + len(fragment) > max_chars:
            break
        parts.append(fragment)
        used += len(fragment) + 1
    if not parts:
        return None
    return ("Reply style: 相似情境下真人的接法（只学长度、节奏和接法，内容别照抄）："
            + "；".join(parts))
