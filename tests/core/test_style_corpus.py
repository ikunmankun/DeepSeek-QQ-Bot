"""语料向量库：分词还原、质量门、cosine 换算、KNN 往返与 fail-soft。"""

import math
from dataclasses import dataclass

import pytest

from app.core.style_corpus import (
    CorpusStyleExample,
    detokenize,
    distance_to_cosine,
    ensure_corpus_vec_table,
    insert_corpus_rows,
    is_quality_context,
    is_quality_reply,
    normalize_vector,
    render_corpus_block,
    retrieve_similar,
)


def test_detokenize_rules():
    assert detokenize("没有 钱 万万 不行 ！ ~") == "没有钱万万不行！~"
    assert detokenize("你 是 who ?") == "你是who?"
    assert detokenize("用 pr 呀") == "用pr呀"
    assert detokenize("3 年前 的") == "3年前的"
    assert detokenize("") == ""


def test_quality_filters():
    assert is_quality_reply("在负一楼")
    assert is_quality_reply("用pr呀")
    assert not is_quality_reply("好")                       # 太短
    assert not is_quality_reply("啊" * 41)                 # 太长
    assert not is_quality_reply("看这个 https://x.com/a")  # 链接
    assert not is_quality_reply("@某人 你看")              # 提及
    assert is_quality_reply("萌萌哒[doge]")                # 单表情可过
    assert not is_quality_reply("[doge][cat] 双表情")      # 双表情拒收
    assert is_quality_context("分子冰淇淋其实不好吃")
    assert not is_quality_context("短的")


def test_distance_to_cosine():
    # 归一化向量：相同 → d=0 → cos=1；正交 → d=√2 → cos=0。
    assert math.isclose(distance_to_cosine(0.0), 1.0)
    assert math.isclose(distance_to_cosine(math.sqrt(2)), 0.0, abs_tol=1e-9)
    assert distance_to_cosine(1.0) == pytest.approx(0.5)


def test_normalize_vector():
    normalized = normalize_vector([3.0, 4.0])
    assert math.isclose(sum(v * v for v in normalized), 1.0)
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]


class _Identity:
    provider = "fake"
    model = "fake-bge"
    dimensions = 512
    version = "v1"


class CharHotProvider:
    """确定性假 embedding：字符集合 one-hot，共享字符越多余弦越高。"""

    def __init__(self) -> None:
        self.identity = _Identity()

    @property
    def available(self) -> bool:
        return True

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 512
        for char in set(text):
            vector[ord(char) % 512] += 1.0
        return normalize_vector(vector)

    def embed_query(self, query: str):
        return self._embed(query)

    def embed_documents(self, documents):
        return [self._embed(str(doc)) for doc in documents]


def _provider() -> CharHotProvider:
    return CharHotProvider()


def test_render_corpus_block():
    hits = [{"text": "在负一楼", "context": ["你在哪"], "similarity": 0.8},
            {"text": "用pr呀", "context": ["剪辑用什么"], "similarity": 0.7}]
    block = render_corpus_block(hits)
    assert block is not None and block.startswith("Reply style:")
    assert "在负一楼" in block and "别照抄" in block
    assert render_corpus_block([]) is None
    long_hits = [{"text": "啊" * 400, "context": [], "similarity": 0.9}]
    assert render_corpus_block(long_hits) is None


def test_retrieve_fail_soft_missing_table(tmp_path):
    from app.storage.db import build_engine

    engine = build_engine(tmp_path / "empty.db")
    try:
        # vec0 表不存在 → 空结果不抛错
        assert retrieve_similar(engine, type("S", (), {})(), "在干嘛") == []
    finally:
        engine.dispose()


def test_insert_and_retrieve_roundtrip(sqlite_engine):
    provider = _provider()
    rows = [
        {"text": "今天吃什么呀", "context_before": ["饿了"], "kind": 0, "gender": "female", "prev_len": 2},
        {"text": "在负一楼", "context_before": ["你在哪"], "kind": 0, "gender": "male", "prev_len": 4},
        {"text": "用pr呀", "context_before": ["剪辑用什么"], "kind": 1, "gender": "female", "prev_len": 5},
    ]
    inserted = insert_corpus_rows(sqlite_engine, provider, rows)
    assert inserted == 3

    # 查询"吃什么"与"今天吃什么呀"共享字符，应命中且相似度达标
    hits = retrieve_similar(sqlite_engine, None, "吃什么", kind=0,
                            limit=2, min_cosine=0.55, provider=provider)
    assert hits, "roundtrip retrieval should hit"
    assert hits[0]["similarity"] >= 0.55
    assert hits[0]["text"] != "吃什么"  # 与查询相同的语料句被剔除

    # kind 分区：kind=1 池检索不到 kind=0 独有的行
    hits_private = retrieve_similar(sqlite_engine, None, "剪辑用什么软件",
                                    kind=1, limit=2, min_cosine=0.55, provider=provider)
    assert all(hit["text"] != "在负一楼" for hit in hits_private)


def test_insert_skips_bad_rows(sqlite_engine):
    provider = _provider()
    rows = [{"text": "正常的回句", "context_before": [], "kind": 0, "gender": "", "prev_len": 0}]
    assert insert_corpus_rows(sqlite_engine, provider, rows) == 1
    # 重复运行续 hit_id，不撞主键
    assert insert_corpus_rows(sqlite_engine, provider, rows) == 1
    with sqlite_engine.connect() as connection:
        from sqlalchemy import text

        count = connection.execute(
            text("SELECT COUNT(*) FROM corpus_style_examples")).fetchone()[0]
        assert count == 2
