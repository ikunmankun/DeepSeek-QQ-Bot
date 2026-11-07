"""证据制人格进化：候选解析、证据合并、跨天晋升与提示词注入。"""

from datetime import datetime, timedelta

from sqlalchemy import select

from app.core.trait_evolution import (
    MAX_CANDIDATES_PER_TICK,
    PROMOTION_MENTIONS,
    _similar,
    _trait_key,
    confirmed_traits,
    merge_trait_candidates,
    parse_trait_candidates,
    promote_due_traits,
    trait_prompt_line,
)
from app.storage.db import session_scope
from app.storage.models import PersonaTraitCandidate

NOW = datetime(2026, 10, 31, 12, 0, 0)


def test_parse_trait_candidates_extracts_json_and_caps():
    raw = (
        "好的，以下是归纳结果：\n"
        '[{"trait": "嘴上嫌弃还是会把事办了", "evidence": "嫌麻烦但帮到底"},\n'
        ' {"trait": "被夸会嘴硬", "evidence": "哼 谁要你夸"}]'
    )
    parsed = parse_trait_candidates(raw)
    assert [item["trait"] for item in parsed] == ["嘴上嫌弃还是会把事办了", "被夸会嘴硬"]


def test_parse_trait_candidates_rejects_junk_and_overlong():
    assert parse_trait_candidates(None) == []
    assert parse_trait_candidates("没有够格的特质") == []
    assert parse_trait_candidates("[]") == []
    # 特质超长截断、超出条数上限丢弃、空白特质剔除
    payload = [{"trait": "x" * 100, "evidence": "e"},
               {"trait": "", "evidence": "e"},
               {"trait": "特质句", "evidence": ""}] * 2
    parsed = parse_trait_candidates(str(payload).replace("'", '"'))
    assert all(len(item["trait"]) <= 24 for item in parsed)
    assert len(parsed) <= MAX_CANDIDATES_PER_TICK


def test_trait_key_ignores_punctuation_and_case():
    assert _trait_key("嘴上嫌弃，还是会把事办了") == _trait_key("嘴上嫌弃还是会把事办了 ")
    assert _trait_key("ABC") == _trait_key("abc")
    assert _trait_key("A") != _trait_key("B")


def test_similar_phrases_merge():
    assert _similar("嘴上嫌弃还是会把事办了", "嘴上嫌弃但是会把事办了")
    assert not _similar("嘴上嫌弃还是会把事办了", "喜欢在深夜刷B站")


def test_merge_counts_evidence_and_promotes_across_days(sqlite_engine):
    trait = "被夸会嘴硬"
    with session_scope(sqlite_engine) as session:
        first_day = NOW - timedelta(days=3)
        merged = merge_trait_candidates(
            session, candidates=[{"trait": trait, "evidence": "哼 谁要你夸"}],
            origin="diary", now=first_day)
        assert merged == 1
        promoted = promote_due_traits(session, now=first_day)
        assert promoted == 0  # 只有 1 条证据，不够晋升
        # 同一天再提两次（措辞略有差异也要合并）：次数够了但没跨天
        merge_trait_candidates(
            session,
            candidates=[{"trait": "被夸会嘴硬", "evidence": "才没有"},
                        {"trait": "被夸了会嘴硬", "evidence": "哼"}],
            origin="diary", now=first_day + timedelta(hours=1))
        assert promote_due_traits(session, now=first_day + timedelta(hours=1)) == 0
    with session_scope(sqlite_engine) as session:
        rows = list(session.scalars(select(PersonaTraitCandidate)))
        assert len(rows) == 1, "相似措辞必须合并成同一条"
        assert rows[0].evidence_count == 3
        assert rows[0].status == "candidate"
        assert len(rows[0].evidence_json) == 3
    # 第三条证据隔了 3 天 → 次数与跨度都够，晋升
    with session_scope(sqlite_engine) as session:
        merge_trait_candidates(
            session, candidates=[{"trait": trait, "evidence": "谁要你夸"}],
            origin="diary", now=NOW)
        promoted = promote_due_traits(session, now=NOW)
        assert promoted == 1
    assert confirmed_traits(sqlite_engine) == [trait]
    line = trait_prompt_line(sqlite_engine)
    assert line is not None and "被夸会嘴硬" in line
    assert "别总结自己" in line


def test_trait_prompt_line_empty_without_confirmed(sqlite_engine):
    assert confirmed_traits(sqlite_engine) == []
    assert trait_prompt_line(sqlite_engine) is None


def test_promotion_requires_min_mentions(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        for index in range(PROMOTION_MENTIONS - 1):
            merge_trait_candidates(
                session, candidates=[{"trait": "稳定特质", "evidence": "e"}],
                origin="diary",
                now=NOW - timedelta(days=10 - index))
        assert promote_due_traits(session, now=NOW) == 0
        # 补上第 3 条证据（跨度 10 天）→ 晋升
        merge_trait_candidates(
            session, candidates=[{"trait": "稳定特质", "evidence": "e"}],
            origin="diary", now=NOW)
        assert promote_due_traits(session, now=NOW) == 1
