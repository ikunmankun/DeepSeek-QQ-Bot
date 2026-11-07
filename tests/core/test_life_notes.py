"""人生笔记：写入去重/裁剪、提示注入、夜间凝练解析、FTS+向量检索。"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import text

from app.core import life_notes
from app.storage.db import session_scope

NOW = datetime(2026, 11, 7, 12, 0)


def test_add_note_dedupes_similar_titles(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        first = life_notes.add_note(session, kind="running_joke", title="下次一定梗的由来",
                                    content="小甲第三次说下次一定，梗正式成型", now=NOW)
        dup = life_notes.add_note(session, kind="running_joke", title="「下次一定」梗的由来",
                                  content="换个说法重复讲同一个故事", now=NOW)
        other = life_notes.add_note(session, kind="relationship", title="和小乙熟起来了",
                                    content="开始互发日常", now=NOW)
    assert first is not None and dup is None and other is not None


def test_add_note_rejects_garbage_and_normalizes_kind(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        assert life_notes.add_note(session, kind="running_joke", title="嗯",
                                   content="太短", now=NOW) is None
        row = life_notes.add_note(session, kind="mystery-kind", title="正常标题",
                                  content="未知类型落回 milestone", now=NOW)
    assert row is not None and row.kind == "milestone"


def test_add_note_trims_overflow_oldest(sqlite_engine):
    titles = ["夜爬山顶看流星", "初雪那天拍的照片", "小甲生日会办成功了",
              "第一次组局吃火锅", "台风天全群在线"]
    with session_scope(sqlite_engine) as session:
        for index, title in enumerate(titles):
            life_notes.add_note(session, kind="milestone", title=title,
                                content=f"内容{index}", now=NOW - timedelta(days=5 - index),
                                max_notes=4)
        rows = life_notes.list_recent(session, limit=10)
    assert len(rows) == 4
    assert all("夜爬山顶" not in row.title for row in rows)  # 最旧被裁


class _FakeEmbedding:
    """确定性假嵌入：文本含'饭'→[1,0]，否则→[0,1]。"""

    available = True

    @property
    def identity(self):  # noqa: D102
        from types import SimpleNamespace

        return SimpleNamespace(provider="fake", model="fake-model",
                               dimensions=2, version="1")

    def embed_query(self, query: str):
        return [1.0, 0.0] if "饭" in str(query) else [0.0, 1.0]

    def embed_documents(self, documents):
        return [[1.0, 0.0] if "饭" in str(doc) else [0.0, 1.0] for doc in documents]


def test_prompt_line_empty_and_filled(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        assert life_notes.prompt_line(session, now=NOW) is None
        life_notes.add_note(session, kind="milestone", title="第一件大事",
                            content="发生了一件值得记的事", now=NOW)
        line = life_notes.prompt_line(session, now=NOW)
    assert line is not None and "第一件大事" in line and "长期记忆" in line


class _FakeLlm:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[list[str]] = []

    def generate_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.response


def test_consolidate_respects_min_interval(sqlite_engine, tmp_path):
    with session_scope(sqlite_engine) as session:
        life_notes.add_note(session, kind="milestone", title="三天前的笔记",
                            content="刚写过", now=NOW - timedelta(days=1))
    llm = _FakeLlm("[]")
    added = life_notes.consolidate_life_notes(sqlite_engine, llm_client=llm,
                                              data_dir=tmp_path, now=NOW,
                                              min_interval_days=3)
    assert added == 0 and llm.prompts == []  # 间隔不够，连 LLM 都不调


def test_consolidate_parses_and_indexes(sqlite_engine, tmp_path):
    # 凝练需要材料：在 data_dir 放一篇当天日记。
    diary_dir = tmp_path / "diary"
    diary_dir.mkdir(parents=True)
    (diary_dir / "2026-11-07.md").write_text("今天大家一直在玩白饭这个梗\n", encoding="utf-8")
    llm = _FakeLlm('[{"kind": "running_joke", "title": "白饭文学的诞生", '
                   '"content": "大家开始管小町叫大白饭，梗来自一次口误", "source_groups": [123]}]')
    added = life_notes.consolidate_life_notes(sqlite_engine, llm_client=llm,
                                              data_dir=tmp_path, now=NOW,
                                              min_interval_days=3)
    assert added == 1
    with session_scope(sqlite_engine) as session:
        rows = life_notes.list_recent(session, limit=3)
        assert rows and rows[0].kind == "running_joke"
        fts_rows = session.execute(
            text("SELECT note_id FROM life_notes_fts")).fetchall()
    assert fts_rows and fts_rows[0][0] == rows[0].id


def test_search_life_notes_fts(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        life_notes.add_note(session, kind="running_joke", title="白饭文学的诞生",
                            content="大家开始管小町叫大白饭", now=NOW)
        life_notes.add_note(session, kind="relationship", title="和小乙的交情",
                            content="开始互发日常分享", now=NOW)
    hits = life_notes.search_life_notes(sqlite_engine, query="白饭 这个梗", limit=3)
    assert hits and "白饭" in hits[0].title + hits[0].content
    assert life_notes.search_life_notes(sqlite_engine, query="", limit=3) == []


def test_search_life_notes_vector_channel(sqlite_engine):
    # 向量通道独立验证：查询措辞避开 FTS 能命中的"白饭"，靠假嵌入的
    # '饭' 判据对上笔记向量。索引同步必须在事务提交后（另开连接写库）。
    from sqlalchemy import text as sqltext

    # 测试用 2 维假嵌入，把 vec 旁表按同维度重建（生产是 512 维真嵌入）。
    with sqlite_engine.begin() as connection:
        connection.execute(sqltext("DROP TABLE IF EXISTS life_notes_vec"))
        connection.execute(sqltext(
            "CREATE VIRTUAL TABLE life_notes_vec "
            "USING vec0(note_id INTEGER PRIMARY KEY, embedding float[2])"))
    with session_scope(sqlite_engine) as session:
        note = life_notes.add_note(session, kind="running_joke", title="白饭梗",
                                   content="白饭这个外号传开了", now=NOW)
        note_id, title, content = note.id, note.title, note.content
    life_notes._sync_note_indexes(sqlite_engine, note_id=note_id,
                                  title=title, content=content,
                                  embedding_provider=_FakeEmbedding())
    # FTS 也能命中，但这里验证向量通道独立有效：换 FTS 查不到的措辞。
    hits = life_notes.search_life_notes(sqlite_engine, query="饭饭相关的话题",
                                        limit=3, embedding_provider=_FakeEmbedding())
    assert hits and "白饭" in hits[0].title + hits[0].content


def test_parse_life_note_json_defensive():
    assert life_notes.parse_life_note_json("不是json") == []
    parsed = life_notes.parse_life_note_json("前缀 [ {\"title\": \"标题\", \"content\": \"内容\", \"kind\": \"promise\"} ] 后缀")
    assert parsed and parsed[0]["kind"] == "promise"
    assert life_notes.parse_life_note_json('[{"title": "只有标题没有内容"}]') == []
    assert life_notes.parse_life_note_json("[]") == []
