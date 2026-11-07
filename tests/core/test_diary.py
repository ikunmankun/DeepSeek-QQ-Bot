"""本地日记工具：按天追加 markdown、带时间与心情头、空文本拒绝。"""

from datetime import datetime

from app.core.diary import DiaryWriter, build_diary_tool
from app.core.time_utils import shanghai_now_naive


class Guard:
    def require_valid(self):
        pass


def test_write_appends_dated_entry_with_mood(tmp_path):
    writer = DiaryWriter(tmp_path)
    tool = build_diary_tool(guard=Guard(), writer=writer,
                            mood_supplier=lambda: "annoyed")
    import asyncio

    async def run():
        return await tool.handler({"text": "今天群里没人说话，有点安静。"})

    result = asyncio.run(run())
    assert result.status == "ok" and result.data["saved"] is True
    day_file = writer.directory / f"{shanghai_now_naive():%Y-%m-%d}.md"
    content = day_file.read_text(encoding="utf-8")
    assert "今天群里没人说话" in content
    assert "## " in content and "心情：annoyed" in content


def test_empty_and_oversized_text_are_blocked(tmp_path):
    import asyncio
    writer = DiaryWriter(tmp_path)
    tool = build_diary_tool(guard=Guard(), writer=writer)

    empty = asyncio.run(tool.handler({"text": "   "}))
    assert empty.status == "blocked" and empty.data["error"] == "empty_diary_text"

    long = asyncio.run(tool.handler({"text": "字" * 2001}))
    assert long.status == "blocked" and long.data["error"] == "diary_too_long"
    assert not (writer.directory / f"{shanghai_now_naive():%Y-%m-%d}.md").exists()


def test_mood_supplier_failure_degrades_to_no_mood(tmp_path):
    import asyncio

    def broken():
        raise RuntimeError("mood down")

    writer = DiaryWriter(tmp_path)
    tool = build_diary_tool(guard=Guard(), writer=writer, mood_supplier=broken)
    result = asyncio.run(tool.handler({"text": "记一笔"}))
    assert result.status == "ok"
    day_file = writer.directory / f"{shanghai_now_naive():%Y-%m-%d}.md"
    assert "心情" not in day_file.read_text(encoding="utf-8")


def test_entries_append_across_days(tmp_path):
    from app.core.diary import DiaryWriter as W
    writer = W(tmp_path)
    writer.write(text="第一篇", now=datetime(2026, 10, 28, 21, 0), mood=None)
    writer.write(text="第二篇", now=datetime(2026, 10, 29, 9, 30), mood="smug")
    assert (writer.directory / "2026-10-28.md").exists()
    day2 = (writer.directory / "2026-10-29.md").read_text(encoding="utf-8")
    assert "## 09:30 · 心情：smug" in day2 and "第二篇" in day2
