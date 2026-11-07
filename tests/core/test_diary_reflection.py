"""每日日记反思：素材收集、反思 prompt、定时计算与写入路径。"""

from datetime import datetime, timedelta
from types import SimpleNamespace

from app.core.affinity import apply_affinity_event
from app.core.diary import (
    DiaryReflectionService,
    DiaryWriter,
    build_reflection_prompt,
    collect_daily_material,
)
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, UserRepository

NOW = datetime(2026, 11, 1, 23, 30)
BOT = 900000101
GROUP = 900000001


def _seed_group_and_bot(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=GROUP, group_name="g",
                                              enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=BOT, nickname="bot", group_card="")
        UserRepository(session).upsert_user(user_id=900000103, nickname="tester",
                                            group_card="")


def test_collect_daily_material_pulls_mood_affinity_and_own_messages(sqlite_engine):
    _seed_group_and_bot(sqlite_engine)
    from app.storage.models import Message

    with session_scope(sqlite_engine) as session:
        apply_affinity_event(session, group_id=GROUP, user_id=900000103,
                             delta=1.0, now=NOW - timedelta(hours=2))
        for i in range(3):
            session.add(Message(
                platform_msg_id=f"bot-{i}", group_id=GROUP, user_id=BOT,
                timestamp=NOW - timedelta(minutes=30 - i),
                plain_text=f"第{i}句回复", raw_json={}, msg_type="text",
                reply_to_msg_id=None, mentioned_bot=False))

    material = collect_daily_material(sqlite_engine, group_ids=[GROUP],
                                      bot_qq=BOT, now=NOW)
    assert "互动最多的人" in material and "900000103" in material
    assert "你自己发了3条消息" in material
    assert "第2句回复" in material


def test_reflection_prompt_gives_freedom_and_no_space_style():
    prompt = build_reflection_prompt(
        persona_text="大肥鱼，傲娇", material="[群1] 今天聊了猫猫",
        now=datetime(2026, 11, 1, 23, 30))
    joined = "\n".join(prompt)
    assert "11月01日" in joined or "11月1日" in joined
    assert "写你真正想记的" in joined
    assert "不是必答题" in joined          # 素材只是参考
    assert "不用空格" in joined
    assert "今天聊了猫猫" in joined


class _FakeLlm:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_text(self, prompt_lines, **kwargs):
        self.calls.append(list(prompt_lines))
        return self.reply


def test_reflection_writes_diary_file_with_fake_llm(tmp_path, sqlite_engine):
    _seed_group_and_bot(sqlite_engine)
    writer = DiaryWriter(tmp_path / "data")
    llm = _FakeLlm("今天群里都在聊猫猫。我一句都没接住，有点不甘心。")
    service = DiaryReflectionService(
        engine=sqlite_engine, settings=SimpleNamespace(data_dir=tmp_path / "data"),
        llm_client=llm, writer=writer, group_ids=[GROUP],
        persona_text="大肥鱼，傲娇", bot_qq=BOT, reflection_time="23:30")
    path = service._reflect()
    assert path is not None and path.exists()
    content = path.read_text(encoding="utf-8")
    assert "不甘心" in content
    assert llm.calls and "睡前写今天的日记" in "\n".join(llm.calls[0])


def test_reflection_skips_empty_reply(tmp_path, sqlite_engine):
    writer = DiaryWriter(tmp_path / "data")
    llm = _FakeLlm("  ")
    service = DiaryReflectionService(
        engine=sqlite_engine, settings=SimpleNamespace(data_dir=tmp_path / "data"),
        llm_client=llm, writer=writer, group_ids=[GROUP],
        persona_text="x", bot_qq=BOT, reflection_time="23:30")
    assert service._reflect() is None
    assert not any(writer.directory.glob("*.md"))


def test_next_run_same_day_before_time_and_tomorrow_after():
    service = DiaryReflectionService(
        engine=None, settings=None, llm_client=None, writer=None,
        group_ids=(), persona_text="", bot_qq=1, reflection_time="23:30")
    earlier = datetime(2026, 11, 1, 20, 0)
    assert service.next_run(earlier) == datetime(2026, 11, 1, 23, 30)
    after = datetime(2026, 11, 1, 23, 31)
    assert service.next_run(after) == datetime(2026, 11, 2, 23, 30)


def test_diary_feature_create_workers_smoke(tmp_path):
    """create_workers 必须能真实构造（曾经漏导入 DiaryWriter 在容器里才炸）。"""
    from app.agent.plugins.builtin import DiaryFeature

    feature = DiaryFeature()
    feature.attach(
        engine=None,
        settings=SimpleNamespace(data_dir=tmp_path, diary_reflection_enabled=True,
                                 diary_reflection_time="23:30"),
        llm_client=object(), group_ids=(900000001,), persona_text="x", bot_qq=900000101)
    workers = feature.create_workers()
    assert len(workers) == 1 and hasattr(workers[0], "run")

    feature._settings = SimpleNamespace(data_dir=tmp_path, diary_reflection_enabled=False)
    assert feature.create_workers() == ()
