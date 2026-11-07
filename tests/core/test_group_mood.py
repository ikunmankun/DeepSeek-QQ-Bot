"""规则版运行时情绪：触发分类、强度累积/衰减、环境底色与提示词注入。"""

from datetime import datetime, timedelta

from app.core.group_mood import (
    apply_mood_trigger,
    classify_mood_trigger,
    effective_mood_state,
    load_group_mood,
    mood_prompt_line,
    should_evaluate_mood,
)
from app.storage.db import session_scope
from app.storage.models import GroupMoodState

DAY = datetime(2026, 10, 25, 14, 0, 0)   # 下午 2 点
NIGHT = datetime(2026, 10, 26, 2, 30, 0)  # 凌晨 2 点半


def test_explicit_direction_always_evaluates():
    assert should_evaluate_mood(text="杂鱼", mentioned_bot=True)
    assert should_evaluate_mood(text="帮我看看", reply_to_bot=True)
    assert should_evaluate_mood(text="小鲸鱼 你真厉害", called_by_name=True)
    # 点名时对方是不是在跟别人说话不重要（@了两个人也照常反应）
    assert should_evaluate_mood(text="杂鱼", mentioned_bot=True, other_addressee=True)


def test_ongoing_exchange_allows_short_jabs_and_second_person_only():
    # 群友斗嘴不 @：机器人刚说过话，接话的短打算当面
    assert should_evaluate_mood(text="杂鱼", bot_spoke_recently=True)
    assert should_evaluate_mood(text="就这", bot_spoke_recently=True)
    assert should_evaluate_mood(text="你好笨", bot_spoke_recently=True)
    # 对别人说的不算
    assert not should_evaluate_mood(text="你好笨", bot_spoke_recently=True, other_addressee=True)
    # 机器人不在对话里，长句不点名不算（聊天里聊游戏"好菜"不该记仇）
    assert not should_evaluate_mood(text="这个游戏平衡做得真菜啊", bot_spoke_recently=True)
    assert not should_evaluate_mood(text="杂鱼", bot_spoke_recently=False)


def test_classify_patterns_are_direction_free():
    assert classify_mood_trigger("你也太厉害了吧")[0] == "smug"
    assert classify_mood_trigger("杂鱼～")[0] == "annoyed"
    assert classify_mood_trigger("帮我看看这个报错咋回事")[0] == "serious"
    # 求助优先于带刺
    assert classify_mood_trigger("你个笨蛋 帮我修一下这个")[0] == "serious"
    assert classify_mood_trigger("今天天气不错") is None


def test_same_mood_stacks_and_different_mood_switches(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        apply_mood_trigger(session, group_id=900000001, mood="annoyed", bump=0.6,
                           reason="被叫杂鱼", now=DAY)
        apply_mood_trigger(session, group_id=900000001, mood="annoyed", bump=0.6,
                           reason="又被笑话", now=DAY + timedelta(minutes=10))
        row = session.get(GroupMoodState, 900000001)
        # 10 分钟后旧强度衰减（45 分钟半衰期）再叠加新事件
        assert row.mood == "annoyed"
        assert 1.0 < row.intensity <= 2.0

        apply_mood_trigger(session, group_id=900000001, mood="smug", bump=0.5,
                           reason="被夸可爱", now=DAY + timedelta(minutes=12))
        row = session.get(GroupMoodState, 900000001)
        assert row.mood == "smug"
        assert row.intensity == 0.5


def test_intensity_decays_back_to_ambient(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        apply_mood_trigger(session, group_id=900000002, mood="annoyed", bump=0.6,
                           reason="被怼", now=DAY)
    row = load_group_mood(sqlite_engine, 900000002)
    # 45 分钟恰好一个半衰期：0.6 → 0.3，仍在门槛之上
    mood, intensity = effective_mood_state(row, DAY + timedelta(minutes=45))
    assert (mood, round(intensity, 2)) == ("annoyed", 0.3)
    # 4 个半衰期后跌破门槛 → 回到白天 ambient（default）
    mood, intensity = effective_mood_state(row, DAY + timedelta(minutes=45 * 4))
    assert (mood, intensity) == ("default", 0.0)


def test_night_ambient_is_sleepy_and_event_mood_wins_over_it():
    assert effective_mood_state(None, NIGHT) == ("sleepy", 0.0)
    assert effective_mood_state(None, DAY) == ("default", 0.0)
    # 半夜被怼：事件情绪压过犯困
    late = datetime(2026, 10, 26, 1, 0, 0)
    fresh = SimpleNamespace_row(mood="annoyed", intensity=0.8, updated_at=late)
    assert effective_mood_state(fresh, late + timedelta(minutes=5))[0] == "annoyed"


class SimpleNamespace_row:
    def __init__(self, *, mood, intensity, updated_at):
        self.mood = mood
        self.intensity = intensity
        self.updated_at = updated_at
        self.reason = ""


def test_prompt_line_speaks_state_without_tone_instructions(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        apply_mood_trigger(session, group_id=900000003, mood="annoyed", bump=0.6,
                           reason="群里有人拿你开玩笑", now=DAY)
    line = mood_prompt_line(sqlite_engine, 900000003, DAY + timedelta(minutes=5))
    assert line is not None
    assert "annoyed" in line and "拿你开玩笑" in line
    # 恼火时联动表情包提示；平静与犯困时不提示
    assert "表情包" in line
    assert mood_prompt_line(sqlite_engine, 900000004, DAY) is None
    night_line = mood_prompt_line(sqlite_engine, 900000004, NIGHT)
    assert night_line is not None and "sleepy" in night_line
    assert "表情包" not in night_line


def test_unknown_mood_row_falls_back_to_ambient():
    row = SimpleNamespace_row(mood="gadget", intensity=1.5, updated_at=DAY)
    assert effective_mood_state(row, DAY) == ("default", 0.0)
