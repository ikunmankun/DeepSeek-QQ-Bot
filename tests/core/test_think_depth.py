"""思考深度路由的规则表与联动行为。"""

from app.core.chat_style import typing_delay_seconds
from app.core.think_depth import (
    current_turn_effort,
    judge_think_depth,
    reset_turn_effort,
    set_turn_effort,
    typing_delay_factor,
)


def test_trivial_replies_go_minimal():
    for text in ("哦", "嗯嗯", "哈哈哈哈哈", "666", "好", "？"):
        decision = judge_think_depth(text)
        assert decision.effort == "minimal", text
        assert decision.level == "minimal"


def test_math_and_code_go_high():
    assert judge_think_depth("帮我算一下 37*89 等于多少").level == "high"
    assert judge_think_depth("解方程 2x+3=11").level == "high"
    assert judge_think_depth("这段python报错了：Traceback (most recent call last)").level == "high"


def test_serious_request_go_high_even_short():
    decision = judge_think_depth("认真想想这个问题")
    assert decision.level == "high"
    assert decision.reason == "serious_request"


def test_explain_and_decision_go_medium():
    assert judge_think_depth("为什么天空是蓝色的").level == "medium"
    assert judge_think_depth("帮我选一个，A还是B，哪个好").level == "medium"


def test_long_multi_question_message_go_high():
    text = "这个问题我纠结很久了？第一个方案是不是成本更高？第二个方案谁来维护？第三个呢？"
    assert judge_think_depth(text).level == "high"


def test_casual_chat_stays_default():
    decision = judge_think_depth("今天天气不错，出来玩了吗")
    assert decision.effort is None
    assert decision.level == "none"


def test_empty_text_is_default():
    assert judge_think_depth("   ").effort is None


def test_energy_coupling_downgrades_hard_questions():
    # 困了：high 降 medium，medium 直接落回默认档。
    decision = judge_think_depth("帮我算一下 37*89", energy=20.0)
    assert decision.effort == "medium"
    assert decision.reason.endswith("+sleepy_downgraded")
    assert judge_think_depth("为什么天空是蓝色的", energy=20.0).effort is None


def test_energy_coupling_exempt_explicit_request():
    # 点名求助：困也得认真答。
    decision = judge_think_depth("帮我算一下 37*89", energy=20.0, explicit_request=True)
    assert decision.effort == "high"


def test_energy_coupling_ignores_trivial_and_default():
    # 极简消息和普通闲聊不受精力影响。
    assert judge_think_depth("哦", energy=10.0).effort == "minimal"
    assert judge_think_depth("今天天气不错", energy=10.0).effort is None


def test_energy_coupling_can_be_disabled():
    decision = judge_think_depth("帮我算一下 37*89", energy=20.0, energy_coupled=False)
    assert decision.effort == "high"


def test_typing_delay_factor_follows_level():
    assert typing_delay_factor("minimal") < 1.0
    assert typing_delay_factor("high") > typing_delay_factor("medium") > 1.0
    assert typing_delay_factor("low") == 1.0
    assert typing_delay_factor("unknown") == 1.0


def test_turn_effort_contextvar_roundtrip():
    token = set_turn_effort("high")
    try:
        assert current_turn_effort() == "high"
        assert typing_delay_factor() > 1.0  # 缺省参数读回合档位
    finally:
        reset_turn_effort(token)
    assert current_turn_effort() == ""
    assert typing_delay_factor() == 1.0


def test_typing_delay_seconds_scales_with_effort():
    settings = type("S", (), {})()
    # 未配置 base/per_char 时延迟恒为 0（旧行为不变）。
    assert typing_delay_seconds(settings, "hi", think_effort="high") == 0.0

    configured = SimpleSettings()
    low = typing_delay_seconds(configured, "一句普通的回复", think_effort="minimal",
                               rng=__import__("random").Random(7))
    high = typing_delay_seconds(configured, "一句普通的回复", think_effort="high",
                                rng=__import__("random").Random(7))
    assert low > 0.0 and high > low


class SimpleSettings:
    """带打字延迟配置的最小 settings 桩。"""

    group_reply_typing_base_seconds = 1.0
    group_reply_typing_base_max_seconds = 1.0
    group_reply_typing_per_char_seconds = 0.02
    group_reply_typing_max_seconds = 0.0
