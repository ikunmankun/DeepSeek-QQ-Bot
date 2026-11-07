"""风格审查：确定性快检、LLM 复审解析与降级、重写提示词。"""

from app.core.style_judge import (
    StyleJudgeResult,
    build_style_judge_prompt,
    build_style_rewrite_prompt,
    judge_reply_style,
    parse_style_judge,
    quick_style_faults,
)


def test_quick_faults_catch_markdown_and_lists():
    assert quick_style_faults("这个**真的**不错")
    assert quick_style_faults("## 今天的总结")
    assert quick_style_faults("- 第一条\n- 第二条")
    assert quick_style_faults("1. 先看这个\n2. 再看那个")
    assert quick_style_faults("这是标题\n# 小节")


def test_quick_faults_catch_customer_service_tone():
    assert any("客服" in issue for issue in quick_style_faults("建议你早点休息哦"))
    assert quick_style_faults("总而言之，今天就是这样啦")
    assert quick_style_faults("希望这能帮到你！")
    assert quick_style_faults("很抱歉给你带来困扰")
    assert quick_style_faults("首先我们要看数据，其次要看趋势")


def test_quick_faults_catch_contrast_pattern():
    assert any("对照" in issue or "而是" in issue
               for issue in quick_style_faults("这不是懒，而是摆烂"))


def test_clean_casual_replies_pass():
    assert quick_style_faults("啊？你们公司还没倒啊") == []
    assert quick_style_faults("哈哈 那你多喝点热水") == []
    assert quick_style_faults("就这？我还以为多大事呢") == []
    assert quick_style_faults("") == []


class JudgeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[str]] = []

    def generate_text(self, prompt_lines, *, conversation_key=None):
        self.calls.append(prompt_lines)
        return self.response


def test_parse_style_judge_pass_and_fail():
    ok = parse_style_judge("STYLE: pass\nISSUES: ")
    assert ok.passed and ok.issues == []
    bad = parse_style_judge("STYLE: fail\nISSUES: 结尾在祝人顺利")
    assert not bad.passed and bad.issues == ["结尾在祝人顺利"]
    # 解析不出格式 → 降级为通过（宁可放过，不可不发）
    assert parse_style_judge("").passed
    assert parse_style_judge("我觉得还行").passed


def test_judge_reply_style_degrades_to_pass_on_error():
    class Exploding:
        def generate_text(self, prompt_lines, *, conversation_key=None):
            raise RuntimeError("provider down")

    assert judge_reply_style(client=Exploding(), bot_name="b",
                             reply="hi", proactive=True).passed


def test_judge_prompt_flags_proactive_kind_and_reply_text():
    client = JudgeLlm("STYLE: fail\nISSUES: 分点了")
    result = judge_reply_style(client=client, bot_name="小町",
                               reply="第一件事……", proactive=True)
    assert not result.passed
    joined = "\n".join(client.calls[0])
    assert "小町" in joined
    assert "第一件事……" in joined
    assert "主动插进群聊" in joined


def test_rewrite_prompt_carries_issues_and_reply():
    lines = build_style_rewrite_prompt(
        prompt_lines=["原始提示", "群聊上下文"],
        reply="建议你早点休息。",
        issues=["客服腔（\"建议你\"）"],
        proactive=False,
    )
    joined = "\n".join(lines)
    assert "原始提示" in joined and "群聊上下文" in joined
    assert "建议你早点休息。" in joined
    assert "客服腔" in joined
    assert "只输出重写后的回复本身" in joined


def test_style_judge_result_issue_text_joins():
    assert StyleJudgeResult(False, ["a", "b"]).issue_text == "a；b"
