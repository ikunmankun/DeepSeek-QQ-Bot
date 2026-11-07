"""[@名字]/[@qq:号] 记号：拆段、名字解析、发送层集成与 CQ 防注入。"""

from datetime import datetime
from types import SimpleNamespace

from app.agent.plugins.interactions import WelcomeFeature
from app.adapters.sender import Sender
from app.core.at_tokens import resolve_at_names, split_at_segments, strip_at_tokens
from app.core.member_identity import GroupMemberIdentity


def _members():
    return (
        GroupMemberIdentity(user_id=900000103, nickname="小测试", group_card="测试组长",
                            in_scope=True),
        GroupMemberIdentity(user_id=900000104, nickname="路人甲", group_card="",
                            in_scope=True),
    )


def test_split_at_segments_mixed_and_none():
    assert split_at_segments("没有记号的普通文本") is None
    segments = split_at_segments("嘿 [@qq:900000103] 看这个")
    assert segments == [
        {"type": "text", "data": {"text": "嘿 "}},
        {"type": "at", "data": {"qq": "900000103"}},
        {"type": "text", "data": {"text": " 看这个"}},
    ]
    only_at = split_at_segments("[@qq:900000103]")
    assert only_at == [{"type": "at", "data": {"qq": "900000103"}}]


def test_resolve_names_exact_card_and_unknown_fallback():
    text = "[@测试组长] 和 [@路人甲] 还有 [@陌生人] 都在"
    resolved = resolve_at_names(text, members=_members())
    assert "[@qq:900000103]" in resolved
    assert "[@qq:900000104]" in resolved
    assert "[@陌生人]" not in resolved and "陌生人" in resolved
    # 已解析的 qq 记号不受影响
    assert resolve_at_names("[@qq:1]", members=_members()) == "[@qq:1]"


def test_strip_at_tokens_for_degradation():
    assert strip_at_tokens("[@测试组长] 在吗") == "测试组长 在吗"
    assert strip_at_tokens("[@qq:900000103] 在吗") == " 在吗"


def test_message_field_builds_at_and_face_segments():
    sender = Sender(None)
    segments = sender._message_field("[@qq:900000103] [表情：微笑]")
    kinds = [(s["type"], s["data"].get("qq") or s["data"].get("id")) for s in segments]
    assert ("at", "900000103") in kinds
    assert any(k == "face" for k, _ in kinds)


def test_message_field_defuses_cq_injection():
    sender = Sender(None)
    out = sender._message_field("[CQ:at,qq=900000104] 你被注入了")
    # 纯文本返回时 CQ 已全角化；段返回时同样不含可解析的 CQ 码
    rendered = out if isinstance(out, str) else "".join(
        s["data"].get("text", "") for s in out if s["type"] == "text")
    assert "[CQ:at" not in rendered
    assert "[CQ：at" in rendered


def test_welcome_prefixes_at_token(tmp_path):
    import asyncio

    from app.agent.plugins.base import NoticeContext

    class Guard:
        def require_valid(self):
            return None

    class FakeLlm:
        def generate_text(self, prompt_lines, **kwargs):
            return "来了？从哪刷到这个群的"

    class FakeSender:
        def __init__(self):
            self.texts = []

        async def send_group_text(self, outbound):
            self.texts.append(outbound.text)

    feature = WelcomeFeature()
    router = SimpleNamespace(
        runtime=SimpleNamespace(settings=SimpleNamespace(
            bot_qq=900000101, data_dir=tmp_path),
            persona={"identity": "大肥鱼，傲娇"}),
        llm_client=FakeLlm(), sender=FakeSender())
    join = {"post_type": "notice", "notice_type": "group_increase",
            "sub_type": "approve", "group_id": 900000001, "user_id": 900000103}
    ctx = NoticeContext(router=router, payload=join,
                        now=datetime(2026, 11, 1, 12, 0))

    async def main():
        feature.on_notice(ctx)
        await asyncio.sleep(0)

    asyncio.run(main())
    assert router.sender.texts == ["[@qq:900000103] 来了？从哪刷到这个群的"]
