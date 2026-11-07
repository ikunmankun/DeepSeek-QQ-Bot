from app.core.qq_faces import (
    FACE_IDS,
    FACE_NAMES,
    face_token,
    message_text,
    resolve_face_id,
    sticker_token,
    text_segments,
)


def test_known_face_ids_are_translated_to_chinese_names() -> None:
    assert face_token("14") == "[表情：微笑]"
    assert face_token(178) == "[表情：斜眼笑]"
    assert face_token("179") == "[表情：doge]"


def test_unknown_face_id_keeps_the_raw_id_for_round_trip() -> None:
    # QQ 隔一阵就加新表情，名字表没跟上时不能把整段信息丢掉。
    assert face_token(999) == "[表情：id999]"
    assert resolve_face_id("id999") is None


def test_empty_face_id_does_not_crash() -> None:
    assert face_token(None) == "[表情：未知]"
    assert face_token("  ") == "[表情：未知]"


def test_market_face_uses_the_summary_field() -> None:
    assert sticker_token("[开心]") == "[表情包：开心]"
    assert sticker_token("") == "[表情包：未知]"


def test_face_names_reverse_lookup_prefers_the_classic_id() -> None:
    # 「可爱」同时挂在 21 和 9786 上，发送时要用经典 id。
    assert FACE_IDS["可爱"] == "21"
    assert FACE_IDS["咖啡"] == "60"
    assert resolve_face_id("可爱") == "21"


def test_message_text_keeps_text_and_face_in_order() -> None:
    message = [
        {"type": "at", "data": {"qq": "123"}},
        {"type": "text", "data": {"text": "哈哈"}},
        {"type": "face", "data": {"id": "13"}},
        {"type": "mface", "data": {"summary": "[笑哭]", "emoji_id": "abc"}},
    ]

    assert message_text(message) == "哈哈[表情：呲牙][表情包：笑哭]"


def test_text_segments_is_none_without_tokens() -> None:
    # 没有记号时必须返回 None，让发送端走原来的纯字符串路径。
    assert text_segments("就一句话") is None
    assert text_segments("") is None


def test_text_segments_splits_tokens_into_face_segments() -> None:
    assert text_segments("行吧[表情：doge]你说得对[表情:笑哭]") == [
        {"type": "text", "data": {"text": "行吧"}},
        {"type": "face", "data": {"id": "179"}},
        {"type": "text", "data": {"text": "你说得对"}},
        {"type": "face", "data": {"id": "182"}},
    ]


def test_text_segments_accepts_half_width_colon_and_numeric_ids() -> None:
    assert text_segments("[表情:14]") == [{"type": "face", "data": {"id": "14"}}]
    assert text_segments("[表情：182]") == [{"type": "face", "data": {"id": "182"}}]


def test_unresolvable_sticker_token_is_dropped() -> None:
    # 商城表情没有可复用的 id，只能丢记号，不能让字面文字漏出去。
    assert text_segments("[表情包：某个自定义表情]") == []
    assert text_segments("好[表情包：某个自定义表情]") == [
        {"type": "text", "data": {"text": "好"}},
    ]


def test_face_table_only_contains_numeric_ids_and_non_empty_names() -> None:
    assert FACE_NAMES and all(key.isdigit() for key in FACE_NAMES)
    assert all(name.strip() for name in FACE_NAMES.values())
