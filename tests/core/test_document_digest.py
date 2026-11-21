"""长文档要点提取单元：分块、宽松解析、map-reduce 降级链路。"""
from types import SimpleNamespace

from app.core.document_digest import (
    MAX_DOCUMENT_CHARS,
    digest_client_enabled,
    digest_points,
    parse_digest,
    parse_points,
    split_text_chunks,
)


class FakeClient:
    """generate_text 按调用序返回预设回复；可在指定序号抛异常。"""

    def __init__(self, replies, fail_at=()):
        self.replies = list(replies)
        self.fail_at = set(fail_at)
        self.calls = []

    def generate_text(self, messages):
        self.calls.append(messages)
        if len(self.calls) - 1 in self.fail_at:
            raise OSError("provider down")
        return self.replies.pop(0)


def test_split_text_chunks_respects_paragraph_boundaries():
    text = "\n\n".join(f"第{i}段。" + "内容" * 200 for i in range(5))
    chunks = split_text_chunks(text, target_chars=1000)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 1100 for chunk in chunks)
    # 只允许丢掉段落分隔的换行，不丢正文。
    assert sum(len(chunk) for chunk in chunks) >= len(text) - 10


def test_split_text_chunks_empty_and_long_paragraph():
    assert split_text_chunks("") == []
    assert split_text_chunks("   ") == []
    long_single = "这是一句话。" * 800  # 无空行的超长整段，按句子切
    chunks = split_text_chunks(long_single, target_chars=1000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1200 for chunk in chunks)


def test_parse_points_tolerates_markdown_wrap_and_noise():
    assert parse_points('```json\n["要点一", "要点二"]\n```') == ["要点一", "要点二"]
    assert parse_points('前置说明 ["要点"] 后缀') == ["要点"]
    assert parse_points("不是json") == []
    assert parse_points(None) == []


def test_parse_digest_requires_points():
    good = '{"gist": "主旨", "points": ["a", "b"]}'
    assert parse_digest(good) == {"gist": "主旨", "points": ["a", "b"]}
    assert parse_digest('{"gist": "主旨", "points": []}') is None
    assert parse_digest("garbage") is None


def test_digest_points_map_reduce_happy_path():
    client = FakeClient([
        '["决策1：改用A方案", "数字2：预算3万"]',
        '["决定3：下周一上线"]',
        '{"gist": "项目调整", "points": ["决策1：改用A方案", "数字2：预算3万", "决定3：下周一上线"]}',
    ])
    text = "\n\n".join("段落" * 1100 for _ in range(2))  # 每段约2200字，合计约4400字 → 2块
    result = digest_points(client=client, text=text)
    assert result["gist"] == "项目调整"
    assert len(result["points"]) == 3
    assert result["stats"] == {"chars": len(text), "chunks": 2, "truncated": False}
    assert len(client.calls) == 3  # 2块map + 1次reduce


def test_digest_points_single_chunk_failure_is_skipped():
    client = FakeClient([
        '["幸存的要点"]',
        '{"gist": "g", "points": ["幸存的要点"]}',
    ], fail_at={0})  # 第1块失败，第2块照常
    text = "\n\n".join("段落" * 1200 for _ in range(2))
    result = digest_points(client=client, text=text)
    assert result["points"] == ["幸存的要点"]
    assert result["gist"] == "g"


def test_digest_points_all_failures_return_error():
    client = FakeClient([], fail_at={0, 1})
    text = "\n\n".join("段落" * 1200 for _ in range(2))
    result = digest_points(client=client, text=text)
    assert result["error"] == "digest_failed"


def test_digest_points_reduce_failure_keeps_map_points():
    client = FakeClient(['["要点A"]', '{"broken'], fail_at={1})
    result = digest_points(client=client, text="段落" * 2000)
    assert result["points"] == ["要点A"]
    assert result["gist"] == ""


def test_digest_points_truncates_overlong_document():
    client = FakeClient(['["p"]', '{"gist": "g", "points": ["p"]}'])
    text = "长" * (MAX_DOCUMENT_CHARS + 5000)
    result = digest_points(client=client, text=text)
    assert result["stats"]["truncated"] is True
    assert "truncated_note" in result
    assert result["stats"]["chars"] == MAX_DOCUMENT_CHARS


def test_digest_points_focus_reaches_map_prompt():
    client = FakeClient(['["p"]', '{"gist": "g", "points": ["p"]}'])
    digest_points(client=client, text="正文" * 3000, focus="支付相关")
    assert "支付相关" in client.calls[0][0]


def test_digest_points_empty_text():
    assert digest_points(client=FakeClient([]), text="") == {"error": "empty_text"}


def test_digest_client_enabled_requires_credentials():
    assert digest_client_enabled(SimpleNamespace(
        llm_base_url="", llm_api_key="k", llm_model="m")) is None
    client = digest_client_enabled(SimpleNamespace(
        llm_base_url="https://api.example.test/v1", llm_api_key="k", llm_model="m"))
    assert client is not None
    assert client.max_output_tokens == 1200
