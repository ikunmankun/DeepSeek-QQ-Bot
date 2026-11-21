"""图片色彩分析与文档要点提取的工具接线：群聊 service_tools + 插件。"""
import io
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from PIL import Image

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.plugins.base import TurnContext
from app.agent.plugins.document_digest import DOCUMENT_DAILY_LIMIT, DocumentDigestFeature
from app.agent.service_tools import ServiceToolContext, build_service_tools
from app.agent.social_contracts import SocialJob, SocialObservation
from app.config import AppSettings
from app.runtime.blocking import BoundedBlockingRunner
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

from test_service_tools import GROUP, ACCOUNT, AUTHOR, Guard


def _blue_tinted_png_bytes():
    image = Image.new("RGB", (80, 80))
    image.putdata([(230, 242, 255)] * 6400)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def seed_text(engine, *, message_id, plain_text, raw_json=None, group_id=GROUP):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic",
                                              enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=AUTHOR, nickname="synthetic author", group_card="")
        row = MessageRepository(session).add_group_message(
            platform_msg_id=message_id, group_id=group_id, user_id=AUTHOR,
            timestamp=datetime.now(UTC), plain_text=plain_text, msg_type="text",
            raw_json=raw_json or {"sender": {"nickname": "synthetic author"}},
            reply_to_msg_id=None, mentioned_bot=False)
        session.flush()
        return row.id


@pytest.fixture
def image_source(sqlite_engine, tmp_path):
    """一张落在群图片缓存里的偏蓝截图 + 对应消息。"""
    cache_dir = tmp_path / "image_cache" / str(GROUP)
    cache_dir.mkdir(parents=True)
    picture = cache_dir / "synthetic.png"
    picture.write_bytes(_blue_tinted_png_bytes())
    raw = {"sender": {"nickname": "synthetic author"},
           "message": [{"type": "image", "data": {"local_path": str(picture)}}]}
    return seed_text(sqlite_engine, message_id="blue-shot", plain_text="看看这张图", raw_json=raw)


@pytest.fixture
def service_context(sqlite_engine, tmp_path, image_source):
    settings = AppSettings.model_construct(bot_qq=ACCOUNT, data_dir=tmp_path,
        napcat_ws_url="ws://127.0.0.1:1", llm_base_url="http://127.0.0.1:1",
        llm_api_key="synthetic", llm_supports_vision_input=True,
        memory_memory_tools_enabled=False, memory_orchestration_v2_enabled=False,
        memory_raw_v3_enabled=False, memory_layered_memory_enabled=False,
        memory_embedding_provider="disabled", memory_compaction_enabled=False,
        memory_query_rewrite_enabled=False)
    runtime = SimpleNamespace(settings=settings, persona={"name": "小町", "identity": "自主兴趣"},
        personas={}, social_policy={"services": {}})
    router = SimpleNamespace(engine=sqlite_engine, runtime=runtime)
    job = SocialJob(f"group:{GROUP}", "group", GROUP, autonomous_tools=True)
    observation = SocialObservation(job, datetime.now(UTC), "synthetic persona",
                                    source_message_ids=(image_source,))
    observed_sources = {image_source}
    store = AutonomousArtifactStore(tmp_path / "artifacts", account_id=ACCOUNT, scope=job.key,
        engine=sqlite_engine, source_message_ids=lambda: tuple(sorted(observed_sources)))
    runner = BoundedBlockingRunner(1)
    value = ServiceToolContext(router, observation, Guard(), observed_sources, runner, store)
    try:
        yield value
    finally:
        runner.close()


@pytest.mark.asyncio
async def test_analyze_image_colors_reports_cold_cast(service_context):
    registry = {tool.name: tool for tool in build_service_tools(service_context)}
    assert "analyze_image_colors" in registry
    tool = registry["analyze_image_colors"]
    outcome = await tool.handler({"source_ids": [service_context.observation.source_message_ids[0]]})
    assert outcome.status == "ok"
    stats = outcome.data["images"][0]["stats"]
    assert stats["cast"]["basis"] == "中性区"
    assert "偏冷" in stats["cast"]["direction"]
    assert stats["cast"]["neutral_rgb"][2] > stats["cast"]["neutral_rgb"][0]
    assert "guidance" in outcome.data


@pytest.mark.asyncio
async def test_analyze_image_colors_rejects_unobserved_sources(service_context):
    registry = {tool.name: tool for tool in build_service_tools(service_context)}
    outcome = await registry["analyze_image_colors"].handler({"source_ids": [999999]})
    assert outcome.status == "blocked"
    assert outcome.data["error"] == "image_source_out_of_scope"


class _FakeBlocking:
    async def run(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class FakeDigestClient:
    def __init__(self):
        self.calls = []

    def generate_text(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return '["决策A：改方案", "数字B：预算3万"]'
        return '{"gist": "测试主旨", "points": ["决策A：改方案", "数字B：预算3万"]}'


def _digest_turn(sqlite_engine, *, now=None):
    guard = Guard()
    job = SocialJob(f"group:{GROUP}", "group", GROUP, autonomous_tools=True)
    return TurnContext(router=SimpleNamespace(engine=sqlite_engine), job=job, guard=guard,
                       store=None, blocking=_FakeBlocking(), interactive=True,
                       now=now or datetime(2026, 11, 18, 12, 0)), guard


def _feature_with_client():
    feature = DocumentDigestFeature()
    feature._client = FakeDigestClient()
    return feature


@pytest.mark.asyncio
async def test_document_digest_reads_full_text_from_sources(sqlite_engine):
    source = seed_text(sqlite_engine, message_id="long-doc",
                       plain_text="这是一份很长的测试文档。" * 50)
    feature = _feature_with_client()
    turn, _ = _digest_turn(sqlite_engine)
    tools = feature.build_tools(turn)
    assert [tool.name for tool in tools] == ["summarize_document"]
    outcome = await tools[0].handler({"source_ids": [source]})
    assert outcome.status == "ok"
    assert outcome.data["gist"] == "测试主旨"
    assert "决策A：改方案" in outcome.data["points"]
    # map 的输入是消息原文全文，不是上下文截断片段。
    map_input = feature._client.calls[0][2]
    assert map_input.startswith("这是一份很长的测试文档。")
    assert len(map_input) > 400


@pytest.mark.asyncio
async def test_document_digest_other_group_source_unreadable(sqlite_engine):
    other_group = GROUP + 1
    foreign_id = seed_text(sqlite_engine, message_id="other-group-doc",
                           plain_text="别的群的文档", group_id=other_group)
    feature = _feature_with_client()
    turn, _ = _digest_turn(sqlite_engine)
    tools = feature.build_tools(turn)
    outcome = await tools[0].handler({"source_ids": [foreign_id]})
    assert outcome.status == "blocked"
    assert outcome.data["error"] == "no_document_text"


@pytest.mark.asyncio
async def test_document_digest_daily_quota_blocks(sqlite_engine):
    seed_text(sqlite_engine, message_id="quota-doc", plain_text="文档正文" * 50)
    feature = _feature_with_client()
    # 配额按天重置：模拟"今天已用满"，同时把计费日钉在今天。
    feature._usage_day = f"{datetime(2026, 11, 18, 12, 0):%Y-%m-%d}"
    feature._usage_count = DOCUMENT_DAILY_LIMIT
    turn, _ = _digest_turn(sqlite_engine, now=datetime(2026, 11, 18, 12, 0))
    tools = feature.build_tools(turn)
    outcome = await tools[0].handler({"text": "这是一段足够长的正文内容，用来触发配额检查。" * 2})
    assert outcome.status == "blocked"
    assert outcome.data["error"] == "document_digest_daily_limit"


@pytest.mark.asyncio
async def test_document_digest_failure_reports_failed(sqlite_engine):
    seed_text(sqlite_engine, message_id="fail-doc", plain_text="文档正文" * 50)
    feature = DocumentDigestFeature()

    class BrokenClient:
        def generate_text(self, messages):
            raise OSError("down")

    feature._client = BrokenClient()
    turn, _ = _digest_turn(sqlite_engine)
    tools = feature.build_tools(turn)
    outcome = await tools[0].handler({"text": "这是一段足够长的正文内容，用于触发提炼失败。" * 2})
    assert outcome.status == "failed"
    assert outcome.data["error"] == "digest_failed"
