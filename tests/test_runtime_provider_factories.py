from types import SimpleNamespace

import pytest

from app.config import AppSettings
from app.runtime.providers import (
    build_episode_post_segment_client,
    build_episode_topic_judge_client,
    build_memory_compaction_client,
    build_proactive_judge_client,
    build_image_reference_search_client,
    build_llm_client,
    build_web_search_client,
)


@pytest.mark.parametrize(
    "builder, expected_identity",
    [
        (build_proactive_judge_client, False),
        (build_episode_topic_judge_client, False),
        (build_episode_post_segment_client, False),
        (build_memory_compaction_client, True),
    ],
)
def test_callable_factory_keeps_injected_fake_client_without_live_provider(
    builder, expected_identity
):
    settings = SimpleNamespace(
        proactive_model_judge_enabled=True,
        memory_episode_topic_judge_enabled=True,
        memory_episode_post_segment_enabled=True,
    )
    fake_client = object()

    def reject_live_provider(**_kwargs):
        raise AssertionError("a test fake must not construct a live provider")

    result = builder(
        settings=settings,
        llm_client=fake_client,
        client_factory=reject_live_provider,
    )
    assert result is fake_client if expected_identity else result is None


def test_external_sol_search_is_independent_of_gemini_chat_and_image_reference_search():
    settings = AppSettings(_env_file=None, BOT_QQ=900000101, OWNER_QQ=900000102,
        NAPCAT_WS_URL="ws://synthetic.invalid/ws", LLM_BASE_URL="https://gemini.example/v1",
        LLM_API_KEY="synthetic-gemini", LLM_MODEL="gemini-3.8-flash-low", LLM_FALLBACK_MODEL="",
        LLM_REASONING_EFFORT="low", LLM_BUILTIN_WEB_SEARCH=False,
        LLM_WEB_SEARCH_MODEL="gpt-6.1-sol", SEARCH_PROVIDER="wyzlab_alpha",
        SEARCH_BASE_URL="https://search.example/v1", SEARCH_API_KEY="synthetic-search",
        WEB_SEARCH_REASONING_EFFORT="low")
    primary = build_llm_client(settings=settings, engine=None)
    search = build_web_search_client(settings)
    references = build_image_reference_search_client(settings)
    try:
        assert primary.model == primary.responses_model == "gemini-3.8-flash-low"
        assert primary.base_url == settings.llm_base_url and primary.api_key == settings.llm_api_key
        for client in (search, references):
            assert client.provider == "wyzlab_alpha"
            assert client.model == "gpt-6.1-sol" and client.reasoning_effort == "low"
            assert client.base_url == settings.search_base_url and client.api_key == settings.search_api_key
            assert client.base_url != primary.base_url and client.api_key != primary.api_key
    finally:
        for client in (primary, search, references):
            client.http_client.close()


def test_ddgs_image_reference_search_keeps_its_existing_model_independent_shape():
    settings = AppSettings(_env_file=None, BOT_QQ=900000101, OWNER_QQ=900000102,
        NAPCAT_WS_URL="ws://synthetic.invalid/ws", LLM_BASE_URL="https://gemini.example/v1",
        LLM_API_KEY="synthetic-gemini", LLM_WEB_SEARCH_MODEL="gpt-6.1-sol",
        SEARCH_PROVIDER="ddgs", SEARCH_API_KEY="", WEB_SEARCH_REASONING_EFFORT="low")
    client = build_image_reference_search_client(settings)
    try:
        assert client.model == "" and client.reasoning_effort == "medium"
        assert client.provider == "ddgs"
    finally:
        client.http_client.close()


def test_decision_client_variants_share_transport_and_usage():
    """按档位的决策 client 变体：共享连接池/用量统计，effort 各自正确。"""

    from app.runtime.providers import DecisionClientVariants

    settings = AppSettings(
        _env_file=None,
        bot_qq=900000099,
        owner_qq=900000101,
        napcat_ws_url="ws://synthetic.invalid/ws",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="test-key",
        llm_model="synthetic",
        llm_fallback_model="synthetic",
        llm_reasoning_effort="low",
    )
    usage_calls = []

    def usage_recorder(usage):
        usage_calls.append(usage)

    from app.providers.llm_client import LlmClient

    primary = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model="synthetic",
        fallback_model="",
        responses_model="synthetic",
        responses_only=True,
        reasoning_effort="low",
        usage_recorder=usage_recorder,
    )
    variants = DecisionClientVariants(settings=settings, primary=primary)
    # 主档位请求变体 → None（直接沿用主 client）。
    assert variants.for_effort("low") is None
    assert variants.for_effort(None) is None
    high = variants.for_effort("high")
    assert high is not None and high.reasoning_effort == "high"
    assert high.http_client is primary.http_client
    assert high.usage_recorder is primary.usage_recorder
    # 同档位缓存复用同一实例；假 client 一律 None。
    assert variants.for_effort("high") is high
    assert DecisionClientVariants(settings=settings, primary=object()).for_effort("high") is None
