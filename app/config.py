from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.style_distill import merge_persona_lists
from app.core.memory_tools import DEFAULT_MEMORY_TOOL_TIMEOUT_SECONDS


# Reasoning votes accepted by every upstream-model call site.  The provider
# switch to DeepSeek introduced ``xhigh``/``max``; ``LlmClient`` already
# normalizes the same vocabulary before sending it upstream.
ReasoningEffort = Literal["", "minimal", "low", "medium", "high", "xhigh", "max"]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    group_agent_enabled: bool = Field(default=True, alias="GROUP_AGENT_ENABLED")
    # Directory of locally hosted sticker/meme images. Non-empty enables the
    # pick_sticker tool in group chats; empty keeps the tool absent.
    local_sticker_dir: str = Field(default="", alias="LOCAL_STICKER_DIR")
    # 每日日记反思：每晚固定时刻自动写一篇（关掉则只靠观察轮随手写）。
    diary_reflection_enabled: bool = Field(default=True, alias="DIARY_REFLECTION_ENABLED")
    diary_reflection_time: str = Field(default="23:30", alias="DIARY_REFLECTION_TIME")
    # B站浏览器自主刷视频（playwright 驱动 /opt/xhs/chrome）。
    bili_browser_enabled: bool = Field(default=True, alias="BILI_BROWSER_ENABLED")
    bili_browser_windows: str = Field(default="12:00", alias="BILI_BROWSER_WINDOWS")
    bili_browser_interests: str = Field(default="动画,游戏,科技,教程", alias="BILI_BROWSER_INTERESTS")
    bili_browser_max_stay_seconds: int = Field(default=480, alias="BILI_BROWSER_MAX_STAY_SECONDS")
    bili_chrome_path: str = Field(default="/opt/xhs/chrome/chrome", alias="BILI_CHROME_PATH")
    # 主动私聊搭话（owner_chitchat worker）：早安窗口 + 闲聊窗口（HH:MM-HH:MM，
    # 逗号分隔，窗口内带抖动）；主人正在私聊时不插话；安静时段 23:00-08:00 顺延。
    owner_chitchat_enabled: bool = Field(default=True, alias="OWNER_CHITCHAT_ENABLED")
    owner_chitchat_morning_window: str = Field(default="08:10-08:50", alias="OWNER_CHITCHAT_MORNING_WINDOW")
    owner_chitchat_windows: str = Field(
        default="12:40-14:30,15:40-17:20,19:00-20:40,20:40-22:30",
        alias="OWNER_CHITCHAT_WINDOWS")
    owner_chitchat_max_daily: int = Field(default=4, alias="OWNER_CHITCHAT_MAX_DAILY")
    owner_chitchat_min_gap_minutes: float = Field(default=150, alias="OWNER_CHITCHAT_MIN_GAP_MINUTES")
    owner_chitchat_idle_minutes: float = Field(default=45, alias="OWNER_CHITCHAT_IDLE_MINUTES")
    # 天气（Open-Meteo 免费接口）：WEATHER_CITY 按城市名 geocoding（支持"广州天河区"
    # 这种写法，查不到会逐级去后缀重试）；WEATHER_LAT/WEATHER_LON 直接给坐标可跳过。
    weather_city: str = Field(default="", alias="WEATHER_CITY")
    weather_lat: float = Field(default=0.0, alias="WEATHER_LAT")
    weather_lon: float = Field(default=0.0, alias="WEATHER_LON")
    # 身体状态模拟（vitality 插件）：饥饿/口渴/精力三维数值，三餐 worker 在群
    # 进程跑并写日记；夜宵是概率事件（饿度达标才吃）。关掉只停排程与提示注入。
    vitality_enabled: bool = Field(default=True, alias="VITALITY_ENABLED")
    vitality_supper_probability: float = Field(default=0.35, alias="VITALITY_SUPPER_PROBABILITY")
    # 语音（TTS）：TTS_API_KEY 为空时功能整体隐藏。provider 支持 mimo/fishaudio/minimax。
    tts_api_key: str = Field(default="", alias="TTS_API_KEY")
    tts_provider: str = Field(default="mimo", alias="TTS_PROVIDER")
    tts_api_url: str = Field(default="https://api.xiaomimimo.com", alias="TTS_API_URL")
    tts_voice: str = Field(default="mimo_default", alias="TTS_VOICE")
    tts_style: str = Field(default="活泼俏皮的少女语气，语速稍快", alias="TTS_STYLE")
    tts_daily_limit: int = Field(default=20, alias="TTS_DAILY_LIMIT")
    tts_group_id: str = Field(default="", alias="TTS_GROUP_ID")      # MiniMax 必需
    tts_model: str = Field(default="mimo-v2.5-tts", alias="TTS_MODEL")
    tts_speed: float = Field(default=1.0, alias="TTS_SPEED")         # >1 更快更活泼  # MiniMax 音色模型
    # 听歌识曲：MUSIC_ID_PROVIDER=audd（默认，用 MUSIC_ID_API_TOKEN）或 xfyun
    # （讯飞「歌曲识别ACRCloud」，用 MUSIC_ID_APP_ID/API_KEY/API_SECRET 三件套；
    # 原生"歌曲识别"只有哼唱 afs，识别不了视频原声 BGM）。凭证缺失时整体隐藏。
    music_id_provider: str = Field(default="audd", alias="MUSIC_ID_PROVIDER")
    music_id_api_token: str = Field(default="", alias="MUSIC_ID_API_TOKEN")
    music_id_app_id: str = Field(default="", alias="MUSIC_ID_APP_ID")
    music_id_api_key: str = Field(default="", alias="MUSIC_ID_API_KEY")
    music_id_api_secret: str = Field(default="", alias="MUSIC_ID_API_SECRET")
    music_id_api_url: str = Field(default="", alias="MUSIC_ID_API_URL")
    music_id_queue_capacity: int = Field(default=4, alias="MUSIC_ID_QUEUE_CAPACITY")
    music_id_timeout_seconds: float = Field(default=240, alias="MUSIC_ID_TIMEOUT_SECONDS")
    # 以图搜图：REVERSE_IMAGE_PROVIDERS 逗号分隔（ascii2d/trace_moe/saucenao，顺序即优先级，
    # 空/auto 用默认链）；off 时整体隐藏。saucenao 需要 REVERSE_IMAGE_SAUCENAO_TOKEN
    # （saucenao.com 免费注册），无凭证自动从链里剔除。
    reverse_image_providers: str = Field(default="ascii2d,trace_moe", alias="REVERSE_IMAGE_PROVIDERS")
    reverse_image_saucenao_token: str = Field(default="", alias="REVERSE_IMAGE_SAUCENAO_TOKEN")
    reverse_image_timeout_seconds: float = Field(default=15.0, alias="REVERSE_IMAGE_TIMEOUT_SECONDS")
    # VPN 代理分流：被墙站点才走（VPN_PROXY_HOSTS 域名后缀清单，含子域），其余直连。
    # 端点支持 http:// 与 socks5://（socks 需镜像装 socksio）；留空 = 全部直连。
    vpn_proxy_url: str = Field(default="", alias="VPN_PROXY_URL")
    vpn_proxy_hosts: str = Field(
        default="saucenao.com,pixiv.net,pximg.net,trace.moe,youtube.com,youtu.be,googlevideo.com,"
                "x.com,twitter.com,twimg.com,danbooru.donmai.us,gelbooru.com",
        alias="VPN_PROXY_HOSTS")
    # 记忆星图：定期把活跃记忆的衰减状态导出到 data/agent_monitor/memory_map.json，
    # 供只读监控页 /memory 画图。只读导出，不影响记忆链路。
    memory_map_enabled: bool = Field(default=True, alias="MEMORY_MAP_ENABLED")
    memory_map_interval_seconds: int = Field(default=300, ge=30, le=86400,
                                             alias="MEMORY_MAP_INTERVAL_SECONDS")
    # 日记本：定期把 data/diary 的近期日记导出到 data/agent_monitor/diary_map.json，
    # 供只读监控页 /diary 翻看。只读导出，不影响日记或聊天链路。
    diary_map_enabled: bool = Field(default=True, alias="DIARY_MAP_ENABLED")
    diary_map_interval_seconds: int = Field(default=300, ge=30, le=86400,
                                            alias="DIARY_MAP_INTERVAL_SECONDS")
    # 语音转文字（ASR）：QQ 语音消息 → 文本，讯飞语音听写，accent 支持方言
    # （cantonese/henanese 等，见讯飞方言清单；mandarin=普通话）。
    # 三件套凭证缺失时整体隐藏；讯飞 app 需在控制台启用"语音听写"服务。
    voice_asr_provider: str = Field(default="xfyun", alias="VOICE_ASR_PROVIDER")
    voice_asr_app_id: str = Field(default="", alias="VOICE_ASR_APP_ID")
    voice_asr_api_key: str = Field(default="", alias="VOICE_ASR_API_KEY")
    voice_asr_api_secret: str = Field(default="", alias="VOICE_ASR_API_SECRET")
    voice_asr_accent: str = Field(default="mandarin", alias="VOICE_ASR_ACCENT")
    voice_asr_language: str = Field(default="zh_cn", alias="VOICE_ASR_LANGUAGE")
    voice_asr_timeout_seconds: float = Field(default=30.0, alias="VOICE_ASR_TIMEOUT_SECONDS")
    # 图表精读：把图表截图交给独立视觉模型（OpenAI 兼容 chat completions，收图）
    # 做结构化解读。主 LLM 是 DeepSeek 官方 API（纯文本收不了图），必须另配视觉
    # 通道：硅基流动/阿里百炼/智谱等都兼容。三件缺失时工具隐藏。
    chart_vision_base_url: str = Field(default="", alias="CHART_VISION_BASE_URL")
    chart_vision_api_key: str = Field(default="", alias="CHART_VISION_API_KEY")
    chart_vision_model: str = Field(default="", alias="CHART_VISION_MODEL")
    chart_vision_timeout_seconds: float = Field(default=60.0, alias="CHART_VISION_TIMEOUT_SECONDS")
    # 长文档要点提取：主模型分块 map-reduce 提要点（纯文本，无需视觉通道）。
    # 关闭后群/私聊的 summarize_document 工具自动隐藏。
    document_digest_enabled: bool = Field(default=True, alias="DOCUMENT_DIGEST_ENABLED")
    agent_runtime: Literal["harness", "legacy"] = Field(default="harness", alias="AGENT_RUNTIME")
    agent_responses_continuation: Literal["off", "auto", "on"] = Field(default="auto", alias="AGENT_RESPONSES_CONTINUATION")
    agent_max_external_writes: int = Field(default=2, ge=1, le=8, alias="AGENT_MAX_EXTERNAL_WRITES")
    group_agent_batch_quiet_seconds: float = Field(default=1.0, ge=0, le=30, alias="GROUP_AGENT_BATCH_QUIET_SECONDS")
    group_agent_batch_max_seconds: float = Field(default=3.0, gt=0, le=60, alias="GROUP_AGENT_BATCH_MAX_SECONDS")
    group_agent_batch_max_messages: int = Field(default=20, ge=1, le=100, alias="GROUP_AGENT_BATCH_MAX_MESSAGES")
    group_agent_max_concurrent_turns: int = Field(default=2, ge=1, le=8, alias="GROUP_AGENT_MAX_CONCURRENT_TURNS")
    group_agent_turn_timeout_seconds: float = Field(default=180, ge=1, le=900, alias="GROUP_AGENT_TURN_TIMEOUT_SECONDS")
    agent_background_turn_timeout_seconds: float = Field(default=900, ge=1, le=900, alias="AGENT_BACKGROUND_TURN_TIMEOUT_SECONDS")
    agent_model_request_timeout_seconds: float = Field(default=120, ge=1, le=300, alias="AGENT_MODEL_REQUEST_TIMEOUT_SECONDS")
    agent_model_first_output_timeout_seconds: float = Field(default=90, ge=1, le=120, alias="AGENT_MODEL_FIRST_OUTPUT_TIMEOUT_SECONDS")
    agent_model_progress_timeout_seconds: float = Field(default=45, ge=1, le=120, alias="AGENT_MODEL_PROGRESS_TIMEOUT_SECONDS")
    group_agent_decision_timeout_seconds: float = Field(default=45, ge=1, le=180, alias="GROUP_AGENT_DECISION_TIMEOUT_SECONDS")
    group_agent_decision_max_output_tokens: int = Field(default=768, ge=256, le=2048, alias="GROUP_AGENT_DECISION_MAX_OUTPUT_TOKENS")
    group_agent_decision_reasoning_effort: ReasoningEffort = Field(default="low", alias="GROUP_AGENT_DECISION_REASONING_EFFORT")
    group_agent_max_pending_per_group: int = Field(default=2000, ge=20, le=20000, alias="GROUP_AGENT_MAX_PENDING_PER_GROUP")
    group_agent_max_message_age_seconds: float = Field(default=300, ge=10, le=3600, alias="GROUP_AGENT_MAX_MESSAGE_AGE_SECONDS")

    social_agent_enabled: bool = Field(default=False, alias="SOCIAL_AGENT_ENABLED")
    social_agent_poll_seconds: float = Field(default=15, ge=5, le=300, alias="SOCIAL_AGENT_POLL_SECONDS")
    social_agent_decision_timeout_seconds: float = Field(default=60, ge=15, le=180, alias="SOCIAL_AGENT_DECISION_TIMEOUT_SECONDS")
    social_agent_max_output_tokens: int = Field(default=1024, ge=256, le=2048, alias="SOCIAL_AGENT_MAX_OUTPUT_TOKENS")
    autonomous_sandbox_token: str = Field(default="", alias="XIAOMACHI_SANDBOX_TOKEN")

    napcat_ws_url: str = Field(alias="NAPCAT_WS_URL")
    llm_base_url: str = Field(alias="LLM_BASE_URL")
    llm_api_key: str = Field(alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-5.4-mini", alias="LLM_MODEL")
    llm_fallback_model: str = Field(default="gpt-5.4", alias="LLM_FALLBACK_MODEL")
    llm_text_endpoint: Literal["chat_completions", "responses"] = Field(
        default="chat_completions",
        alias="LLM_TEXT_ENDPOINT",
    )
    llm_builtin_web_search: bool = Field(default=False, alias="LLM_BUILTIN_WEB_SEARCH")
    llm_builtin_web_search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        alias="LLM_BUILTIN_WEB_SEARCH_CONTEXT_SIZE",
    )
    llm_web_search_model: str = Field(default="", alias="LLM_WEB_SEARCH_MODEL")
    llm_web_search_api_key: str = Field(default="", alias="LLM_WEB_SEARCH_API_KEY")
    llm_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        alias="LLM_REASONING_EFFORT",
    )
    # 思考深度路由：按消息复杂度动态覆盖本回合 reasoning effort
    # （简单消息秒答、难题深思，模拟人脑）；精力联动=困了懒得深想
    # （被点名/主人直接求助豁免）。详见 app/core/think_depth.py。
    think_depth_enabled: bool = Field(default=True, alias="THINK_DEPTH_ENABLED")
    think_depth_energy_coupled: bool = Field(
        default=True,
        alias="THINK_DEPTH_ENERGY_COUPLED",
    )
    # 语料向量库（PersonalDialog 蒸馏）：回复时按当前消息检索"相似情境下
    # 真人的接法"注入风格参考。向量库未建/嵌入通道不可用时自动零注入。
    corpus_style_retrieval_enabled: bool = Field(
        default=True,
        alias="CORPUS_STYLE_RETRIEVAL_ENABLED",
    )
    corpus_style_max_examples: int = Field(
        default=2,
        ge=1,
        le=3,
        alias="CORPUS_STYLE_MAX_EXAMPLES",
    )
    corpus_style_min_cosine: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        alias="CORPUS_STYLE_MIN_COSINE",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=10,
        alias="LLM_TIMEOUT_SECONDS",
    )
    llm_supports_vision_input: bool = Field(default=True, alias="LLM_SUPPORTS_VISION_INPUT")
    llm_vision_model: str = Field(default="", alias="LLM_VISION_MODEL")
    group_image_base_url: str = Field(default="", alias="GROUP_IMAGE_BASE_URL")
    group_image_api_key: str = Field(default="", alias="GROUP_IMAGE_API_KEY")
    group_image_chat_base_url: str = Field(default="", alias="GROUP_IMAGE_CHAT_BASE_URL")
    group_image_chat_api_key: str = Field(default="", alias="GROUP_IMAGE_CHAT_API_KEY")
    group_image_transport: Literal["responses", "images"] = Field(
        default="responses",
        alias="GROUP_IMAGE_TRANSPORT",
    )
    group_image_model: str = Field(default="gpt-image-2", alias="GROUP_IMAGE_MODEL")
    group_image_edit_model: str = Field(default="", alias="GROUP_IMAGE_EDIT_MODEL")
    # Standard Responses separates the mainline model from image_generation.model.
    # Empty preserves the existing relay-specific image-model-as-outer contract.
    group_image_responses_model: str = Field(default="", alias="GROUP_IMAGE_RESPONSES_MODEL")
    group_image_generations_endpoint: str = Field(
        default="/images/generations",
        alias="GROUP_IMAGE_GENERATIONS_ENDPOINT",
    )
    group_image_edits_endpoint: str = Field(default="/images/edits", alias="GROUP_IMAGE_EDITS_ENDPOINT")
    group_image_size: str = Field(default="auto", alias="GROUP_IMAGE_SIZE")
    group_image_quality: str = Field(default="high", alias="GROUP_IMAGE_QUALITY")
    group_image_output_format: str = Field(default="png", alias="GROUP_IMAGE_OUTPUT_FORMAT")
    group_image_queue_capacity: int = Field(default=3, alias="GROUP_IMAGE_QUEUE_CAPACITY")
    group_image_timeout_seconds: float = Field(default=900.0, alias="GROUP_IMAGE_TIMEOUT_SECONDS")
    bot_qq: int = Field(alias="BOT_QQ")
    owner_qq: int = Field(alias="OWNER_QQ")
    private_chat_qqs: str = Field(default="", alias="PRIVATE_CHAT_QQS")
    private_chat_all_friends: bool = Field(default=False, alias="PRIVATE_CHAT_ALL_FRIENDS")
    search_provider: str = Field(default="tavily", alias="SEARCH_PROVIDER")
    search_base_url: str = Field(default="https://api.tavily.com/search", alias="SEARCH_BASE_URL")
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")
    search_timeout_seconds: float = Field(default=8.0, alias="SEARCH_TIMEOUT_SECONDS")
    agent_web_search_timeout_seconds: float = Field(default=45.0, ge=5, le=90, alias="AGENT_WEB_SEARCH_TIMEOUT_SECONDS")
    agent_web_search_results: int = Field(default=8, ge=1, le=20, alias="AGENT_WEB_SEARCH_RESULTS")
    agent_web_task_max_calls: int = Field(default=16, ge=8, le=24, alias="AGENT_WEB_TASK_MAX_CALLS")
    agent_task_turn_timeout_seconds: float = Field(default=600, ge=180, le=900, alias="AGENT_TASK_TURN_TIMEOUT_SECONDS")
    agent_web_task_max_chars: int = Field(default=96000, ge=48000, le=128000, alias="AGENT_WEB_TASK_MAX_CHARS")
    web_page_proxy_url: str = Field(default="", alias="WEB_PAGE_PROXY_URL")
    x_crawler_session_file: str = Field(default="", alias="X_CRAWLER_SESSION_FILE", repr=False)
    x_crawler_proxy_url: str = Field(default="http://127.0.0.1:18797", alias="X_CRAWLER_PROXY_URL")
    download_media_retention_hours: float = Field(default=24, ge=1, le=168, alias='DOWNLOAD_MEDIA_RETENTION_HOURS')
    download_media_cache_max_mib: int = Field(default=512, ge=128, le=8192, alias='DOWNLOAD_MEDIA_CACHE_MAX_MIB')
    agent_context_compaction_enabled: bool = Field(default=True, alias="AGENT_CONTEXT_COMPACTION_ENABLED")
    agent_context_compaction_threshold_ratio: float = Field(default=0.62, ge=0.2, le=0.8,
        alias="AGENT_CONTEXT_COMPACTION_THRESHOLD_RATIO")
    agent_context_retain_tokens: int = Field(default=32000, ge=1000, le=64000, alias="AGENT_CONTEXT_RETAIN_TOKENS")
    agent_context_summary_tokens: int = Field(default=8192, ge=512, le=8192, alias="AGENT_CONTEXT_SUMMARY_TOKENS")
    agent_context_history_tokens: int = Field(default=24000, ge=4000, le=64000, alias="AGENT_CONTEXT_HISTORY_TOKENS")
    agent_context_soft_input_tokens: int = Field(default=96000, ge=8000, le=200000, alias="AGENT_CONTEXT_SOFT_INPUT_TOKENS")
    agent_context_target_input_tokens: int = Field(default=64000, ge=4000, le=160000, alias="AGENT_CONTEXT_TARGET_INPUT_TOKENS")
    agent_context_hard_input_tokens: int = Field(default=160000, ge=12000, le=240000, alias="AGENT_CONTEXT_HARD_INPUT_TOKENS")
    agent_memory_context_tokens: int = Field(default=8000, ge=1500, le=24000, alias="AGENT_MEMORY_CONTEXT_TOKENS")
    agent_background_history_tokens: int = Field(default=8000, ge=2000, le=24000, alias="AGENT_BACKGROUND_HISTORY_TOKENS")
    agent_monitor_enabled: bool = Field(default=False, alias="AGENT_MONITOR_ENABLED")
    web_search_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = Field(
        default="medium", alias="WEB_SEARCH_REASONING_EFFORT")
    search_region: str = Field(default="wt-wt", alias="SEARCH_REGION")
    search_backend: str = Field(default="auto", alias="SEARCH_BACKEND")
    context_recent_limit: int = Field(default=60, alias="CONTEXT_RECENT_LIMIT")
    context_summary_limit: int = Field(default=3, alias="CONTEXT_SUMMARY_LIMIT")
    context_history_limit: int = Field(default=8, alias="CONTEXT_HISTORY_LIMIT")
    # Live group messages older than this are archived (context/memory) but not
    # answered: QQ bridges replay everything that queued up while the session
    # was offline.  0 disables the gate.
    group_message_max_age_seconds: int = Field(
        default=300,
        ge=0,
        alias="GROUP_MESSAGE_MAX_AGE_SECONDS",
    )
    # Reply delivery shape.  These live outside the persona so the persona file
    # only describes style and personality: normal replies stay short, and only
    # an over-long answer is split into several QQ messages.
    group_reply_split_enabled: bool = Field(
        default=True,
        alias="GROUP_REPLY_SPLIT_ENABLED",
    )
    group_reply_split_max_messages: int = Field(
        default=3,
        ge=1,
        le=6,
        alias="GROUP_REPLY_SPLIT_MAX_MESSAGES",
    )
    group_reply_split_max_chars: int = Field(
        default=64,
        ge=8,
        alias="GROUP_REPLY_SPLIT_MAX_CHARS",
    )
    group_reply_split_min_delay_seconds: float = Field(
        default=0.8,
        ge=0.0,
        alias="GROUP_REPLY_SPLIT_MIN_DELAY_SECONDS",
    )
    group_reply_split_max_delay_seconds: float = Field(
        default=2.5,
        ge=0.0,
        alias="GROUP_REPLY_SPLIT_MAX_DELAY_SECONDS",
    )
    # 拟人打字延迟：回复发出前的"阅读+打字"停顿（默认 0 = 关闭，测试基线不变）。
    # 全 0 时无延迟；启用后 = uniform(base, base_max) + per_char × 文本长度，
    # 再截到 max。只作用于纯文字群回复的第一段，段间节奏仍由 split delay 管。
    group_reply_typing_base_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        alias="GROUP_REPLY_TYPING_BASE_SECONDS",
    )
    group_reply_typing_base_max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        alias="GROUP_REPLY_TYPING_BASE_MAX_SECONDS",
    )
    group_reply_typing_per_char_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="GROUP_REPLY_TYPING_PER_CHAR_SECONDS",
    )
    group_reply_typing_max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        alias="GROUP_REPLY_TYPING_MAX_SECONDS",
    )
    # 私聊同款打字延迟：DM 期望比群聊快一点，默认参数也小一号。
    private_reply_typing_base_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        alias="PRIVATE_REPLY_TYPING_BASE_SECONDS",
    )
    private_reply_typing_base_max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        alias="PRIVATE_REPLY_TYPING_BASE_MAX_SECONDS",
    )
    private_reply_typing_per_char_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="PRIVATE_REPLY_TYPING_PER_CHAR_SECONDS",
    )
    private_reply_typing_max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        alias="PRIVATE_REPLY_TYPING_MAX_SECONDS",
    )
    memory_compaction_enabled: bool = Field(default=True, alias="MEMORY_COMPACTION_ENABLED")
    memory_compaction_batch_size: int = Field(default=50, alias="MEMORY_COMPACTION_BATCH_SIZE")
    memory_compaction_max_facts: int = Field(default=24, alias="MEMORY_COMPACTION_MAX_FACTS")
    memory_compaction_retry_limit: int = Field(default=3, alias="MEMORY_COMPACTION_RETRY_LIMIT")
    memory_compaction_backfill_windows: int = Field(default=24, alias="MEMORY_COMPACTION_BACKFILL_WINDOWS")
    memory_compaction_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        alias="MEMORY_COMPACTION_REASONING_EFFORT",
    )
    memory_compaction_max_output_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        alias="MEMORY_COMPACTION_MAX_OUTPUT_TOKENS",
    )
    memory_orchestration_v2_enabled: bool = Field(default=False, alias="MEMORY_ORCHESTRATION_V2_ENABLED")
    memory_orchestration_shadow_mode: bool = Field(default=False, alias="MEMORY_ORCHESTRATION_SHADOW_MODE")
    memory_raw_v3_enabled: bool = Field(default=False, alias="MEMORY_RAW_V3_ENABLED")
    memory_embedding_provider: Literal["local", "openai_compatible", "disabled"] = Field(
        default="local", alias="MEMORY_EMBEDDING_PROVIDER"
    )
    memory_embedding_device: Literal["auto", "cuda", "cpu"] = Field(
        default="cpu", alias="MEMORY_EMBEDDING_DEVICE"
    )
    memory_embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", alias="MEMORY_EMBEDDING_MODEL")
    memory_embedding_dimensions: int = Field(default=512, alias="MEMORY_EMBEDDING_DIMENSIONS")
    memory_embedding_cache_dir: Path = Field(
        default=Path("/workspace/data/models"), alias="MEMORY_EMBEDDING_CACHE_DIR"
    )
    memory_embedding_local_files_only: bool = Field(
        default=False, alias="MEMORY_EMBEDDING_LOCAL_FILES_ONLY"
    )
    memory_embedding_base_url: str = Field(default="", alias="MEMORY_EMBEDDING_BASE_URL")
    memory_embedding_api_key: str = Field(default="", alias="MEMORY_EMBEDDING_API_KEY")
    memory_embedding_version: str = Field(default="", alias="MEMORY_EMBEDDING_VERSION")
    memory_embedding_timeout_seconds: float = Field(default=10.0, alias="MEMORY_EMBEDDING_TIMEOUT_SECONDS")
    memory_retrieval_channel_timeout_seconds: float = Field(
        default=45.0, gt=0, le=60, alias="MEMORY_RETRIEVAL_CHANNEL_TIMEOUT_SECONDS"
    )
    memory_episode_idle_minutes: int = Field(default=10, alias="MEMORY_EPISODE_IDLE_MINUTES")
    memory_episode_max_messages: int = Field(default=70, alias="MEMORY_EPISODE_MAX_MESSAGES")
    memory_episode_max_tokens: int = Field(default=8000, alias="MEMORY_EPISODE_MAX_TOKENS")
    memory_episode_topic_judge_enabled: bool = Field(
        default=False,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_ENABLED",
    )
    memory_episode_topic_judge_context_messages: int = Field(
        default=20,
        ge=2,
        le=30,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_CONTEXT_MESSAGES",
    )
    memory_episode_topic_judge_start_messages: int = Field(
        default=50,
        ge=1,
        le=100,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_START_MESSAGES",
    )
    memory_episode_topic_judge_interval: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_INTERVAL",
    )
    memory_episode_topic_judge_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        alias="MEMORY_EPISODE_TOPIC_JUDGE_REASONING_EFFORT",
    )
    memory_episode_topic_judge_max_output_tokens: int = Field(
        default=128,
        ge=32,
        le=512,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_MAX_OUTPUT_TOKENS",
    )
    memory_episode_post_segment_enabled: bool = Field(
        default=False,
        alias="MEMORY_EPISODE_POST_SEGMENT_ENABLED",
    )
    memory_episode_post_segment_min_messages: int = Field(
        default=25,
        ge=5,
        le=120,
        alias="MEMORY_EPISODE_POST_SEGMENT_MIN_MESSAGES",
    )
    memory_episode_post_segment_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        alias="MEMORY_EPISODE_POST_SEGMENT_REASONING_EFFORT",
    )
    memory_episode_post_segment_max_output_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        alias="MEMORY_EPISODE_POST_SEGMENT_MAX_OUTPUT_TOKENS",
    )
    memory_current_default_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        alias="MEMORY_CURRENT_DEFAULT_TTL_HOURS",
    )
    memory_chunk_max_tokens: int = Field(default=2400, alias="MEMORY_CHUNK_MAX_TOKENS")
    memory_chunk_overlap_messages: int = Field(default=8, alias="MEMORY_CHUNK_OVERLAP_MESSAGES")
    memory_chunk_max_messages: int = Field(default=40, ge=1, alias="MEMORY_CHUNK_MAX_MESSAGES")
    memory_query_rewrite_enabled: bool = Field(default=False, alias="MEMORY_QUERY_REWRITE_ENABLED")
    memory_query_rewrite_timeout_seconds: float = Field(default=3.0, alias="MEMORY_QUERY_REWRITE_TIMEOUT_SECONDS")
    memory_query_rewrite_max_output_tokens: int = Field(
        default=256, alias="MEMORY_QUERY_REWRITE_MAX_OUTPUT_TOKENS"
    )
    memory_llm_rerank_enabled: bool = Field(default=False, alias="MEMORY_LLM_RERANK_ENABLED")
    memory_normal_context_budget_tokens: int = Field(default=48000, alias="MEMORY_NORMAL_CONTEXT_BUDGET_TOKENS")
    memory_detail_context_budget_tokens: int = Field(default=64000, alias="MEMORY_DETAIL_CONTEXT_BUDGET_TOKENS")
    memory_recent_context_budget_tokens: int = Field(default=10000, alias="MEMORY_RECENT_CONTEXT_BUDGET_TOKENS")
    memory_history_context_budget_tokens: int = Field(
        default=24000,
        alias="MEMORY_HISTORY_CONTEXT_BUDGET_TOKENS",
    )
    memory_context_budget_chars: int = Field(
        default=24000,
        alias="MEMORY_CONTEXT_BUDGET_CHARS",
    )
    memory_adaptive_context_enabled: bool = Field(
        default=False,
        alias="MEMORY_ADAPTIVE_CONTEXT_ENABLED",
    )
    memory_adaptive_context_budget_chars: int = Field(
        default=48000,
        gt=0,
        alias="MEMORY_ADAPTIVE_CONTEXT_BUDGET_CHARS",
    )
    memory_recent_protected_min_tokens: int = Field(
        default=1200,
        ge=0,
        alias="MEMORY_RECENT_PROTECTED_MIN_TOKENS",
    )
    memory_history_protected_min_tokens: int = Field(
        default=2400,
        ge=0,
        alias="MEMORY_HISTORY_PROTECTED_MIN_TOKENS",
    )
    memory_recent_protected_min_messages: int = Field(
        default=1,
        ge=0,
        alias="MEMORY_RECENT_PROTECTED_MIN_MESSAGES",
    )
    memory_history_protected_min_messages: int = Field(
        default=1,
        ge=0,
        alias="MEMORY_HISTORY_PROTECTED_MIN_MESSAGES",
    )
    memory_adaptive_max_recent_messages: int = Field(
        default=60,
        gt=0,
        alias="MEMORY_ADAPTIVE_MAX_RECENT_MESSAGES",
    )
    memory_adaptive_max_history_messages: int = Field(
        default=300,
        gt=0,
        alias="MEMORY_ADAPTIVE_MAX_HISTORY_MESSAGES",
    )
    memory_layered_memory_enabled: bool = Field(
        default=False,
        alias="MEMORY_LAYERED_MEMORY_ENABLED",
    )
    memory_memory_tools_enabled: bool = Field(
        default=False,
        alias="MEMORY_MEMORY_TOOLS_ENABLED",
    )
    proactive_model_judge_enabled: bool = Field(
        default=False,
        alias="PROACTIVE_MODEL_JUDGE_ENABLED",
    )
    proactive_judge_model: str = Field(
        default="",
        alias="PROACTIVE_JUDGE_MODEL",
    )
    proactive_judge_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        alias="PROACTIVE_JUDGE_REASONING_EFFORT",
    )
    proactive_judge_max_output_tokens: int = Field(
        default=256,
        ge=64,
        le=2048,
        alias="PROACTIVE_JUDGE_MAX_OUTPUT_TOKENS",
    )
    proactive_judge_context_messages: int = Field(
        default=20,
        ge=1,
        le=20,
        alias="PROACTIVE_JUDGE_CONTEXT_MESSAGES",
    )
    proactive_judge_max_chars_per_message: int = Field(
        default=120,
        ge=20,
        le=1000,
        alias="PROACTIVE_JUDGE_MAX_CHARS_PER_MESSAGE",
    )
    proactive_recent_messages_limit: int = Field(
        default=60,
        ge=1,
        le=200,
        alias="PROACTIVE_RECENT_MESSAGES_LIMIT",
    )
    proactive_image_max_count: int = Field(
        default=2,
        ge=1,
        le=10,
        alias="PROACTIVE_IMAGE_MAX_COUNT",
    )
    style_judge_enabled: bool = Field(
        default=True,
        alias="STYLE_JUDGE_ENABLED",
    )
    style_judge_sample_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        alias="STYLE_JUDGE_SAMPLE_RATE",
    )
    silence_learning_enabled: bool = Field(
        default=True,
        alias="SILENCE_LEARNING_ENABLED",
    )
    # 人生笔记（自传体长期记忆）：夜间凝练节奏与存量软上限。
    life_note_enabled: bool = Field(
        default=True,
        alias="LIFE_NOTE_ENABLED",
    )
    life_note_min_interval_days: int = Field(
        default=3,
        ge=1,
        le=30,
        alias="LIFE_NOTE_MIN_INTERVAL_DAYS",
    )
    life_note_max_count: int = Field(
        default=200,
        ge=10,
        le=2000,
        alias="LIFE_NOTE_MAX_COUNT",
    )
    # 功能愿望私信主人：夜间提炼新愿望后每天最多一条私聊，安静时段顺延。
    owner_wish_notice_enabled: bool = Field(
        default=True,
        alias="OWNER_WISH_NOTICE_ENABLED",
    )
    owner_wish_notice_voice_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        alias="OWNER_WISH_NOTICE_VOICE_PROBABILITY",
    )
    owner_wish_notice_max_voice_daily: int = Field(
        default=2,
        ge=0,
        le=20,
        alias="OWNER_WISH_NOTICE_MAX_VOICE_DAILY",
    )
    # 事实向量候选通道：query embedding 对群内真向量暴力余弦补候选的条数。
    memory_vector_candidate_limit: int = Field(
        default=10,
        ge=0,
        le=50,
        alias="MEMORY_VECTOR_CANDIDATE_LIMIT",
    )
    # 余额看护：DeepSeek 余额跌破阈值时私聊主人撒娇要饭钱。
    balance_alert_enabled: bool = Field(
        default=True,
        alias="BALANCE_ALERT_ENABLED",
    )
    balance_alert_threshold_yuan: float = Field(
        default=5.0,
        ge=0.0,
        le=1000.0,
        alias="BALANCE_ALERT_THRESHOLD_YUAN",
    )
    balance_alert_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        alias="BALANCE_ALERT_INTERVAL_HOURS",
    )
    trait_evolution_enabled: bool = Field(
        default=True,
        alias="TRAIT_EVOLUTION_ENABLED",
    )
    memory_memory_tool_max_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        alias="MEMORY_MEMORY_TOOL_MAX_ROUNDS",
    )
    memory_decision_envelope_enabled: bool = Field(
        default=True,
        alias="MEMORY_DECISION_ENVELOPE_ENABLED",
    )
    memory_memory_tool_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="MEMORY_MEMORY_TOOL_MAX_RESULTS",
    )
    memory_memory_tool_timeout_seconds: float = Field(
        default=DEFAULT_MEMORY_TOOL_TIMEOUT_SECONDS,
        gt=0,
        le=120,
        alias="MEMORY_MEMORY_TOOL_TIMEOUT_SECONDS",
    )
    memory_member_fact_supplement_limit: int = Field(
        default=20,
        ge=1,
        le=30,
        alias="MEMORY_MEMBER_FACT_SUPPLEMENT_LIMIT",
    )
    memory_fact_semantic_ranking_enabled: bool = Field(
        default=False,
        alias="MEMORY_FACT_SEMANTIC_RANKING_ENABLED",
    )
    memory_fact_semantic_candidates: int = Field(
        default=60,
        ge=1,
        le=200,
        alias="MEMORY_FACT_SEMANTIC_CANDIDATES",
    )
    memory_search_compact_budget_tokens: int = Field(
        default=6000,
        ge=1024,
        alias="MEMORY_SEARCH_COMPACT_BUDGET_TOKENS",
    )
    memory_search_auto_budget_tokens: int = Field(
        default=12000,
        ge=2048,
        alias="MEMORY_SEARCH_AUTO_BUDGET_TOKENS",
    )
    memory_max_evidence_messages: int = Field(
        default=300,
        alias="MEMORY_MAX_EVIDENCE_MESSAGES",
    )
    memory_recent_intent_candidate_limit: int = Field(
        default=48,
        ge=1,
        le=300,
        alias="MEMORY_RECENT_INTENT_CANDIDATE_LIMIT",
    )
    memory_fts_candidate_limit: int = Field(default=30, alias="MEMORY_FTS_CANDIDATE_LIMIT")
    memory_vector_candidate_limit: int = Field(default=30, alias="MEMORY_VECTOR_CANDIDATE_LIMIT")
    memory_final_episode_limit: int = Field(default=12, alias="MEMORY_FINAL_EPISODE_LIMIT")
    llm_context_window_tokens: int = Field(default=258000, alias="LLM_CONTEXT_WINDOW_TOKENS")
    llm_max_output_tokens: int = Field(default=8192, alias="LLM_MAX_OUTPUT_TOKENS")
    llm_context_safety_margin_tokens: int = Field(default=32768, alias="LLM_CONTEXT_SAFETY_MARGIN_TOKENS")
    llm_tool_context_reserve_tokens: int = Field(default=32768, alias="LLM_TOOL_CONTEXT_RESERVE_TOKENS")
    config_dir: Path = Path("configs")
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def validate_adaptive_memory_budget(self) -> AppSettings:
        if not self.memory_adaptive_context_enabled:
            return self
        if (
            self.memory_recent_protected_min_messages
            > self.memory_adaptive_max_recent_messages
        ):
            raise ValueError("adaptive recent protected minimum exceeds message safety cap")
        if (
            self.memory_history_protected_min_messages
            > self.memory_adaptive_max_history_messages
        ):
            raise ValueError("adaptive history protected minimum exceeds message safety cap")
        if (
            self.memory_recent_protected_min_tokens
            + self.memory_history_protected_min_tokens
            > self.memory_normal_context_budget_tokens
        ):
            raise ValueError("adaptive protected token minima exceed normal memory budget")
        return self

    @property
    def memory_recent_snapshot_limit(self) -> int:
        if not self.memory_adaptive_context_enabled:
            return int(self.context_recent_limit)
        return max(
            int(self.context_recent_limit),
            int(self.memory_adaptive_max_recent_messages),
        )

    @property
    def memory_effective_context_budget_chars(self) -> int:
        if self.memory_adaptive_context_enabled:
            return int(self.memory_adaptive_context_budget_chars)
        return int(self.memory_context_budget_chars)

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def private_chat_whitelist(self) -> set[int]:
        values = {self.owner_qq}
        if self.private_chat_qqs:
            values.update(int(item.strip()) for item in self.private_chat_qqs.split(",") if item.strip())
        return values


@dataclass(slots=True)
class RuntimeConfig:
    settings: AppSettings
    persona: dict[str, Any]
    group_policy: dict[str, Any]
    safety: dict[str, Any]
    personas: dict[str, dict[str, Any]] = field(default_factory=dict)
    social_policy: dict[str, Any] = field(default_factory=dict)


def group_recent_context_limit(runtime: RuntimeConfig, group_id: int) -> int:
    """Per-group automatic message window, bounded by the Agent's 30-row cap."""
    defaults = runtime.group_policy.get("default_group_behavior", {})
    group = runtime.group_policy.get("groups", {}).get(str(group_id), {})
    value = group.get("recent_context_limit", defaults.get("recent_context_limit"))
    if isinstance(value, bool):
        return 30
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return 30
    return min(limit, 30) if limit > 0 else 30


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_runtime_config(settings: AppSettings) -> RuntimeConfig:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    from app.core.persona_engine import normalize_persona
    # 本机补丁：允许 voice/backstory/rules/style/examples/avoid 这套结构化写法，
    # 在入口统一映射成渲染器认识的字段名，下游（路由、社交、质量门）无需感知。
    persona = normalize_persona(_read_yaml(settings.config_dir / "persona.yaml"))
    personas: dict[str, dict[str, Any]] = {"default": dict(persona)}
    persona_variants_path = settings.config_dir / "persona_variants.yaml"
    if persona_variants_path.exists():
        variant_config = _read_yaml(persona_variants_path)
        variants = variant_config.get("personas", {})
        if not isinstance(variants, dict):
            raise ValueError("expected personas mapping in persona_variants.yaml")
        for persona_key, overlay in variants.items():
            if not isinstance(overlay, dict):
                raise ValueError(
                    f"expected mapping for persona variant {persona_key!r}"
                )
            personas[str(persona_key)] = _deep_merge_mapping(persona, overlay)
    personas_dir = settings.config_dir / "personas"
    if personas_dir.is_dir():
        for persona_path in sorted(personas_dir.glob("*.yaml")):
            try:
                persona_profile = _read_yaml(persona_path)
            except ValueError as exc:
                raise ValueError(f"invalid persona profile {persona_path.name}: {exc}") from exc
            personas[persona_path.stem] = persona_profile
    live_personas_dir = settings.data_dir / "personas"
    if live_personas_dir.is_dir():
        for persona_path in sorted(live_personas_dir.glob("*.yaml")):
            if persona_path.name.endswith(".live.yaml"):
                continue
            persona_key = persona_path.stem
            persona_profile = _read_yaml(persona_path)
            personas[persona_key] = persona_profile
        for live_path in sorted(live_personas_dir.glob("*.live.yaml")):
            persona_key = live_path.name.removesuffix(".live.yaml")
            if persona_key not in personas:
                continue
            live_profile = _read_yaml(live_path)
            personas[persona_key] = _deep_merge_mapping(
                personas[persona_key], live_profile
            )
    group_policy = _read_yaml(settings.config_dir / "groups.yaml")
    local_groups_path = settings.config_dir / "groups.local.yaml"
    if local_groups_path.exists():
        local_policy = _read_yaml(local_groups_path)
        if isinstance(local_policy, dict):
            group_policy = _deep_merge_mapping(group_policy, local_policy)
    safety = _read_yaml(settings.config_dir / "safety.yaml")
    social_path = settings.config_dir / "social.yaml"
    social_policy = _read_yaml(social_path) if social_path.exists() else {}
    social_local_path = settings.config_dir / "social.local.yaml"
    if social_local_path.exists():
        social_policy = _deep_merge_mapping(social_policy, _read_yaml(social_local_path))
    return RuntimeConfig(
        settings=settings,
        persona=persona,
        group_policy=group_policy,
        safety=safety,
        personas=personas,
        social_policy=social_policy,
    )


def _deep_merge_mapping(base: dict, overlay: dict) -> dict:
    """Merge overlay over base, recursing into nested mappings."""
    merged = dict(base)
    for key, value in overlay.items():
        if key == "groups" and isinstance(value, dict):
            # 生产覆盖文件提供完整、真实的群表，整体替换占位群表。
            merged[key] = value
        elif key in {"facts", "external_relations"} and isinstance(value, list):
            merged[key] = merge_persona_lists(merged.get(key), value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged
