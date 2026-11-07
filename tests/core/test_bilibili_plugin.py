"""B 站插件：引用提取、链接解析入上下文、观察轮热榜、刷B站工具、群兴趣。"""

import httpx
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import select

from app.agent.plugins.bilibili import BilibiliFeature
from app.agent.plugins.base import InboundContext
from app.core.bilibili_patrol import (
    BiliInfoCache,
    VideoInfo,
    bump_group_interest,
    decayed_interests,
    extract_bilibili_refs,
    fetch_popular,
    fetch_video_info,
    hot_prompt_line,
    interest_prompt_line,
    share_prompt_line,
)
from app.storage.db import session_scope
from app.storage.models import BilibiliGroupInterest

NOW = datetime(2026, 10, 31, 12, 0, 0)


def test_extract_refs_dedupes_and_limits():
    text = "看这个 BV1xx411c7mD 和 b23.tv/abc 还有 av170001，BV1xx411c7mD 再看一遍"
    assert extract_bilibili_refs(text) == ["BV1xx411c7mD", "b23.tv/abc", "av170001"]
    assert extract_bilibili_refs("普通聊天，没有链接") == []


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _view_payload(bvid, title="测试视频", author="测试UP", desc="这是一个测试简介",
                  zone=""):
    payload = {"code": 0, "data": {"bvid": bvid, "title": title, "desc": desc,
             "owner": {"name": author}, "aid": 170001, "duration": 300}}
    if zone:
        payload["data"]["tname"] = zone
    return payload


def test_fetch_video_info_and_popular():
    def handler(request: httpx.Request) -> httpx.Response:
        if "web-interface/view" in str(request.url):
            return httpx.Response(200, json=_view_payload("BV1xx411c7mD"))
        if "web-interface/popular" in str(request.url):
            return httpx.Response(200, json={"code": 0, "data": {"list": [
                {"bvid": "BV1hot00001", "title": "热门视频一", "owner": {"name": "UP一"},
                 "desc": "简介一"},
                {"bvid": "BV1hot00002", "title": "热门视频二", "owner": {"name": "UP二"},
                 "desc": "简介二"}]}})
        return httpx.Response(404)

    with _mock_client(handler) as client:
        info = fetch_video_info("BV1xx411c7mD", client=client)
        assert info is not None and info.title == "测试视频" and info.author == "测试UP"
        hot = fetch_popular(client=client, limit=2)
        assert [v.title for v in hot] == ["热门视频一", "热门视频二"]


def test_cache_ttl_and_share_window():
    cache = BiliInfoCache()
    info = VideoInfo("BV1", "标题", "UP", "简介", "https://b23.tv/BV1")
    cache.put_info("BV1", info, now=0)
    assert cache.get_info("BV1", now=100) is info
    assert cache.get_info("BV1", now=10**9) is None
    cache.put_share(1, info, now=0)
    assert cache.recent_share(1, now=60) is info
    assert cache.recent_share(1, now=10**6) is None


def _feature_ctx(sqlite_engine_unused=None, *, text, group_id=900000001):
    event = SimpleNamespace(
        user_id=900000103, group_id=group_id, plain_text=text,
        mentioned_bot=False, reply_to_msg_id=None, raw_payload={},
        timestamp=None, images=[])
    messages = SimpleNamespace(
        get_by_platform_msg_id=lambda platform_msg_id: None,
        has_group_messages_from=lambda **kwargs: True)
    router = SimpleNamespace(runtime=SimpleNamespace(settings=SimpleNamespace(
        bot_qq=900000101, data_dir="/tmp/bili-test-no-cookies")))
    return InboundContext(router=router, session=None, messages=messages, event=event, now=None)


def test_feature_enriches_shared_link_into_prompt():
    feature = BilibiliFeature()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_view_payload("BV1xx411c7mD", title="猫猫视频"))

    def fake_client():
        return _mock_client(handler)

    feature._client = fake_client
    ctx = _feature_ctx(text="你们看这个 https://www.bilibili.com/video/BV1xx411c7mD")
    feature.on_inbound(ctx)
    assert calls, "检测到链接应当取一次视频信息"

    lines = feature.prompt_lines(kind="group", group_id=900000001, now=None, observing=False)
    assert any("猫猫视频" in line for line in lines)
    # 私聊不注入
    assert feature.prompt_lines(kind="private", group_id=1, now=None, observing=False) == ()


def test_observing_turn_gets_hot_line():
    feature = BilibiliFeature()
    feature._hot = [VideoInfo("BV1h", "热门一", "UP", "", "u")]
    lines = feature.prompt_lines(kind="group", group_id=1, now=None, observing=True)
    assert any("B站此刻热门" in line and "热门一" in line for line in lines)


def test_browse_tool_hot_and_video(tmp_path):
    feature = BilibiliFeature()

    def handler(request: httpx.Request) -> httpx.Response:
        if "web-interface/popular" in str(request.url):
            return httpx.Response(200, json={"code": 0, "data": {"list": [
                {"bvid": "BV1h1", "title": "热门", "owner": {"name": "UP"}, "desc": ""}]}})
        return httpx.Response(200, json=_view_payload("BV1xx411c7mD"))

    feature._client = lambda: _mock_client(handler)

    import asyncio

    class Guard:
        def require_valid(self):
            return None

    turn = SimpleNamespace(guard=Guard())
    tool = {t.name: t for t in feature.build_tools(turn)}["browse_bilibili"]
    hot = asyncio.run(tool.handler({"action": "hot"}))
    assert hot.status == "ok" and hot.data["videos"][0]["title"] == "热门"
    video = asyncio.run(tool.handler({"action": "video", "query": "BV1xx411c7mD"}))
    assert video.status == "ok" and video.data["videos"][0]["title"] == "测试视频"
    bad = asyncio.run(tool.handler({"action": "video", "query": "没有链接"}))
    assert bad.status == "blocked"


# ---- 群兴趣画像 --------------------------------------------------------

def test_bump_interest_accumulates_and_decays(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        bump_group_interest(session, group_id=1, zone="游戏", now=NOW - timedelta(days=40))
        bump_group_interest(session, group_id=1, zone="游戏", now=NOW - timedelta(days=20))
        bump_group_interest(session, group_id=1, zone="生活", now=NOW)
    with session_scope(sqlite_engine) as session:
        rows = {row.zone: row for row in session.scalars(select(BilibiliGroupInterest))}
    assert rows["游戏"].score == 2.0 and rows["游戏"].share_count == 2
    # 40 天没动的游戏区衰减后仍可召回；生活区新鲜分高
    zones = decayed_interests(sqlite_engine, group_id=1, now=NOW)
    assert "生活" in zones and "游戏" in zones
    # 半衰期 30 天：只有一次分享且 60 天没人提的分区淡出
    with session_scope(sqlite_engine) as session:
        bump_group_interest(session, group_id=2, zone="知识", now=NOW - timedelta(days=60))
    assert decayed_interests(sqlite_engine, group_id=2, now=NOW) == []


def test_interest_prompt_line_requires_fresh_taste(sqlite_engine):
    assert interest_prompt_line(sqlite_engine, group_id=3, now=NOW) is None
    with session_scope(sqlite_engine) as session:
        bump_group_interest(session, group_id=3, zone="游戏", now=NOW)
    line = interest_prompt_line(sqlite_engine, group_id=3, now=NOW)
    assert line is not None and "游戏" in line and "别硬往话题上凑" in line


def test_zone_capped_per_group(sqlite_engine):
    from app.core.bilibili_patrol import MAX_INTEREST_ZONES
    with session_scope(sqlite_engine) as session:
        for index in range(MAX_INTEREST_ZONES + 2):
            bump_group_interest(session, group_id=4, zone=f"分区{index}", now=NOW)
        rows = list(session.scalars(select(BilibiliGroupInterest).where(
            BilibiliGroupInterest.group_id == 4)))
    assert len(rows) <= MAX_INTEREST_ZONES


def test_feature_records_interest_on_shared_link():
    """engine 就绪时分享链接会给分区加分；engine 缺失时静默跳过。"""

    class _FakeScalarResult:
        def __init__(self, items):
            self._items = items

        def all(self):
            return list(self._items)

        def first(self):
            return self._items[0] if self._items else None

    feature = BilibiliFeature()
    info = VideoInfo("BV1", "t", "a", "s", "u", zone="游戏")

    # engine 未接：直接跳过，不碰数据库
    feature._remember_interest(1, info, now=NOW)

    import contextlib

    import app.storage.db as db_module

    store = {}

    class Store:
        def add(self, row):
            store[row.zone] = row

        def get(self, model, key):
            return None

        def scalars(self, stmt):
            return _FakeScalarResult([])

        def delete(self, row):
            return None

    @contextlib.contextmanager
    def fake_session_scope(engine):
        yield Store()

    original = db_module.session_scope
    db_module.session_scope = fake_session_scope
    try:
        feature._engine = object()
        feature._remember_interest(1, info, now=NOW)
    finally:
        db_module.session_scope = original
        feature._engine = None
    assert store["游戏"].share_count == 1
    assert store["游戏"].score == 1.0


def test_shared_link_with_zone_updates_interest_table(sqlite_engine):
    feature = BilibiliFeature()
    feature.attach_engine(sqlite_engine)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_view_payload("BV1xx411c7mD", zone="游戏"))

    feature._client = lambda: _mock_client(handler)
    ctx = _feature_ctx(text="你们看这个 https://www.bilibili.com/video/BV1xx411c7mD")
    ctx.event.timestamp = NOW
    feature.on_inbound(ctx)
    with session_scope(sqlite_engine) as session:
        row = session.scalars(select(BilibiliGroupInterest).where(
            BilibiliGroupInterest.group_id == 900000001)).first()
    assert row is not None and row.zone == "游戏"
