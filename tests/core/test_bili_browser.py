"""B 站浏览器：窗口调度、兴趣挑选、findings 注入。"""

import random
from datetime import datetime
from types import SimpleNamespace

from app.agent.plugins.bili_browser import BilibiliBrowserFeature
from app.core.bili_browser import (
    format_findings,
    next_window,
    parse_windows,
    pick_entries,
)


def test_parse_windows_valid_invalid_and_fallback():
    assert parse_windows("12:00,18:00") == [(12, 0), (18, 0)]
    assert parse_windows("9:05") == [(9, 5)]
    assert parse_windows("abc,25:00") == [(12, 0)]     # 非法片段跳过 → 默认兜底
    assert parse_windows("") == [(12, 0)]


def test_next_window_rolls_to_tomorrow():
    windows = parse_windows("12:00,18:00")
    assert next_window(datetime(2026, 11, 1, 9, 0), windows) == datetime(2026, 11, 1, 12, 0)
    assert next_window(datetime(2026, 11, 1, 13, 0), windows) == datetime(2026, 11, 1, 18, 0)
    assert next_window(datetime(2026, 11, 1, 19, 0), windows) == datetime(2026, 11, 2, 12, 0)


def test_pick_entries_prefers_interest_hits_then_random():
    entries = [{"title": " cooking 教程"}, {"title": "动画新番速报"},
               {"title": "游戏实况"}, {"title": "随机内容"}]
    rng = random.Random(7)
    picked = pick_entries(entries, ["动画", "游戏"], rng=rng, limit=2)
    titles = [e["title"] for e in picked]
    assert "动画新番速报" in titles and "游戏实况" in titles


def test_format_findings_truncates():
    out = format_findings("标题", "很长" * 200)
    assert out.startswith("刷B站看到《标题》：") and len(out) < 240


def test_prompt_lines_inject_fresh_findings_only(tmp_path):
    import time as time_mod

    settings = SimpleNamespace(
        data_dir=tmp_path, bili_browser_enabled=True,
        bili_browser_windows="12:00", bili_browser_interests="动画",
        bili_browser_max_stay_seconds=60, bili_chrome_path="",
        owner_qq=900000101, llm_model="m", llm_api_key="k",
        llm_base_url="http://x")
    feature = BilibiliBrowserFeature()
    feature.attach(settings=settings, sender=None, llm_client=None,
                   persona_text="x")
    # 无 service（chrome 缺失）→ 不注入
    assert feature.prompt_lines(kind="group", group_id=1, now=None, observing=True) == ()

    feature._service = SimpleNamespace(latest_findings=[
        (time_mod.monotonic(), "刷B站看到《猫猫》：评论区在夸猫")])
    lines = feature.prompt_lines(kind="group", group_id=1, now=None, observing=True)
    assert any("猫猫" in line for line in lines)
    # 私聊/非观察轮不注入
    assert feature.prompt_lines(kind="private", group_id=1, now=None, observing=True) == ()
    assert feature.prompt_lines(kind="group", group_id=1, now=None, observing=False) == ()
