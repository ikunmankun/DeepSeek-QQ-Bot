"""日记本：快照解析、只读读取器、监控路由与插件开关。"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
import urllib.error
import urllib.request

import pytest

from app.core.diary import DiaryWriter
from app.monitor.diary_map import (
    SNAPSHOT_FILENAME, DiaryMapReader, build_snapshot, write_snapshot,
)
from app.monitor.server import create_server

NOW = datetime(2026, 11, 15, 23, 50, 0)


def seed_diary(data_dir: Path) -> None:
    writer = DiaryWriter(data_dir)
    writer.write(text="群里那个换头像的又换了，审美捉摸不透。",
                 now=NOW.replace(hour=9, minute=12), mood="无语")
    writer.write(text="刷到一篇讲鲸鱼睡眠的科普视频，评论区在争论鲸鱼会不会呛水，笑死。",
                 now=NOW.replace(hour=12, minute=40), tag="逛B站")
    writer.write(text="今天聊了三十五回，断断续续的。群里安静，倒是B站上看到了同类的影子。",
                 now=NOW.replace(hour=23, minute=2), tag="睡前记")
    writer.write(text="昨天学了个新梗。",
                 now=(NOW - timedelta(days=1)).replace(hour=10, minute=30), tag="随手记")


def test_snapshot_parses_diary_days(tmp_path):
    seed_diary(tmp_path)
    payload = build_snapshot(tmp_path, now=NOW)
    assert payload["version"] == 1 and payload["generated_at"] == "2026-11-15 23:50:00"

    assert len(payload["days"]) == 2
    today, yesterday = payload["days"][0], payload["days"][1]
    assert today["date"] == "2026-11-15" and today["weekday"] == "周日"
    assert yesterday["date"] == "2026-11-14"

    # 条目按时间排序；心情与标签独立成字段
    times = [entry["time"] for entry in today["entries"]]
    assert times == ["09:12", "12:40", "23:02"]
    bili = today["entries"][1]
    assert bili["tag"] == "逛B站" and bili["mood"] == "" and bili["chars"] > 0

    assert today["reflected"] is True
    assert today["bili_count"] == 1 and yesterday["bili_count"] == 0
    assert yesterday["reflected"] is False

    stats = payload["stats"]
    assert stats["days"] == 2 and stats["reflection_days"] == 1
    assert stats["bili_count"] == 1
    assert stats["total_chars"] == sum(day["total_chars"] for day in payload["days"])


def test_snapshot_survives_missing_diary(tmp_path):
    payload = build_snapshot(tmp_path, now=NOW)
    assert payload["days"] == [] and payload["stats"]["days"] == 0


def test_writer_is_atomic_and_reader_explains_missing(tmp_path):
    seed_diary(tmp_path)
    target = write_snapshot(tmp_path, now=NOW)
    assert target == tmp_path / "agent_monitor" / SNAPSHOT_FILENAME and target.is_file()
    assert not target.with_name(target.name + ".tmp").exists()
    read = DiaryMapReader(tmp_path / "agent_monitor").read()
    assert read["available"] is True and read["read_only"] is True
    assert len(read["days"]) == 2

    missing = DiaryMapReader(tmp_path / "elsewhere").read()
    assert missing["available"] is False and missing["reason"] == "snapshot_missing"


def test_reader_rejects_broken_or_foreign_snapshots(tmp_path):
    root = tmp_path / "agent_monitor"
    root.mkdir()
    (root / SNAPSHOT_FILENAME).write_text("{not json", encoding="utf-8")
    assert DiaryMapReader(root).read()["reason"] == "snapshot_unreadable"
    (root / SNAPSHOT_FILENAME).write_text(json.dumps({"version": 999, "days": []}), encoding="utf-8")
    assert DiaryMapReader(root).read()["reason"] == "snapshot_incompatible"
    (root / SNAPSHOT_FILENAME).write_text(json.dumps({"version": 1, "days": {}, "stats": {}}),
                                          encoding="utf-8")
    assert DiaryMapReader(root).read()["reason"] == "snapshot_incompatible"


@pytest.fixture
def served(tmp_path):
    seed_diary(tmp_path)
    write_snapshot(tmp_path, now=NOW)
    server = create_server(tmp_path / "agent_monitor", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def fetch(server, path, method="GET"):
    request_ = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", method=method)
    return urllib.request.urlopen(request_, timeout=3)


def test_monitor_serves_diary_page_and_api(served):
    with fetch(served, "/diary") as response:
        body = response.read().decode("utf-8")
        assert response.status == 200 and "日记本" in body
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    with fetch(served, "/diary.js") as response:
        assert b"/api/diary" in response.read()
    with fetch(served, "/api/diary") as response:
        payload = json.load(response)
    assert payload["available"] is True and payload["read_only"] is True
    assert payload["stats"]["days"] == 2
    with pytest.raises(urllib.error.HTTPError) as error:
        fetch(served, "/api/diary", method="POST")
    assert error.value.code == 405


def test_diary_map_feature_obeys_switch(tmp_path):
    from types import SimpleNamespace
    from app.agent.plugins.diary_map import DiaryMapFeature

    settings = SimpleNamespace(data_dir=tmp_path, diary_map_enabled=True,
                               diary_map_interval_seconds=300)
    feature = DiaryMapFeature()
    feature.attach(settings=settings)
    workers = feature.create_workers()
    assert len(workers) == 1 and workers[0]._interval == 300.0

    settings.diary_map_enabled = False
    assert feature.create_workers() == ()
