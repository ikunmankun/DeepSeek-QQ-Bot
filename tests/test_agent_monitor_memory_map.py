"""记忆星图：快照数值、只读读取器、监控路由与插件开关。"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
import urllib.error
import urllib.request

import pytest

from app.monitor.memory_map import (
    SNAPSHOT_FILENAME, MemoryMapReader, build_snapshot, write_snapshot,
)
from app.monitor.server import create_server
from app.storage.db import session_scope
from app.storage.models import MemoryItem

NOW = datetime(2026, 11, 11, 20, 0, 0)


def seed(engine):
    def memory(**overrides):
        base = dict(scope_type="group", scope_id="900000001", subject_type="member",
                    subject_id="900000101", memory_kind="fact", source_msg_id="synthetic",
                    canonical_key="", content="合成记忆")
        base.update(overrides)
        return MemoryItem(**base)

    with session_scope(engine) as session:
        session.add(memory(memory_kind="taboo", content="不要戳她睡觉",
                           last_seen_at=NOW - timedelta(days=400)))
        session.add(memory(memory_kind="fact", content="群友在做记忆星图",
                           importance=3, mention_count=3,
                           valid_from=NOW - timedelta(days=10), last_seen_at=NOW))
        session.add(memory(memory_kind="event", content="很久以前的一次吵架",
                           importance=1, mention_count=1,
                           last_seen_at=NOW - timedelta(days=200)))
        session.add(memory(memory_kind="preference", content="喜欢大白饭",
                           importance=2, mention_count=1,
                           last_seen_at=NOW - timedelta(days=1)))


def test_snapshot_classifies_decay_states(sqlite_engine):
    seed(sqlite_engine)
    payload = build_snapshot(sqlite_engine, now=NOW)
    assert payload["version"] == 1 and payload["generated_at"] == "2026-11-11 20:00:00"
    nodes = {node["content"]: node for node in payload["nodes"]}

    taboo = nodes["不要戳她睡觉"]
    assert taboo["immutable"] is True and taboo["faded"] is False
    assert taboo["forget_in_days"] is None and taboo["archive_in_days"] is None

    fact = nodes["群友在做记忆星图"]
    assert fact["promoted"] is True and fact["immutable"] is True
    assert fact["mention_count"] == 3

    event = nodes["很久以前的一次吵架"]
    assert event["immutable"] is False and event["faded"] is True
    assert event["strength"] < 0.5 and event["archive_in_days"] > 0

    preference = nodes["喜欢大白饭"]
    assert preference["faded"] is False and preference["strength"] > 0.9

    stats = payload["stats"]
    assert stats["total"] == 4
    assert stats["immutable"] == 2 and stats["promoted"] == 1 and stats["faded"] == 1
    assert stats["by_kind"]["event"] == 1
    # 越接近归档越靠前；永不遗忘的排在最后
    assert payload["nodes"][0]["content"] == "很久以前的一次吵架"


def test_snapshot_handler_survives_empty_database(sqlite_engine):
    payload = build_snapshot(sqlite_engine, now=NOW)
    assert payload["nodes"] == [] and payload["stats"]["total"] == 0


def test_writer_is_atomic_and_reader_explains_missing(tmp_path, sqlite_engine):
    seed(sqlite_engine)
    target = write_snapshot(sqlite_engine, tmp_path, now=NOW)
    assert target == tmp_path / "agent_monitor" / SNAPSHOT_FILENAME and target.is_file()
    assert not target.with_name(target.name + ".tmp").exists()
    read = MemoryMapReader(tmp_path / "agent_monitor").read()
    assert read["available"] is True and read["read_only"] is True
    assert len(read["nodes"]) == 4

    missing = MemoryMapReader(tmp_path / "elsewhere").read()
    assert missing["available"] is False and missing["reason"] == "snapshot_missing"


def test_reader_rejects_broken_or_foreign_snapshots(tmp_path):
    root = tmp_path / "agent_monitor"
    root.mkdir()
    (root / SNAPSHOT_FILENAME).write_text("{not json", encoding="utf-8")
    assert MemoryMapReader(root).read()["reason"] == "snapshot_unreadable"
    (root / SNAPSHOT_FILENAME).write_text(json.dumps({"version": 999, "nodes": []}), encoding="utf-8")
    assert MemoryMapReader(root).read()["reason"] == "snapshot_incompatible"
    (root / SNAPSHOT_FILENAME).write_text(json.dumps({"version": 1, "nodes": {}, "stats": {}}),
                                          encoding="utf-8")
    assert MemoryMapReader(root).read()["reason"] == "snapshot_incompatible"


@pytest.fixture
def served(tmp_path, sqlite_engine):
    seed(sqlite_engine)
    write_snapshot(sqlite_engine, tmp_path, now=NOW)
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


def test_monitor_serves_memory_page_and_api(served):
    with fetch(served, "/memory") as response:
        body = response.read().decode("utf-8")
        assert response.status == 200 and "记忆星图" in body
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    with fetch(served, "/memory.js") as response:
        assert b"/api/memory-map" in response.read()
    with fetch(served, "/api/memory-map") as response:
        payload = json.load(response)
    assert payload["available"] is True and payload["read_only"] is True
    assert payload["stats"]["total"] == 4
    with pytest.raises(urllib.error.HTTPError) as error:
        fetch(served, "/api/memory-map", method="POST")
    assert error.value.code == 405


def test_memory_map_feature_obeys_switch(sqlite_engine, tmp_path):
    from types import SimpleNamespace
    from app.agent.plugins.memory_map import MemoryMapFeature

    settings = SimpleNamespace(data_dir=tmp_path, memory_map_enabled=True,
                               memory_map_interval_seconds=300)
    feature = MemoryMapFeature()
    feature.attach(engine=sqlite_engine, settings=settings)
    workers = feature.create_workers()
    assert len(workers) == 1 and workers[0]._interval == 300.0

    settings.memory_map_enabled = False
    assert feature.create_workers() == ()
