"""自改 V1：提案库校验（白名单/预算/语法）、approve 门控、工具 owner 闸。"""
import json
from types import SimpleNamespace

import pytest

from app.core.selfmod import SelfModStore, build_selfmod_tools


@pytest.fixture
def store(tmp_path):
    return SelfModStore(tmp_path / "data")


def _files(*pairs):
    return [{"path": path, "content": content} for path, content in pairs]


def test_create_validates_allowlist(store):
    with pytest.raises(ValueError, match="白名单"):
        store.create_proposal(title="x", motivation="y",
                              files=_files(("infra/wsl/.env", "K=1")))
    with pytest.raises(ValueError, match="白名单"):
        store.create_proposal(title="x", motivation="y",
                              files=_files(("app/../scripts/evil.py", "pass")))
    with pytest.raises(ValueError, match="白名单"):
        store.create_proposal(title="x", motivation="y",
                              files=_files(("scripts/run_tests.sh", "rm -rf /")))


def test_create_validates_syntax_and_budget(store):
    with pytest.raises(ValueError, match="语法错误"):
        store.create_proposal(title="x", motivation="y",
                              files=_files(("app/a.py", "def broken(:\n")))
    big = "\n".join(["x = 1"] * 400)
    with pytest.raises(ValueError, match="超预算"):
        store.create_proposal(title="x", motivation="y",
                              files=_files(("app/a.py", big)))


def test_create_writes_proposal_and_status_flow(store):
    result = store.create_proposal(title="修正则", motivation="关键词漏匹配",
        tests_note="补一条回归", files=_files(
            ("app/core/group_mood.py", "STICKER_WAR_HINT = 'x'\n"),
            ("tests/core/test_group_mood.py", "def test_x():\n    assert True\n")))
    pid = result["id"]
    meta = json.loads((store.pending / pid / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "draft" and meta["approved"] is False
    assert len(meta["files"]) == 2
    assert (store.files_dir(pid) / "app/core/group_mood.py").exists()

    # 未测试不能 approve
    with pytest.raises(ValueError, match="测试未通过"):
        store.mark_approved(pid)
    store.results.mkdir(parents=True, exist_ok=True)
    (store.results / f"{pid}.json").write_text(json.dumps(
        {"status": "failed", "summary": "红"}), encoding="utf-8")
    with pytest.raises(ValueError, match="测试未通过"):
        store.mark_approved(pid)
    (store.results / f"{pid}.json").write_text(json.dumps(
        {"status": "passed", "summary": "绿"}), encoding="utf-8")
    meta = store.mark_approved(pid)
    assert meta["approved"] is True
    # 重复 approve 幂等（applied 标记由 worker 落库后才置位）
    meta = store.mark_approved(pid)
    assert meta["approved"] is True


def test_list_and_result(store):
    created = store.create_proposal(title="t", motivation="m",
                                    files=_files(("app/a.py", "x = 1\n")))
    listing = store.list_proposals()
    assert listing[0]["id"] == created["id"] and listing[0]["test_status"] is None


class _Runner:
    async def run(self, fn, *args):
        return fn(*args)


class _Guard:
    def require_valid(self):
        return None


def _tools(tmp_path, actor, owner):
    store = SelfModStore(tmp_path / "data")
    return store, build_selfmod_tools(store=store, actor_id=actor, owner_qq=owner,
                                      runner=_Runner())


async def test_owner_gate_blocks_non_owner(tmp_path):
    store, tools = _tools(tmp_path, actor=111, owner=100)
    draft = tools[0]
    result = await draft.handler({"title": "t", "motivation": "m",
        "files": _files(("app/a.py", "x = 1\n"))})
    assert result.status == "blocked" and result.data["error"] == "owner_only"
    approve = tools[2]
    result = await approve.handler({"proposal_id": "whatever"})
    assert result.status == "blocked" and result.data["error"] == "owner_only"


async def test_draft_and_approve_happy_path(tmp_path):
    store, tools = _tools(tmp_path, actor=100, owner=100)
    draft, status, approve = tools
    result = await draft.handler({"title": "改提示词", "motivation": "斗图不接图",
        "tests_note": "", "files": _files(("configs/persona.yaml", "name: 鲸鱼娘\n"))})
    assert result.status == "ok" and result.data["queued"] is True
    pid = result.data["id"]
    store.results.mkdir(parents=True, exist_ok=True)
    (store.results / f"{pid}.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    ok = await approve.handler({"proposal_id": pid})
    assert ok.status == "ok" and ok.data["approved"] is True
    listed = await status.handler({})
    assert listed.data["proposals"][0]["test_status"] == "passed"


async def test_draft_rejects_bad_file(tmp_path):
    store, tools = _tools(tmp_path, actor=100, owner=100)
    result = await tools[0].handler({"title": "t", "motivation": "m",
        "files": _files(("app/a.py", "def broken(:\n"))})
    assert result.status == "blocked" and result.data["error"] == "proposal_invalid"
