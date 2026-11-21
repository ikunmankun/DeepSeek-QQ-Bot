"""自改 V1 的 WSL 侧 worker（root 运行）：认领提案 → 整树快照叠加 → 一次性容器跑
全量测试 → （主人已批准且全绿）落回真实仓库并触发 rebuild。

启动（幂等，重复启动会因单实例锁退出）：
  wsl.exe --user root -e bash -c "nohup python3 /mnt/e/QQ-Bot/scripts/selfmod_worker.py >> /var/lib/docker/volumes/xiaomachi-bot-data/_data/selfmod/worker.log 2>&1 &"

rebuild-xiaomachi-wsl.bat 末尾会自动拉起本 worker。
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/mnt/e/QQ-Bot")
# 注意：compose 的项目名是 xiaomachi-bot，卷 xiaomachi_data 的真实目录是
# /var/lib/docker/volumes/xiaomachi-bot-data。2026-11 前这里写的是
# xiaomachi-data（不存在），worker 因此一直起不来、也看不到提案。
DATA = Path("/var/lib/docker/volumes/xiaomachi-bot-data/_data")
SELFMOD = DATA / "selfmod"
PENDING = SELFMOD / "pending"
RESULTS = SELFMOD / "results"
BACKUP = SELFMOD / "backup"
HISTORY = SELFMOD / "history"
LOCK = SELFMOD / "worker.lock"
IMAGE = "xiaomachi-bot:local"
SNAPSHOT_ITEMS = ("app", "configs", "scripts", "tests", "pyproject.toml")
TEST_CMD = ("cp -r /src/app /src/configs /src/scripts /src/tests /src/pyproject.toml /work/ && "
            "cd /work && python -m pytest tests/agent tests/core tests/adapters -q -p no:cacheprovider "
            "-o asyncio_mode=auto")
APPLY_ALLOWLIST = ("app/", "tests/", "configs/persona")
TEST_TIMEOUT = 2700
REBUILD_TIMEOUT = 900


def acquire_lock() -> bool:
    SELFMOD.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or 0)
            os.kill(pid, 0)
            return False  # 还活着，退出
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # 死锁文件，接管
    LOCK.write_text(str(os.getpid()))
    return True


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_meta(proposal_id: str) -> dict | None:
    path = PENDING / proposal_id / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_meta(proposal_id: str, meta: dict) -> None:
    (PENDING / proposal_id / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def proposal_files(proposal_id: str) -> list[tuple[str, Path]]:
    root = PENDING / proposal_id / "files"
    out = []
    for file in sorted(root.rglob("*")):
        if file.is_file():
            rel = file.relative_to(root).as_posix()
            if rel.startswith(APPLY_ALLOWLIST):
                out.append((rel, file))
    return out


def snapshot_tree(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in SNAPSHOT_ITEMS:
        src = REPO / item
        if src.is_dir():
            shutil.copytree(src, dst / item, ignore=shutil.ignore_patterns(
                "__pycache__", ".pytest_cache", "*.pyc", "test-work"))
        elif src.exists():
            shutil.copy2(src, dst / item)


def run_tests(snapshot: Path) -> tuple[str, str]:
    cmd = ["docker", "run", "--rm", "-v", f"{snapshot}:/src:ro", IMAGE, "sh", "-c", TEST_CMD]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        return "timeout", str(exc)[-800:]
    tail = "\n".join((run.stdout or "").splitlines()[-40:])
    if run.returncode == 0:
        return "passed", tail
    return "failed", tail or (run.stderr or "")[-800:]


def apply_proposal(proposal_id: str, files: list[tuple[str, Path]]) -> list[str]:
    """落回真实仓库；被覆盖的原文件先进 BACKUP/<id>/ 以便回滚。"""
    overwritten = []
    for rel, src in files:
        target = REPO / rel
        if target.exists():
            backup = BACKUP / proposal_id / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        overwritten.append(rel)
    return overwritten


def trigger_rebuild() -> tuple[str, str]:
    cmd_exe = Path("/mnt/c/Windows/System32/cmd.exe")
    if not cmd_exe.exists():
        return "failed", "cmd.exe not found"
    try:
        run = subprocess.run([str(cmd_exe), "/c", "E:\\QQ-Bot\\rebuild-xiaomachi-wsl.bat"],
                             capture_output=True, text=True, timeout=REBUILD_TIMEOUT)
        tail = "\n".join((run.stdout or "").splitlines()[-15:])
        ok = "Rebuild complete" in (run.stdout or "")
        return ("passed" if ok else "failed"), tail
    except subprocess.TimeoutExpired:
        return "timeout", "rebuild timeout"


def process_one(proposal_id: str) -> None:
    meta = read_meta(proposal_id)
    if meta is None or meta.get("status") != "draft":
        return
    log(f"claim {proposal_id} title={meta.get('title')}")
    meta["status"] = "testing"
    write_meta(proposal_id, meta)
    work = Path("/tmp") / f"selfmod-{proposal_id}"
    try:
        snapshot_tree(work)
        files = proposal_files(proposal_id)
        if not files:
            raise ValueError("提案没有任何白名单内文件")
        for rel, src in files:
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        status, tail = run_tests(work)
        summary = "全量测试通过" if status == "passed" else f"测试未通过（{status}）"
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{proposal_id}.json").write_text(json.dumps({
            "status": status, "summary": summary, "tail": tail,
            "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        meta = read_meta(proposal_id) or meta
        meta["status"] = status
        write_meta(proposal_id, meta)
        log(f"tests {proposal_id}: {status}")
    except Exception as exc:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{proposal_id}.json").write_text(json.dumps({
            "status": "error", "summary": f"worker 异常: {type(exc).__name__}",
            "tail": str(exc)[-800:]}, ensure_ascii=False, indent=1), encoding="utf-8")
        meta = read_meta(proposal_id) or meta
        meta["status"] = "error"
        write_meta(proposal_id, meta)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def apply_one(proposal_id: str) -> None:
    meta = read_meta(proposal_id)
    if meta is None or meta.get("status") != "passed" or not meta.get("approved") or meta.get("applied"):
        return
    log(f"apply {proposal_id}")
    files = proposal_files(proposal_id)
    overwritten = apply_proposal(proposal_id, files)
    rebuild_status, rebuild_tail = trigger_rebuild()
    meta["applied"] = True
    meta["applied_files"] = overwritten
    meta["rebuild_status"] = rebuild_status
    write_meta(proposal_id, meta)
    HISTORY.mkdir(parents=True, exist_ok=True)
    (HISTORY / f"{time.strftime('%Y%m%d-%H%M%S')}-{proposal_id}.json").write_text(json.dumps({
        "id": proposal_id, "title": meta.get("title"), "files": overwritten,
        "rebuild_status": rebuild_status, "rebuild_tail": rebuild_tail,
        "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"applied {proposal_id}: {overwritten} rebuild={rebuild_status}")


def main() -> None:
    if not acquire_lock():
        log("another worker is running; exit")
        return
    log("selfmod worker started")
    while True:
        try:
            if PENDING.exists():
                for meta_path in sorted(PENDING.glob("*/meta.json")):
                    pid = meta_path.parent.name
                    meta = read_meta(pid)
                    if meta and meta.get("status") == "draft":
                        process_one(pid)
                    elif meta and meta.get("status") == "passed" and meta.get("approved") and not meta.get("applied"):
                        apply_one(pid)
        except Exception as exc:
            log(f"loop error: {type(exc).__name__}: {str(exc)[:300]}")
        time.sleep(60)


if __name__ == "__main__":
    main()
