"""自改 V1：大肥鱼起草代码修改提案，进程外 worker 跑测试，主人 apply 发布。

流程：draft_code_change（主人私聊/自己主动）→ data/selfmod/pending/<id>/
（meta.json + files/）→ WSL 侧 scripts/selfmod_worker.py 认领：整树拷贝快照 →
叠加提案文件 → 一次性容器跑全量测试 → results/<id>.json 回写 → 全绿后
主人 approve → worker 把文件落回真实仓库并触发 rebuild。

红线（worker 与本模块双重校验，谁都不许绕）：
* 群聊内容永远只能产生"建议文本"，永远到不了文件写入这一层
* 路径白名单 app/ tests/ configs/persona*.yaml；infra/、scripts/、data/、.env、.git 全禁
* 单提案 ≤6 个文件、新增 ≤300 行、单文件 ≤80KB
* 发布必须 meta.approved=true（approve_code_change 仅主人可设）且测试 passed
"""
from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from app.agent.tool_agent import AgentTool, ToolOutcome, validate_arguments

logger = logging.getLogger(__name__)

ALLOWED_PREFIXES = ("app/", "tests/", "configs/persona")
MAX_FILES = 6
MAX_ADDED_LINES = 300
MAX_FILE_BYTES = 80 * 1024
_BAD_PATH = re.compile(r"(^|/)\.\.(/|$)|^\.\.|//|\\\\|\x00")


def _validate_path(rel: str) -> str | None:
    """规范化并校验白名单；返回规范路径或 None。"""
    rel = str(rel or "").strip().replace("\\", "/").lstrip("/")
    if not rel or _BAD_PATH.search(rel):
        return None
    if not rel.startswith(ALLOWED_PREFIXES):
        return None
    return rel


def _syntax_check(rel: str, content: str) -> str | None:
    if rel.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as exc:
            return f"{rel}: 语法错误 line {exc.lineno}: {exc.msg}"
    return None


class SelfModStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "selfmod"
        self.pending = self.root / "pending"
        self.results = self.root / "results"
        self.backup = self.root / "backup"
        self.history = self.root / "history"

    # ---- proposal lifecycle ----

    def create_proposal(self, *, title: str, motivation: str,
                        files: list[dict], tests_note: str = "",
                        author: str = "owner") -> dict:
        """校验并落盘一个提案；返回 {id, warnings} 或抛 ValueError。"""
        title = str(title or "").strip()[:80]
        motivation = str(motivation or "").strip()[:2000]
        if not title or not motivation:
            raise ValueError("title 和 motivation 必填")
        if not isinstance(files, list) or not files:
            raise ValueError("files 不能为空")
        if len(files) > MAX_FILES:
            raise ValueError(f"单提案最多 {MAX_FILES} 个文件")
        proposal_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        warnings: list[str] = []
        added_lines = 0
        cleaned: list[tuple[str, str]] = []
        for item in files:
            rel = _validate_path(item.get("path"))
            if rel is None:
                raise ValueError(f"路径不在白名单或非法: {item.get('path')!r}（允许 {ALLOWED_PREFIXES}）")
            content = str(item.get("content") or "")
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError(f"{rel}: 单文件超 {MAX_FILE_BYTES} 字节")
            added_lines += sum(1 for _ in content.splitlines())
            error = _syntax_check(rel, content)
            if error:
                raise ValueError(error)
            cleaned.append((rel, content))
        if added_lines > MAX_ADDED_LINES:
            raise ValueError(f"单提案新增行数 {added_lines} 超预算 {MAX_ADDED_LINES}")
        target = self.pending / proposal_id
        files_dir = target / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in cleaned:
            out = files_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
        (target / "meta.json").write_text(json.dumps({
            "id": proposal_id, "title": title, "motivation": motivation,
            "tests_note": str(tests_note or "")[:1000], "author": author,
            "files": [rel for rel, _ in cleaned],
            "added_lines": added_lines, "status": "draft", "approved": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"id": proposal_id, "added_lines": added_lines, "warnings": warnings}

    def set_status(self, proposal_id: str, status: str) -> None:
        meta_path = self.pending / proposal_id / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = status
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def mark_approved(self, proposal_id: str) -> dict:
        meta_path = self.pending / proposal_id / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result = self.read_result(proposal_id)
        if not result or result.get("status") != "passed":
            raise ValueError("测试未通过（或还没跑完），不能发布")
        if meta.get("applied"):
            raise ValueError("该提案已应用")
        meta["approved"] = True
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return meta

    def read_result(self, proposal_id: str) -> dict | None:
        result_path = self.results / f"{proposal_id}.json"
        if not result_path.exists():
            return None
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_proposals(self, limit: int = 10) -> list[dict]:
        out = []
        if not self.pending.exists():
            return out
        for meta_path in sorted(self.pending.glob("*/meta.json"), reverse=True)[:limit]:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            result = self.read_result(meta["id"])
            out.append({"id": meta["id"], "title": meta["title"], "status": meta.get("status"),
                        "approved": bool(meta.get("approved")), "files": meta.get("files", []),
                        "test_status": (result or {}).get("status"),
                        "summary": (result or {}).get("summary", "")})
        return out

    def files_dir(self, proposal_id: str) -> Path:
        return self.pending / proposal_id / "files"

    def record_history(self, proposal_id: str, entry: dict) -> None:
        self.history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (self.history / f"{stamp}-{proposal_id}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")


def build_selfmod_tools(*, store: SelfModStore, actor_id: int, owner_qq: int,
                        runner) -> tuple[AgentTool, ...]:
    """主人私聊专属工具；群聊永不注册（防注入的第一道闸）。"""

    def require_owner() -> bool:
        return int(actor_id) == int(owner_qq) and int(owner_qq) > 0

    schema_files = {"type": "array", "minItems": 1, "maxItems": MAX_FILES, "items": {
        "type": "object", "properties": {"path": {"type": "string", "maxLength": 200},
                                         "content": {"type": "string", "maxLength": 90000}},
        "required": ["path", "content"], "additionalProperties": False}}

    async def draft(args):
        if not require_owner():
            return ToolOutcome({"error": "owner_only"}, "blocked")
        def write():
            try:
                return store.create_proposal(title=args.get("title"), motivation=args.get("motivation"),
                    files=args.get("files") or [], tests_note=args.get("tests_note") or "")
            except ValueError as exc:
                return {"error": "proposal_invalid", "detail": str(exc)[:200]}
        result = await runner.run(write)
        if result.get("error"):
            return ToolOutcome(result, "blocked")
        return ToolOutcome({"queued": True, **result,
            "guidance": "提案已进待测队列，WSL worker 会整树拷贝后跑全量测试（约15分钟）。"
                        "用 check_selfmod_status 查进度；全绿后向主人报告，等主人说'应用'再调 approve_code_change。"}, "ok")

    async def status(args):
        if not require_owner():
            return ToolOutcome({"error": "owner_only"}, "blocked")
        def read():
            pid = str(args.get("proposal_id") or "").strip()
            if pid:
                meta_path = store.pending / pid / "meta.json"
                if not meta_path.exists():
                    return {"error": "proposal_not_found"}
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return {"proposals": [{**meta, "result": store.read_result(pid)}]}
            return {"proposals": store.list_proposals()}
        return ToolOutcome(await runner.run(read), "ok")

    async def approve(args):
        if not require_owner():
            return ToolOutcome({"error": "owner_only"}, "blocked")
        def mark():
            try:
                return {"approved": True, **store.mark_approved(str(args.get("proposal_id") or ""))}
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                return {"error": "approve_rejected", "detail": str(exc)[:160]}
        result = await runner.run(mark)
        return ToolOutcome(result, "blocked" if result.get("error") else "ok")

    return (
        AgentTool("draft_code_change", "起草一个代码自改提案（仅限当前主人）：给出动机、改动文件（相对仓库根的完整路径+完整新内容）和测试说明。路径白名单 app/ tests/ configs/persona*.yaml；单提案≤6文件≤300新增行；.py 会做语法校验。提案不直接生效：worker 整树拷贝跑全量测试，全绿+主人批准后才发布", {
            "type": "object",
            "properties": {"title": {"type": "string", "maxLength": 80},
                           "motivation": {"type": "string", "maxLength": 2000},
                           "tests_note": {"type": "string", "maxLength": 1000},
                           "files": schema_files},
            "required": ["title", "motivation", "files"], "additionalProperties": False},
            draft, "write"),
        AgentTool("check_selfmod_status", "查自改提案状态：不传 id 列最近10个；传 id 看单个提案的测试结果与进度", {
            "type": "object", "properties": {"proposal_id": {"type": "string", "maxLength": 64}},
            "additionalProperties": False}, status, "read"),
        AgentTool("approve_code_change", "主人明确说'应用/发布'某提案后调用：校验测试已全绿并标记 approved，WSL worker 随后把文件落回真实仓库并触发 rebuild 重启。测试没过时会被拒绝", {
            "type": "object", "properties": {"proposal_id": {"type": "string", "maxLength": 64}},
            "required": ["proposal_id"], "additionalProperties": False}, approve, "write"),
    )
