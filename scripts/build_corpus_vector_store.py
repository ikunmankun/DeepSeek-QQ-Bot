"""从 PersonalDialog 语料构建真人接法向量库（corpus_style_examples + vec0）。

流式扫描 datasets/dialogues_train.jsonl，按质量门筛"回句 + 上句情境"对，
嵌入（本地 bge-small-zh，L2 归一化）后双写：
* corpus_style_examples（元数据，启动自建）
* corpus_style_vec（vec0 虚拟表，SQL 侧 KNN；重建=--rebuild drop+重导）

选择口径：回句 3-40 字、上句 5-160 字做情境、无 URL/@、表情 ≤1、精确去重、
同首二字多样性封顶（防饭圈刷屏占满库）。kind=1 额外标女性声子集（私聊甜心
面检索用），kind=0 是全量池。目标行数默认 40000。

在 bot 容器内运行（fastembed 与模型缓存都在那里）：
    docker exec -e PYTHONPATH=/workspace xiaomachi-bot python \
        /workspace/scripts/build_corpus_vector_store.py [--rebuild] [--max-rows 40000]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "datasets" / "dialogues_train.jsonl"
sys.path.insert(0, str(REPO_ROOT))
# python -I 会移除脚本目录，同目录借用 analyze 脚本的纯函数要显式补 path。
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 纯函数从 analyze 脚本借（同目录、纯 stdlib）——collect 阶段宿主机可能没有
# pydantic/sqlalchemy，app.* 的导入全部留到嵌入路径里惰性进行。
from analyze_personal_dialog import detokenize  # noqa: E402

REPLY_MIN_CHARS, REPLY_MAX_CHARS = 3, 40
CONTEXT_MIN_CHARS, CONTEXT_MAX_CHARS = 5, 160
PER_PREFIX_CAP_DEFAULT = 60
_EMOTICON = re.compile(r"\[[^\[\]]{1,10}\]")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_MENTION = re.compile(r"@[^\s@]{1,20}")


def is_quality_reply(text: str) -> bool:
    if not (REPLY_MIN_CHARS <= len(text) <= REPLY_MAX_CHARS):
        return False
    if _URL.search(text) or _MENTION.search(text):
        return False
    return len(_EMOTICON.findall(text)) <= 1


def is_quality_context(text: str) -> bool:
    return CONTEXT_MIN_CHARS <= len(text) <= CONTEXT_MAX_CHARS


def collect_rows(max_rows: int, per_prefix_cap: int, max_lines: int) -> list[dict]:
    """流式扫描语料，产出通过质量门的 (回句, 情境) 行。"""

    rows: list[dict] = []
    seen_texts: set[str] = set()
    prefix_counter: Counter[str] = Counter()
    started = time.time()
    processed = 0
    with DATASET.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_lines and line_no > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            dialog = item.get("dialog")
            profiles = item.get("profile") or []
            uids = item.get("uid") or []
            if not isinstance(dialog, list) or len(dialog) < 2:
                continue
            gender_by_uid = {}
            for index, profile in enumerate(profiles):
                if isinstance(profile, dict):
                    gender_by_uid[index] = str(profile.get("gender") or "").strip().lower()
            texts: list[str] = []
            genders: list[str] = []
            for turn_index, turn in enumerate(dialog):
                if not isinstance(turn, list) or not turn:
                    continue
                text = detokenize("".join(str(t) for t in turn))
                if not text:
                    continue
                texts.append(text)
                uid = uids[turn_index] if turn_index < len(uids) and isinstance(uids[turn_index], int) else 0
                genders.append(gender_by_uid.get(uid, ""))
            for index in range(1, len(texts)):
                reply, context = texts[index], texts[index - 1]
                if not is_quality_reply(reply) or not is_quality_context(context):
                    continue
                if reply in seen_texts:
                    continue
                prefix = reply[:2]
                if prefix_counter[prefix] >= per_prefix_cap:
                    continue
                gender = genders[index] if index < len(genders) else ""
                seen_texts.add(reply)
                prefix_counter[prefix] += 1
                rows.append({
                    "text": reply,
                    "context_before": [context],
                    "kind": 1 if gender == "female" else 0,
                    "gender": gender,
                    "prev_len": len(context),
                })
                if len(rows) >= max_rows:
                    print(f"row target reached at line {line_no}", file=sys.stderr)
                    return rows
            processed += 1
            if processed % 500_000 == 0:
                elapsed = time.time() - started
                print(f"scanned {processed} dialogs, rows={len(rows)} ({elapsed:.0f}s)",
                      file=sys.stderr)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="先 drop vec0 表与元数据表再重建")
    parser.add_argument("--max-rows", type=int, default=40_000)
    parser.add_argument("--per-prefix-cap", type=int, default=PER_PREFIX_CAP_DEFAULT)
    parser.add_argument("--max-lines", type=int, default=0, help="最多扫描行数（0=全量）")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--collect-only", metavar="OUT",
                        help="只筛样本不嵌入（宿主机跑，1.8GB 在宿主机）：写 jsonl 后退出")
    parser.add_argument("--from-samples", metavar="PATH",
                        help="跳过扫描，从样本 jsonl 嵌入入库（容器内跑，数据在宿主机时用这条链路）")
    args = parser.parse_args()

    if args.collect_only:
        if not DATASET.exists():
            print(f"dataset not found: {DATASET}", file=sys.stderr)
            return 1
        rows = collect_rows(args.max_rows, args.per_prefix_cap, args.max_lines)
        out = Path(args.collect_only)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        kinds = Counter(row["kind"] for row in rows)
        print(f"collected rows={len(rows)} kinds={dict(kinds)} -> {out}", file=sys.stderr)
        return 0

    from app.config import AppSettings
    from app.storage.db import build_engine

    settings = AppSettings()
    engine = build_engine(settings.sqlite_path)

    from app.core.fuzzy_recall import get_shared_embedding_provider
    from app.core.style_corpus import (
        ensure_corpus_vec_table,
        insert_corpus_rows,
        retrieve_similar,
    )

    provider = get_shared_embedding_provider(settings)
    if provider is None or not getattr(provider, "available", False):
        print("embedding provider unavailable", file=sys.stderr)
        return 1
    dimensions = int(getattr(provider.identity, "dimensions", 0) or 0)
    print(f"embedding provider ready: {provider.identity}", file=sys.stderr)

    if args.rebuild:
        from sqlalchemy import text

        with engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS corpus_style_vec"))
            connection.execute(text("DROP TABLE IF EXISTS corpus_style_examples"))
        print("rebuild: old tables dropped", file=sys.stderr)
    # 元数据表由 Base.create_all 维护，但脚本路径可能早于启动——手动兜底建表。
    from app.storage.models import Base, CorpusStyleExample  # noqa: E402

    Base.metadata.create_all(engine, tables=[CorpusStyleExample.__table__])
    if not ensure_corpus_vec_table(engine, dimensions):
        print("vec0 table unavailable", file=sys.stderr)
        return 1

    if args.from_samples:
        samples_path = Path(args.from_samples)
        if not samples_path.exists():
            print(f"samples not found: {samples_path}", file=sys.stderr)
            return 1
        rows = []
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"loaded samples rows={len(rows)}", file=sys.stderr)
    else:
        rows = collect_rows(args.max_rows, args.per_prefix_cap, args.max_lines)
    kind_counts = Counter(row["kind"] for row in rows)
    print(f"rows={len(rows)} kinds={dict(kind_counts)}", file=sys.stderr)
    if not rows:
        return 1
    started = time.time()
    inserted = insert_corpus_rows(engine, provider, rows, batch_size=args.batch_size)
    print(f"inserted={inserted} in {time.time() - started:.0f}s", file=sys.stderr)
    if inserted <= 0:
        return 1
    # 简单探针：确认 KNN 通路
    for probe in ("在干嘛", "哈哈哈", "晚安"):
        hits = retrieve_similar(engine, settings, probe, kind=0, limit=2)
        top = hits[0]["similarity"] if hits else "-"
        print(f"probe {probe!r}: hits={len(hits)} top_sim={top}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
