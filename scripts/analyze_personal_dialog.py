"""PersonalDialog 语料的真人网聊风格统计画像（流式全量扫描）。

数据集：腾讯 AI Lab PersonalDialog（datasets/dialogues_train.jsonl，微博衍生，
带说话人画像 tag/loc/gender）。设计要点：
* 逐行 json.loads 流式扫描，只保留聚合值与有界 Counter，不落任何原始语料；
* 语料文本是分词带空格的，先按规则还原：CJK 相邻直接拼接、ASCII 词间留
  一个空格、标点紧贴（"没有 钱 万万 不行 ！ ~" → "没有钱万万不行！~"）；
* 长度/长度比用固定种子蓄水池采样（200k 样本）算分位数，内存有界；
* 起手式/收尾式/短句 Counter 超过上限自动剪枝保 top，防止百万级 distinct 爆内存。

用法：
    python scripts/analyze_personal_dialog.py [--max-lines N] [--sample-rate K]
产出：
    .analysis/personal_dialog_stats.md    人读报告
    .analysis/personal_dialog_stats.json  聚合原始值
    .analysis/personal_dialog_samples.jsonl  蒸馏用高质量对话样本（复用）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "datasets" / "dialogues_train.jsonl"
OUT_DIR = REPO_ROOT / ".analysis"

RESERVOIR_SIZE = 200_000
COUNTER_CAP = 120_000
SEED = 42

_EMOTICON = re.compile(r"\[[^\[\]]{1,10}\]")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_MENTION = re.compile(r"@[^\s@]{1,20}")

#: 句尾语气词清单（对照 persona 的"语气词尾巴"与"一条消息顶多一个语气词"规则）。
PARTICLES = ("吧", "呀", "哦", "噢", "喔", "啦", "哈", "呢", "嘛", "呗", "咯",
             "哟", "唷", "诶", "欸", "噻", "嘞", "捏", "嗷")

#: 起笑前缀（口径与 app/core/style_distill.compute_style_stats 对齐并扩充）。
LAUGH_PREFIXES = ("哈哈", "笑死", "草", "绷", "难绷", "乐", "寄", "艹",
                  "??", "？？")

#: 无内容追问式起手（对照反八股"在吗/干嘛/咋了式没内容的开场收尾"）。
CONTENTLESS_OPENERS = ("在吗", "在嘛", "干嘛", "咋了", "怎么样", "说吧", "然后呢",
                       "真的假的", "为什么", "为啥", "什么意思")

#: 例句样本筛选：太长像段落、含链接/提及的不要（蒸馏语料要干净短句）。
SAMPLE_MIN_CHARS = 4
SAMPLE_MAX_CHARS = 60
SAMPLE_TARGET = 4000


def detokenize(text: str) -> str:
    """还原分词空格：ASCII 词间留一个空格，CJK/标点直接拼接。"""

    tokens = text.split()
    if not tokens:
        return ""
    parts = [tokens[0]]
    for token in tokens[1:]:
        prev = parts[-1]
        if prev and _is_ascii_alnum(prev[-1]) and token and _is_ascii_alnum(token[0]):
            parts.append(" " + token)
        else:
            parts.append(token)
    return "".join(parts)


def _is_ascii_alnum(char: str) -> bool:
    return char.isascii() and char.isalnum()


class Reservoir:
    """固定种子蓄水池采样：流式数据上算分位数。"""

    def __init__(self, size: int, seed: int = SEED) -> None:
        self.size = size
        self.rng = random.Random(seed)
        self.items: list[float] = []
        self.seen = 0

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(value)
            return
        slot = self.rng.randint(1, self.seen)
        if slot <= self.size:
            self.items[slot - 1] = value

    def percentiles(self, ratios: tuple[float, ...]) -> dict[str, float]:
        values = sorted(self.items)
        result: dict[str, float] = {}
        for ratio in ratios:
            key = f"p{int(ratio * 100)}"
            if not values:
                result[key] = 0.0
                continue
            index = min(len(values) - 1, int(len(values) * ratio))
            result[key] = round(values[index], 2)
        return result


class BoundedCounter:
    """超过 cap 自动剪到 top 半数的计数器（防百万级 distinct 撑爆内存）。"""

    def __init__(self, cap: int = COUNTER_CAP) -> None:
        self.cap = cap
        self.counter: Counter[str] = Counter()
        self.pruned_total = 0

    def add(self, key: str) -> None:
        self.counter[key] += 1
        if len(self.counter) > self.cap:
            keep = dict(self.counter.most_common(self.cap // 2))
            dropped = sum(self.counter.values()) - sum(keep.values())
            self.pruned_total += dropped
            self.counter = Counter(keep)

    def top(self, n: int) -> list[tuple[str, int]]:
        return self.counter.most_common(n)

    def __len__(self) -> int:
        return len(self.counter)


class Stats:
    """聚合容器：一次扫描累积全部统计维度。"""

    def __init__(self) -> None:
        self.dialogs = 0
        self.utterances = 0
        self.turns_counter = Counter()
        self.speaker_counter = Counter()
        self.lengths = Reservoir(RESERVOIR_SIZE)
        self.reply_ratios = Reservoir(RESERVOIR_SIZE)
        self.short_reply_count = 0
        self.long_prev_count = 0
        self.particle_end = Counter()
        self.particle_any = Counter()
        self.stacked_particle = 0          # 一条里 ≥2 个语气词
        self.punct_end = Counter()
        self.double_punct = 0              # ？？/！！ 连用
        self.bracket_tail = 0              # 句尾（…）备注
        self.starters = BoundedCounter()
        self.closers = BoundedCounter()
        self.short_utterances = BoundedCounter()
        self.laugh_start = 0
        self.contentless_open = 0
        self.emoticon_count = 0
        self.url_count = 0
        self.mention_count = 0
        self.female_lengths = Reservoir(RESERVOIR_SIZE // 2)
        self.female_particle_end = Counter()
        self.female_stacked = 0
        self.female_utterances = 0
        self.tag_counter = BoundedCounter()
        # 蒸馏样本：质量过滤后的「上文 → 回句」对
        self.samples: list[dict] = []
        self.sample_pool: list[dict] = []
        self.sample_rng = random.Random(SEED)

    # --- 单句级统计 -------------------------------------------------------
    def feed_utterance(self, text: str, gender: str) -> None:
        self.utterances += 1
        length = len(text)
        self.lengths.add(length)

        stripped = text.rstrip("。.！!？?～~… ")
        last = stripped[-1:] if stripped else ""
        if last in PARTICLES:
            self.particle_end[last] += 1
            if gender == "female":
                self.female_particle_end[last] += 1
        found = [p for p in PARTICLES if p in text]
        if len(found) >= 2:
            self.stacked_particle += 1
            if gender == "female":
                self.female_stacked += 1
        for p in set(found):
            self.particle_any[p] += 1

        if text.endswith(("？？", "！！")):
            self.double_punct += 1
        if text.endswith(("。", ".", "！?", "!?", "?！", "！")) and text.endswith(("！", "!")):
            self.punct_end["！"] += 1
        elif text.endswith(("？", "?")):
            self.punct_end["？"] += 1
        elif text.endswith(("~", "～")):
            self.punct_end["~"] += 1
        elif text.endswith(("…", "。。", "。。。")):
            self.punct_end["…"] += 1
        elif text.endswith(("）", ")")) and ("（" in text or "(" in text):
            self.bracket_tail += 1
            self.punct_end["（备注）"] += 1
        elif text.endswith(("。", ".")):
            self.punct_end["。"] += 1

        if text.startswith(LAUGH_PREFIXES):
            self.laugh_start += 1
        if any(text.startswith(op) for op in CONTENTLESS_OPENERS) and length <= 12:
            self.contentless_open += 1

        if _EMOTICON.search(text):
            self.emoticon_count += 1
        if _URL.search(text):
            self.url_count += 1
        if _MENTION.search(text):
            self.mention_count += 1

        # 起手式（≤20 字句取首段）、收尾式（≤15 字句取尾段）
        if 2 <= length <= 20:
            head = re.split(r"[，,。.！!？?~～… ]", text, maxsplit=1)[0]
            if head:
                self.starters.add(head)
        if 2 <= length <= 15:
            tail = re.split(r"[，,。.！!？?~～…]", text)[-1]
            tail = tail.strip() or text[-2:]
            self.closers.add(tail[-4:])

        if length <= 8:
            self.short_utterances.add(text)

        if gender == "female":
            self.female_lengths.add(length)
            self.female_utterances += 1

    # --- 对话级统计 -------------------------------------------------------
    def feed_dialog(self, utterances: list[str], genders: list[str]) -> None:
        self.dialogs += 1
        self.turns_counter[min(len(utterances), 12)] += 1
        for utt, gender in zip(utterances, genders):
            self.speaker_counter[gender] += 1
            self.feed_utterance(utt, gender)
        # 回复长度比：上一句 ≥10 字时，回句长度比 ≤0.3 记"短回"
        for i in range(1, len(utterances)):
            prev, reply = utterances[i - 1], utterances[i]
            if len(prev) >= 10:
                self.long_prev_count += 1
                ratio = len(reply) / max(1, len(prev))
                self.reply_ratios.add(round(ratio, 3))
                if ratio <= 0.3:
                    self.short_reply_count += 1

    # --- 蒸馏样本 ---------------------------------------------------------
    def maybe_sample(self, utterances: list[str], tags: list[str], gender: str) -> None:
        """质量过滤后的对话进样本池，最后蓄水池到 SAMPLE_TARGET。"""
        if not (2 <= len(utterances) <= 8):
            return
        if any(len(u) < SAMPLE_MIN_CHARS or len(u) > SAMPLE_MAX_CHARS for u in utterances):
            return
        joined = "".join(utterances)
        if _URL.search(joined) or _MENTION.search(joined):
            return
        emoticon_hits = len(_EMOTICON.findall(joined))
        if emoticon_hits > 1 or emoticon_hits and len(joined) < 20:
            return
        record = {
            "dialog": utterances,
            "gender": gender,
            "tags": tags[:3],
        }
        if len(self.sample_pool) < SAMPLE_TARGET * 4:
            self.sample_pool.append(record)
        else:
            slot = self.sample_rng.randint(1, len(self.sample_pool))
            if slot <= SAMPLE_TARGET * 4:
                self.sample_pool[slot - 1] = record

    def finalize_samples(self) -> None:
        self.sample_rng.shuffle(self.sample_pool)
        self.samples = self.sample_pool[:SAMPLE_TARGET]


def percentile_note(reservoir: Reservoir) -> dict:
    return reservoir.percentiles((0.1, 0.25, 0.5, 0.75, 0.9, 0.99))


def pct(part: int, total: int) -> str:
    return f"{part / total * 100:.2f}%" if total else "0%"


def per_kilo(part: int, total: int) -> str:
    return f"{part / total * 1000:.1f}" if total else "0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lines", type=int, default=0, help="最多扫描的对话行数（0=全量）")
    parser.add_argument("--sample-rate", type=int, default=1, help="每 K 行处理一行（1=全量）")
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"dataset not found: {DATASET}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(exist_ok=True)

    stats = Stats()
    started = time.time()
    processed = 0
    with DATASET.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if args.max_lines and line_no > args.max_lines:
                break
            if args.sample_rate > 1 and line_no % args.sample_rate:
                continue
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
            if not isinstance(dialog, list) or not dialog:
                continue
            gender_by_uid = {}
            tag_by_uid = {}
            for index, profile in enumerate(profiles):
                if isinstance(profile, dict):
                    gender_by_uid[index] = str(profile.get("gender") or "").strip().lower()
                    raw_tags = profile.get("tag") or []
                    if isinstance(raw_tags, str):
                        raw_tags = raw_tags.split(";")
                    tag_by_uid[index] = [t for t in (str(x).strip() for x in raw_tags) if t]

            utterances: list[str] = []
            genders: list[str] = []
            speaker_tags: list[str] = []
            dominant_gender = ""
            for turn_index, turn in enumerate(dialog):
                if not isinstance(turn, list) or not turn:
                    continue
                raw = "".join(str(t) for t in turn)
                text = detokenize(raw)
                if not text:
                    continue
                uid = uids[turn_index] if turn_index < len(uids) and isinstance(uids[turn_index], int) else 0
                gender = gender_by_uid.get(uid, "")
                utterances.append(text)
                genders.append(gender)
                if gender == "female":
                    dominant_gender = "female"
                speaker_tags.extend(tag_by_uid.get(uid, []))
            if not utterances:
                continue
            stats.feed_dialog(utterances, genders)
            stats.maybe_sample(utterances, speaker_tags, dominant_gender)
            for tag in speaker_tags:
                stats.tag_counter.add(tag)
            processed += 1
            if processed % 200_000 == 0:
                elapsed = time.time() - started
                print(f"processed {processed} dialogs ({elapsed:.0f}s, "
                      f"{processed / max(1, elapsed):.0f}/s)", file=sys.stderr)

    stats.finalize_samples()
    elapsed = time.time() - started
    print(f"done: {processed} dialogs in {elapsed:.0f}s", file=sys.stderr)
    write_outputs(stats, processed)
    return 0


def write_outputs(stats: Stats, processed: int) -> None:
    total = stats.utterances
    length_stats = percentile_note(stats.lengths)
    female_length_stats = percentile_note(stats.female_lengths)
    ratio_stats = percentile_note(stats.reply_ratios)

    agg = {
        "dialogs": stats.dialogs,
        "processed_dialogs": processed,
        "utterances": total,
        "lengths": length_stats,
        "avg_len": round(sum(stats.lengths.items) / max(1, len(stats.lengths.items)), 2),
        "turns_distribution": dict(stats.turns_counter),
        "gender_of_speakers": dict(stats.speaker_counter),
        "particle_end": dict(stats.particle_end),
        "particle_any": dict(stats.particle_any.most_common()),
        "particle_end_per_kilo": {k: per_kilo(v, total) for k, v in stats.particle_end.items()},
        "stacked_particle_rate": pct(stats.stacked_particle, total),
        "female": {
            "utterances": stats.female_utterances,
            "lengths": female_length_stats,
            "particle_end": dict(stats.female_particle_end),
            "particle_end_per_kilo": {k: per_kilo(v, stats.female_utterances)
                                      for k, v in stats.female_particle_end.items()},
            "stacked_particle_rate": pct(stats.female_stacked, stats.female_utterances),
        },
        "punct_end": {k: pct(v, total) for k, v in stats.punct_end.items()},
        "double_punct_rate": pct(stats.double_punct, total),
        "bracket_tail_rate": pct(stats.bracket_tail, total),
        "laugh_start_rate": pct(stats.laugh_start, total),
        "contentless_open_rate": pct(stats.contentless_open, total),
        "emoticon_rate": pct(stats.emoticon_count, total),
        "url_rate": pct(stats.url_count, total),
        "mention_rate": pct(stats.mention_count, total),
        "reply_ratio": ratio_stats,
        "short_reply_rate": pct(stats.short_reply_count, max(1, stats.long_prev_count)),
        "starters_top50": stats.starters.top(50),
        "closers_top50": stats.closers.top(50),
        "short_utterances_top200": stats.short_utterances.top(200),
        "tags_top30": stats.tag_counter.top(30),
        "samples_written": len(stats.samples),
    }
    (OUT_DIR / "personal_dialog_stats.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT_DIR / "personal_dialog_samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in stats.samples:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- Markdown 报告 ---------------------------------------------------
    lines: list[str] = []
    lines.append("# PersonalDialog 真人网聊风格统计画像")
    lines.append("")
    lines.append(f"- 扫描对话数：{stats.dialogs:,}（处理 {processed:,}）")
    lines.append(f"- 句子总数：{total:,}")
    lines.append(f"- 平均句长：{agg['avg_len']} 字")
    lines.append(f"- 句长分布：{length_stats}")
    lines.append(f"- 女性声句子：{stats.female_utterances:,}（{pct(stats.female_utterances, total)}），"
                 f"句长分布：{female_length_stats}")
    lines.append(f"- 回复长度比（上一句≥10字）：{ratio_stats}，短回率（≤0.3）{agg['short_reply_rate']}")
    lines.append(f"- 语气词堆叠率（一条≥2个）：全量 {agg['stacked_particle_rate']}，"
                 f"女性 {agg['female']['stacked_particle_rate']}")
    lines.append(f"- 起笑前缀：{agg['laugh_start_rate']}，无内容追问式开场：{agg['contentless_open_rate']}")
    lines.append(f"- [表情] 占位符：{agg['emoticon_rate']}，URL：{agg['url_rate']}，@提及：{agg['mention_rate']}")
    lines.append(f"- 双标点连用（？？/！！）：{agg['double_punct_rate']}，句尾（备注）：{agg['bracket_tail_rate']}")
    lines.append("")
    lines.append("## 句尾语气词（每千句次数）")
    lines.append("")
    lines.append("| 语气词 | 全量 | 女性声 |")
    lines.append("|---|---|---|")
    for particle in PARTICLES:
        lines.append(f"| {particle} | {per_kilo(stats.particle_end.get(particle, 0), total)} "
                     f"| {per_kilo(stats.female_particle_end.get(particle, 0), stats.female_utterances)} |")
    lines.append("")
    lines.append("## 句尾标点分布（占全部句子）")
    lines.append("")
    for key, value in sorted(agg["punct_end"].items(), key=lambda kv: -float(kv[1][:-1])):
        lines.append(f"- {key}：{value}")
    lines.append("")
    lines.append("## 起手式 top-50（≤20 字句的首段）")
    lines.append("")
    for word, count in stats.starters.top(50):
        lines.append(f"- {word}（{count:,}）")
    lines.append("")
    lines.append("## 收尾式 top-50（≤15 字句的尾段，取末 4 字）")
    lines.append("")
    for word, count in stats.closers.top(50):
        lines.append(f"- {word}（{count:,}）")
    lines.append("")
    lines.append("## 短句（≤8 字）top-200")
    lines.append("")
    for word, count in stats.short_utterances.top(200):
        lines.append(f"- {word}（{count:,}）")
    lines.append("")
    lines.append("## 兴趣标签 top-30")
    lines.append("")
    lines.append("、".join(f"{word}({count})" for word, count in stats.tag_counter.top(30)))
    lines.append("")
    (OUT_DIR / "personal_dialog_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"reports written to {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
