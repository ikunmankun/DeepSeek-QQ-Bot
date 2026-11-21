"""从 PersonalDialog 语料蒸馏真人网聊风格规则，反哺大肥鱼两个人设面。

前置：先跑 scripts/analyze_personal_dialog.py 产出
    .analysis/personal_dialog_samples.jsonl（4000 条高质量对话样本）
    .analysis/personal_dialog_stats.json（聚合统计）

流程（三阶段 LLM，全部 fail-soft，任何阶段失败保留前段产出）：
  1. 规则蒸馏：统计+样本+现行规则 → supported / new_habits / revisions / persona_mismatches
  2. 留出校验：没用过的样本对照初稿 → 修订
  3. 例句改写：真人短句/模式 → 大肥鱼（群聊）与甜心肥鱼（私聊）两套语气候选

在 bot 容器内运行（LLM_* 环境变量由 compose 注入）：
    docker exec -e PYTHONPATH=/workspace xiaomachi-bot python \
        /workspace/scripts/distill_personal_dialog_style.py
产出 .analysis/personal_dialog_style_rules.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / ".analysis"
SAMPLES_PATH = ANALYSIS / "personal_dialog_samples.jsonl"
STATS_PATH = ANALYSIS / "personal_dialog_stats.json"
OUT_PATH = ANALYSIS / "personal_dialog_style_rules.md"
PERSONA_YAML = REPO_ROOT / "configs" / "persona.yaml"
VARIANTS_YAML = REPO_ROOT / "configs" / "persona_variants.yaml"

sys.path.insert(0, str(REPO_ROOT))

from app.core.style_distill import parse_fenced_json, parse_persona_yaml  # noqa: E402


def build_llm_client():
    """容器内用 compose 注入的 LLM_* 环境变量建主档 client。"""

    from app.providers.llm_client import LlmClient

    model = os.environ.get("LLM_MODEL", "deepseek-flash")
    return LlmClient(
        base_url=os.environ["LLM_BASE_URL"].rstrip("/"),
        api_key=os.environ["LLM_API_KEY"],
        model=model,
        fallback_model="",
        responses_model=model,
        responses_only=True,
        # 辅助调用 effort 必须 low/medium（2026-11-18 教训：high 思考会把输出
        # 预算烧光，一个字正文都产不出）；产出清单规模也按预算收紧。
        reasoning_effort="medium",
        max_output_tokens=16384,
        timeout_seconds=240.0,
    )


def load_lines(path: Path, limit: int) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(item)
            if limit and len(items) >= limit:
                break
    return items


def samples_block(samples: list[dict], budget_chars: int) -> str:
    """把样本对话压成 transcript 块，字符预算内均匀取样。"""

    lines: list[str] = []
    used = 0
    step = max(1, len(samples) // 1200)
    picked = samples[::step][:1200]
    for item in picked:
        dialog = item.get("dialog") or []
        line = " ┃ ".join(dialog)
        entry = f"[{item.get('gender', '')}] {line}"
        if used + len(entry) > budget_chars:
            break
        lines.append(entry)
        used += len(entry) + 1
    return "\n".join(lines)


def read_rule_text(path: Path, start_marker: str, end_marker: str) -> str:
    """从人设 YAML 里截取规则段（简化：整段注入，控制预算）。"""

    text = path.read_text(encoding="utf-8")
    return text[:6000]


def stage1_rules(client, stats: dict, samples: list[dict]) -> dict:
    rules_text = read_rule_text(PERSONA_YAML, "", "")
    variants_text = read_rule_text(VARIANTS_YAML, "", "")
    stats_block = json.dumps(stats, ensure_ascii=False)[:6000]
    transcript = samples_block(samples, 45000)
    prompt = [
        "你是网聊风格分析专家。下面是大规模真人中文网络对话语料（PersonalDialog，"
        "微博衍生、非角色扮演）的机械统计与抽样对话，以及一个 AI 角色现行说话规则"
        "（群聊面：傲娇机灵鲸鱼娘「大肥鱼」；私聊面：甜心肥鱼，只对主人软糯粘人）。\n"
        "任务：对照真人语料检验现行规则，产出数据支持的风格修订建议。"
        "注意：这个角色的定位是『像真群友』，语料里的真人短促、接梗、不客套的模式最有参考价值；"
        "但语料是微博评论区风格，与 QQ 群好友闲聊不完全等价，需要判断取舍。\n"
        "只输出一个 ```yaml 代码块，字段：\n"
        "- supported: 现行规则被语料证实的条目，6-10 条，每条含 rule（现行条目摘要）、"
        "evidence（语料证据，引用统计数字或样本模式）\n"
        "- new_habits: 建议新增的说话习惯，4-6 条，每条含 rule（写成可执行的指令句）、"
        "face（default/sweet/both）、evidence（语料观察）\n"
        "- revisions: 建议修订的现行条目，仅当语料明确反对时才提，每条含 original、"
        "revised、reason；没有就不写\n"
        "- persona_mismatches: 真人常做但现行规则禁止/缺失、值得主人权衡的点，"
        "每条含 observation、conflict、trade_off；没有就不写\n"
        "硬要求：每条都必须锚定语料证据，禁止凭空发挥；不要重复现行规则里已经写过的内容。\n\n"
        f"机械统计：\n{stats_block}\n\n"
        f"现行群聊面规则（persona.yaml）：\n{rules_text[:3500]}\n\n"
        f"现行私聊甜心面规则（persona_variants.yaml 的 dafy_sweet）：\n{variants_text[:3500]}\n\n"
        f"抽样对话（[性别] 句子 ┃ 句子 ...）：\n{transcript}"
    ]
    raw = client.generate_text(prompt, reasoning_effort="high")
    return parse_persona_yaml(raw)


def stage2_refine(client, draft: dict, holdout: list[dict]) -> dict:
    transcript = samples_block(holdout, 30000)
    prompt = [
        "下面是针对真人网聊语料蒸馏出的风格规则建议初稿，以及一组**没有参与蒸馏**的留出样本。\n"
        "请对照留出样本校验：删掉样本里观察不到的泛化条目，把写得太满的收窄，"
        "补充留出样本里明显存在但初稿漏掉的模式。只输出修订后的完整 ```yaml 代码块，字段不变。\n\n"
        f"初稿：\n{json.dumps(draft, ensure_ascii=False, indent=2)[:8000]}\n\n"
        f"留出样本：\n{transcript}"
    ]
    raw = client.generate_text(prompt, reasoning_effort="high")
    return parse_persona_yaml(raw)


def stage3_examples(client, stats: dict, samples: list[dict]) -> dict:
    short_top = "\n".join(
        f"- {word}（{count}）" for word, count in (stats.get("short_utterances_top200") or [])[:120]
    )
    starters = "\n".join(
        f"- {word}（{count}）" for word, count in (stats.get("starters_top50") or [])[:40]
    )
    transcript = samples_block(samples, 20000)
    prompt = [
        "下面是真人网聊语料的高频短句、起手式与对话样本。请把其中**可迁移的结构模式**"
        "（长度、节奏、语气词用法、没接住的反应方式）改写成两套 AI 角色的自然例句：\n"
        "- 面向群聊的鲸鱼娘「大肥鱼」：傲娇机灵、嘴硬、爱白饭和 token、B站冲浪、懂反串；"
        "参考她现有例句的味道：「尾巴大而已，别乱叫」「?吃饭还有次数限制的」「题发来，别在人身上鉴定了」\n"
        "- 面向私聊的「甜心肥鱼」：只对主人软糯粘人、贴贴、白饭排第二主人排第一；"
        "参考味道：「来啦来啦，我在这儿呢」「我还以为你把我忘了，饭都少吃半碗」\n"
        "硬规则：不照抄原句；内容必须贴合角色设定（白饭/token/尾巴/B站/群聊日常）；"
        "保留真人的长度与节奏特征；每条一句、不带解释。\n"
        "只输出一个 ```yaml 代码块：\n"
        "- dafy: 大肥鱼群聊例句候选，8 条，格式 `触发 → 回复`\n"
        "- sweet: 甜心肥鱼私聊例句候选，8 条，格式 `触发 → 回复`\n"
        "- pattern_notes: 从语料里看到的、改写时遵循的结构模式说明，3-6 条\n\n"
        f"高频短句 top：\n{short_top}\n\n"
        f"起手式 top：\n{starters}\n\n"
        f"对话样本：\n{transcript}"
    ]
    raw = client.generate_text(prompt, reasoning_effort="high")
    return parse_persona_yaml(raw)


def render_report(draft: dict, refined: dict | None, examples: dict | None) -> str:
    lines = ["# PersonalDialog 风格规则蒸馏", ""]
    body = refined or draft
    for section in ("supported", "new_habits", "revisions", "persona_mismatches"):
        items = body.get(section) or draft.get(section) or []
        lines.append(f"## {section}（{len(items)}）")
        lines.append("")
        lines.append(json.dumps(items, ensure_ascii=False, indent=2))
        lines.append("")
    if examples:
        lines.append("## 例句候选（dafy 群聊 / sweet 私聊）")
        lines.append("")
        lines.append(json.dumps(examples, ensure_ascii=False, indent=2))
        lines.append("")
    if refined:
        lines.append("## 备注")
        lines.append("")
        lines.append("- 本报告经过留出样本二次校验（stage2 refine）。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-refine", action="store_true", help="跳过留出校验（调试用）")
    parser.add_argument("--holdout-ratio", type=float, default=0.25)
    args = parser.parse_args()

    if not SAMPLES_PATH.exists() or not STATS_PATH.exists():
        print("先跑 scripts/analyze_personal_dialog.py", file=sys.stderr)
        return 1
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    samples = load_lines(SAMPLES_PATH, limit=0)
    cut = int(len(samples) * (1.0 - args.holdout_ratio))
    train, holdout = samples[:cut], samples[cut:]
    print(f"samples total={len(samples)} train={len(train)} holdout={len(holdout)}", file=sys.stderr)

    client = build_llm_client()
    print("stage1: 规则蒸馏...", file=sys.stderr)
    draft = stage1_rules(client, stats, train)
    refined = None
    if not args.skip_refine:
        print("stage2: 留出校验...", file=sys.stderr)
        try:
            refined = stage2_refine(client, draft, holdout)
        except Exception as exc:
            print(f"stage2 failed (keep draft): {type(exc).__name__}: {exc}", file=sys.stderr)
    print("stage3: 例句改写...", file=sys.stderr)
    examples = None
    try:
        examples = stage3_examples(client, stats, train)
    except Exception as exc:
        print(f"stage3 failed (skip examples): {type(exc).__name__}: {exc}", file=sys.stderr)

    OUT_PATH.write_text(render_report(draft, refined, examples), encoding="utf-8")
    print(f"written: {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
