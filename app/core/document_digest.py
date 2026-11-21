"""长文档要点提取：分块"map 提要点" + "reduce 合并"，供 summarize_document 工具使用。

为什么不用一次全文调用：主模型上下文有限且长输入容易让要点变成泛泛
总结。分块后每块只做"捞要点"这一件小事，块内的具体数字、名字、结论
都会被保留；reduce 只做去重合并，不重新发明内容。整条链路都是文本
管道——输入是纯文本，模型输出宽松解析成 JSON 数组，解析失败当空处理
（与 trait_evolution 同一套宽容风格），绝不把解析失败抛给聊天回合。

调用方给一个有 ``generate_text(messages) -> str`` 的 client 即可；
client 的异常在块内捕获降级，全部失败时返回 error，不中断会话。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: 单次工具可处理的文档总长（字）。超出截断并在结果里如实标注，
#: 不假装读过全文。
MAX_DOCUMENT_CHARS = 60000
#: map 阶段每块的目标长度：按段落边界切，宁可短不跨段。
CHUNK_TARGET_CHARS = 4000
#: map 最多调 LLM 的块数——超过直接不读了，控制一次工具调用的成本。
MAX_CHUNKS = 8
#: 最终要点条数与单条长度。
MAX_POINTS = 8
MAX_POINT_CHARS = 80

MAP_PROMPT = (
    "你在帮人从长文档里提取重点。下面是文档的一段（第{index}/{total}块）。"
    "提取这一段真正重要的信息：结论、数字、决定、要求、待办、异常。"
    f"最多5条，每条不超过{MAX_POINT_CHARS}个字，短句，保留具体数字和名词。"
    "废话、寒暄、重复内容不要。没有要点就返回空数组。\n"
    '只输出 JSON 数组：["要点1", "要点2"]'
)

REDUCE_PROMPT = (
    "下面是从一篇长文档各段提取的要点（可能有重复或相近的）。"
    "把它们去重、合并成最终的重点清单："
    f"最多{MAX_POINTS}条，每条不超过{MAX_POINT_CHARS}个字，按重要性排序。"
    "再给一句话主旨（gist，30字内）。文档观点有冲突就保留冲突的两面，"
    "别替文档下它没下的结论。\n"
    '只输出 JSON：{"gist": "一句话主旨", "points": ["要点1", "要点2"]}'
)


def split_text_chunks(text: str, *, target_chars: int = CHUNK_TARGET_CHARS) -> list[str]:
    """按段落边界把文本切成约 target_chars 的块；空块丢弃。"""
    normalized = str(text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if not paragraphs:
        paragraphs = [normalized]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        # 单段超长（比如没有空行的整段文章）按句子再切。
        if len(paragraph) > target_chars:
            pieces = re.split(r"(?<=[。！？!?；;\n])", paragraph)
        else:
            pieces = [paragraph]
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current and current_len + len(piece) > target_chars:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks


def parse_points(raw: str | None) -> list[str]:
    """宽松解析 map 阶段输出的 JSON 数组；解析不出就当空。"""
    if not raw:
        return []
    match = re.search(r"\[.*\]", str(raw), re.DOTALL)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    points = []
    for item in payload[:MAX_POINTS]:
        point = re.sub(r"\s+", " ", str(item or "")).strip()[:MAX_POINT_CHARS]
        if point:
            points.append(point)
    return points


def parse_digest(raw: str | None) -> dict[str, Any] | None:
    """宽松解析 reduce 阶段输出的 JSON 对象。"""
    if not raw:
        return None
    match = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    gist = re.sub(r"\s+", " ", str(payload.get("gist") or "")).strip()[:80]
    points = parse_points(json.dumps(payload.get("points") or [], ensure_ascii=False))
    if not points:
        return None
    return {"gist": gist, "points": points}


def digest_client_enabled(settings):
    """按配置构建要点提取的辅助 LLM client；未配置或不可用时返回 None。

    主模型、独立小输出预算、不重试——辅助调用要快失败，不能拖慢聊天回合。
    群聊插件与私聊服务共用这一个工厂，行为保持一致。
    """
    if not getattr(settings, "document_digest_enabled", True):
        return None
    base_url = str(getattr(settings, "llm_base_url", "") or "")
    api_key = str(getattr(settings, "llm_api_key", "") or "")
    if not base_url or not api_key:
        return None
    from app.providers.llm_client import LlmClient

    try:
        return LlmClient(
            base_url=base_url,
            api_key=api_key,
            model=settings.llm_model,
            fallback_model="",
            responses_only=True,
            responses_model=settings.llm_model,
            max_output_tokens=1200,
            timeout_seconds=60.0,
            # 辅助小任务用 low：推理 token 太多会吃光输出预算（同 trait_evolution 教训）。
            reasoning_effort="low",
            request_max_attempts=1,
        )
    except Exception:
        logger.warning("document_digest_client_unavailable", exc_info=True)
        return None


def digest_points(*, client, text: str, focus: str = "") -> dict[str, Any]:
    """同步执行 map-reduce；返回 {gist, points, stats} 或 {error}。

    单块失败跳过（记 warning）；全部块失败返回 error。focus 是用户
    关心的方向（可空），会附加到每块的提取问题里。
    """
    full_text = str(text or "").strip()
    if not full_text:
        return {"error": "empty_text"}
    truncated = False
    if len(full_text) > MAX_DOCUMENT_CHARS:
        full_text = full_text[:MAX_DOCUMENT_CHARS]
        truncated = True
    chunks = split_text_chunks(full_text)[:MAX_CHUNKS]
    if not chunks:
        return {"error": "empty_text"}
    stats = {"chars": len(full_text), "chunks": len(chunks), "truncated": truncated}

    focus_line = f"用户特别关心：{focus.strip()[:120]}。优先保留跟它相关的要点。" if focus.strip() else ""
    collected: list[str] = []
    failed_chunks = 0
    for index, chunk in enumerate(chunks, start=1):
        prompt = MAP_PROMPT.format(index=index, total=len(chunks))
        if focus_line:
            prompt += "\n" + focus_line
        try:
            raw = client.generate_text([prompt, "以下是文档片段：", chunk])
        except Exception:
            logger.warning("document_digest_map_failed chunk=%s/%s", index, len(chunks))
            failed_chunks += 1
            continue
        collected.extend(parse_points(raw))
    if not collected:
        return {"error": "digest_failed", "stats": stats,
                "note": "所有分块都没能提取出要点" if failed_chunks else "文档里没有提取到有效要点"}

    # reduce 一次合并；去重交给提示词+顺序保留，模型自己合并相近表述。
    merged_raw = "\n".join(f"- {point}" for point in dict.fromkeys(collected))
    try:
        digest = parse_digest(client.generate_text([REDUCE_PROMPT, "各段要点：", merged_raw]))
    except Exception:
        logger.warning("document_digest_reduce_failed")
        digest = None
    if digest is None:
        # reduce 失败不丢结果：直接把 map 要点截到上限返回。
        digest = {"gist": "", "points": list(dict.fromkeys(collected))[:MAX_POINTS]}
    if truncated:
        digest["truncated_note"] = f"原文超过{MAX_DOCUMENT_CHARS}字，只处理了前{MAX_DOCUMENT_CHARS}字"
    digest["stats"] = stats
    return digest
