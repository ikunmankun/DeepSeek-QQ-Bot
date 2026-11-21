"""图表精读：把图表截图交给视觉模型做结构化解读。

为什么单独一个工具：DeepSeek 官方 API 是纯文本（收不了图），
`LLM_SUPPORTS_VISION_INPUT=true` 且 vision_model 未配置时模型根本看不见图；
即便能看，密集图表也需要专门的提取式提问（轴/图例/系列/趋势/数值）
而不是让闲聊模型自由发挥。本模块走独立的 OpenAI 兼容视觉通道
（硅基流动/阿里百炼/智谱等都兼容），输入原始像素，输出一段结构化解读，
agent 拿到后用自己的口吻接话。

凭证未配置时上层隐藏工具；接口错误抛 OSError。
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CHART_PROMPT = (
    "这是一张聊天群里分享的图表截图。请仔细读图并输出一段结构化解读，供另一个"
    "AI 助手转述给群友。要求覆盖：\n"
    "1. 图表类型与主题（标题/副标题若有）\n"
    "2. 坐标轴：X 轴和 Y 轴分别是什么、单位、刻度范围\n"
    "3. 图例：每个系列的颜色/名称\n"
    "4. 关键数值：最大值、最小值、转折点，尽量读出具体数字\n"
    "5. 趋势与结论：整体上升/下降/波动，各系列之间谁高谁低，图表想表达什么\n"
    "6. 若图里有文字说明、数据标签、表格，把它们念出来\n"
    "图像可能不清晰：看不清的地方明确说“看不清”，不要编造具体数字。\n"
    "用中文输出，条理清楚，300字以内。"
)


def analyze_chart_image(image_bytes: bytes, content_type: str, *, base_url: str,
                        api_key: str, model: str, timeout_seconds: float = 60.0) -> str:
    """OpenAI 兼容视觉通道：图片 base64 + 图表提取式提问 → 解读文本。

    接口/网络错误抛 OSError；返回的解读文本可能包含“看不清”的诚实说明。
    """
    base_url = base_url.rstrip("/")
    data_uri = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": CHART_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        "max_tokens": 1200,
        "temperature": 0.2,
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout_seconds,
        trust_env=False,
    )
    response.raise_for_status()
    payload_out = response.json()
    choices = payload_out.get("choices") or []
    text = str(((choices[0].get("message") or {}).get("content") or "").strip()) if choices else ""
    if not text:
        raise OSError("vision model returned empty analysis")
    return text


def sniff_image_type(path: Path) -> str:
    """按文件头判断 MIME（napcat 原件可能没有扩展名）。"""
    head = Path(path).read_bytes()[:12] if Path(path).is_file() else b""
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head[:4] == b"GIF8":
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"
