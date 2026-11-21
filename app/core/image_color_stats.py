"""图片客观色彩分析：亮度/对比度/偏色/饱和度/主色调，纯 PIL 像素统计。

为什么不用视觉模型：群友问"这张图颜色偏没偏"是客观测量问题——
视觉模型只会说"看起来有点冷"，且主模型（DeepSeek）收不了图、
独立视觉通道还得额外配凭证。像素统计任何模型都能转述：偏色方向和
强度有真实数字支撑，不是模型的主观印象。口径约定：

* 偏色看"中性灰区"——只统计低饱和、中等亮度的像素的 RGB 均值差，
  画面本身的内容色（蓝天、草地）不会污染白平衡判断；中性区不足时
  退回全图灰世界口径并如实标注（含内容色，仅供参考）；
* 对比度用亮度标准差 + P10-P95 范围，不受极端噪点影响；
* 全部输出可测量的数字（均值/占比/百分比），不编造色温开尔文
  这类估不准的精确值——给"偏冷/偏暖 + 偏离百分比"。
"""
from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageStat

#: 统计用的工作分辨率：长边缩到这个值，65k 像素遍历毫无压力，
#: 统计口径（占比/均值）对分辨率不敏感。
WORK_SIZE = 256

#: 亮度分界（0-255）：暗部/高光占比用。
DARK_CUTOFF = 40
HIGHLIGHT_CUTOFF = 220

#: 中性灰区判定：饱和度低于它、亮度（灰度口径）在暗部高光之间的像素才
#: 参与白平衡统计。亮度上限放宽到 250——白底截图的背景（亮度240+）
#: 恰恰是最可靠的白平衡参照，一刀切到"中间调"会把它们全部排除。
NEUTRAL_SAT_MAX = 0.14
NEUTRAL_LUMA_MIN = 40
NEUTRAL_LUMA_MAX = 250

#: 偏色强度分级：中性区 RGB 相对灰点 (r+g+b)/3 的最大通道偏离。
CAST_LEVELS = ((0.04, "几乎不偏"), (0.10, "轻微偏色"), (0.20, "明显偏色"))
CAST_SEVERE = "严重偏色"


def _percentile(hist: list[int], fraction: float) -> int:
    """从 256-bin 直方图取分位值。"""
    total = sum(hist)
    if not total:
        return 0
    target = total * fraction
    cumulative = 0
    for value, count in enumerate(hist):
        cumulative += count
        if cumulative >= target:
            return value
    return 255


def _cast_direction(r: float, g: float, b: float) -> str:
    """给中性区 RGB 均值一个"偏什么色"的短描述。"""
    gray = (r + g + b) / 3.0
    dr, dg, db = r - gray, g - gray, b - gray
    warm = dr - db  # 红蓝差决定冷暖
    parts: list[str] = []
    if abs(warm) >= 6:
        parts.append("偏暖" if warm > 0 else "偏冷")
    if dg >= 8 and dg >= dr and dg >= db:
        parts.append("偏绿")
    elif dr >= 10 and dr > dg and abs(warm) < 6:
        parts.append("偏红")
    elif db >= 10 and db > dg and abs(warm) < 6:
        parts.append("偏蓝")
    return "、".join(parts) if parts else "中性"


def _dominant_colors(pixels: list[tuple[int, int, int]], total: int) -> list[list[Any]]:
    """主色调：16 色量化后取占比最高的前 3 色（hex + 占比）。"""
    if not pixels or not total:
        return []
    counts: dict[tuple[int, int, int], int] = {}
    # 量化到 32 级再计数，避免同色微抖动拆散占比。
    for r, g, b in pixels:
        key = (r >> 5, g >> 5, b >> 5)
        counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [["#{:02X}{:02X}{:02X}".format(r << 5, g << 5, b << 5), round(n / total, 2)]
            for (r, g, b), n in top]


def compute_color_stats(data: bytes) -> dict[str, Any]:
    """对图片字节做客观色彩统计；损坏或空图抛 OSError/ValueError。"""
    with Image.open(io.BytesIO(data)) as picture:
        picture.load()  # 动图取第一帧
        rgb = picture.convert("RGB")
        width, height = rgb.size
        scale = WORK_SIZE / max(width, height)
        if scale < 1.0:
            rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        gray = rgb.convert("L")
        hsv = rgb.convert("HSV")

        stat = ImageStat.Stat(gray)
        luma_mean = stat.mean[0]
        luma_std = stat.stddev[0]
        luma_hist = gray.histogram()

        total = sum(luma_hist)
        dark_ratio = sum(luma_hist[:DARK_CUTOFF]) / max(1, total)
        highlight_ratio = sum(luma_hist[HIGHLIGHT_CUTOFF:]) / max(1, total)
        p10 = _percentile(luma_hist, 0.10)
        p95 = _percentile(luma_hist, 0.95)

        rgb_stat = ImageStat.Stat(rgb)
        avg_r, avg_g, avg_b = (round(v, 1) for v in rgb_stat.mean)

        sat_hist = hsv.histogram()[256:512]
        pixel_total = max(1, sum(sat_hist))
        vivid_ratio = sum(sat_hist[96:]) / pixel_total

        # 中性灰区：低饱和 + 中等亮度。这里的 R/G/B 差值才是白平衡信号。
        # 截图/白底照片的大片背景几乎都落在这里，轻度偏色能直接测出；
        # 整图强滤镜会让"原灰区"饱和度超标被排除——此时退回灰世界假设
        # （全图通道均值差）：对内容色占比大的图会高估，强度打对折保守报。
        neutral_pixels: list[tuple[int, int, int]] = []
        all_pixels = list(rgb.getdata())
        sat_data = [s for (_, s, _) in hsv.getdata()]
        gray_data = list(gray.getdata())
        sat_max_255 = round(NEUTRAL_SAT_MAX * 255)
        for (r, g, b), s, lum in zip(all_pixels, sat_data, gray_data):
            if s <= sat_max_255 and NEUTRAL_LUMA_MIN <= lum <= NEUTRAL_LUMA_MAX:
                neutral_pixels.append((r, g, b))
        neutral_rgb = None
        discount = 0.0
        if len(neutral_pixels) >= max(64, len(all_pixels) * 0.02):
            n = len(neutral_pixels)
            nr = sum(p[0] for p in neutral_pixels) / n
            ng = sum(p[1] for p in neutral_pixels) / n
            nb = sum(p[2] for p in neutral_pixels) / n
            neutral_rgb = [round(nr), round(ng), round(nb)]
            cast_basis = "中性区"
            discount = 1.0
        elif all_pixels:
            nr, ng, nb = ImageStat.Stat(rgb).mean
            cast_basis = "全图口径"
            discount = 0.5
        if all_pixels:
            gray_point = (nr + ng + nb) / 3.0
            strength = discount * max(abs(nr - gray_point), abs(ng - gray_point),
                                      abs(nb - gray_point)) / max(1.0, gray_point)
            cast_direction = _cast_direction(nr, ng, nb)
        else:
            strength, cast_direction = 0.0, "图片没有可统计的像素"
        if neutral_rgb is None and all_pixels:
            neutral_rgb = [round(nr), round(ng), round(nb)]

        dominant = _dominant_colors(all_pixels, len(all_pixels))

    if strength < CAST_LEVELS[0][0]:
        cast_level = CAST_LEVELS[0][1]
    elif strength < CAST_LEVELS[1][0]:
        cast_level = CAST_LEVELS[1][1]
    elif strength < CAST_LEVELS[2][0]:
        cast_level = CAST_LEVELS[2][1]
    else:
        cast_level = CAST_SEVERE

    if luma_std < 28:
        contrast_level = "低（画面发灰）"
    elif luma_std > 60:
        contrast_level = "高（明暗反差大）"
    else:
        contrast_level = "正常"

    if luma_mean < 60:
        brightness_level = "整体偏暗"
    elif luma_mean > 175:
        brightness_level = "整体偏亮"
    else:
        brightness_level = "正常"

    summary: list[str] = []
    if all_pixels:
        neutral_label = "中性区" if cast_basis == "中性区" else "全图"
        if cast_basis == "中性区":
            summary.append(f"中性区平均RGB {neutral_rgb}，{cast_direction}"
                           f"（强度约{strength:.0%}，{cast_level}）")
        else:
            summary.append(f"{neutral_label}口径平均RGB {neutral_rgb}，{cast_direction}"
                           f"（强度约{strength:.0%}，{cast_level}；含画面内容色，仅供参考）")
    else:
        summary.append("图片没有可统计的像素")
    summary.append(f"平均亮度{luma_mean:.0f}/255（{brightness_level}），"
                   f"暗部占{dark_ratio:.0%}，高光占{highlight_ratio:.0%}"
                   + ("，可能过曝" if highlight_ratio > 0.25 else ""))
    summary.append(f"对比度{luma_std:.0f}（{contrast_level}），明暗范围 P10={p10} ~ P95={p95}")
    sat_mean = ImageStat.Stat(hsv).mean[1] / 255.0
    summary.append(f"平均饱和度{sat_mean:.2f}，浓艳色占{vivid_ratio:.0%}")
    if dominant:
        summary.append("主色调：" + "、".join(f"{color}（{share:.0%}）" for color, share in dominant))

    return {
        "size": [width, height],
        "brightness": {"mean": round(luma_mean, 1), "level": brightness_level,
                       "dark_ratio": round(dark_ratio, 3),
                       "highlight_ratio": round(highlight_ratio, 3)},
        "contrast": {"std": round(luma_std, 1), "p10": p10, "p95": p95, "level": contrast_level},
        "channel_avg": {"r": avg_r, "g": avg_g, "b": avg_b},
        "saturation": {"mean": round(sat_mean, 3), "vivid_ratio": round(vivid_ratio, 3)},
        "cast": {"direction": cast_direction, "strength": round(strength, 3), "level": cast_level,
                 "basis": cast_basis if all_pixels else "none", "neutral_rgb": neutral_rgb},
        "dominant": dominant,
        "summary": summary,
    }
