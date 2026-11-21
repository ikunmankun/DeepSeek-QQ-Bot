"""图片客观色彩分析单元：合成图验证偏色/亮度/对比度口径。"""
import io

import pytest
from PIL import Image

from app.core.image_color_stats import compute_color_stats


def _png(pixels):
    """把像素行列表转成 PNG 字节；像素是 (r, g, b) 元组。"""
    height = len(pixels)
    width = len(pixels[0])
    image = Image.new("RGB", (width, height))
    image.putdata([pixel for row in pixels for pixel in row])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _flat(color, size=64):
    return _png([[color] * size for _ in range(size)])


def test_neutral_slightly_blue_screenshot_detected_as_cold_cast():
    # 截图场景：大片浅蓝白底（饱和度约0.10，落进中性区）+ 少量深色文字。
    pixels = [[(230, 242, 255)] * 64 for _ in range(60)]
    pixels += [[(30, 30, 30)] * 64 for _ in range(4)]
    stats = compute_color_stats(_png(pixels))
    assert stats["cast"]["basis"] == "中性区"
    assert "偏冷" in stats["cast"]["direction"]
    assert stats["cast"]["strength"] > 0.04
    assert stats["cast"]["neutral_rgb"][2] > stats["cast"]["neutral_rgb"][0]


def test_strong_uniform_filter_falls_back_to_whole_image():
    # 整图加蓝色滤镜：原灰区饱和度超标，中性区清空 → 全图灰世界口径。
    stats = compute_color_stats(_flat((100, 100, 150)))
    assert stats["cast"]["basis"] == "全图口径"
    assert "偏冷" in stats["cast"]["direction"]
    assert stats["cast"]["strength"] > 0.05  # 打对折后仍显著


def test_true_gray_image_reports_no_cast():
    stats = compute_color_stats(_flat((128, 128, 128)))
    assert stats["cast"]["direction"] == "中性"
    assert stats["cast"]["strength"] < 0.04
    assert stats["cast"]["level"] == "几乎不偏"
    assert stats["brightness"]["level"] == "正常"


def test_dark_and_bright_images_report_brightness():
    dark = compute_color_stats(_flat((20, 18, 16)))
    assert dark["brightness"]["level"] == "整体偏暗"
    assert dark["brightness"]["mean"] < 60
    bright = compute_color_stats(_flat((250, 250, 250)))
    assert bright["brightness"]["level"] == "整体偏亮"
    assert bright["brightness"]["highlight_ratio"] > 0.9
    assert "可能过曝" in bright["summary"][1]


def test_high_contrast_checkerboard():
    pixels = [[(0, 0, 0) if (x + y) % 2 == 0 else (255, 255, 255)
               for x in range(64)] for y in range(64)]
    stats = compute_color_stats(_png(pixels))
    assert stats["contrast"]["level"] == "高（明暗反差大）"
    assert stats["contrast"]["std"] > 60
    assert stats["contrast"]["p10"] <= 10 and stats["contrast"]["p95"] >= 245


def test_low_contrast_gray_wash():
    # 整体灰蒙蒙：亮度挤在窄区间，标准差很小。
    pixels = [[(118, 120, 122)] * 64 for _ in range(64)]
    stats = compute_color_stats(_png(pixels))
    assert "低" in stats["contrast"]["level"]
    assert stats["contrast"]["std"] < 5


def test_dominant_colors_and_vivid_content():
    # 高饱和内容图：主色调应取到红，浓艳占比接近 1。
    pixels = [[(200, 30, 30)] * 64 for _ in range(64)]
    stats = compute_color_stats(_png(pixels))
    dominant = [color for color, _ in stats["dominant"]]
    assert dominant and dominant[0].startswith("#")
    assert stats["saturation"]["vivid_ratio"] > 0.5


def test_non_square_and_resize():
    # 长图缩放不崩；返回原始尺寸。
    pixels = [[(120, 120, 120)] * 40 for _ in range(300)]
    stats = compute_color_stats(_png(pixels))
    assert stats["size"] == [40, 300]


def test_corrupt_bytes_raise():
    with pytest.raises(Exception):
        compute_color_stats(b"not an image at all")
