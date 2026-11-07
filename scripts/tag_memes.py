"""用视觉模型给本地表情库批量打情绪标签，并重命名成 ``编号_关键词.webp``。

pick_sticker 的 keyword 参数按文件名匹配，纯数字编号等于没有检索能力；
本脚本把 ``001.webp`` 这类文件重命名为 ``001_无语_看戏.webp``，让模型能
按情绪挑图。已带关键词的文件自动跳过，可安全重复执行。

设计为在 bot 容器内运行（依赖 httpx，网络姿态与机器人一致）：

    docker run --rm --network host --env-file infra/wsl/.env \
        -v /mnt/e/QQ-Bot:/workspace -w /workspace xiaomachi-bot:local \
        python scripts/tag_memes.py --limit 3          # 干跑看效果
    ... python scripts/tag_memes.py                    # 干跑全部
    ... python scripts/tag_memes.py --apply            # 真正重命名
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path

import httpx

IMAGE_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg", ".gif"}
MEDIA_TYPES = {
    ".webp": "image/webp", ".png": "image/png",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
}
_UNTAGGED = re.compile(r"^\d+$")
_INVALID_FILENAME = re.compile(r'[\\/:*?"<>|\r\n\t]')
_TAG_SPLIT = re.compile(r"[_、,，;；\s]+")

PROMPT = (
    "这是QQ群里使用的表情包贴图，主角通常是一条蓝色卡通鲸鱼。"
    "请根据画面内容给出2到3个简短中文关键词，描述它表达的情绪或适用场景。"
    "参考词（不限于此）：无语 笑死 看戏 得意 贴贴 摸鱼 疑惑 害怕 干饭 睡觉 "
    "吃瓜 装无辜 赞 拒绝 思考 麻了 哭 生气 加油 懵 摆烂 狂喜 鄙视 委屈 害羞。"
    "只输出关键词本身，用单个下划线_分隔，不要解释、不要句号。"
)


def caption(image_path: Path, *, base_url: str, api_key: str, model: str, client: httpx.Client) -> str:
    media_type = MEDIA_TYPES[image_path.suffix.lower()]
    raw = image_path.read_bytes()
    data_url = f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"
    response = client.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        content = str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected payload: {str(payload)[:300]}")
    if not content.strip():
        raise RuntimeError(f"empty content: {str(payload)[:300]}")
    return content


def tags_to_stem_suffix(raw: str, *, max_tags: int = 3, max_tag_chars: int = 8) -> str:
    """把模型输出清洗成文件名安全的下划线后缀；无效输出返回空串。"""

    parts = [part for part in _TAG_SPLIT.split(str(raw or "").strip()) if part]
    cleaned: list[str] = []
    for part in parts[: max_tags + 2]:
        tag = _INVALID_FILENAME.sub("", part).strip("。．.！!？?~～")
        if not tag or len(tag) > max_tag_chars:
            continue
        if tag not in cleaned:
            cleaned.append(tag)
        if len(cleaned) >= max_tags:
            break
    return "_".join(cleaned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=os.environ.get("LOCAL_STICKER_DIR") or "memes")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个未打标文件（0=全部）")
    parser.add_argument("--apply", action="store_true", help="真正重命名（默认干跑）")
    parser.add_argument("--sleep", type=float, default=0.4, help="两次请求间隔秒数")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    args = parser.parse_args()

    base_url = (os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com").strip()
    api_key = (os.environ.get("LLM_API_KEY") or "").strip()
    model = args.model.strip()
    if not api_key or not model:
        print("缺少 LLM_API_KEY 或 LLM_MODEL 环境变量", file=sys.stderr)
        return 2

    directory = Path(args.directory)
    pending = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and _UNTAGGED.match(path.stem)
    )
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"待打标 {len(pending)} 个文件（目录 {directory}，模型 {model}，{'APPLY' if args.apply else 'DRY-RUN'}）")

    renamed = failed = 0
    with httpx.Client() as client:
        for index, path in enumerate(pending, 1):
            suffix = raw_output = ""
            for attempt in (1, 2, 3):
                try:
                    raw_output = caption(path, base_url=base_url, api_key=api_key,
                                         model=model, client=client)
                    suffix = tags_to_stem_suffix(raw_output)
                    break
                except Exception as exc:
                    if attempt == 3:
                        print(f"[{index}/{len(pending)}] {path.name} 失败: {type(exc).__name__}: {str(exc)[:120]}")
                        failed += 1
                    else:
                        time.sleep(1.5 * attempt)
            if not suffix:
                if args.apply or args.limit:
                    print(f"[{index}/{len(pending)}] {path.name} 空标签，原始输出: {raw_output!r}")
                continue
            target = path.with_name(f"{path.stem}_{suffix}{path.suffix.lower()}")
            print(f"[{index}/{len(pending)}] {path.name} -> {target.name}")
            if args.apply:
                path.rename(target)
            renamed += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"完成：成功 {renamed}，失败 {failed}，未处理 {len(pending) - renamed - failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
