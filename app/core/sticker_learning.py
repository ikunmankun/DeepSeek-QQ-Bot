"""偷表情包：群友发表情自动收藏、视觉打标、供 pick_sticker 按情绪复用。

链路：入站钩子把群图按内容哈希收进 data/sticker_library/learned/（去重、
限额）→ 周期 worker 用视觉模型给未打标的新图写 <hash>.json 标签 →
pick_sticker 的目录目录里多出 learned 库，keyword 匹配标签。群友会发现
自己的表情被它学走了——这就是活物感。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from app.core.time_utils import shanghai_now_naive

logger = logging.getLogger(__name__)

LEARNED_DIR_NAME = "sticker_library/learned"
IMAGE_SUFFIXES = {".webp", ".png", ".jpg", ".jpeg", ".gif"}
MEDIA_TYPES = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg", ".gif": "image/gif"}
MAX_STICKER_BYTES = 8 * 1024 * 1024
MAX_LEARNED_FILES = 500
#: 单轮打标上限：控制视觉调用的突发量。
TAG_BATCH_PER_TICK = 8

TAG_PROMPT = (
    "这是QQ群里会用的表情包。先给出2到3个简短中文关键词描述它表达的情绪或适用场景，"
    "用单个下划线_分隔；然后接一个竖线||，再用不超过20字说明这张图的梗含义或笑点"
    "（是知名梗就说梗名，看不懂梗就描述画面内容）。只输出这一行，不要解释。"
    "示例：无语_看戏_吃瓜||熊猫头趴桌摆烂，表达无语看戏"
)

#: 标签段和含义段的分隔符。
_MEANING_SPLIT = "||"

_TAG_SPLIT = re.compile(r"[_、,，;；\s]+")


class StickerLearner:
    """收藏与打标；目录约定：<data_dir>/sticker_library/learned/。"""

    def __init__(self, data_dir: Path) -> None:
        self.learned_dir = Path(data_dir) / LEARNED_DIR_NAME

    def collect_from_event(self, event: Any, *, bot_qq: int) -> int:
        """把一条入站消息里的图片收进学习库；返回新收藏张数。"""

        if event.user_id == bot_qq:
            return 0
        self.learned_dir.mkdir(parents=True, exist_ok=True)
        collected = 0
        for image in list(getattr(event, "images", None) or [])[:4]:
            local_path = getattr(image, "local_path", None)
            if not local_path:
                continue
            try:
                path = Path(local_path)
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if not raw or len(raw) > MAX_STICKER_BYTES:
                continue
            if sum(1 for _ in self.learned_dir.iterdir()) >= MAX_LEARNED_FILES:
                return collected
            digest = hashlib.sha256(raw).hexdigest()[:12]
            target = self.learned_dir / f"{digest}{path.suffix.lower()}"
            if target.exists():
                continue
            target.write_bytes(raw)
            collected += 1
        return collected

    def pending_tagging(self) -> list[Path]:
        """已收藏但还没有标签文件的图。"""

        if not self.learned_dir.exists():
            return []
        pending = []
        for path in sorted(self.learned_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not path.with_suffix(".json").exists():
                pending.append(path)
        return pending

    def tag_one(self, path: Path, *, base_url: str, api_key: str, model: str,
                client: httpx.Client, sidecar_dir: Path | None = None) -> str:
        """用视觉模型给一张图生成下划线分隔的情绪标签（+梗含义）并落盘。

        sidecar_dir：标签文件的落盘目录；默认写在图片旁边。memes/ 是只读
        卷挂载，它的含义文件统一放 data/meme_meanings/。
        """
        sidecar_parent = Path(sidecar_dir) if sidecar_dir is not None else path.parent
        sidecar_parent.mkdir(parents=True, exist_ok=True)
        media_type = MEDIA_TYPES.get(path.suffix.lower(), "image/webp")
        data_url = f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        response = client.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TAG_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        raw = str(response.json()["choices"][0]["message"]["content"] or "").strip()
        meaning = ""
        if _MEANING_SPLIT in raw:
            raw, _, meaning_raw = raw.partition(_MEANING_SPLIT)
            meaning = re.sub(r"[\r\n]", "", meaning_raw).strip("。．.！!？?~～ ")[:40]
        tags = _TAG_SPLIT.split(raw)
        cleaned = []
        for part in tags[:3]:
            tag = re.sub(r'[\\/:*?"<>|\r\n]', "", part).strip("。．.！!？?~～ ")[:8]
            if tag and tag not in cleaned:
                cleaned.append(tag)
        if not cleaned:
            raise ValueError("empty tags")
        sidecar = sidecar_parent / (path.stem + ".json")
        sidecar.write_text(json.dumps({"tags": "_".join(cleaned), "meaning": meaning},
                                      ensure_ascii=False),
                           encoding="utf-8")
        return "_".join(cleaned)

    def load_tags(self, path: Path, *, sidecar_dir: Path | None = None) -> str:
        sidecar = (Path(sidecar_dir) / (path.stem + ".json")) if sidecar_dir is not None \
            else path.with_suffix(".json")
        if not sidecar.exists():
            return ""
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        tags = str(data.get("tags", ""))
        meaning = str(data.get("meaning", "") or "")
        return f"{tags}||{meaning}" if meaning else tags

    def load_meaning(self, path: Path, *, sidecar_dir: Path | None = None) -> str:
        sidecar = (Path(sidecar_dir) / (path.stem + ".json")) if sidecar_dir is not None \
            else path.with_suffix(".json")
        if not sidecar.exists():
            return ""
        try:
            return str(json.loads(sidecar.read_text(encoding="utf-8")).get("meaning", "") or "")
        except (OSError, ValueError):
            return ""


class StickerTaggingService:
    """周期 worker：给学习库里的新图打标签（每轮小批量），顺带回填 memes 库梗含义。

    memes/ 是只读卷挂载，含义文件统一写到 data/meme_meanings/<编号>.json；
    542 张库存按每轮 4 张慢慢补齐，不 burst。
    """

    def __init__(self, *, learner: StickerLearner, settings: Any,
                 interval_seconds: float = 600.0,
                 memes_dir: Path | None = None,
                 meme_meanings_dir: Path | None = None,
                 memes_batch_per_tick: int = 4) -> None:
        self.learner = learner
        self.settings = settings
        self.interval_seconds = float(interval_seconds)
        self.memes_dir = Path(memes_dir) if memes_dir else None
        self.meanings_dir = Path(meme_meanings_dir) if meme_meanings_dir else None
        self.memes_batch_per_tick = int(memes_batch_per_tick)

    async def run(self) -> None:
        while True:
            try:
                tagged = await asyncio.to_thread(self._tick)
                if tagged:
                    logger.info("sticker_learning_tagged count=%s", tagged)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("sticker_tagging_tick_failed")
            await asyncio.sleep(self.interval_seconds)

    def _tick(self) -> int:
        tagged = self._tick_learned()
        tagged += self._tick_memes()
        return tagged

    def _vision_credentials(self) -> tuple[str, str, str] | None:
        base_url = (getattr(self.settings, "llm_base_url", "") or "").strip()
        api_key = (getattr(self.settings, "llm_api_key", "") or "").strip()
        model = (getattr(self.settings, "llm_model", "") or "").strip()
        if not api_key or not model:
            return None
        return base_url, api_key, model

    def _tag_with_retry(self, client: httpx.Client, path: Path, creds, sidecar_dir: Path | None) -> bool:
        base_url, api_key, model = creds
        for attempt in (1, 2):
            try:
                self.learner.tag_one(path, base_url=base_url, api_key=api_key,
                                     model=model, client=client, sidecar_dir=sidecar_dir)
                return True
            except Exception as exc:
                if attempt == 2:
                    logger.warning("sticker_tag_failed file=%s error=%s",
                                   path.name, type(exc).__name__)
        return False

    def _tick_learned(self) -> int:
        pending = self.learner.pending_tagging()[:TAG_BATCH_PER_TICK]
        if not pending:
            return 0
        creds = self._vision_credentials()
        if creds is None:
            return 0
        tagged = 0
        with httpx.Client() as client:
            for path in pending:
                if self._tag_with_retry(client, path, creds, None):
                    tagged += 1
        return tagged

    def _tick_memes(self) -> int:
        """给 memes 编号库补梗含义；已有含义文件的跳过，每轮小批量。"""
        if self.memes_dir is None or self.meanings_dir is None:
            return 0
        if not self.memes_dir.exists():
            return 0
        pending = []
        for path in sorted(self.memes_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if (self.meanings_dir / (path.stem + ".json")).exists():
                continue
            pending.append(path)
            if len(pending) >= self.memes_batch_per_tick:
                break
        if not pending:
            return 0
        creds = self._vision_credentials()
        if creds is None:
            return 0
        tagged = 0
        with httpx.Client() as client:
            for path in pending:
                if self._tag_with_retry(client, path, creds, self.meanings_dir):
                    tagged += 1
        return tagged
