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

#: 坏输出模式：模型给汉字插递增序号（"反(1)差(2)_催(4)更(5)"）。出现两处
#: 及以上就整体剥掉 (数字)，避免标签和含义被编号污染。
_CHAR_INDEX_PATTERN = re.compile(r"\(\d+\)")


def _strip_char_index_noise(text: str) -> str:
    if not text:
        return text
    found = _CHAR_INDEX_PATTERN.findall(text)
    if len(found) < 2:
        return text
    return _CHAR_INDEX_PATTERN.sub("", text).strip("。．.！!？?~～ ")

#: 视觉打标的输出预算。推理模型（如 deepseek-flash）会先烧掉一段
#: reasoning 再吐 content：200 会全部耗在思考里（finish=length、
#: content 恒空），实测 ~1400 才能稳定拿到完整"标签||含义"。
TAG_MAX_TOKENS = 1400

#: 空含义 sidecar 的重打次数上限：防"每轮空烧视觉调用"的死循环复发。
#: 达到上限后这张图就保持占位，等模型/网络真正恢复再人工清零。
EMPTY_MEANING_RETRY_CAP = 3


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
                "max_tokens": TAG_MAX_TOKENS,
                "temperature": 0.3,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("choices", [{}])[0].get("message", {})
        raw = str(message.get("content") or "").strip()
        if not raw:
            # 推理模型偶发 content 空但 reasoning 尾部已给出结论：只认
            # "标签_标签||含义"格式（标签段必须带下划线），否则宁失败重试。
            reasoning = str(message.get("reasoning_content") or "").strip()
            tail = reasoning[-200:].replace("\n", " ").strip()
            candidates = re.findall(
                r"([^\s|，。：:]{2,30}_[^\s|，。：:]{1,28})\|\|", tail)
            if candidates:
                raw = (candidates[-1] + _MEANING_SPLIT
                       + tail.rsplit(_MEANING_SPLIT, 1)[-1].strip())
        meaning = ""
        if _MEANING_SPLIT in raw:
            raw, _, meaning_raw = raw.partition(_MEANING_SPLIT)
            meaning = _strip_char_index_noise(
                re.sub(r"[\r\n]", "", meaning_raw).strip("。．.！!？?~～ "))[:40]
        tags = _TAG_SPLIT.split(_strip_char_index_noise(raw))
        cleaned = []
        for part in tags[:3]:
            tag = _strip_char_index_noise(
                re.sub(r'[\\/:*?"<>|\r\n]', "", part).strip("。．.！!？?~～ ")[:8])
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
                 memes_batch_per_tick: int = 4,
                 client_factory: Any = None) -> None:
        self.learner = learner
        self.settings = settings
        self.interval_seconds = float(interval_seconds)
        self.memes_dir = Path(memes_dir) if memes_dir else None
        self.meanings_dir = Path(meme_meanings_dir) if meme_meanings_dir else None
        self.memes_batch_per_tick = int(memes_batch_per_tick)
        # 测试可注入 httpx 客户端工厂；生产恒为 httpx.Client。
        self._client_factory = client_factory or httpx.Client

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

    def _tag_with_retry(self, client: httpx.Client, path: Path, creds,
                        sidecar_dir: Path | None) -> Exception | None:
        """两次尝试打标；成功返回 None，失败返回最后一次的异常。"""

        base_url, api_key, model = creds
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                self.learner.tag_one(path, base_url=base_url, api_key=api_key,
                                     model=model, client=client, sidecar_dir=sidecar_dir)
                return None
            except Exception as exc:
                last = exc
                if attempt == 2:
                    logger.warning("sticker_tag_failed file=%s error=%s",
                                   path.name, type(exc).__name__)
        return last

    def _write_sidecar(self, sidecar_dir: Path | None, path: Path,
                       *, tags: list[str], meaning: str = "",
                       retry_count: int | None = None) -> None:
        """落一个标签文件；retry_count 非 None 时记录空含义重打次数。"""

        target = (Path(sidecar_dir) / (path.stem + ".json")) if sidecar_dir is not None \
            else path.with_suffix(".json")
        payload = {"tags": "_".join(tags), "meaning": meaning}
        if retry_count is not None:
            payload["empty_meaning_retries"] = retry_count
        target.write_text(json.dumps(payload, ensure_ascii=False),
                          encoding="utf-8")

    def _sidecar_needs_meaning(self, sidecar: Path) -> bool:
        """sidecar 存在但 meaning 为空、且还没耗尽重打次数。"""

        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True  # 损坏的 sidecar 视同占位，重打覆盖
        if str(data.get("meaning") or "").strip():
            return False
        return int(data.get("empty_meaning_retries") or 0) < EMPTY_MEANING_RETRY_CAP

    def _fallback_tags_from_filename(self, path: Path) -> list[str]:
        """memes 编号库的标签就写在文件名里（041_别急_装无辜_画饼.webp），兜底解析。"""

        stem = re.sub(r"^\d+", "", path.stem)
        cleaned: list[str] = []
        for part in stem.split("_"):
            tag = re.sub(r'[\\/:*?"<>|\r\n]', "", part).strip("。．.！!？?~～ ")[:8]
            if tag and tag not in cleaned:
                cleaned.append(tag)
        return cleaned[:3]

    def _tick_learned(self) -> int:
        pending = list(self.learner.pending_tagging()[:TAG_BATCH_PER_TICK])
        # 已有 sidecar 但 meaning 为空的占位也重打（同一重试上限防死循环）。
        if self.learner.learned_dir.exists():
            for path in sorted(self.learner.learned_dir.iterdir()):
                if len(pending) >= TAG_BATCH_PER_TICK:
                    break
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                sidecar = path.with_suffix(".json")
                if sidecar.exists() and self._sidecar_needs_meaning(sidecar):
                    pending.append(path)
        if not pending:
            return 0
        creds = self._vision_credentials()
        if creds is None:
            return 0
        tagged = 0
        with self._client_factory() as client:
            for path in pending:
                failure = self._tag_with_retry(client, path, creds, None)
                if failure is None:
                    tagged += 1
                elif isinstance(failure, ValueError):
                    # 确定性失败（模型输出解析不出标签）：落空标签占位并计数，
                    # 到 EMPTY_MEANING_RETRY_CAP 为止，不让同一张图每轮空烧。
                    sidecar = path.with_suffix(".json")
                    retries = 0
                    try:
                        retries = int(json.loads(sidecar.read_text(encoding="utf-8"))
                                      .get("empty_meaning_retries") or 0)
                    except (OSError, ValueError):
                        pass
                    self._write_sidecar(None, path, tags=[],
                                        retry_count=retries + 1)
        return tagged

    def _tick_memes(self) -> int:
        """给 memes 编号库补梗含义，两档队列：没含义文件的优先，已有文件但
        meaning 为空的占位也重打（带重试上限）。每轮小批量，不 burst。"""
        if self.memes_dir is None or self.meanings_dir is None:
            return 0
        if not self.memes_dir.exists():
            return 0
        pending = []
        for path in sorted(self.memes_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sidecar = self.meanings_dir / (path.stem + ".json")
            if not sidecar.exists():
                pending.append(path)
            elif self._sidecar_needs_meaning(sidecar):
                pending.append(path)
            if len(pending) >= self.memes_batch_per_tick:
                break
        if not pending:
            return 0
        creds = self._vision_credentials()
        if creds is None:
            return 0
        tagged = 0
        with self._client_factory() as client:
            for path in pending:
                failure = self._tag_with_retry(client, path, creds, self.meanings_dir)
                if failure is None:
                    tagged += 1
                    continue
                if not isinstance(failure, ValueError):
                    continue  # 网络/服务端错误：不动 sidecar，下轮自然重试
                # 确定性失败：已有 sidecar 的记录重打次数（到上限为止）；
                # 没有 sidecar 的落文件名兜底标签占位。
                sidecar = self.meanings_dir / (path.stem + ".json")
                if sidecar.exists():
                    try:
                        data = json.loads(sidecar.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        data = {}
                    self._write_sidecar(self.meanings_dir, path,
                                        tags=str(data.get("tags") or "").split("_"),
                                        retry_count=int(data.get("empty_meaning_retries") or 0) + 1)
                else:
                    self._write_sidecar(self.meanings_dir, path,
                                        tags=self._fallback_tags_from_filename(path),
                                        retry_count=1)
        return tagged
