from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

import httpx

from app.core.image_optimizer import optimize_image_bytes


logger = logging.getLogger(__name__)


async def refresh_image_urls(
    raw_payload: dict[str, Any],
    call_api: Callable[[str, dict[str, Any]], Awaitable[Any]],
    *,
    timeout_seconds: float = 8.0,
) -> int:
    """入库前把 napcat 给的图片链接换新鲜：CDN 链接带短时效 rkey，到手常已
    死（2026-10-29 起实测全 400，图片缓存/表情收藏因此断流）。

    经 OneBot ``get_image`` 按 file_id 换新链接；napcat 若回本地路径（bot
    对 napcat 的 ntqq/cache 目录有只读挂载，路径一致可直接读）也一并带上。
    每条消息至多换 4 张；失败自吞（换不到就走原链接，行为与从前一致）。
    返回成功刷新的张数。
    """

    message = raw_payload.get("message") if isinstance(raw_payload, dict) else None
    if not isinstance(message, list):
        return 0
    refreshed = 0
    for item in message[:8]:
        if not (isinstance(item, dict) and item.get("type") == "image"):
            continue
        data = item.setdefault("data", {})
        if not isinstance(data, dict):
            continue
        if str(data.get("local_path") or "").strip():
            continue  # napcat 已给本地文件，不用换
        file_id = str(data.get("file") or "").strip()
        if not file_id:
            continue
        try:
            response = await asyncio.wait_for(
                call_api("get_image", {"file": file_id}), timeout=timeout_seconds)
        except Exception:
            continue
        payload = response.get("data") or {} if isinstance(response, dict) else {}
        if not isinstance(payload, dict):
            continue
        fresh_url = str(payload.get("url") or "").strip()
        if fresh_url.startswith("http"):
            data["url"] = fresh_url
            refreshed += 1
        napcat_file = str(payload.get("file") or "").strip()
        if napcat_file.startswith("/"):
            # napcat 本地绝对路径：bot 以相同路径只读挂载，下载兜底可直接读
            try:
                if Path(napcat_file).is_file():
                    data["local_path"] = napcat_file
                    refreshed += 1
            except OSError:
                pass
    if refreshed:
        logger.debug("image_url_refreshed count=%s", refreshed)
    return refreshed


def _guess_media_type(*, image_url: str, response: httpx.Response) -> str | None:
    content_type = response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    if content_type.startswith("image/"):
        return content_type
    guessed_type, _encoding = mimetypes.guess_type(image_url)
    if guessed_type and guessed_type.startswith("image/"):
        return guessed_type
    return None


def _guess_suffix(*, media_type: str, file_id: str | None, image_url: str) -> str:
    guessed_suffix = mimetypes.guess_extension(media_type)
    if guessed_suffix:
        return guessed_suffix
    if file_id:
        file_suffix = Path(file_id).suffix
        if file_suffix:
            return file_suffix
    url_suffix = Path(image_url).suffix
    if url_suffix:
        return url_suffix
    return ".img"


def cache_images_in_raw_payload(
    raw_payload: dict[str, Any],
    *,
    cache_dir: Path,
    http_client: httpx.Client | None = None,
) -> None:
    message = raw_payload.get("message")
    if not isinstance(message, list):
        return

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=15.0, follow_redirects=True)
    target_dir = cache_dir / str(raw_payload.get("group_id", "unknown"))
    target_dir.mkdir(parents=True, exist_ok=True)
    message_id = str(raw_payload.get("message_id", "message"))
    image_items = [
        (index, item)
        for index, item in enumerate(message)
        if isinstance(item, dict) and item.get("type") == "image"
    ]

    try:
        if not image_items:
            return
        with ThreadPoolExecutor(
            max_workers=min(4, len(image_items)),
            thread_name_prefix="image-cache",
        ) as pool:
            futures = [
                pool.submit(
                    _download_and_cache_one,
                    index=index,
                    item=item,
                    target_dir=target_dir,
                    message_id=message_id,
                    http_client=client,
                )
                for index, item in image_items
            ]
            for future in futures:
                future.result()
    finally:
        if owns_client:
            client.close()


def _download_and_cache_one(
    *,
    index: int,
    item: dict[str, Any],
    target_dir: Path,
    message_id: str,
    http_client: httpx.Client,
) -> None:
    if not isinstance(item, dict) or item.get("type") != "image":
        return
    data = item.setdefault("data", {})
    if not isinstance(data, dict):
        return

    local_path_text = str(data.get("local_path", "")).strip()
    if local_path_text and Path(local_path_text).exists():
        return

    image_url = str(data.get("url", "")).strip()
    if not image_url:
        return

    try:
        response = http_client.get(image_url)
        response.raise_for_status()
    except Exception:
        logger.exception("image_cache_download_failed message_id=%s url=%s", message_id, image_url)
        return

    media_type = _guess_media_type(image_url=image_url, response=response)
    if media_type is None or not response.content:
        logger.warning(
            "image_cache_invalid_content message_id=%s url=%s content_type=%s size=%s",
            message_id,
            image_url,
            response.headers.get("content-type"),
            len(response.content),
        )
        return

    content = response.content
    optimized = optimize_image_bytes(content, media_type=media_type)
    if optimized is not None:
        content, media_type = optimized

    suffix = _guess_suffix(
        media_type=media_type,
        file_id=str(data.get("file", "")).strip() or None,
        image_url=image_url,
    )
    cached_path = target_dir / f"{message_id}-{index}{suffix}"
    cached_path.write_bytes(content)
    data["local_path"] = str(cached_path)
