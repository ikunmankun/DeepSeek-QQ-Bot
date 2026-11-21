import io
from pathlib import Path

import httpx
from PIL import Image

from app.core.image_cache import cache_images_in_raw_payload


def test_cache_images_in_raw_payload_downloads_image_and_persists_local_path(tmp_path) -> None:
    payload = {
        "message_id": "image-msg-1",
        "group_id": 10001,
        "message": [
            {"type": "text", "data": {"text": "look"}},
            {
                "type": "image",
                "data": {
                    "file": "cat.png",
                    "url": "https://img.example.test/cat.png",
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://img.example.test/cat.png"
        return httpx.Response(
            200,
            content=b"png-bytes",
            headers={"content-type": "image/png"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    cache_images_in_raw_payload(payload, cache_dir=tmp_path, http_client=client)

    image_data = payload["message"][1]["data"]
    cached_path = Path(image_data["local_path"])
    assert cached_path.exists()
    assert cached_path.read_bytes() == b"png-bytes"
    assert cached_path.parent == tmp_path / "10001"
    assert cached_path.suffix == ".png"


def test_cache_images_in_raw_payload_downloads_multiple_images_in_parallel_and_compresses(tmp_path) -> None:
    image = Image.effect_noise((1600, 1200), 100).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    payload = {
        "message_id": "multi-image-msg",
        "group_id": 10001,
        "message": [
            {
                "type": "image",
                "data": {
                    "file": "one.png",
                    "url": "https://img.example.test/one.png",
                },
            },
            {
                "type": "image",
                "data": {
                    "file": "two.png",
                    "url": "https://img.example.test/two.png",
                },
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            content=png_bytes,
            headers={"content-type": "image/png"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    cache_images_in_raw_payload(payload, cache_dir=tmp_path, http_client=client)

    for image_data in (payload["message"][0]["data"], payload["message"][1]["data"]):
        cached_path = Path(image_data["local_path"])
        assert cached_path.exists()
        assert cached_path.suffix == ".jpg"
        optimized = cached_path.read_bytes()
        assert optimized != png_bytes
        decoded = Image.open(io.BytesIO(optimized))
        assert max(decoded.size) <= 1280


def test_refresh_image_urls_swaps_dead_url(tmp_path) -> None:
    """napcat 给的 CDN 链到手即死（rkey 过期）：入库前经 get_image 换新鲜链接。"""
    import asyncio
    from app.core.image_cache import refresh_image_urls

    payload = {"message": [{"type": "image", "data": {
        "file": "ABC.webp", "url": "https://multimedia.nt.qq.com.cn/dead?rkey=old"}}]}
    calls = []

    async def fake_call_api(action, params):
        calls.append((action, params))
        return {"data": {"url": "https://multimedia.nt.qq.com.cn/fresh?rkey=new",
                         "file": "ABC.webp"}}

    refreshed = asyncio.run(refresh_image_urls(payload, fake_call_api))
    assert refreshed == 1
    assert calls == [("get_image", {"file": "ABC.webp"})]
    assert payload["message"][0]["data"]["url"].endswith("rkey=new")


def test_refresh_image_urls_uses_napcat_local_file(tmp_path) -> None:
    """get_image 回 napcat 本地路径且文件可读（ro 挂载路径一致）：直接带 local_path。"""
    import asyncio
    from app.core.image_cache import refresh_image_urls

    napcat_file = tmp_path / "pic.webp"
    napcat_file.write_bytes(b"fake")
    payload = {"message": [{"type": "image", "data": {"file": "ABC.webp"}}]}

    async def fake_call_api(action, params):
        return {"data": {"file": str(napcat_file), "url": ""}}

    refreshed = asyncio.run(refresh_image_urls(payload, fake_call_api))
    assert refreshed == 1
    assert payload["message"][0]["data"]["local_path"] == str(napcat_file)


def test_refresh_image_urls_skips_local_and_failures(tmp_path) -> None:
    """已有本地路径的跳过；API 失败自吞不抛。"""
    import asyncio
    from app.core.image_cache import refresh_image_urls

    local = tmp_path / "already.webp"
    local.write_bytes(b"x")
    payload = {"message": [
        {"type": "image", "data": {"file": "A.webp", "local_path": str(local)}},
        {"type": "image", "data": {"file": "B.webp"}},
    ]}

    async def failing_call_api(action, params):
        raise RuntimeError("ws down")

    refreshed = asyncio.run(refresh_image_urls(payload, failing_call_api))
    assert refreshed == 0
    # 已有 local_path 的没被动过，也没有凭空造出 url
    assert "url" not in payload["message"][0]["data"]
    assert payload["message"][0]["data"]["local_path"] == str(local)
