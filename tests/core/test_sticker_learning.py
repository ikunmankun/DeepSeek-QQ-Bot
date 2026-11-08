"""偷表情包：收藏去重/限额、打标落盘、pick_sticker 学习库匹配。"""

import json
from pathlib import Path
from types import SimpleNamespace

from app.agent.local_stickers import build_sticker_tools
from app.core.sticker_learning import StickerLearner
from PIL import Image


def _png_bytes(color, size=(10, 10)):
    import io

    body = io.BytesIO()
    Image.new("RGB", size, color).save(body, format="PNG")
    return body.getvalue()


def _event(text="", images=(), user_id=900000103, group_id=900000001):
    return SimpleNamespace(user_id=user_id, group_id=group_id, plain_text=text,
                           mentioned_bot=False, reply_to_msg_id=None,
                           raw_payload={}, timestamp=None, images=list(images))


class _FakeImage:
    def __init__(self, local_path):
        self.local_path = local_path
        self.url = ""


def test_collect_dedupes_by_content_and_skips_bot(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(_png_bytes("crimson"))
    learner = StickerLearner(tmp_path / "data")

    bot_event = _event(images=[_FakeImage(str(src))], user_id=900000101)
    assert learner.collect_from_event(bot_event, bot_qq=900000101) == 0

    event = _event(images=[_FakeImage(str(src)), _FakeImage(str(src))])
    assert learner.collect_from_event(event, bot_qq=900000101) == 1
    assert learner.collect_from_event(event, bot_qq=900000101) == 0
    assert len(list(learner.learned_dir.iterdir())) == 1


def test_collect_caps_library_size_and_skips_oversize(tmp_path):
    learner = StickerLearner(tmp_path / "data")
    from app.core.sticker_learning import MAX_LEARNED_FILES

    learner.learned_dir.mkdir(parents=True)
    for i in range(MAX_LEARNED_FILES):
        (learner.learned_dir / f"{i:04d}.png").write_bytes(b"x")
    src = tmp_path / "a.png"
    src.write_bytes(_png_bytes("navy"))
    assert learner.collect_from_event(_event(images=[_FakeImage(str(src))]),
                                      bot_qq=1) == 0

    big = tmp_path / "big.png"
    big.write_bytes(b"y" * (9 * 1024 * 1024))
    (learner.learned_dir / "0000.png").unlink()
    assert learner.collect_from_event(_event(images=[_FakeImage(str(big))]),
                                      bot_qq=1) == 0


def test_tag_one_writes_sidecar(tmp_path):
    import httpx

    learner = StickerLearner(tmp_path / "data")
    learner.learned_dir.mkdir(parents=True)
    img = learner.learned_dir / "abc123.png"
    img.write_bytes(_png_bytes("seagreen"))

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "无语、看戏，绷不住"}}]}

    class FakeClient:
        def post(self, *args, **kwargs):
            return FakeResponse()

    tags = learner.tag_one(img, base_url="http://x", api_key="k", model="m",
                           client=FakeClient())
    assert tags == "无语_看戏_绷不住"
    assert json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))["tags"] == "无语_看戏_绷不住"
    assert learner.pending_tagging() == []
    httpx.Client  # 确认依赖在


def test_pick_sticker_matches_learned_sidecar_tags(tmp_path):
    import asyncio

    main = tmp_path / "main"
    main.mkdir()
    (main / "037_无语.webp").write_bytes(_png_bytes("gold"))
    learned = tmp_path / "learned"
    learned.mkdir()
    (learned / "abc123def456.png").write_bytes(_png_bytes("khaki"))
    (learned / "abc123def456.json").write_text(
        json.dumps({"tags": "看戏_吃瓜"}), encoding="utf-8")

    class Guard:
        def require_valid(self):
            return None

    class FakeRunner:
        async def run(self, fn, *args):
            return fn(*args)

    class Store:
        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

        def add(self, staging, origin):
            target = self.root / Path(staging).name
            target.write_bytes(Path(staging).read_bytes())
            return "art-1"

        def get(self, artifact_id):
            return str(self.root / "artifact.png")

    store = Store(tmp_path / "store")
    tool = build_sticker_tools(runner=FakeRunner(), guard=Guard(), store=store,
                               directory=main, extra_directory=learned)[0]
    result = asyncio.run(tool.handler({"keyword": "吃瓜"}))
    assert result.status == "ok"
    assert result.data["meme_id"] == "abc123def456"
    assert result.data["matched"] == 1


def test_tag_one_stores_meaning_after_double_pipe(tmp_path):
    learner = StickerLearner(tmp_path / "data")
    learner.learned_dir.mkdir(parents=True)
    img = learner.learned_dir / "def456.png"
    img.write_bytes(_png_bytes("olive"))

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {
                "content": "看戏_吃瓜_乐||两只熊猫头对视，冷眼看热闹的梗"}}]}

    class FakeClient:
        def post(self, *args, **kwargs):
            return FakeResponse()

    tags = learner.tag_one(img, base_url="http://x", api_key="k", model="m",
                           client=FakeClient())
    assert tags == "看戏_吃瓜_乐"
    sidecar = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["tags"] == "看戏_吃瓜_乐"
    assert sidecar["meaning"] == "两只熊猫头对视，冷眼看热闹的梗"
    assert learner.load_meaning(img) == "两只熊猫头对视，冷眼看热闹的梗"


def test_pick_sticker_keyword_also_matches_meaning(tmp_path):
    import asyncio

    main = tmp_path / "main"
    main.mkdir()
    (main / "037_无语.webp").write_bytes(_png_bytes("gold"))
    learned = tmp_path / "learned"
    learned.mkdir()
    (learned / "facepalm987.png").write_bytes(_png_bytes("khaki"))
    (learned / "facepalm987.json").write_text(
        json.dumps({"tags": "无语_看戏", "meaning": "熊猫头捂脸，草泥马狂奔的梗"}),
        encoding="utf-8")

    class Guard:
        def require_valid(self):
            return None

    class FakeRunner:
        async def run(self, fn, *args):
            return fn(*args)

    class Store:
        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

        def add(self, staging, origin):
            target = self.root / Path(staging).name
            target.write_bytes(Path(staging).read_bytes())
            return "art-1"

        def get(self, artifact_id):
            return str(self.root / "artifact.png")

    store = Store(tmp_path / "store")
    tool = build_sticker_tools(runner=FakeRunner(), guard=Guard(), store=store,
                               directory=main, extra_directory=learned)[0]
    result = asyncio.run(tool.handler({"keyword": "捂脸"}))
    assert result.status == "ok"
    assert result.data["meme_id"] == "facepalm987"


def test_find_learned_meme_tool_matches_tags_and_meaning(tmp_path):
    from app.agent.plugins.builtin import _build_learned_meme_tool

    learner = StickerLearner(tmp_path / "data")
    learner.learned_dir.mkdir(parents=True)
    (learner.learned_dir / "aaa111.png").write_bytes(_png_bytes("teal"))
    (learner.learned_dir / "aaa111.json").write_text(
        json.dumps({"tags": "狂喜_得意", "meaning": "鲸鱼张嘴吃token的梗"}),
        encoding="utf-8")
    (learner.learned_dir / "bbb222.png").write_bytes(_png_bytes("salmon"))
    (learner.learned_dir / "bbb222.json").write_text(
        json.dumps({"tags": "无语", "meaning": ""}), encoding="utf-8")

    tool = _build_learned_meme_tool(learner)
    hit = tool.handler({"keyword": "token"})
    assert hit.status == "ok"
    assert [m["file"] for m in hit.data["matches"]] == ["aaa111"]
    assert hit.data["matches"][0]["meme_id"] == "aaa111"

    miss = tool.handler({"keyword": "不存在的梗"})
    assert miss.status == "ok" and miss.data["matches"] == []

    empty = tool.handler({})
    assert empty.status == "blocked" and empty.data["error"] == "keyword_required"
