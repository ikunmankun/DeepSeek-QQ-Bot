"""以图搜图单元：三个来源解析、链路降级、构建器隐藏开关。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.reverse_image_search import (
    ReverseImageSearcher, resolve_provider_chain, search_ascii2d, search_saucenao,
    search_trace_moe,
)
from app.runtime.providers import build_reverse_image_searcher


class _FakeResponse:
    def __init__(self, payload=None, text="", url="https://ascii2d.net/search/color/abc123", status_code=200):
        self._payload = payload
        self.text = text
        self.url = url
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


def _image(tmp_path):
    image = tmp_path / "sticker.jpg"
    image.write_bytes(b"fake-bytes")
    return image


def test_saucenao_parses_match(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris
    captured = {}

    def fake_post(url, data=None, files=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["token"] = data["api_token"]
        captured["trust_env"] = kwargs.get("trust_env")
        return _FakeResponse(payload={"results": [
            {"header": {"similarity": "92.17", "thumbnail": "https://tn/1.jpg"},
             "data": {"title": "蓝鲸娘", "member_name": "画师甲",
                      "ext_urls": [{"url": "https://www.pixiv.net/artworks/123"}]}},
            {"header": {"similarity": "70.00"}, "data": {"title": "其它"}},
        ]})

    monkeypatch.setattr(ris.httpx, "post", fake_post)
    hits = search_saucenao(_image(tmp_path), saucenao_token="tok-1")
    assert captured["url"] == "https://saucenao.com/search.php"
    assert captured["token"] == "tok-1"
    assert captured["trust_env"] is False
    assert len(hits) == 2
    assert hits[0].provider == "saucenao"
    assert hits[0].title == "蓝鲸娘"
    assert hits[0].author == "画师甲"
    assert hits[0].source_url == "https://www.pixiv.net/artworks/123"
    assert hits[0].similarity == pytest.approx(92.17)


def test_saucenao_without_token_makes_no_request(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris

    def fail_post(*args, **kwargs):
        raise AssertionError("should not call httpx")

    monkeypatch.setattr(ris.httpx, "post", fail_post)
    assert search_saucenao(_image(tmp_path), saucenao_token="") == []


def test_saucenao_http_error_raises_oserror(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris
    monkeypatch.setattr(ris.httpx, "post", lambda *a, **k: _FakeResponse(status_code=403))
    with pytest.raises(Exception):
        search_saucenao(_image(tmp_path), saucenao_token="tok")


_ASCII2D_PAGE = """
<html><body>
<div class="row item-box py-2 border-bottom mx-1">
  <div class="cols-md-3"><img src="/thumb/1.jpg"></div>
  <div class="cols-md-9 detail-box">
    <h6><a href="https://www.pixiv.net/artworks/456" target="_blank">蓝色鲸鱼娘立绘</a>
      <small>736x1104 JPEG 234.5KB</small></h6>
    <hr>
    <small><a href="https://www.pixiv.net/users/789">画师乙</a> | <a href="https://www.pixiv.net/artworks/456">pixiv</a></small>
  </div>
</div>
<div class="row item-box py-2 border-bottom mx-1">
  <div class="cols-md-9 detail-box">
    <h6><a href="https://ascii2d.net/redir/9" target="_blank">第二条</a><small>800x600 PNG</small></h6>
  </div>
</div>
</body></html>
"""


def test_ascii2d_parses_color_page_with_bovw_entry(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris
    captured = {}

    def fake_post(url, files=None, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["follow"] = kwargs.get("follow_redirects")
        return _FakeResponse(text=_ASCII2D_PAGE, url="https://ascii2d.net/search/color/abc123def")

    monkeypatch.setattr(ris.httpx, "post", fake_post)
    hits = search_ascii2d(_image(tmp_path))
    assert captured["url"] == "https://ascii2d.net/search/multi"
    assert captured["follow"] is True
    assert len(hits) == 2
    assert hits[0].provider == "ascii2d"
    assert hits[0].title == "蓝色鲸鱼娘立绘"
    assert hits[0].source_url == "https://www.pixiv.net/artworks/456"
    assert hits[0].author == "画师乙"
    assert "736x1104" in hits[0].detail
    assert hits[0].series_url == "https://ascii2d.net/search/bovw/abc123def"
    # ascii2d 自身跳转链接不算出处，但结构里没有外链时要兜底取第一个锚点
    assert hits[1].source_url.startswith("https://")


def test_ascii2d_structure_change_returns_empty(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris
    monkeypatch.setattr(ris.httpx, "post",
                        lambda *a, **k: _FakeResponse(text="<html>captcha page</html>",
                                                      url="https://ascii2d.net/search/color/xx"))
    assert search_ascii2d(_image(tmp_path)) == []


def test_trace_moe_parses_anime_scene(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris
    monkeypatch.setattr(ris.httpx, "post", lambda *a, **k: _FakeResponse(payload={"result": [
        {"anilist": {"id": 21, "title": {"native": "海贼王", "english": "One Piece"},
                     "synonyms": ["OP"]}, "episode": 4, "from": 755.2, "similarity": 0.9234},
        {"anilist": {"id": 22, "title": {"native": "限制级"}, "isAdult": True},
         "episode": 1, "from": 10.0, "similarity": 0.9},
    ]}))
    hits = search_trace_moe(_image(tmp_path))
    assert len(hits) == 1
    assert hits[0].provider == "trace_moe"
    assert hits[0].title == "海贼王"
    assert hits[0].source_url == "https://anilist.co/anime/21"
    assert hits[0].similarity == pytest.approx(92.34)
    assert "第4集" in hits[0].detail


def test_chain_first_provider_failure_falls_through(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris

    def fail_saucenao(image, *, saucenao_token="", timeout=15.0, **kwargs):
        raise OSError("saucenao_error:limit")

    # _PROVIDERS 是导入期固化的函数表，必须改表而不是改模块属性
    monkeypatch.setitem(ris._PROVIDERS, "saucenao", fail_saucenao)
    monkeypatch.setitem(ris._PROVIDERS, "ascii2d",
                        lambda image, *, saucenao_token="", timeout=15.0, **kwargs:
                        [ris.ReverseImageHit(provider="ascii2d", title="命中")])
    searcher = ReverseImageSearcher(providers=("saucenao", "ascii2d"), saucenao_token="tok")
    hits = searcher.search(_image(tmp_path))
    assert len(hits) == 1 and hits[0].provider == "ascii2d"


def test_chain_all_empty_returns_empty(tmp_path):
    searcher = ReverseImageSearcher(providers=("ascii2d", "trace_moe"))
    assert searcher.search(_image(tmp_path)) == []


def test_resolve_provider_chain_rules():
    assert resolve_provider_chain("") == ("ascii2d", "trace_moe")
    assert resolve_provider_chain("auto") == ("ascii2d", "trace_moe")
    assert resolve_provider_chain("off") == ()
    assert resolve_provider_chain("OFF") == ()
    assert resolve_provider_chain("saucenao,ascii2d", saucenao_token="") == ("ascii2d",)
    assert resolve_provider_chain("saucenao,ascii2d", saucenao_token="t") == ("saucenao", "ascii2d")
    assert resolve_provider_chain("trace.moe, 未知源, ascii2d") == ("trace_moe", "ascii2d")


def test_builder_hides_feature_when_off():
    settings = SimpleNamespace(reverse_image_providers="off", reverse_image_saucenao_token="",
                               reverse_image_timeout_seconds=15.0)
    assert build_reverse_image_searcher(settings=settings) is None


def test_vpn_proxy_matches_blocked_hosts_only():
    from app.core.proxy_routing import resolve_vpn_hosts, vpn_proxy_for_url
    hosts = resolve_vpn_hosts("saucenao.com, trace.moe，.pximg.net")
    assert hosts == ("saucenao.com", "trace.moe", "pximg.net")
    proxy = "socks5://1.2.3.4:1080"
    assert vpn_proxy_for_url("https://saucenao.com/search.php", proxy_url=proxy, hosts=hosts) == proxy
    assert vpn_proxy_for_url("https://i.pximg.net/a.jpg", proxy_url=proxy, hosts=hosts) == proxy
    assert vpn_proxy_for_url("https://notsaucenao.com/", proxy_url=proxy, hosts=hosts) == ""
    assert vpn_proxy_for_url("https://api.audd.io/", proxy_url=proxy, hosts=hosts) == ""
    assert vpn_proxy_for_url("https://trace.moe/", proxy_url="", hosts=hosts) == ""
    assert resolve_vpn_hosts("") == ()


def test_searcher_passes_vpn_proxy_to_matched_provider(tmp_path, monkeypatch):
    import app.core.reverse_image_search as ris
    captured = {}

    def fake_post(url, files=None, timeout=None, trust_env=None, proxy=None, **kwargs):
        captured["url"], captured["proxy"] = url, proxy
        return _FakeResponse(text="<html>no items</html>", url="https://ascii2d.net/search/color/abc")

    monkeypatch.setattr(ris.httpx, "post", fake_post)
    searcher = ReverseImageSearcher(providers=("ascii2d",),
        vpn_proxy_url="socks5://1.2.3.4:1080", vpn_hosts=("ascii2d.net",))
    assert searcher.search(_image(tmp_path)) == []
    assert captured["proxy"] == "socks5://1.2.3.4:1080"

    direct = ReverseImageSearcher(providers=("ascii2d",))
    assert direct.search(_image(tmp_path)) == []
    assert captured["proxy"] is None


def test_builder_keeps_chain_and_drops_tokenless_saucenao():
    settings = SimpleNamespace(reverse_image_providers="saucenao,ascii2d",
                               reverse_image_saucenao_token="", reverse_image_timeout_seconds=15.0)
    searcher = build_reverse_image_searcher(settings=settings)
    assert searcher is not None and searcher.providers == ("ascii2d",)
    settings = SimpleNamespace(reverse_image_providers="saucenao,ascii2d",
                               reverse_image_saucenao_token="tok", reverse_image_timeout_seconds=10.0)
    searcher = build_reverse_image_searcher(settings=settings)
    assert searcher.providers == ("saucenao", "ascii2d")
    assert searcher.timeout_seconds == 10.0
