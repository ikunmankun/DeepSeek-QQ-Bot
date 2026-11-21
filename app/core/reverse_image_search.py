"""以图搜图：来源与同系列检索。

路线（按配置顺序取第一个出结果的，和识曲的兜底链同一思路）：
* ascii2d（默认）：multipart 上传 → 302 到 color 页；解析 HTML 结果，
  bovw 同 hash 页是"以图找相似/同系列"入口。无鉴权、无官方 API，结构变了就返回空。
* trace_moe（默认）：动漫截图专精，能给出番名/集数/时间点；匿名有频控。
* saucenao：动漫图鉴源最全（pixiv/danbooru 画师+ID），需 REVERSE_IMAGE_SAUCENAO_TOKEN，
  免费档限额，凭证为空时自动从链里剔除。

全部 httpx 同步 + trust_env=False 直连：容器代理环境变量对这些站是死路
（与 music_id.py 同一课，2026-11 实测）。接口错误抛 OSError，无结果返回 []。
"""
from __future__ import annotations

import html as _html
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from app.core.proxy_routing import vpn_proxy_for_url

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS = "ascii2d,trace_moe"
_SAUCE_URL = "https://saucenao.com/search.php"
_ASCII2D_URL = "https://ascii2d.net/search/multi"
_TRACE_URL = "https://trace.moe/api/search"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QQBot-reverse-image/1.0"
_MAX_HITS_PER_PROVIDER = 4
_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".webp": "image/webp", ".gif": "image/gif"}


@dataclass(slots=True)
class ReverseImageHit:
    provider: str
    title: str = ""
    author: str = ""
    source_url: str = ""     # 最可信的出处页
    similarity: float = 0.0  # 0-100
    detail: str = ""         # 集数/时间点/尺寸等补充
    series_url: str = ""     # 同系列/更多相似结果入口
    thumbnail_url: str = ""


def _mime_of(image: Path) -> str:
    return _MIME_BY_SUFFIX.get(image.suffix.lower(), "image/jpeg")


def _clean(text: str) -> str:
    return _html.unescape(str(text or "")).strip()


@dataclass(frozen=True, slots=True)
class ReverseImageSearcher:
    providers: tuple[str, ...]
    saucenao_token: str = ""
    timeout_seconds: float = 15.0
    vpn_proxy_url: str = ""
    vpn_hosts: tuple[str, ...] = ()

    def search(self, image: Path) -> list[ReverseImageHit]:
        """按链路顺序搜，第一个出结果的源胜出；全失败/全空返回 []。"""
        failures: list[str] = []
        for name in self.providers:
            handler = _PROVIDERS.get(name)
            if handler is None:
                continue
            try:
                hits = handler(image, saucenao_token=self.saucenao_token,
                               timeout=self.timeout_seconds,
                               vpn_proxy_url=self.vpn_proxy_url,
                               vpn_hosts=self.vpn_hosts)
            except Exception as exc:  # 单源挂掉不拖垮整条链
                failures.append(f"{name}:{type(exc).__name__}")
                logger.info("reverse_image_provider_failed provider=%s error=%s",
                            name, str(exc)[:120])
                continue
            if hits:
                return hits
        if failures and len(failures) == len(self.providers):
            logger.info("reverse_image_all_providers_failed errors=%s", ",".join(failures))
        return []


def search_saucenao(image: Path, *, saucenao_token: str = "", timeout: float = 15.0,
                    vpn_proxy_url: str = "", vpn_hosts: tuple[str, ...] = ()) -> list[ReverseImageHit]:
    """SauceNAO JSON API。凭证/限额错误抛 OSError（HTTP 403/429）。"""
    if not saucenao_token.strip():
        return []
    with image.open("rb") as stream:
        response = httpx.post(
            _SAUCE_URL,
            data={"api_token": saucenao_token, "output_type": "2", "numres": "5"},
            files={"file": (image.name, stream, _mime_of(image))},
            timeout=timeout, trust_env=False,
            proxy=vpn_proxy_for_url(_SAUCE_URL, proxy_url=vpn_proxy_url, hosts=vpn_hosts) or None,
        )
    response.raise_for_status()
    results = response.json().get("results") or []
    hits: list[ReverseImageHit] = []
    for item in results[:_MAX_HITS_PER_PROVIDER]:
        header = item.get("header") or {}
        data = item.get("data") or {}
        creator = data.get("member_name") or data.get("creator") or ""
        if isinstance(creator, (list, tuple)):
            creator = "、".join(str(one) for one in creator if one)
        urls = [str(one.get("url") or "") for one in (data.get("ext_urls") or [])
                if isinstance(one, dict)]
        try:
            similarity = float(str(header.get("similarity") or 0))
        except ValueError:
            similarity = 0.0
        title = _clean(data.get("title") or data.get("source") or data.get("material") or "")
        if not title and not urls:
            continue
        hits.append(ReverseImageHit(
            provider="saucenao",
            title=title,
            author=_clean(creator),
            source_url=urls[0] if urls else "",
            similarity=similarity,
            detail=str(data.get("part") or data.get("chars") or "").strip()[:80],
            thumbnail_url=str(header.get("thumbnail") or "").strip(),
        ))
    return hits


_ANCHOR_RE = re.compile(r'<a href="(https?://[^"]+)"[^>]*>([^<]{0,160})</a>')
_SIZE_RE = re.compile(r'<small[^>]*>\s*(\d{2,5}x\d{2,5}[^<]{0,60})</small>')


def search_ascii2d(image: Path, *, saucenao_token: str = "", timeout: float = 15.0,
                   vpn_proxy_url: str = "", vpn_hosts: tuple[str, ...] = ()) -> list[ReverseImageHit]:
    """Ascii2D color 页解析。无官方 API：页面结构变化时返回 []，由链路落到下一源。"""
    with image.open("rb") as stream:
        response = httpx.post(
            _ASCII2D_URL,
            files={"file": (image.name, stream, _mime_of(image))},
            headers={"User-Agent": _UA},
            timeout=timeout, trust_env=False, follow_redirects=True,
            proxy=vpn_proxy_for_url(_ASCII2D_URL, proxy_url=vpn_proxy_url, hosts=vpn_hosts) or None,
        )
    response.raise_for_status()
    page = response.text
    color_url = str(response.url)
    hash_part = color_url.rstrip("/").rsplit("/", 1)[-1]
    bovw_url = f"https://ascii2d.net/search/bovw/{hash_part}" if "/search/color/" in color_url else ""
    blocks = re.split(r'class="row item-box', page)[1:]
    hits: list[ReverseImageHit] = []
    for block in blocks[:_MAX_HITS_PER_PROVIDER]:
        anchors = _ANCHOR_RE.findall(block[:2500])
        if not anchors:
            continue
        external = [(url, _clean(text)) for url, text in anchors
                    if "ascii2d.net" not in url]
        picked = external[0] if external else anchors[0]
        title, source_url = picked[1], picked[0]
        size_match = _SIZE_RE.search(block[:2500])
        author = ""
        user_match = re.search(r'<a href="https?://[^"]*(?:users|member)[^"]*"[^>]*>([^<]{1,80})</a>', block[:2500])
        if user_match:
            author = _clean(user_match.group(1))
        if not title:
            title = source_url.rsplit("/", 1)[-1][:80]
        hits.append(ReverseImageHit(
            provider="ascii2d",
            title=title[:120],
            author=author,
            source_url=source_url,
            detail=_clean(size_match.group(1)) if size_match else "",
            series_url=bovw_url,
        ))
    return hits


def search_trace_moe(image: Path, *, saucenao_token: str = "", timeout: float = 15.0,
                     vpn_proxy_url: str = "", vpn_hosts: tuple[str, ...] = ()) -> list[ReverseImageHit]:
    """trace.moe：动漫截图 → 番名/集数/时间点。匿名频控超限抛 OSError。"""
    with image.open("rb") as stream:
        response = httpx.post(
            _TRACE_URL,
            files={"image": (image.name, stream, _mime_of(image))},
            timeout=timeout, trust_env=False,
            proxy=vpn_proxy_for_url(_TRACE_URL, proxy_url=vpn_proxy_url, hosts=vpn_hosts) or None,
        )
    response.raise_for_status()
    results = response.json().get("result") or []
    hits: list[ReverseImageHit] = []
    for item in results[:_MAX_HITS_PER_PROVIDER]:
        anilist = item.get("anilist") or {}
        if anilist.get("isAdult"):
            continue
        titles = anilist.get("title") or {}
        title = _clean(titles.get("native") or titles.get("english") or titles.get("romaji") or "")
        if not title:
            continue
        try:
            similarity = float(item.get("similarity") or 0) * 100.0
        except (TypeError, ValueError):
            similarity = 0.0
        detail_parts = []
        if item.get("episode"):
            detail_parts.append(f"第{item['episode']}集")
        if item.get("from"):
            seconds = int(float(item["from"]))
            detail_parts.append(f"{seconds // 60:02d}:{seconds % 60:02d} 附近")
        anime_id = anilist.get("id")
        hits.append(ReverseImageHit(
            provider="trace_moe",
            title=title[:120],
            author=_clean((anilist.get("synonyms") or [""])[0])[:80] if anilist.get("synonyms") else "",
            source_url=f"https://anilist.co/anime/{anime_id}" if anime_id else "",
            similarity=round(similarity, 2),
            detail=" ".join(detail_parts),
            series_url="",
            thumbnail_url=str(item.get("image") or "").strip(),
        ))
    return hits


_PROVIDERS = {"saucenao": search_saucenao, "ascii2d": search_ascii2d, "trace_moe": search_trace_moe}


def resolve_provider_chain(spec: str, *, saucenao_token: str = "") -> tuple[str, ...]:
    """解析 REVERSE_IMAGE_PROVIDERS：空/auto 用默认链，off/none 禁用，未知名忽略；
    saucenao 无凭证时剔除，链空则上层整个隐藏功能。"""
    spec = str(spec or "").strip().lower()
    if spec in ("", "auto", "default"):
        ordered = ["ascii2d", "trace_moe"]
    else:
        ordered = []
        for part in re.split(r"[,，\s]+", spec):
            name = {"tracemoe": "trace_moe", "trace.moe": "trace_moe"}.get(part, part)
            if name in ("off", "none", "disabled"):
                return ()
            if name in _PROVIDERS and name not in ordered:
                ordered.append(name)
    if "saucenao" in ordered and not str(saucenao_token or "").strip():
        ordered.remove("saucenao")
    return tuple(ordered)


def hit_to_payload(hit: ReverseImageHit) -> dict:
    return asdict(hit)
