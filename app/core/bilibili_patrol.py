"""B 站插件的核心逻辑：链接识别、热门榜拉取、信息缓存、群兴趣画像。

参照流行插件（链接解析 / bilichat 视频总结 / 热榜快报）的公开能力，
只读公开 JSON API：不登录、不带 cookie、不下媒体。视频元数据复用
providers.shared_page_text.read_bilibili_text 的既有实现。

群兴趣（对照"兴趣是长出来的"）：群友分享过什么视频，机器人就慢慢知道
这个群在看什么——按视频分区（tname）计分，读取时按半衰期向 0 衰减，
久没人聊的口味自然淡出。只做只读的"看什么"，没有点赞/评论等写操作。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select

from app.core.time_utils import stored_as_utc
from app.storage.models import BilibiliGroupInterest

_BV = re.compile(r"BV[0-9A-Za-z]{10}")
_AV = re.compile(r"\bav(\d{6,})\b", re.IGNORECASE)
_B23 = re.compile(r"b23\.tv/[0-9A-Za-z]+")

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

INFO_TTL_SECONDS = 6 * 3600
HOT_TTL_SECONDS = 6 * 3600
SHARE_TTL_SECONDS = 15 * 60


def extract_bilibili_refs(text: str) -> list[str]:
    """从一条消息里提取 B 站引用（BV/av/短链），按出现顺序去重限 3 个。"""

    normalized = str(text or "")
    found: list[tuple[int, str]] = []
    for match in _BV.finditer(normalized):
        found.append((match.start(), match.group(0)))
    for match in _AV.finditer(normalized):
        found.append((match.start(), f"av{match.group(1)}"))
    for match in _B23.finditer(normalized):
        found.append((match.start(), match.group(0)))
    deduped = list(dict.fromkeys(ref for _, ref in sorted(found)))
    return deduped[:3]


@dataclass(frozen=True)
class VideoInfo:
    bvid: str
    title: str
    author: str
    summary: str
    url: str
    zone: str = ""  # 分区名（tname），群兴趣画像的计数单位


def _clean(value: Any, size: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:size]


def fetch_video_info(ref: str, *, client: httpx.Client) -> VideoInfo | None:
    """取一个 BV/av/短链的视频元数据；失败返回 None（调用方决定降级）。"""

    try:
        if ref.startswith("BV"):
            bvid = ref
        else:
            response = client.get(f"https://www.bilibili.com/video/{ref}",
                                  headers=_HEADERS, follow_redirects=True, timeout=10.0)
            match = _BV.search(response.url.path or "") or _BV.search(str(response.url))
            if response.status_code != 200 or match is None:
                return None
            bvid = match.group(0)
        response = client.get("https://api.bilibili.com/x/web-interface/view",
                              params={"bvid": bvid}, headers=_HEADERS, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            return None
        data = payload.get("data") or {}
        title = _clean(data.get("title"), 300)
        if not title:
            return None
        return VideoInfo(
            bvid=str(data.get("bvid") or bvid),
            title=title,
            author=_clean((data.get("owner") or {}).get("name"), 60),
            summary=_clean(data.get("desc"), 220),
            url=f"https://www.bilibili.com/video/{data.get('bvid') or bvid}",
            zone=_clean(data.get("tname"), 48),
        )
    except (httpx.HTTPError, ValueError, OSError):
        return None


def fetch_popular(*, client: httpx.Client, limit: int = 6) -> list[VideoInfo]:
    """B 站综合热门榜（公开 API，只读）。"""

    response = client.get("https://api.bilibili.com/x/web-interface/popular",
                          params={"ps": 20, "pn": 1}, headers=_HEADERS, timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        return []
    videos: list[VideoInfo] = []
    for item in (payload.get("data") or {}).get("list") or []:
        title = _clean(item.get("title"), 300)
        bvid = str(item.get("bvid") or "")
        if not title or not bvid:
            continue
        videos.append(VideoInfo(
            bvid=bvid, title=title,
            author=_clean((item.get("owner") or {}).get("name"), 60),
            summary=_clean(item.get("desc"), 160),
            url=f"https://www.bilibili.com/video/{bvid}",
            zone=_clean(item.get("tname"), 48),
        ))
        if len(videos) >= limit:
            break
    return videos


class BiliInfoCache:
    """进程内缓存：BV → 视频信息（TTL 6h）；每群最近分享（TTL 15min）。"""

    def __init__(self) -> None:
        self._info: dict[str, tuple[float, VideoInfo]] = {}
        self._shares: dict[int, tuple[float, VideoInfo]] = {}

    def get_info(self, ref: str, *, now: float) -> VideoInfo | None:
        entry = self._info.get(ref)
        if entry is None or now - entry[0] > INFO_TTL_SECONDS:
            return None
        return entry[1]

    def put_info(self, ref: str, info: VideoInfo, *, now: float) -> None:
        if len(self._info) > 128:
            self._info.clear()
        self._info[ref] = (now, info)

    def put_share(self, group_id: int, info: VideoInfo, *, now: float) -> None:
        self._shares[group_id] = (now, info)

    def recent_share(self, group_id: int, *, now: float) -> VideoInfo | None:
        entry = self._shares.get(group_id)
        if entry is None or now - entry[0] > SHARE_TTL_SECONDS:
            return None
        return entry[1]


def hot_prompt_line(videos: list[VideoInfo]) -> str:
    items = "；".join(f"《{v.title}》({v.author}) {v.url}" for v in videos[:4])
    return (f"B站此刻热门（想安利哪个就把它的链接一起甩到群里，像随手分享，别像播报员）：{items}。"
            "分享前用工具看一眼简介再说话，链接直接贴在回复文本里，QQ会自动渲染成卡片。")


def share_prompt_line(info: VideoInfo) -> str:
    return (f"对方刚分享了B站视频：《{info.title}》（UP主：{info.author}）。"
            f"简介：{info.summary or '无'}。回应要接住这个视频的话题，别复述简介。")


# ---- 群兴趣画像（按分区计分，读取时衰减） ------------------------------

#: 兴趣半衰期：口味变得比记忆快，一个月没人聊的分区就淡出。
INTEREST_HALF_LIFE_DAYS = 30.0
#: 单分区分数上限（防单刷）。
INTEREST_SCORE_CAP = 20.0
#: 注入提示词的最低衰减分：低于这个值说明这个口味最近根本没出现。
INTEREST_PROMPT_MIN_SCORE = 0.5
#: 每群最多记几个分区，防止蹭过一次的冷门分区占表。
MAX_INTEREST_ZONES = 8


def bump_group_interest(session, *, group_id: int, zone: str, now: datetime,
                        weight: float = 1.0) -> None:
    """记一次群友分享：对应分区加分。失败由调用方自吞（on_inbound 里）。"""

    zone = _clean(zone, 48)
    if not zone:
        return
    row = session.get(BilibiliGroupInterest, (int(group_id), zone))
    if row is None:
        # 分区数封顶：只挤掉分数最低的，不无限扩表。
        count = len(
            session.scalars(
                select(BilibiliGroupInterest.zone).where(
                    BilibiliGroupInterest.group_id == int(group_id)
                )
            ).all()
        )
        if count >= MAX_INTEREST_ZONES:
            weakest = session.scalars(
                select(BilibiliGroupInterest)
                .where(BilibiliGroupInterest.group_id == int(group_id))
                .order_by(BilibiliGroupInterest.score.asc())
                .limit(1)
            ).first()
            if weakest is not None and float(weakest.score or 0.0) <= weight:
                session.delete(weakest)
            else:
                return
        row = BilibiliGroupInterest(
            group_id=int(group_id), zone=zone,
            score=0.0, share_count=0,
            first_seen_at=now, last_seen_at=now,
        )
        session.add(row)
    row.score = min(INTEREST_SCORE_CAP, float(row.score or 0.0) + float(weight))
    row.share_count = int(row.share_count or 0) + 1
    row.last_seen_at = now


def decayed_interests(engine, *, group_id: int, now: datetime,
                      limit: int = 3) -> list[str]:
    """当前还"热"的分区（按 30 天半衰期衰减后的分数排序）。"""

    try:
        from app.storage.db import session_scope

        with session_scope(engine) as session:
            rows = list(
                session.scalars(
                    select(BilibiliGroupInterest).where(
                        BilibiliGroupInterest.group_id == int(group_id)
                    )
                )
            )
    except Exception:
        return []
    scored: list[tuple[str, float]] = []
    for row in rows:
        try:
            elapsed_days = (
                stored_as_utc(now) - stored_as_utc(row.last_seen_at)
            ).total_seconds() / 86400.0
        except (TypeError, ValueError):
            continue
        score = float(row.score or 0.0) * 2.0 ** (-max(0.0, elapsed_days) / INTEREST_HALF_LIFE_DAYS)
        if score >= INTEREST_PROMPT_MIN_SCORE:
            scored.append((str(row.zone), score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [zone for zone, _score in scored[: max(1, limit)]]


def interest_prompt_line(engine, *, group_id: int, now: datetime) -> str | None:
    """给观察轮的兴趣行；没有足够新鲜的口味时返回 None。"""

    zones = decayed_interests(engine, group_id=group_id, now=now)
    if not zones:
        return None
    return (f"这个群最近常看B站的：{'、'.join(zones)}（都是群友分享过的分区）。"
            "刷到相关内容可以留意，别硬往话题上凑。")
