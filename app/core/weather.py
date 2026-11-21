"""天气查询：Open-Meteo 免费公开接口，无 key。

* WEATHER_CITY 按城市名 geocoding 拿经纬度并进程内缓存；带"区/县"的写法
  （如"广州天河区"）先原样查，查不到就逐级去后缀、退到前两个字重试；
  WEATHER_LAT/WEATHER_LON 可直接指定坐标跳过 geocoding。
* 接口按域名分流：命中 VPN_PROXY_HOSTS 时显式走代理端点，否则直连
  （与反向搜图同一套路，不吃容器代理 env）。
* 任何一步失败都返回 None，调用方自行降级，这里不抛异常。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.proxy_routing import resolve_vpn_hosts, vpn_proxy_for_url

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: WMO weather code → 中文短描述（Open-Meteo 标准码表节选）。
WMO_DESCRIPTIONS = {
    0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
    45: "有雾", 48: "雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "密集毛毛雨",
    56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨带冰雹", 99: "强雷阵雨带冰雹",
}


@dataclass(frozen=True)
class WeatherReport:
    city: str
    description: str
    temp_now: float | None
    feels_like: float | None
    temp_min: float | None
    temp_max: float | None
    precip_probability: int | None  # %
    precip_sum: float | None  # mm
    wind_kmh: float | None
    humidity: float | None

    @property
    def line(self) -> str:
        """一行人话摘要，直接可塞提示词或播报。"""

        parts = [f"{self.city}{self.description}"]
        if self.temp_now is not None:
            parts.append(f"气温{self.temp_now:.0f}°C")
        if self.feels_like is not None and self.temp_now is not None \
                and abs(self.feels_like - self.temp_now) >= 3:
            parts.append(f"体感{self.feels_like:.0f}°C")
        if self.temp_min is not None and self.temp_max is not None:
            parts.append(f"今天{self.temp_min:.0f}~{self.temp_max:.0f}°C")
        if self.precip_probability is not None and self.precip_probability >= 30:
            parts.append(f"降水概率{self.precip_probability}%")
        if self.precip_sum is not None and self.precip_sum >= 0.5:
            parts.append(f"预计雨量{self.precip_sum:.0f}mm")
        if self.wind_kmh is not None and self.wind_kmh >= 20:
            parts.append(f"风{self.wind_kmh:.0f}km/h")
        if self.humidity is not None and self.humidity >= 85:
            parts.append("很潮湿")
        return "，".join(parts)


def describe_code(code: Any) -> str:
    try:
        return WMO_DESCRIPTIONS.get(int(code), "天气不明")
    except (TypeError, ValueError):
        return "天气不明"


def city_candidates(city: str) -> list[str]:
    """城市名检索候选：原样 → 去后缀 → 再去 → 前两个字。"""

    base = str(city or "").strip()
    if not base:
        return []
    candidates = [base]
    current = base
    for _ in range(2):
        stripped = current.rstrip("市区县镇旗")
        if not stripped or stripped == current:
            break
        current = stripped
        if current not in candidates:
            candidates.append(current)
    short = base[:2]
    if short and short not in candidates:
        candidates.append(short)
    return candidates


def _http_get_json(settings, url: str, params: dict, *, transport: httpx.BaseTransport | None = None) -> dict:
    proxy = vpn_proxy_for_url(
        url,
        proxy_url=str(getattr(settings, "vpn_proxy_url", "") or ""),
        hosts=resolve_vpn_hosts(str(getattr(settings, "vpn_proxy_hosts", "") or "")),
    )
    with httpx.Client(timeout=12.0, trust_env=False, proxy=proxy or None,
                      transport=transport) as client:
        resp = client.get(url, params=params,
                          headers={"User-Agent": "xiaomachi-bot/1.0"})
        resp.raise_for_status()
        return resp.json()


_GEO_CACHE: dict[str, tuple[float, float, str]] = {}


def geocode_city(settings, city: str, *, transport: httpx.BaseTransport | None = None) -> tuple[float, float, str] | None:
    """城市名 → (纬度, 经度, 解析出的地名)；查不到返回 None。"""

    cached = _GEO_CACHE.get(city)
    if cached is not None:
        return cached
    for candidate in city_candidates(city):
        try:
            data = _http_get_json(settings, GEOCODING_URL,
                {"name": candidate, "count": 1, "language": "zh", "format": "json"},
                transport=transport)
        except Exception:
            continue
        results = data.get("results") or []
        if not results:
            continue
        hit = results[0]
        try:
            resolved = (float(hit["latitude"]), float(hit["longitude"]),
                        str(hit.get("name") or candidate))
        except (KeyError, TypeError, ValueError):
            continue
        _GEO_CACHE[city] = resolved
        return resolved
    return None


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_weather(settings, *, transport: httpx.BaseTransport | None = None) -> WeatherReport | None:
    """按配置拿当前天气 + 今日预报；任何失败返回 None。"""

    city = str(getattr(settings, "weather_city", "") or "").strip()
    lat0 = _num(getattr(settings, "weather_lat", 0.0)) or 0.0
    lon0 = _num(getattr(settings, "weather_lon", 0.0)) or 0.0
    try:
        if lat0 and lon0:
            lat, lon = lat0, lon0
        else:
            if not city:
                return None
            located = geocode_city(settings, city, transport=transport)
            if located is None:
                logger.warning("weather_geocode_failed city=%s", city)
                return None
            lat, lon, resolved = located
            city = resolved or city
        data = _http_get_json(settings, FORECAST_URL, {
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,precipitation_sum",
            "timezone": "Asia/Shanghai", "forecast_days": 1,
        }, transport=transport)
        current = data.get("current") or {}
        daily = data.get("daily") or {}

        def _first(key: str) -> Any:
            values = daily.get(key) or []
            return values[0] if values else None

        precip_probability = _num(_first("precipitation_probability_max"))
        humidity = _num(current.get("relative_humidity_2m"))
        return WeatherReport(
            city=city, description=describe_code(current.get("weather_code")),
            temp_now=_num(current.get("temperature_2m")),
            feels_like=_num(current.get("apparent_temperature")),
            temp_min=_num(_first("temperature_2m_min")),
            temp_max=_num(_first("temperature_2m_max")),
            precip_probability=int(precip_probability) if precip_probability is not None else None,
            precip_sum=_num(_first("precipitation_sum")),
            wind_kmh=_num(current.get("wind_speed_10m")),
            humidity=humidity,
        )
    except Exception:
        logger.warning("weather_fetch_failed", exc_info=True)
        return None
