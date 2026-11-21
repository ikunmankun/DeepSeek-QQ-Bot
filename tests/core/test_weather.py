"""天气模块：城市候选、码表、摘要行、Open-Meteo 拉取（MockTransport，不出网）。"""
import json
from types import SimpleNamespace

import httpx

import app.core.weather as weather_module
from app.core.weather import WeatherReport, city_candidates, describe_code, fetch_weather


def _settings(**overrides):
    base = dict(weather_city="广州天河区", weather_lat=0.0, weather_lon=0.0,
                vpn_proxy_url="", vpn_proxy_hosts="")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_city_candidates_strips_suffixes_step_by_step():
    assert city_candidates("广州天河区") == ["广州天河区", "广州天河", "广州"]


def test_city_candidates_short_city_stays_single():
    assert city_candidates("杭州") == ["杭州"]


def test_city_candidates_empty():
    assert city_candidates("") == []
    assert city_candidates("   ") == []


def test_describe_code_covers_common_and_unknown():
    assert describe_code(0) == "晴"
    assert describe_code(61) == "小雨"
    assert describe_code(95) == "雷阵雨"
    assert describe_code(None) == "天气不明"
    assert describe_code("bad") == "天气不明"


def test_report_line_compact_and_conditional():
    report = WeatherReport(city="天河", description="小雨", temp_now=18.4,
                           feels_like=15.0, temp_min=14.0, temp_max=22.0,
                           precip_probability=70, precip_sum=3.2,
                           wind_kmh=9.0, humidity=90)
    line = report.line
    assert "天河小雨" in line
    assert "气温18°C" in line
    assert "体感15°C" in line
    assert "14~22°C" in line
    assert "降水概率70%" in line
    assert "很潮湿" in line
    assert "km/h" not in line  # 微风不提


def test_report_line_skips_low_probability_rain():
    report = WeatherReport(city="天河", description="晴", temp_now=26.0,
                           feels_like=27.0, temp_min=24.0, temp_max=31.0,
                           precip_probability=10, precip_sum=0.0,
                           wind_kmh=12.0, humidity=60)
    line = report.line
    assert "降水概率" not in line
    assert "体感" not in line  # 温差小于 3 度不提体感
    assert "很潮湿" not in line


def test_fetch_weather_geocodes_then_reads_forecast():
    weather_module._GEO_CACHE.clear()
    urls = []

    def handler(request):
        urls.append(str(request.url))
        if "geocoding-api" in str(request.url):
            payload = {"results": [{"latitude": 23.12, "longitude": 113.36, "name": "天河区"}]}
        else:
            payload = {
                "current": {"temperature_2m": 20.0, "apparent_temperature": 18.0,
                            "relative_humidity_2m": 88, "weather_code": 61,
                            "wind_speed_10m": 24.0},
                "daily": {"temperature_2m_max": [26.0], "temperature_2m_min": [17.0],
                          "precipitation_probability_max": [80], "precipitation_sum": [4.5]},
            }
        return httpx.Response(200, content=json.dumps(payload).encode())

    report = fetch_weather(_settings(), transport=httpx.MockTransport(handler))
    assert report is not None
    assert report.city == "天河区"
    assert report.description == "小雨"
    assert report.temp_now == 20.0
    assert "降水概率80%" in report.line
    assert any("geocoding-api.open-meteo.com" in url for url in urls)
    assert any("latitude=23.1200" in url for url in urls)


def test_fetch_weather_falls_back_to_shorter_city_name():
    weather_module._GEO_CACHE.clear()
    seen_names = []

    def handler(request):
        name = dict(request.url.params).get("name")
        seen_names.append(name)
        if name == "广州":
            payload = {"results": [{"latitude": 23.13, "longitude": 113.26, "name": "广州"}]}
            return httpx.Response(200, content=json.dumps(payload).encode())
        return httpx.Response(200, content=b'{"results": []}')

    report = fetch_weather(_settings(weather_city="广州天河区"),
                           transport=httpx.MockTransport(handler))
    assert report is not None
    assert report.city == "广州"
    # 预报请求也过同一个 handler（无 name 参数），过滤后再比对 geocoding 顺序
    assert [name for name in seen_names if name] == ["广州天河区", "广州天河", "广州"]


def test_fetch_weather_uses_coordinates_without_geocoding():
    weather_module._GEO_CACHE.clear()
    urls = []

    def handler(request):
        urls.append(str(request.url))
        payload = {"current": {"temperature_2m": 31.0, "apparent_temperature": 36.0,
                               "relative_humidity_2m": 70, "weather_code": 1,
                               "wind_speed_10m": 10.0},
                   "daily": {"temperature_2m_max": [33.0], "temperature_2m_min": [26.0],
                             "precipitation_probability_max": [5], "precipitation_sum": [0.0]}}
        return httpx.Response(200, content=json.dumps(payload).encode())

    report = fetch_weather(_settings(weather_city="天河", weather_lat=23.1, weather_lon=113.3),
                           transport=httpx.MockTransport(handler))
    assert report is not None
    assert report.city == "天河"
    assert not any("geocoding-api" in url for url in urls)


def test_fetch_weather_returns_none_without_location():
    weather_module._GEO_CACHE.clear()
    assert fetch_weather(_settings(weather_city="")) is None


def test_fetch_weather_returns_none_when_geocode_fails():
    weather_module._GEO_CACHE.clear()

    def handler(request):
        return httpx.Response(200, content=b'{"results": []}')

    assert fetch_weather(_settings(weather_city="不存在的地方"),
                         transport=httpx.MockTransport(handler)) is None
