"""按主机名分流：被墙站点走 VPN 代理端点，其余直连。

用法：VPN_PROXY_URL 填代理端点（http://host:port 或 socks5://host:port，
socks 端点需要镜像里装 socksio）；VPN_PROXY_HOSTS 是域名后缀清单（逗号分隔，
含全部子域）。未配置端点 → 永远直连（与历史行为一致）。

为什么是显式传参而不是 env：识曲/搜图这类客户端是 trust_env=False 直连的
（容器里的 mihomo 对部分站点是死路，2026-11 实测），只有命中清单的请求才
把代理端点显式递给 httpx。
"""
from __future__ import annotations

from urllib.parse import urlsplit


def resolve_vpn_hosts(spec: str) -> tuple[str, ...]:
    """解析 VPN_PROXY_HOSTS：域名后缀，小写去重，容忍中文逗号。"""
    hosts: list[str] = []
    for part in str(spec or "").replace("，", ",").lower().split(","):
        part = part.strip().lstrip(".")
        if part and part not in hosts:
            hosts.append(part)
    return tuple(hosts)


def vpn_proxy_for_url(url: str, *, proxy_url: str, hosts: tuple[str, ...]) -> str:
    """命中清单返回代理端点串，否则空串（httpx 的 proxy= 传空即直连）。"""
    proxy_url = str(proxy_url or "").strip()
    if not proxy_url or not hosts:
        return ""
    hostname = (urlsplit(str(url)).hostname or "").lower().rstrip(".")
    if not hostname:
        return ""
    for suffix in hosts:
        if hostname == suffix or hostname.endswith("." + suffix):
            return proxy_url
    return ""
