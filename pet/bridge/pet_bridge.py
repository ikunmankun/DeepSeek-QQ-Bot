"""大肥鱼桌宠桥：把 bot 的真实状态喂给 dsh-pet（身体归它，灵魂归我们）。

约定（对 dsh-pet v0.4.8 独立模式逆向确认）：
- 端口每次启动随机（127.0.0.1 回环），按进程枚举发现
- token 来自 pet/dsh-pet-token.txt（start-dsh-pet.bat 用 DSH_PET_TOKEN 注入，
  app.asar 已打本地补丁接受该环境变量）
- broadcast：POST /dsh-pet-7340/broadcast?pet=main&surface=desktop
  {text, duration(ms), anim(可选，动作中文名)}

用法：
  python pet_bridge.py say "文本" [--anim 动作名] [--duration 6000]
  python pet_bridge.py anim "吃白饭"
  python pet_bridge.py watch    # 常驻：记忆星图变化 → 桌宠播报
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_FILE = HERE.parent / "dsh-pet-token.txt"
MONITOR = "http://127.0.0.1:8765"
PET = "main"
SURFACE = "desktop"


def _listen_ports():
    import psutil
    ports = set()
    for proc in psutil.process_iter(["name"]):
        if (proc.info["name"] or "").lower() != "dsh-pet.exe":
            continue
        try:
            for conn in proc.net_connections(kind="tcp"):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.ip == "127.0.0.1":
                    ports.add(conn.laddr.port)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return sorted(ports)


def api_base():
    for port in _listen_ports():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    raise RuntimeError("dsh-pet 没在运行：先跑 start-dsh-pet.bat")


def _token():
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


def _post(base, path, body):
    req = urllib.request.Request(
        f"{base}{path}?pet={PET}&surface={SURFACE}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"X-DSH-Pet-Token": _token(), "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.load(r)


def say(text, anim=None, duration=6000):
    base = api_base()
    body = {"text": text[:600], "duration": int(duration)}
    if anim:
        body["anim"] = anim
    return _post(base, "/dsh-pet-7340/broadcast", body)


def play(anim_name):
    base = api_base()
    return _post(base, "/dsh-pet-7340/anim", {"name": anim_name})


def memory_summary():
    with urllib.request.urlopen(MONITOR + "/api/memory-map", timeout=8) as r:
        data = json.load(r)
    if not data.get("available"):
        return None
    stats = data["stats"]
    if not stats.get("total"):
        return None
    return (f"现在记得 {stats['total']} 件事，"
            f"{stats['immutable']} 件永不遗忘，{stats['faded']} 件在淡忘")


def pet_events():
    with urllib.request.urlopen(MONITOR + "/api/pet-events", timeout=8) as r:
        return json.load(r)


def list_todos(base):
    req = urllib.request.Request(
        f"{base}/dsh-pet-7340/todos?pet={PET}&surface={SURFACE}",
        headers={"X-DSH-Pet-Token": _token()})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.load(r).get("todos", [])


def _todo_method(method, path, body=None):
    base = api_base()
    tail = f"/{path}" if path else ""
    req = urllib.request.Request(
        f"{base}/dsh-pet-7340/todos{tail}?pet={PET}&surface={SURFACE}",
        data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
        headers={"X-DSH-Pet-Token": _token(), "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.load(r)


def sync_todos(desired_texts, synced_ids):
    """把 bot 的日程待办同步到桌宠头顶：只动我们自己同步过的条目。"""
    base = api_base()
    existing = list_todos(base)
    existing_by_id = {t["id"]: t for t in existing}
    existing_texts = {t["text"] for t in existing}
    changed = 0
    # 删除：我们同步过、但 bot 侧已不存在（完成/过期）的
    for todo_id in list(synced_ids):
        item = existing_by_id.get(todo_id)
        if item is None:
            synced_ids.pop(todo_id, None)
            continue
        if item["text"] not in desired_texts:
            _todo_method("DELETE", todo_id)
            synced_ids.pop(todo_id, None)
            changed += 1
    # 新增：bot 侧有、桌宠没有的
    for text in sorted(desired_texts - existing_texts):
        try:
            created = _todo_method("POST", "", {"text": text, "dueAt": None})
            synced_ids[created["todo"]["id"]] = True
            changed += 1
        except Exception as exc:
            print("[bridge] todo create failed:", exc, flush=True)
    return changed


def watch(interval_seconds=600):
    last_key = None
    print("[bridge] watching memory map + pet events ...", flush=True)
    seen_message_ids = set()
    seen_wish_ids = set()
    seen_group_ids = set()
    synced_todo_ids = {}
    last_whisper = 0.0
    last_group_react = 0.0
    events_tick = 0
    memory_tick = 0
    first_events_tick = True
    while True:
        now = time.time()
        try:
            events = pet_events()
            if events.get("available"):
                # 主人私聊：挥手表新消息（限流 3 分钟一条）
                fresh = [m for m in events.get("owner_messages", [])
                         if m["id"] not in seen_message_ids]
                for m in events.get("owner_messages", []):
                    seen_message_ids.add(m["id"])
                if fresh and now - last_whisper > 180:
                    snippet = fresh[-1]["text"][:60]
                    say("主人在私聊叫我了：" + snippet, anim="点击回应-元气挥手")
                    last_whisper = now
                    print("[bridge] owner message reacted:", snippet, flush=True)
                # 愿望实现：女仆舞庆祝
                for wish in events.get("wishes_done", []):
                    if wish["id"] in seen_wish_ids:
                        continue
                    seen_wish_ids.add(wish["id"])
                    say("愿望实现了！" + wish["content"][:60], anim="优雅女仆舞", duration=8000)
                    print("[bridge] wish celebrated:", wish["content"][:40], flush=True)
                # 群聊动静：bot 自己回复或被 @（限流 8 分钟一条；首轮只记录不反应）
                group_fresh = []
                for g in events.get("group_activity", []):
                    if g["id"] in seen_group_ids:
                        continue
                    seen_group_ids.add(g["id"])
                    if not first_events_tick:
                        group_fresh.append(g)
                if group_fresh and now - last_group_react > 480:
                    last = group_fresh[-1]
                    if last.get("by_bot"):
                        say("刚在群里回了句：" + last["text"][:50], anim="原地漂浮踏步")
                    else:
                        say("群里有人叫我：" + last["text"][:50], anim="东张西望")
                    last_group_react = now
                    print("[bridge] group reacted:", last["text"][:40], flush=True)
                # 待办清单同步（每 4 个 tick 跑一次）
                if events_tick % 4 == 0:
                    desired = {t["text"] for t in events.get("todos", [])}
                    changed = sync_todos(desired, synced_todo_ids)
                    if changed:
                        print("[bridge] todos synced, changed =", changed, flush=True)
            # 记忆星图播报（低频）
            if now - memory_tick > interval_seconds:
                memory_tick = now
                summary = memory_summary()
                if summary:
                    key = summary
                    if key != last_key:
                        say(summary)
                        last_key = key
                        print("[bridge] whispered:", summary, flush=True)
        except Exception as exc:
            print("[bridge] watch error:", exc, flush=True)
        first_events_tick = False
        events_tick += 1
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description="大肥鱼桌宠桥")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_say = sub.add_parser("say", help="让桌宠说一句话")
    p_say.add_argument("text")
    p_say.add_argument("--anim", default=None)
    p_say.add_argument("--duration", type=int, default=6000)
    p_anim = sub.add_parser("anim", help="播一段指定动作")
    p_anim.add_argument("name")
    sub.add_parser("watch", help="常驻：记忆星图变化 → 桌宠播报")
    args = parser.parse_args()
    if args.cmd == "say":
        print(json.dumps(say(args.text, args.anim, args.duration), ensure_ascii=False))
    elif args.cmd == "anim":
        print(json.dumps(play(args.name), ensure_ascii=False))
    elif args.cmd == "watch":
        watch()


if __name__ == "__main__":
    main()
