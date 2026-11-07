"""QQ 原生表情（小黄脸 / 商城表情）与文本记号的互转。

OneBot 把 QQ 自带表情放在 ``face`` 段里，只给一个数字 id；原来的
``_flatten_message`` 直接忽略这类段，模型收到的就是空字符串——用户发表情，
在它眼里等于什么都没说。

这里统一成 ``[表情：微笑]`` 这样的记号：

* 进模型上下文：模型能读懂语气，也能在回复里照抄这个记号；
* 出模型回复：发送端把记号还原成真正的 ``face`` 段，QQ 里显示原生表情。

id → 名称取自 NapCat 的 QQ 表情表（0~395 的小黄脸，外加 97xx / 12xxxx 的
Emoji 码位别名）。同一个名字可能有多个 id（如「可爱」= 21 / 9786），
反向查表时优先取较小的经典 id。
"""

from __future__ import annotations

import re
from typing import Any

FACE_LABEL = "表情"
STICKER_LABEL = "表情包"

#: 需要翻译成文本的表情类消息段。
FACE_SEGMENT_TYPES = frozenset({"face", "mface", "marketface"})

FACE_NAMES: dict[str, str] = {
    "0": "惊讶", "1": "撇嘴", "2": "色", "3": "发呆", "4": "得意",
    "5": "流泪", "6": "害羞", "7": "闭嘴", "8": "睡", "9": "大哭",
    "10": "尴尬", "11": "发怒", "12": "调皮", "13": "呲牙", "14": "微笑",
    "15": "难过", "16": "酷", "18": "抓狂", "19": "吐", "20": "偷笑",
    "21": "可爱", "22": "白眼", "23": "傲慢", "24": "饥饿", "25": "困",
    "26": "惊恐", "27": "流汗", "28": "憨笑", "29": "悠闲", "30": "奋斗",
    "31": "咒骂", "32": "疑问", "33": "嘘", "34": "晕", "35": "折磨",
    "36": "衰", "37": "骷髅", "38": "敲打", "39": "再见", "41": "发抖",
    "42": "爱情", "43": "跳跳", "46": "猪头", "49": "拥抱", "53": "蛋糕",
    "56": "刀", "59": "便便", "60": "咖啡", "63": "玫瑰", "64": "凋谢",
    "66": "爱心", "67": "心碎", "74": "太阳", "75": "月亮", "76": "赞",
    "77": "踩", "78": "握手", "79": "胜利", "85": "飞吻", "86": "怄火",
    "89": "西瓜", "96": "冷汗", "97": "擦汗", "98": "抠鼻", "99": "鼓掌",
    "100": "糗大了", "101": "坏笑", "102": "左哼哼", "103": "右哼哼", "104": "哈欠",
    "105": "鄙视", "106": "委屈", "107": "快哭了", "108": "阴险", "109": "左亲亲",
    "110": "吓", "111": "可怜", "112": "菜刀", "114": "篮球", "116": "示爱",
    "118": "抱拳", "119": "勾引", "120": "拳头", "121": "差劲", "123": "NO",
    "124": "OK", "125": "转圈", "129": "挥手", "137": "鞭炮", "144": "喝彩",
    "146": "爆筋", "147": "棒棒糖", "169": "手枪", "171": "茶", "172": "眨眼睛",
    "173": "泪奔", "174": "无奈", "175": "卖萌", "176": "小纠结", "177": "喷血",
    "178": "斜眼笑", "179": "doge", "181": "戳一戳", "182": "笑哭", "183": "我最美",
    "185": "羊驼", "187": "幽灵", "201": "点赞", "212": "托腮", "262": "脑阔疼",
    "263": "沧桑", "264": "捂脸", "265": "辣眼睛", "266": "哦哟", "267": "头秃",
    "268": "问号脸", "269": "暗中观察", "270": "emm", "271": "吃瓜", "272": "呵呵哒",
    "273": "我酸了", "277": "汪汪", "281": "无眼笑", "282": "敬礼", "283": "狂笑",
    "284": "面无表情", "285": "摸鱼", "286": "魔鬼笑", "287": "哦", "289": "睁眼",
    "293": "摸锦鲤", "294": "期待", "295": "拿到红包", "297": "拜谢", "298": "元宝",
    "299": "牛啊", "300": "胖三斤", "302": "左拜年", "303": "右拜年", "305": "右亲亲",
    "306": "牛气冲天", "307": "喵喵", "311": "打call", "312": "变形", "314": "仔细分析",
    "317": "菜汪", "318": "崇拜", "319": "比心", "320": "庆祝", "323": "嫌弃",
    "324": "吃糖", "325": "惊吓", "326": "生气", "332": "举牌牌", "333": "烟花",
    "334": "虎虎生威", "336": "豹富", "337": "花朵脸", "338": "我想开了", "339": "舔屏",
    "341": "打招呼", "342": "酸Q", "343": "我方了", "344": "大怨种", "345": "红包多多",
    "346": "你真棒棒", "347": "大展宏兔", "349": "坚强", "350": "贴贴", "351": "敲敲",
    "352": "咦", "353": "拜托", "354": "尊嘟假嘟", "355": "耶", "356": "666",
    "357": "裂开", "392": "龙年快乐", "393": "新年中龙", "394": "新年大龙", "395": "略略略",
    # Emoji 码位别名（QQ 里同一个表情的另一套 id）
    "9749": "咖啡", "9786": "可爱", "10024": "闪光",
    "127801": "玫瑰", "127817": "西瓜", "127822": "苹果", "127827": "草莓",
    "127836": "拉面", "127838": "面包", "127847": "刨冰", "127866": "啤酒",
    "127867": "干杯", "127881": "庆祝", "128074": "拳头", "128076": "好的",
    "128077": "厉害", "128079": "鼓掌", "128157": "礼物", "128164": "睡觉",
    "128166": "水", "128168": "吹气", "128170": "肌肉", "128293": "火",
    "128513": "呲牙", "128514": "激动", "128516": "高兴", "128522": "嘿嘿",
    "128524": "羞涩", "128527": "哼哼", "128530": "不屑", "128531": "汗",
    "128532": "失落", "128536": "飞吻", "128538": "亲亲", "128540": "淘气",
    "128541": "吐舌", "128557": "大哭", "128560": "紧张", "128563": "瞪眼",
}

FACE_IDS: dict[str, str] = {}
for _face_id, _face_name in FACE_NAMES.items():
    FACE_IDS.setdefault(_face_name, _face_id)
del _face_id, _face_name

#: ``[表情：微笑]`` / ``[表情:微笑]`` / ``[表情包：开心]``
TOKEN_PATTERN = re.compile(r"\[(表情包|表情)\s*[：:]\s*([^\[\]：:]{1,20})\]", re.UNICODE)


def face_token(face_id: Any) -> str:
    """把 ``face`` 段里的 id 译成模型可读的记号。

    查不到名字时保留原始 id（``[表情：id999]``），既不丢信息，也让
    :func:`resolve_face_id` 有机会原样还原。
    """

    raw = str(face_id if face_id is not None else "").strip()
    if not raw:
        return f"[{FACE_LABEL}：未知]"
    name = FACE_NAMES.get(raw)
    return f"[{FACE_LABEL}：{name}]" if name else f"[{FACE_LABEL}：id{raw}]"


def sticker_token(summary: Any) -> str:
    """商城表情 / 表情包的 ``summary`` 通常已经写成 ``[开心]``，去掉方括号即可。"""

    raw = str(summary if summary is not None else "").strip().strip("[]【】")
    raw = re.sub(r"\s+", " ", raw)[:20].strip()
    return f"[{STICKER_LABEL}：{raw}]" if raw else f"[{STICKER_LABEL}：未知]"


def segment_text(item: Any) -> str:
    """把一个表情类消息段渲染成文本记号；其它类型返回空串。"""

    if not isinstance(item, dict) or item.get("type") not in FACE_SEGMENT_TYPES:
        return ""
    data = item.get("data")
    if not isinstance(data, dict):
        data = {}
    if item.get("type") == "face":
        return face_token(data.get("id"))
    return sticker_token(data.get("summary") or data.get("name") or data.get("emoji_id"))


def message_text(message: Any) -> str:
    """把整条消息里的文字段和表情段拼成 plain text（保持原始顺序）。"""

    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, list):
        return ""
    parts: list[str] = []
    for item in message:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            data = item.get("data")
            if isinstance(data, dict):
                text = str(data.get("text", ""))
                if text:
                    parts.append(text)
        elif kind in FACE_SEGMENT_TYPES:
            parts.append(segment_text(item))
    return "".join(parts).strip()


def resolve_face_id(value: Any) -> str | None:
    """把记号里的名称还原成可发送的 face id；无法还原时返回 ``None``。"""

    token = str(value if value is not None else "").strip()
    if not token:
        return None
    if token.isdigit():
        return token if token in FACE_NAMES else None
    if token[:2].lower() == "id" and token[2:].isdigit():
        candidate = token[2:]
        return candidate if candidate in FACE_NAMES else None
    normalized = re.sub(r"\s+", "", token).lower()
    for name, face_id in FACE_IDS.items():
        if name.lower() == normalized:
            return face_id
    return None


def text_segments(text: str) -> list[dict[str, Any]] | None:
    """把回复文本里的表情记号拆成 OneBot 消息段。

    文本里没有记号时返回 ``None``，调用方保持原来的纯字符串发送方式，
    行为完全不变。有记号时返回段数组；还原不了的表情包记号会被丢弃
    （商城表情没有可复用的 id），全部被丢弃时返回空列表。
    """

    raw = str(text or "")
    if not TOKEN_PATTERN.search(raw):
        return None
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in TOKEN_PATTERN.finditer(raw):
        prefix = raw[cursor:match.start()]
        if prefix:
            segments.append({"type": "text", "data": {"text": prefix}})
        cursor = match.end()
        face_id = resolve_face_id(match.group(2))
        if face_id is not None:
            segments.append({"type": "face", "data": {"id": face_id}})
    tail = raw[cursor:]
    if tail:
        segments.append({"type": "text", "data": {"text": tail}})
    return segments
