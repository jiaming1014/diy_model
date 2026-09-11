"""text_utils.py：純文字規則助手（關鍵字分類、日期、查詢清洗）｜新手教學版。

【這支程式在做什麼？（白話版）】
聊天核心裡有一堆「看一眼就能分類」的小規則：
這句話是在問日期？是閒聊？還是要即時資料？
把這些純規則抽到這裡，chat_core 就能專心管對話流程。

【P16 模組化】
原本這些函式住在 chat_core.py（近千行），抽出來後：
- 沒有副作用、不碰網路、不碰模型，最好測試。
- chat_core 以 `from text_utils import ...` 重新匯出同名函式，舊寫法照用。

【注意】
- 這裡讀 config.py 的即時值（如 USER_MAX_CHARS），不是快照，改 env 立即生效。
"""

from datetime import datetime
import re
from typing import Final
from zoneinfo import ZoneInfo

import config as _config

TAIPEI_TZ: Final[str] = "Asia/Taipei"
_WEEKDAY_ZH: Final[tuple[str, ...]] = ("一", "二", "三", "四", "五", "六", "日")
_DATE_KEYWORDS: Final[tuple[str, ...]] = (
    "今天", "今日", "現在", "日期", "幾號", "星期", "禮拜", "時間", "幾點", "年月日",
)
_WEATHER_KEYWORDS: Final[tuple[str, ...]] = (
    "天氣", "氣溫", "溫度", "下雨", "降雨", "降水", "颱風", "預報", "濕度", "晴", "多雲", "陰天",
)
_CHITCHAT: Final[frozenset[str]] = frozenset(
    {"你好", "您好", "嗨", "嗨嗨", "哈囉", "早安", "午安", "晚安", "謝謝", "感謝",
     "再見", "掰掰", "拜拜", "ok", "okay", "hi", "hello", "hey"}
)
# 優化：即時類關鍵字，問句含這些就直接上網搜，不花時間翻本地筆記
# 本地筆記通常沒有這些；相反先搜網更快且答案更新
# P0 修正：移除過寬的泛時間詞（2026／今年／現況等），避免「今年筆記在哪」被誤判跳過 RAG；
# 年齡類改交給工具迴圈的 LLM 自行判斷，不再強制即時。
_REALTIME_KEYWORDS: Final[tuple[str, ...]] = (
    "新聞", "最新", "股價", "匯率", "比特幣", "加密貨幣", "天氣", "氣溫",
    "颱風", "地震", "即時", "公告", "發布", "上市", "發表", "演出",
    "票價", "賽事", "比分", "排名", "榜單",
)
# 本地意圖關鍵字：含這些優先視為查本地筆記，不強制即時（除非同時命中強即時詞）
_LOCAL_KEYWORDS: Final[tuple[str, ...]] = (
    "筆記", "專案", "資料夾", "嵌入", "收藏", "本地", "notes",
)
_STRIP_EDGE_RE: Final[re.Pattern[str]] = re.compile(r"^[\s，。！？、；：,.!?;:～~\-—]+|[\s，。！？、；：,.!?;:～~\-—]+$")
_QUERY_FILLER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(請問|請問一下|幫我查一下|幫我找一下|查一下|找一下|謝謝|麻煩)[，,。\s]*|[？?！!啊呢吧喔哦]+$"
)


def _current_year() -> str:
    """工具描述用的年份，每年自動跟進，不再寫死 2025｜新手：菜單上的年份不用每年手改。」"""
    return str(datetime.now(ZoneInfo(TAIPEI_TZ)).year)


def _today_str() -> str:
    """回傳台灣時間今天，如 2026-09-07 星期日。」"""
    now = datetime.now(ZoneInfo(TAIPEI_TZ))
    weekday = _WEEKDAY_ZH[now.weekday()]
    return f"{now:%Y-%m-%d} 星期{weekday}"


def _strip_edge(text: str) -> str:
    """去頭尾空白與中英標點，統一短句判斷的輸入。」"""
    return _STRIP_EDGE_RE.sub("", text.strip())


def _is_date_query(query: str) -> bool:
    """短句＋含日期關鍵字才算查日期，避免長問句被誤判。」"""
    text = _strip_edge(query)
    if not text:
        return False
    # 去空白後再量長度，避免標點／空格撐大誤判
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 30:
        return False
    if any(kw in text for kw in _WEATHER_KEYWORDS):
        return False
    return any(kw in text for kw in _DATE_KEYWORDS)


def _is_chitchat(query: str) -> bool:
    """問候、道謝、道別等無需事實的短句，不強制搜尋。」"""
    text = _strip_edge(query).lower()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 20:
        return False
    # 含事實關鍵字不算閒聊，避免「你好請問天氣」被誤判跳過搜尋
    if any(kw in text for kw in _WEATHER_KEYWORDS + _REALTIME_KEYWORDS + _DATE_KEYWORDS):
        # 純問候才放行：完全等於問候詞才算
        return text in _CHITCHAT
    if text in _CHITCHAT:
        return True
    return any(text.startswith(w) and len(text) - len(w) <= 3 for w in _CHITCHAT if w)


def _needs_realtime(query: str) -> bool:
    """問句是否明確要求即時資料｜新手：新聞、股價、天氣這種直接上網，別翻筆記本。」"""
    text = _strip_edge(query).lower()
    if not text:
        return False
    # 與日期關鍵字隔離：只問「今天幾號」是日期捷徑，不算即時
    if _is_date_query(query):
        return False
    if not any(kw in text for kw in _REALTIME_KEYWORDS):
        return False
    # 本地意圖優先：含筆記／專案等詞視為查本地，不強制即時，交給工具迴圈自行決定
    if any(kw in text for kw in _LOCAL_KEYWORDS):
        return False
    return True


def _truncate_user_msg(msg: str) -> str:
    """使用者輸入截斷防爆：超 USER_MAX_CHARS 留頭＋註記，避免單輪撐爆上下文。」"""
    max_chars = _config.USER_MAX_CHARS
    text = msg.strip() if isinstance(msg, str) else str(msg or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…（過長已截斷）"


def _clean_query_for_search(query: str) -> str:
    """規則式查詢清洗：去口語填充詞＋壓空白＋截斷，提高快取命中與檢索召回。」"""
    max_chars = _config.SEARCH_QUERY_MAX_CHARS
    text = _strip_edge(query.strip() if isinstance(query, str) else str(query or ""))
    text = _QUERY_FILLER_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def _normalize_url(url: str) -> str:
    """URL 去重鍵：小寫＋去尾斜線＋去追蹤參數。」"""
    u = (url or "").strip().lower()
    u = re.sub(r"[?#].*$", "", u).rstrip("/")
    return u
