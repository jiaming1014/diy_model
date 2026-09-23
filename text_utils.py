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
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import config as _config

TAIPEI_TZ: Final[str] = "Asia/Taipei"
_WEEKDAY_ZH: Final[tuple[str, ...]] = ("一", "二", "三", "四", "五", "六", "日")
_DATE_KEYWORDS: Final[tuple[str, ...]] = (
    "今天", "今日", "現在", "日期", "幾號", "星期", "禮拜", "時間", "幾點", "年月日",
)
# 實詞：單獨出現即算查日期；軟詞（今天／今日／現在）需搭配實詞，否則今日頭條、現在流行什麼會被誤判
_DATE_CORE_KEYWORDS: Final[tuple[str, ...]] = (
    "日期", "幾號", "星期", "禮拜", "時間", "幾點", "年月日",
)
_DATE_SOFT_KEYWORDS: Final[tuple[str, ...]] = ("今天", "今日", "現在")
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
    "新聞", "最新", "股價", "匯率", "比特幣", "加密貨幣", "天氣", "氣溫", "下雨", "下雪",
    "颱風", "地震", "即時", "公告", "發布", "上市", "發表", "演出",
    "票價", "賽事", "比分", "排名", "榜單",
)
# 本地意圖關鍵字：含這些優先視為查本地筆記，不強制即時，交給工具迴圈自行決定是否上網
_LOCAL_KEYWORDS: Final[tuple[str, ...]] = (
    "筆記", "專案", "資料夾", "嵌入", "收藏", "本地", "notes", "note",
)
# P19 工具意圖：工作區檔案操作，命中即跳過 RAG（本地筆記幫不上寫檔）
# P22 收緊：拿掉過寬的「寫一」「存到」，避免「寫一封信」「存到哪」誤判為寫檔請求
_WORKSPACE_KEYWORDS: Final[tuple[str, ...]] = (
    "ai_workspace", "ai workspace", "工作區",
    "新增檔案", "建立檔案", "修改檔案", "寫入檔案", "寫成檔案", "存成檔案", "轉成檔案",
    "新增資料夾", "建立資料夾", "建資料夾",
    "寫個", "存成", "存檔",
)
# P19 工具意圖：音樂播放，命中即跳過 RAG（聽歌不需翻筆記）
_MUSIC_KEYWORDS: Final[tuple[str, ...]] = (
    "youtube", "油管",
    "播歌", "放歌", "聽歌", "放首",
    "首歌", "歌曲", "單曲",
    "播音樂", "放音樂", "聽音樂", "播放音樂",
)
# P22 補漏判：播放動詞＋音樂名詞的組合也算，涵蓋「播周杰倫的歌」等常見說法
_MUSIC_PLAY_VERBS: Final[tuple[str, ...]] = ("播", "點", "放", "聽")
_MUSIC_NOUNS: Final[tuple[str, ...]] = ("歌", "音樂", "單曲", "專輯", "youtube", "油管")
_STRIP_EDGE_RE: Final[re.Pattern[str]] = re.compile(r"^[\s，。！？、；：,.!?;:～~\-—]+|[\s，。！？、；：,.!?;:～~\-—]+$")
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_QUERY_FILLER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(請問一下|請問|幫我查一下|幫我找一下|查一下|找一下|謝謝|麻煩)[，,。\s]*|[？?！!啊呢吧喔哦]+$"
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
    """短句＋含日期關鍵字才算查日期，避免長問句被誤判。

    軟詞（今天／今日／現在）單獨出現不算，需搭配實詞或短到只剩它，
    否則「今日頭條」「現在流行什麼」會被誤回今天幾號。
    """
    text = _strip_edge(query)
    if not text:
        return False
    # 去空白後再量長度，避免標點／空格撐大誤判
    compact = _WS_RE.sub("", text)
    if len(compact) > 30:
        return False
    if any(kw in text for kw in _WEATHER_KEYWORDS):
        return False
    if any(kw in text for kw in _DATE_CORE_KEYWORDS):
        return True
    if any(kw in text for kw in _DATE_SOFT_KEYWORDS):
        return len(compact) <= 4
    return False


def _is_chitchat(query: str) -> bool:
    """問候、道謝、道別等無需事實的短句，不強制搜尋。」"""
    text = _strip_edge(query).lower()
    if not text:
        return False
    compact = _WS_RE.sub("", text)
    if len(compact) > 20:
        return False
    # 含事實關鍵字不算閒聊，避免「你好請問天氣」被誤判跳過搜尋
    if any(kw in text for kw in _WEATHER_KEYWORDS + _REALTIME_KEYWORDS + _DATE_KEYWORDS):
        # 純問候才放行：完全等於問候詞才算
        return text in _CHITCHAT
    if text in _CHITCHAT:
        return True
    # 問候詞開頭＋多不超過 3 字也算（如 嗨嗨嗨、謝謝你），再長就是有事要問了
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
    text = msg.strip() if isinstance(msg, str) else ("" if msg is None else str(msg))
    # 管線／轉貼可能帶進孤立代理字（surrogates），會炸 json 存檔與模型序列化，入口先清掉
    text = text.encode("utf-8", "ignore").decode("utf-8")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…（過長已截斷）"


def _clean_query_for_search(query: str, max_chars: int | None = None) -> str:
    """規則式查詢清洗：去口語填充詞＋壓空白＋截斷，提高快取命中與檢索召回。

    max_chars 沒給走搜尋預算；RAG 檢索可傳 RAG 預算用足額度。
    """
    budget = _config.SEARCH_QUERY_MAX_CHARS if max_chars is None else max(1, int(max_chars))
    text = _strip_edge(query.strip() if isinstance(query, str) else ("" if query is None else str(query)))
    text = _QUERY_FILLER_RE.sub("", text).strip()
    text = _WS_RE.sub(" ", text).strip()
    return text[:budget].strip()


# 去重時忽略的追蹤參數（utm 家族＋各平台點擊標記），其餘 query 保留以免不同文章誤判同一頁
# OPT-11：片段與尾符正則預編譯，_normalize_url 熱路徑不再現編譯
_FRAG_RE: Final[re.Pattern[str]] = re.compile(r"#.*$")
_TRAIL_QS_RE: Final[re.Pattern[str]] = re.compile(r"[?&]$")
_TRACKING_PARAM_RE: Final[re.Pattern[str]] = re.compile(
    r"([?&])(?:utm(?:_[a-z_]+)?|gclid|gbraid|wbraid|fbclid|msclkid|mc_cid|mc_eid|igshid)(=[^&]*)?",
    re.IGNORECASE,
)


_CRED_IN_URL_RE: Final[re.Pattern[str]] = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<userinfo>[^/@\s]+)@")


def redact_url_creds(text: str) -> str:
    """把文字中的 URL 帳密遮蔽成 scheme://***@host（L1）。

    錯誤訊息、日誌與 --health 輸出可能夾帶 QDRANT_URL；若使用者以
    https://user:pass@host 形式設定（Qdrant 基本驗證常見寫法），
    原文會把密碼印出來。這裡做通用遮蔽，不需知道設定了什麼。
    """
    if not text:
        return text
    return _CRED_IN_URL_RE.sub(lambda m: f"{m.group('scheme')}***@", text)


def _normalize_url(url: str) -> str:
    """URL 去重鍵：去 # 片段與追蹤參數＋去尾斜線；只小寫 scheme＋host（DNS 不分大小寫），path／query 保大小寫。」"""
    raw = url.strip() if isinstance(url, str) else ("" if url is None else str(url))
    u = _FRAG_RE.sub("", raw)
    # scheme＋host 小寫即可，path／query 原樣保留（Linux 路徑大小寫敏感）
    try:
        parts = urlsplit(u)
        if parts.netloc:
            netloc = parts.netloc if "@" in parts.netloc else parts.netloc.lower()
            u = urlunsplit((parts.scheme.lower(), netloc, parts.path, parts.query, ""))
    except Exception:
        u = u.lower()
    u = _TRACKING_PARAM_RE.sub(r"\1", u)
    u = u.replace("?&", "?")
    while "&&" in u:
        u = u.replace("&&", "&")
    u = _TRAIL_QS_RE.sub("", u).rstrip("/")
    return u


# 含「遊戲」的問句不走寬鬆的「存檔」匹配（遊戲進度存檔是查資料不是寫檔）
_GAME_WORDS: Final[tuple[str, ...]] = ("遊戲", "游戲", "game")
_WORKSPACE_STRONG_KEYWORDS: Final[tuple[str, ...]] = tuple(kw for kw in _WORKSPACE_KEYWORDS if kw != "存檔")
# 窄語境動作關鍵字：強關鍵字再去掉單純提及（工作區／目錄名），只剩動手做的動作
_WORKSPACE_ACTION_KEYWORDS: Final[tuple[str, ...]] = tuple(
    kw for kw in _WORKSPACE_STRONG_KEYWORDS if kw not in ("ai_workspace", "ai workspace", "工作區")
)


def _needs_workspace(query: str) -> bool:
    """是否為工作區檔案操作（寫檔／建資料夾）｜新手：動手做檔案，就別浪費時間翻筆記。

    窄語境（遊戲／聽說）：單純提及不算，只認動作關鍵字；自訂目錄名提及也算。
    """
    return _needs_workspace_norm(_strip_edge(query).lower())


def _needs_workspace_norm(text: str) -> bool:
    """工作區判定核心（吃已正規化文本），供 _detect_intent 共用一次正規化。」"""
    if not text:
        return False
    if any(w in text for w in _GAME_WORDS) or any(w in text for w in _HEARSAY_WORDS):
        return any(kw in text for kw in _WORKSPACE_ACTION_KEYWORDS)
    if any(kw in text for kw in _WORKSPACE_KEYWORDS):
        return True
    # 自訂目錄名：改名後提及新目錄名也算意圖（預設名已在關鍵字表內，不重複）
    ws = (_config.WORKSPACE_DIRNAME or "").strip().lower()
    return bool(ws) and ws not in ("ai_workspace", "ai workspace") and ws in text


# 含「聽說」的問句不走寬鬆組合判斷（「聽說這首歌…」是打聽消息不是點歌），只認精確音樂關鍵字
_HEARSAY_WORDS: Final[tuple[str, ...]] = ("聽說", "听说", "據說", "据说")
_MUSIC_EXACT_KEYWORDS: Final[frozenset[str]] = frozenset({
    "youtube", "油管", "播歌", "放歌", "聽歌", "播音樂", "放音樂", "聽音樂", "播放音樂",
})


def _needs_music(query: str) -> bool:
    """是否為音樂播放請求｜新手：聽歌不需翻筆記，直接開 YouTube。

    P22：除關鍵字硬比對外，「播放動詞＋音樂名詞」組合也算，涵蓋口語說法。
    排除：含聽說只認精確關鍵字；含寫／創作且無明確音樂標記不算；
    放首排除前字為開（開放首先）。
    """
    return _needs_music_norm(_strip_edge(query).lower())


def _needs_music_norm(text: str) -> bool:
    """音樂判定核心（吃已正規化文本），供 _detect_intent 共用一次正規化。」"""
    if not text:
        return False
    if any(w in text for w in _HEARSAY_WORDS):
        return any(kw in text for kw in _MUSIC_EXACT_KEYWORDS)
    if (any(w in text for w in ("寫", "写", "創作", "创作")) and "播放" not in text
            and not any(kw in text for kw in _MUSIC_EXACT_KEYWORDS)):
        # 寫歌／創作是作曲查資料，不是點歌；有明確音樂標記（YouTube／播歌…）時不排除，交上層判衝突
        return False
    for kw in _MUSIC_KEYWORDS:
        if kw == "放首":
            # 「開放／开放首先登記」不是點歌：排除前字為開的命中（簡體开一起認）
            if re.search(r"(?<![開开])放首", text):
                return True
            continue
        if kw in text:
            return True
    return any(v in text for v in _MUSIC_PLAY_VERBS) and any(n in text for n in _MUSIC_NOUNS)


def _detect_intent(query: str) -> str | None:
    """工具意圖分流：music／workspace／None（一般問答，工具全給）。

    偵測不到回 None，上層走完整工具清單，寧可多送 token 也不讓功能失效。
    P21：兩者都命中（如含 YouTube 又要寫檔）回 None 給完整清單，避免誤刪工具。
    OPT-20：正規化一次後共用內部判定，每訊息省一次 strip＋lower。
    """
    text = _strip_edge(query).lower()
    want_music = _needs_music_norm(text)
    want_workspace = _needs_workspace_norm(text)
    if want_music and want_workspace:
        return None
    if want_music:
        return "music"
    if want_workspace:
        return "workspace"
    return None
