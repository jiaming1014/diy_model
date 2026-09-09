"""chat_core.py（原 ch8_29.py）：具工具呼叫能力的 Ollama 聊天核心｜新手教學版。

【這支程式是什麼？一句話】
它是聊天機器人的大腦，負責決定要直接回答、查日期、還是上網查資料。

【新手導讀（先看這裡，再看程式）】
1. 對話歷史：用 ChatState 記住最近幾組問答，就像小抄，讓模型有上下文不會失憶。
2. 日期捷徑：短句問日期（今天幾號、星期幾）直接用本機時間回答，不呼叫模型，比較快又不會抄錯。
3. 閒聊判斷：問候語（你好、謝謝）不需要搜尋，直接信任模型回答，省時間。
4. 工具流程：宣告 get_today、search_web 兩個工具，讓模型自己決定要不要查，就像給它兩張求救卡。
5. 降級策略：工具流程失敗或模型沒叫工具時，補一次傳統搜尋或直接回答，保證不中斷（備案的概念）。
6. 串流輸出：chat_w 是產生器（generator），一片一片 yield 文字，呼叫端可即時顯示，像直播字幕。

【程式跑起來的三條路（chat_w 裡）】
A. 日期路：短問句＋日期關鍵字 → 直接回今天日期。
B. 工具路：讓模型選工具 → 查日期／查網頁 → 再串流回答。
C. 傳統路：工具壞掉或關掉搜尋時走這裡，直接問模型，保證一定有話回你。
"""

# --- 標準函式庫匯入 ---
from collections.abc import Iterator  # 會一片一片回傳字串的串流函式｜新手：想成「會吐很多小紙條的機器」
from collections.abc import Mapping  # dict 與唯讀映射的共同介面，取欄位時共用
from dataclasses import dataclass, field  # 把相關狀態包成一包，取代散落的全域變數
from datetime import datetime  # 取得現在時間（配合台灣時區算今天日期）
from typing import Final, Literal, TypedDict  # Final：約好不改；Literal：限定字串；TypedDict：字典長相
from typing import NotRequired  # TypedDict 中可有可無的鍵
from zoneinfo import ZoneInfo  # ZoneInfo("Asia/Taipei") 取得台灣時間
import json  # 解析工具參數（arguments 可能是 JSON 字串）
import logging  # 取代 print，讓降級訊息可分級、不污染串流輸出
import re  # 正則統一清標點，比多層 strip 更穩
import threading  # 歷史紀錄加鎖，避免多執行緒同時改 hist 打架
import time  # 重試退避睡眠用

# --- 第三方套件匯入 ---
import ollama  # 呼叫本機 Ollama 模型做聊天（跟 AI 講話的電話）
from ddgs import DDGS  # DuckDuckGo 搜尋，上網查資料的工具

logger = logging.getLogger(__name__)  # 本模組的日誌器，上層決定要印多細

# 優化：嘗試匯入本地 RAG 檢索，缺檔或缺套件時降級為空函式，對話不受影響
try:
    from rag_qdrant import search_local as _rag_search  # 本地筆記向量檢索
except Exception:  # noqa: BROAD_EXCEPT_OK - rag 模組缺失時仍要能純網路問答
    def _rag_search(query: str, limit: int = 3) -> list[dict[str, str]]:
        return []


# --- 全域常數設定（唯一真相在 config.py，這裡保留同名，對外寫法不變）---
from config import HIST_MAX_CHARS as HIST_MAX_CHARS
from config import MAX_TOOL_ROUNDS as MAX_TOOL_ROUNDS
from config import OLLAMA_MODEL as OLLAMA_MODEL
from config import OLLAMA_RETRIES as OLLAMA_RETRIES
from config import OLLAMA_TIMEOUT as OLLAMA_TIMEOUT
from config import RAG_ENABLE as RAG_ENABLE
from config import RAG_MAX_CHARS as RAG_MAX_CHARS
from config import RAG_MAX_RESULTS as RAG_MAX_RESULTS
from config import SEARCH_CACHE_MAX as _SEARCH_CACHE_MAX
from config import SEARCH_CACHE_TTL as _SEARCH_CACHE_TTL
from config import SEARCH_MAX_RESULTS as SEARCH_MAX_RESULTS
from config import SEARCH_QUERY_MAX_CHARS as SEARCH_QUERY_MAX_CHARS
from config import SEARCH_TIMEOUT as SEARCH_TIMEOUT

# 優化：建立帶逾時的共用 client，避免 ollama.chat 等不到就永遠卡住
_ollama = ollama.Client(timeout=OLLAMA_TIMEOUT)

Role = Literal["system", "user", "assistant", "tool"]
ToolCall = tuple[str, dict[str, str]]  # (工具名, {參數名: 參數值})


class ChatMessage(TypedDict):
    """聊天訊息：role 是誰講的，content 是講什麼。"""

    role: Role
    content: str
    tool_calls: NotRequired[list[object]]


class SearchResult(TypedDict):
    """搜尋結果：標題＋摘要＋連結，排版給模型看。」"""

    title: str
    snippet: str
    url: str


@dataclass
class ChatState:
    """可攜的對話狀態，多會話隔離、好測試｜新手：每位客人拿自己的小抄本，互不干擾。」"""

    hist: list[ChatMessage] = field(default_factory=list)
    last_sources: list[SearchResult] = field(default_factory=list)
    last_rag: list[dict[str, str]] = field(default_factory=list)


# --- 對話記憶與規則關鍵字 ---
_default_state = ChatState()  # 預設會話，chat_w 未傳 state 時使用（相容舊匯入）
hist: list[ChatMessage] = _default_state.hist  # 相容別名：與 _default_state.hist 同一物件，請勿重綁
backtrace: Final[int] = 4
_last_sources: list[SearchResult] = _default_state.last_sources  # 相容別名：同物件，原地修改
_last_rag: list[dict[str, str]] = _default_state.last_rag  # 相容別名：同物件，原地修改
_hist_lock = threading.Lock()
_search_cache: dict[str, list[SearchResult]] = {}
_search_cache_time: dict[str, float] = {}
_search_cache_lock = threading.Lock()
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
# 本地筆記通常沒有這些；相反先搜網更快、答案更新
_REALTIME_KEYWORDS: Final[tuple[str, ...]] = (
    "新聞", "最新", "今天新聞", "股價", "匯率", "比特幣", "加密貨幣", "天氣", "氣溫",
    "颱風", "地震", "即時", "發生", "事件", "公告", "發布", "上市", "發表", "演出",
    "票價", "賽事", "比分", "排名", "榜單", "老闆", "當下", "目前狀況", "2026", "今年",
    "幾歲", "年齡", "出生", "多大了", "現年", "現況", "近況",
)
_STRIP_EDGE_RE: Final[re.Pattern[str]] = re.compile(r"^[\s，。！？、；：,.!?;:～~\-—]+|[\s，。！？、；：,.!?;:～~\-—]+$")


def _current_year() -> str:
    """工具描述用的年份，每年自動跟進，不再寫死 2025｜新手：菜單上的年份不用每年手改。」"""
    return str(datetime.now(ZoneInfo(TAIPEI_TZ)).year)


_tools_cache: dict[str, list[dict[str, object]]] = {}  # 優化：依年份快取，避免每次重建


def _tools() -> list[dict[str, object]]:
    """動態產生工具清單，年份隨現在走，但同一 session 內只算一次。」"""
    year = _current_year()
    cached = _tools_cache.get(year)
    if cached is not None:
        return cached
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_today",
                "description": "取得台灣今天的日期與星期，不需參數。問到今天、現在、日期、星期幾時優先呼叫。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": f"需要即時資訊、新聞、{year} 之後發生的事實時呼叫。用關鍵字查繁中網頁，回傳標題、摘要、來源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "繁中搜尋關鍵字"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    _tools_cache[year] = tools
    return tools


# 保留舊常數名供外部匯入，內容動態產生
TOOLS: Final[list[dict[str, object]]] = _tools()


def _resolve_state(state: ChatState | None) -> ChatState:
    """未傳 state 回預設單例，有傳用傳入的｜新手：沒帶小抄本就用全班共用那本。」"""
    return state if state is not None else _default_state


def _resolve_model(model: str | None) -> str:
    """統一解析聊天模型：未指定用 OLLAMA_MODEL｜新手：沒交代用哪支電話就打預設那支。」"""
    return model or OLLAMA_MODEL


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
    if not text or len(text) > 30:
        return False
    if any(kw in text for kw in _WEATHER_KEYWORDS):
        return False
    return any(kw in text for kw in _DATE_KEYWORDS)


def _is_chitchat(query: str) -> bool:
    """問候、道謝、道別等無需事實的短句，不強制搜尋。」"""
    text = _strip_edge(query).lower()
    if not text or len(text) > 20:
        return False
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
    return any(kw in text for kw in _REALTIME_KEYWORDS)


def _get_field(message: object, key: str) -> object:
    """同時相容 Mapping（含 dict）與物件的欄位取值｜新手：不管哪種包裝紙都用同一把剪刀拆。」"""
    if isinstance(message, Mapping):
        return message.get(key)
    return getattr(message, key, None)


def _trim_hist(state: ChatState | list[ChatMessage] | None = None, target: list[ChatMessage] | None = None) -> None:
    """裁歷史：先按組數裁，再按字數預算從舊裁｜新手：小抄太厚先撕整頁，還是厚就撕舊的字。

    優化：第一參數吃 ChatState 或裸 list（測試抓出的易誤用點），list 視為要裁的緩衝。
    """
    first: ChatState | list[ChatMessage] | None = state
    if isinstance(first, list):
        buf = first  # 直接傳 list：就裁它
    else:
        st = _resolve_state(first)
        buf = target if target is not None else st.hist
    with _hist_lock:
        while len(buf) > 2 * backtrace:
            del buf[0:2]
        # 優化：字數預算，避免長問答撐爆上下文
        total = sum(len(m.get("content", "")) for m in buf)
        while buf and total > HIST_MAX_CHARS:
            removed = buf.pop(0)
            total -= len(removed.get("content", ""))
            # 保持 user→assistant 成對：若剩奇數且開頭是 assistant 補撕一則
            if len(buf) % 2 == 1 and buf and buf[0].get("role") == "assistant":
                removed2 = buf.pop(0)
                total -= len(removed2.get("content", ""))


def _remember(user_msg: str, assistant_msg: str, state: ChatState | None = None, target: list[ChatMessage] | None = None) -> None:
    """把一組問答記入指定狀態，預設寫全域相容別名。」"""
    st = _resolve_state(state)
    buf = target if target is not None else st.hist
    with _hist_lock:
        buf.append({"role": "user", "content": user_msg})
        buf.append({"role": "assistant", "content": assistant_msg})
    _trim_hist(st, buf)


def _cache_get(key: str) -> list[SearchResult] | None:
    """TTL 快取讀取，過期回 None｜新手：小抄放 5 分鐘就丟掉，免得拿舊新聞回答。」"""
    with _search_cache_lock:
        hit = _search_cache.get(key)
        if hit is None:
            return None
        if time.monotonic() - _search_cache_time.get(key, 0.0) > _SEARCH_CACHE_TTL:
            _search_cache.pop(key, None)
            _search_cache_time.pop(key, None)
            return None
        return list(hit)


def _cache_put(key: str, value: list[SearchResult]) -> None:
    """TTL 快取寫入，超量淘汰最舊。」"""
    with _search_cache_lock:
        _search_cache[key] = list(value)
        _search_cache_time[key] = time.monotonic()
        if len(_search_cache) > _SEARCH_CACHE_MAX:
            oldest = next(iter(_search_cache))
            _search_cache.pop(oldest, None)
            _search_cache_time.pop(oldest, None)


def _search_web(query: str, max_results: int = SEARCH_MAX_RESULTS, state: ChatState | None = None) -> list[SearchResult]:
    """DDGS 網頁搜尋，失敗回空讓上層降級｜優化：原地更新 state，不重綁全域名稱。」"""
    st = _resolve_state(state)
    # 優化：原地清空，保持與模組別名同一物件，不再 `= []` 脫鉤
    with _hist_lock:
        st.last_sources.clear()
    short_query = query.strip()[:SEARCH_QUERY_MAX_CHARS].strip()
    if not short_query:
        return []
    cached = _cache_get(short_query)
    if cached is not None:
        with _hist_lock:
            st.last_sources.extend(cached)
        return list(cached)
    try:
        try:
            ddgs_ctx = DDGS(timeout=SEARCH_TIMEOUT)
        except TypeError:
            ddgs_ctx = DDGS()
        with ddgs_ctx as ddgs:
            raw = list(ddgs.text(short_query, region="tw-twn", max_results=max_results))
    except Exception as e:  # noqa: BROAD_EXCEPT_OK - httpx 錯誤型別不一，邊界統一降級
        logger.warning("網頁搜尋失敗，已降級為無結果：%s", e)
        return []
    results: list[SearchResult] = []
    for item in raw:
        title = str(item.get("title", "") or "")
        snippet = str(item.get("body", "") or "")
        url = str(item.get("href", "") or "")
        if title or snippet:
            results.append({"title": title, "snippet": snippet, "url": url})
    with _hist_lock:
        st.last_sources.extend(results)
    _cache_put(short_query, results)
    return results


def get_last_sources(state: ChatState | None = None) -> list[SearchResult]:
    """回傳最近一次搜尋結果（唯讀複本）。"""
    return list(_resolve_state(state).last_sources)


def get_last_rag(state: ChatState | None = None) -> list[dict[str, str]]:
    """回傳最近一次本地筆記命中（唯讀複本）。"""
    return list(_resolve_state(state).last_rag)


def _retrieve_rag(query: str, state: ChatState | None = None) -> list[dict[str, str]]:
    """查本地筆記並記住結果；關閉開關、閒聊、空字串時直接回空。」"""
    st = _resolve_state(state)
    with _hist_lock:
        st.last_rag.clear()
    if not RAG_ENABLE:
        return []
    if _is_chitchat(query) or _is_date_query(query):
        return []
    try:
        hits = _rag_search(query, limit=RAG_MAX_RESULTS)
    except Exception as e:  # noqa: BROAD_EXCEPT_OK - 嵌入／Qdrant 斷線降級為無命中
        logger.warning("本地檢索失敗，已降級為無命中：%s", e)
        return []
    with _hist_lock:
        st.last_rag.extend(hits)
    return hits


def _format_rag_results(hits: list[dict[str, str]]) -> str:
    """把本地筆記拼成模型看得懂的參考文字；無命中回空字串。

    優化：加入 RAG_MAX_CHARS 字數預算，單筆也截斷，防止長筆記撐爆上下文。
    """
    if not hits:
        return ""
    lines = ["以下為本地筆記（僅供參考，若與問題無關請忽略，回答請用繁體中文）："]
    budget = RAG_MAX_CHARS
    for i, h in enumerate(hits, start=1):
        text = str(h.get("text", "") or "")
        # 每則先截到單筆上限（預算的 1/2），避免一則獨佔
        per_hit = max(200, budget // 2)
        if len(text) > per_hit:
            text = text[:per_hit] + "…"
        header = f"[筆記{i}｜{h.get('source', '')}]\n{text}"
        if budget <= 0:
            break
        if len(header) > budget:
            lines.append(header[:budget] + "…")
            break
        lines.append(header)
        budget -= len(header)
    return "\n\n".join(lines)


def _format_search_results(results: list[SearchResult]) -> str:
    """把搜尋結果拼成模型看得懂的事實文字。」"""
    if not results:
        return "搜尋無結果。"
    lines = ["以下為已發生的事實："]
    for res in results:
        lines.append(f"標題：{res['title']}\n摘要：{res['snippet']}\n來源：{res['url']}")
    lines.append("請依照上述事實回答問題。")
    return "\n\n".join(lines)


def _extract_tool_calls(message: object) -> list[ToolCall]:
    """相容 dict 與物件兩種回傳，取出 (工具名, 參數)。"""
    raw_calls = _get_field(message, "tool_calls")
    if not raw_calls:
        return []
    calls: list[ToolCall] = []
    items = raw_calls if isinstance(raw_calls, list) else []
    for item in items:
        func = _get_field(item, "function")
        name = _get_field(func, "name") if func is not None else None
        args = _get_field(func, "arguments") if func is not None else None
        if not name:
            continue
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                args_dict = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                args_dict = {}
        elif isinstance(args, dict):
            args_dict = args
        else:
            args_dict = {}
        clean: dict[str, str] = {str(k): str(v) for k, v in args_dict.items()}
        calls.append((str(name), clean))
    return calls


def _assistant_text(message: object) -> str:
    """取出 assistant 文字，None 回空字串。」"""
    raw = _get_field(message, "content")
    return str(raw) if raw is not None else ""


def _assistant_tool_message(message: object) -> ChatMessage:
    """保留 tool_calls 的 assistant 訊息，保持 user→assistant→tool 順序。」"""
    raw_calls = _get_field(message, "tool_calls")
    msg: ChatMessage = {"role": "assistant", "content": _assistant_text(message)}
    if isinstance(raw_calls, list) and raw_calls:
        msg["tool_calls"] = list(raw_calls)
    return msg


def _run_tool(name: str, args: dict[str, str], user_msg: str, state: ChatState | None = None) -> str:
    """執行單一工具，回傳餵給模型的文字結果。」"""
    if name == "get_today":
        return f"今天是 {_today_str()}（台灣時間）。"
    if name == "search_web":
        query = args.get("query", user_msg).strip() or user_msg
        return _format_search_results(_search_web(query, state=state))
    logger.warning("收到未知工具呼叫：%s，已要求模型直接回答", name)
    return f"未知工具：{name}，請直接回答使用者問題。"


def _call_chat_once(messages: list[ChatMessage], model: str, tools: list[dict[str, object]] | None = None) -> object:
    """呼叫一次 ollama.chat，tools 為 None 表示純問答。」"""
    if tools is None:
        return _ollama.chat(model=model, messages=messages)
    return _ollama.chat(model=model, messages=messages, tools=tools)


def _call_chat_with_retry(messages: list[ChatMessage], model: str, tools: list[dict[str, object]] | None = None) -> object:
    """帶指數退避重試的非串流呼叫｜新手：電話打不通等一下再撥，越等越久。」"""
    last_err: Exception | None = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            return _call_chat_once(messages, model, tools)
        except Exception as e:  # noqa: BROAD_EXCEPT_OK - 連線／模型錯誤邊界重試
            last_err = e
            logger.warning("ollama.chat 第 %d 次失敗：%s", attempt + 1, e)
            if attempt < OLLAMA_RETRIES:
                time.sleep(0.5 * (2 ** attempt))  # 0.5s、1s 退避
    assert last_err is not None
    raise last_err


def _stream_reply(messages: list[ChatMessage], model: str | None = None) -> Iterator[str]:
    """串流取得 ollama 回覆片段，有文字才 yield。

    優化：捕獲串流中斷（雲端模型網路不穩），已產生的片段仍保留、
    讓上層 _remember 正常記入歷史，不因中斷丟失整輪。
    """
    use_model = _resolve_model(model)
    try:
        stream = _ollama.chat(model=use_model, messages=messages, stream=True)
    except Exception as e:  # noqa: BROAD_EXCEPT_OK - 建立串流失敗也要降級
        logger.warning("串流建立失敗（%s），回覆為空", e)
        return
    try:
        for chunk in stream:
            try:
                message_obj = _get_field(chunk, "message")
                raw_content = _get_field(message_obj, "content") if message_obj is not None else None
                content = str(raw_content) if raw_content is not None else None
                if content:
                    yield content
            except Exception as e:  # noqa: BROAD_EXCEPT_OK - 單一 chunk 解析失敗，跳過不炸整輪
                logger.warning("串流 chunk 解析失敗（%s），已跳過", e)
                continue
    except Exception as e:  # noqa: BROAD_EXCEPT_OK - 串流中斷（雲端斷線），保留已生成片段
        logger.warning("串流中斷（%s），已保留已生成的片段", e)
        return


def _system_content(messages: list[ChatMessage], today: str) -> str:
    """安全取系統提示，缺失回預設｜新手：黑板第一行不見就補一張新的。」"""
    if messages and messages[0].get("role") == "system":
        c = messages[0].get("content", "")
        if isinstance(c, str) and c.strip():
            return c
    return f"今天是 {today}（台灣時間）。請一律使用繁體中文（台灣用語）回答。"


def _build_base_messages(sys_msg: str, user_msg: str, today: str, rag_block: str, state: ChatState | None = None) -> list[ChatMessage]:
    """組基本提示詞：系統＋歷史＋（本地筆記）＋本次問題。」"""
    st = _resolve_state(state)
    sys_with_date: ChatMessage = {
        "role": "system",
        "content": f"{sys_msg}\n今天是 {today}（台灣時間），回答時間問題以此為準。請一律使用繁體中文（台灣用語）回答。",
    }
    with _hist_lock:
        hist_copy = list(st.hist)
    if rag_block:
        return [sys_with_date] + hist_copy + [{"role": "user", "content": f"{rag_block}\n\n---\n\n使用者問題：{user_msg}"}]
    return [sys_with_date] + hist_copy + [{"role": "user", "content": user_msg}]


def _run_tool_loop(base: list[ChatMessage], user_msg: str, use_model: str, state: ChatState | None = None) -> list[ChatMessage]:
    """跑最多 MAX_TOOL_ROUNDS 輪工具呼叫，回傳含工具結果的訊息串。」"""
    st = _resolve_state(state)
    messages: list[ChatMessage] = list(base)
    for _ in range(MAX_TOOL_ROUNDS):
        first = _call_chat_with_retry(messages, use_model, _tools())
        msg_obj = first["message"] if isinstance(first, dict) else _get_field(first, "message")
        calls = _extract_tool_calls(msg_obj)
        if not calls:
            break
        messages.append(_assistant_tool_message(msg_obj))
        for tool_name, tool_args in calls:
            messages.append({"role": "tool", "content": _run_tool(tool_name, tool_args, user_msg, state=st)})
    return messages


def _rag_threshold() -> float:
    """讀重排門檻：優先吃 reranker 運行值（測試可 mock），缺失回 -inf（= 不設限）。"""
    try:
        import reranker as _r

        return float(_r.RERANK_THRESHOLD)
    except Exception:
        return float("-inf")


_RAG_VECTOR_FLOOR: Final[float] = 0.5  # 未設門檻時，0~1 餘弦分數的內建地板，擋明顯不相關命中


def _rag_adequate(hits: list[dict[str, str]] | None) -> bool:
    """RAG 是否真夠用：有命中且最高分過門檻才算夠，避免不相關筆記誤判為夠用。"""
    if not hits:
        return False
    scores: list[float] = []
    for h in hits:
        try:
            v = h.get("score", None)
            if v is None or v == "":
                continue
            f = float(v)  # type: ignore[arg-type]
            if f != f or f == float("inf"):
                continue  # nan 與缺分數哨兵當無分數
            scores.append(f)
        except (TypeError, ValueError):
            continue
    if not scores:
        return True  # 無分數可判，沿用舊行為（有命中即夠）
    try:
        thresh = _rag_threshold()
    except Exception:
        thresh = float("-inf")
    if thresh > float("-inf"):
        return max(scores) >= thresh
    # 未設門檻：僅對 0~1 餘弦分數用內建地板擋明顯不相關；其他量尺沿用舊行為
    if all(0.0 <= s <= 1.0 for s in scores):
        return max(scores) >= _RAG_VECTOR_FLOOR
    return True


def _handle_tool_flow(messages: list[ChatMessage], user_msg: str, today: str, rag_block: str, use_model: str, state: ChatState | None = None, rag_hits: list[dict[str, str]] | None = None) -> Iterator[str]:
    """工具流程收尾：必要時補傳統搜尋，再串流回傳。」

    優化：RAG 已有命中（本地資料夠）時不再無條件補搜網；
    只有 RAG 空手、或明確要求即時資料，才補一次網路搜尋。
    """
    st = _resolve_state(state)
    # 是否有本機足夠依據：本地筆記有命中且分數過門檻，或問句本身不需外部事實（閒聊）
    if rag_hits is not None:
        local_adequate = _rag_adequate(rag_hits) or _is_chitchat(user_msg)
    else:
        local_adequate = bool(rag_block) or _is_chitchat(user_msg)
    # 明確要求即時資料時，本地命中不足以取代網路，仍要補搜
    wants_realtime = _needs_realtime(user_msg)
    needs_legacy_search = all(m.get("role") != "tool" for m in messages) and not local_adequate
    if (needs_legacy_search or wants_realtime) and not _is_chitchat(user_msg):
        facts = _format_search_results(_search_web(user_msg, state=st))
        parts = [f"今天是 {today}（台灣時間）。"]
        if rag_block:
            parts.append(rag_block)
        parts.append(facts)
        parts.append(f"使用者問題：{user_msg}")
        content = "\n\n---\n\n".join(parts)
        sys_with_date: ChatMessage = {"role": "system", "content": _system_content(messages, today)}
        with _hist_lock:
            hist_copy = list(st.hist)
        messages = [sys_with_date] + hist_copy + [{"role": "user", "content": content}]
    reply_full = ""
    for reply in _stream_reply(messages, model=use_model):
        reply_full += reply
        yield reply
    _remember(user_msg, reply_full, state=st)


def _handle_legacy_flow(sys_msg: str, user_msg: str, today: str, use_model: str, state: ChatState | None = None) -> Iterator[str]:
    """傳統降級流程：不做工具增強，直接問模型。」"""
    st = _resolve_state(state)
    sys_with_date: ChatMessage = {
        "role": "system",
        "content": f"{sys_msg}\n今天是 {today}（台灣時間），回答時間問題以此為準。請一律使用繁體中文（台灣用語）回答。",
    }
    with _hist_lock:
        hist_copy = list(st.hist)
    reply_full = ""
    for reply in _stream_reply([sys_with_date] + hist_copy + [{"role": "user", "content": user_msg}], model=use_model):
        reply_full += reply
        yield reply
    _remember(user_msg, reply_full, state=st)


def _handle_date_query(user_msg: str, state: ChatState | None = None) -> str:
    """日期捷徑：短句問日期直接組今天回答。」"""
    st = _resolve_state(state)
    reply_full = f"今天是 {_today_str()}（台灣時間）。"
    _remember(user_msg, reply_full, state=st)
    return reply_full


def chat_w(sys_msg: str, user_msg: str, search_g: bool = True, model: str | None = None, state: ChatState | None = None) -> Iterator[str]:
    """主聊天函式（產生器）：日期捷徑→工具流程→傳統降級，最後串流回傳並記入歷史。

    state 為 None 用全域預設（相容舊呼叫），傳入 ChatState 即多會話隔離。
    """
    st = _resolve_state(state)
    use_model = _resolve_model(model)
    # 修正：每輪先清空上一輪殘留，避免本輪未搜網時 CLI 誤印舊來源
    with _hist_lock:
        st.last_rag.clear()
        st.last_sources.clear()
    if _is_date_query(user_msg):
        yield _handle_date_query(user_msg, state=st)
        return
    today = _today_str()
    _trim_hist(st)
    # 優化：明確要求即時資料的問句，跳過本地筆記檢索（RAG + reranker），直接進工具／搜尋流程
    # 本地筆記通常沒有新聞、股價、天氣等即時內容；先搜網更快且答案更新
    if _needs_realtime(user_msg):
        rag_hits = []
    else:
        rag_hits = _retrieve_rag(user_msg, state=st)
    rag_block = _format_rag_results(rag_hits)
    base = _build_base_messages(sys_msg, user_msg, today, rag_block, state=st)
    if search_g is True:
        try:
            messages = _run_tool_loop(base, user_msg, use_model, state=st)
            yield from _handle_tool_flow(messages, user_msg, today, rag_block, use_model, state=st, rag_hits=rag_hits)
            return
        except Exception as e:  # noqa: BROAD_EXCEPT_OK - 工具流程失敗降級為舊流程
            logger.warning("工具流程降級為傳統流程：%s", e)
            pass
    yield from _handle_legacy_flow(sys_msg, user_msg, today, use_model, state=st)
