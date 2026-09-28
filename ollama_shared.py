"""ollama_shared.py：共用 Ollama client（依 timeout 快取）｜新手教學版。

【為什麼要共用？】
以前 chat_core、rag_qdrant、reranker 各建一個 ollama.Client，
同一個 timeout 卻有三份連線池。現在依 timeout 快取：
同 timeout 拿同一顆，不同 timeout 各拿一顆，記憶體只多一個小字典。

【生命週期】
快取內的 client 不主動關閉（可能多個模組同時在用），
程序結束隨 GC 回收。timeout 種類通常只有一種；種類異常變多時按插入順序淘汰最舊，
上限 _MAX_CLIENTS 顆，被淘汰的實例若他處仍有引用則照常用（只出快取表）。
"""

import concurrent.futures
import threading
from typing import Any, Final

_MAX_CLIENTS: Final[int] = 8  # 正常只有 1 種 timeout，上限擋浮點抖動等異常成長

_clients: dict[float, Any] = {}  # timeout → client，同一個 timeout 只留一顆（插入有序，最舊在前）
_lock = threading.Lock()  # 保護 _clients，多模組同時取用不打架


def _evict_oldest_locked() -> None:
    """已持有鎖時清到上限內：踢最早插入（dict 迭代首位），不 close（可能他處仍在用）。"""
    while len(_clients) > _MAX_CLIENTS:
        oldest = next(iter(_clients))
        del _clients[oldest]


def get_shared_client(timeout: float) -> Any:
    """依 timeout 取共用 client，同 timeout 回同一顆；建不出拋錯交上層降級。

    非有限逾時（nan／inf）直接拋錯：nan 當鍵永遠 miss 會無限建新 client。
    """
    key = float(timeout)
    # nan 自己不等於自己（nan != nan 恆真），這是標準判法；inf 另行擋下
    if key != key or key in (float("inf"), float("-inf")):
        raise ValueError(f"timeout 須為有限數值（收到 {timeout!r}）")
    with _lock:  # 快路徑：快取已有就回，不進慢路徑
        hit = _clients.get(key)
        if hit is not None:
            return hit
    import ollama  # 延遲匯入：缺套件時由呼叫端 except 降級

    fresh = ollama.Client(timeout=key)
    with _lock:  # 慢路徑再進鎖：避免兩執行緒各建一顆
        hit = _clients.get(key)
        if hit is not None:  # 別人先建好了，丟掉自己這顆改用他的
            try:
                fresh.close()
            except Exception:
                pass
            return hit
        _clients[key] = fresh
        _evict_oldest_locked()
        return fresh


def remember(timeout: float, client: Any) -> None:
    """把外部建好的 client 登記進快取（timeout 變更路徑用），方便後續共用。"""
    key = float(timeout)
    if key != key or key in (float("inf"), float("-inf")):
        raise ValueError(f"timeout 須為有限數值（收到 {timeout!r}）")
    with _lock:
        _clients[key] = client  # 覆蓋同 timeout 舊值，舊的由呼叫端負責關
        _evict_oldest_locked()


def is_managed(client: Any) -> bool:
    """是否為快取內的共用實例；是的話呼叫端不應自行 close。"""
    with _lock:
        return any(client is c for c in _clients.values())  # 比身分（is），不比內容


def ollama_model_names(models: object) -> list[str]:
    """從 ollama list() 回傳值抽模型名，認 dict／list／回傳物件（含 .models 的 ListResponse）。

    舊版只認 dict／list，ollama 0.6 真回物件時名字永遠比對不到，只能保守當可用。
    """
    items: list[object] = []
    try:
        if isinstance(models, dict):
            items = models.get("models", []) or []
        elif isinstance(models, list):
            items = list(models)
        else:
            items = list(getattr(models, "models", None) or [])
    except Exception:
        return []
    names: list[str] = []
    for m in items:
        try:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model")
            elif isinstance(m, str):
                n = m
            else:
                n = getattr(m, "model", None) or getattr(m, "name", None)
            if n:
                names.append(str(n))
        except Exception:
            continue
    return names


# 模組級共用執行緒池：probe_model 每次新建池會重複建立/銷毀執行緒，共用省開銷
_PROBE_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_PROBE_POOL_LOCK = threading.Lock()


def _get_probe_pool() -> concurrent.futures.ThreadPoolExecutor:
    """取得模組級共用執行緒池（延遲建立，max_workers=1 足以為 list() 加逾時）。"""
    global _PROBE_POOL
    with _PROBE_POOL_LOCK:
        if _PROBE_POOL is None:
            _PROBE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return _PROBE_POOL


def probe_model(model: str, timeout: float, timeout_s: float = 5.0) -> bool:
    """偵測指定模型是否在 Ollama 就緒，逾時當不可用｜新手：--health 出發前點名，缺誰早知道。

    名單比對沿用重排偵測的寬鬆規則（大小寫不敏感、任一包含即算）。
    """
    try:
        client = get_shared_client(timeout)
        ex = _get_probe_pool()
        models = ex.submit(client.list).result(timeout=timeout_s)
    except Exception:
        return False
    names = ollama_model_names(models)
    if not names:
        return True  # 連線成功但形狀不明，保守當可用（實際呼叫失敗時上層會再降級）
    want = (model or "").strip().lower()
    return bool(want) and any(want in n.lower() or n.lower() in want for n in names if n)


def extract_ollama_message(msg: object) -> str:
    """從 Ollama 回覆物件取出 message 文字，認 dict／回傳物件（含 .message 的 ChatResponse）。

    舊版各模組各自判斷 msg 是 dict 還是物件，格式漂移時要改多處；
    統一抽到這裡，Ollama SDK 改回覆格式只需改一處。
    """
    try:
        if isinstance(msg, dict):
            raw = msg.get("message")
        else:
            raw = getattr(msg, "message", None)
        if isinstance(raw, dict):
            return str(raw.get("content") or "")
        return str(raw) if raw is not None else ""
    except Exception:
        return ""


def clear() -> None:
    """測試用：清空快取（避免 mock 實例污染後續測試）。"""
    with _lock:
        _clients.clear()  # 只清表不 close：測試替身不一定有 close
