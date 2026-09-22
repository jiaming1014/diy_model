"""ollama_shared.py：共用 Ollama client（依 timeout 快取）｜新手教學版。

【為什麼要共用？】
以前 chat_core、rag_qdrant、reranker 各建一個 ollama.Client，
同一個 timeout 卻有三份連線池。現在依 timeout 快取：
同 timeout 拿同一顆，不同 timeout 各拿一顆，記憶體只多一個小字典。

【生命週期】
快取內的 client 不主動關閉（可能多個模組同時在用），
程序結束隨 GC 回收。timeout 種類通常只有一種，不會無限成長。
"""

import threading
from typing import Any

_clients: dict[float, Any] = {}  # timeout → client，同一個 timeout 只留一顆
_lock = threading.Lock()  # 保護 _clients，多模組同時取用不打架


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
        return fresh


def remember(timeout: float, client: Any) -> None:
    """把外部建好的 client 登記進快取（timeout 變更路徑用），方便後續共用。"""
    key = float(timeout)
    if key != key or key in (float("inf"), float("-inf")):
        raise ValueError(f"timeout 須為有限數值（收到 {timeout!r}）")
    with _lock:
        _clients[key] = client  # 覆蓋同 timeout 舊值，舊的由呼叫端負責關


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


def probe_model(model: str, timeout: float, timeout_s: float = 5.0) -> bool:
    """探測指定模型是否在 Ollama 就緒，逾時當不可用｜新手：--health 出發前點名，缺誰早知道。

    名單比對沿用重排探測的寬鬆規則（大小寫不敏感、任一包含即算）。
    """
    try:
        import concurrent.futures

        client = get_shared_client(timeout)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            models = ex.submit(client.list).result(timeout=timeout_s)
    except Exception:
        return False
    names = ollama_model_names(models)
    if not names:
        return True  # 連線成功但形狀不明，保守當可用（實際呼叫失敗時上層會再降級）
    want = (model or "").strip().lower()
    return bool(want) and any(want in n.lower() or n.lower() in want for n in names if n)


def clear() -> None:
    """測試用：清空快取（避免 mock 實例污染後續測試）。"""
    with _lock:
        _clients.clear()  # 只清表不 close：測試替身不一定有 close
