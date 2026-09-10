"""reranker.py：本地筆記重排模組（CrossEncoder 優先，Ollama LLM 備援）｜新手教學版。

【這支程式在做什麼？（白話版）】
向量檢索像用漁網撈魚：撈得快，但小魚小蝦也會進來。
重排（rerank）就是請評審逐條打分，把最對題的 3 條排前面。

【兩種評審】
1. CrossEncoder（優先）：專門打分的本地小模型，中文推 bge-reranker-v2-m3，
   又快又準，但要 pip install sentence-transformers。
2. Ollama LLM（備援）：沒裝 CrossEncoder 時，用你本來就有的 Ollama 模型
   幫每條打 0~10 分，不用裝新套件，慢一點但零依賴。

【降級保證】
兩種評審都不在，就回原順序，不讓對話中斷（跟 rag_qdrant 的哲學一致）。

【常用環境變數】
- RERANK_ENABLE=1/0：總開關，預設 1（開），設 0 直接回原順序
- RERANK_BACKEND=auto/crossencoder/llm/none：強制指定評審，預設 auto
- RERANK_MODEL：CrossEncoder 模型名，預設 BAAI/bge-reranker-v2-m3
- RERANK_LLM_MODEL：LLM 備援用的 Ollama 模型，預設跟著 OLLAMA_MODEL
- RERANK_THRESHOLD：最低分，低於此分直接丟掉，預設不設限
- RERANK_SNIPPET_CHARS：LLM 每條看幾個字，預設 300
"""

import concurrent.futures  # 給 probe 加逾時，避免 ollama.list 卡住
import json
import logging
import re
import threading

# 設定唯一真相在 config.py，這裡保留同名，對外寫法與測試 mock 不變
import config as _config_module
from config import OLLAMA_TIMEOUT as _ollama_timeout
from config import RERANK_BACKEND as RERANK_BACKEND
from config import RERANK_BATCH as RERANK_BATCH
from config import RERANK_ENABLE as RERANK_ENABLE
from config import RERANK_LLM_MODEL as RERANK_LLM_MODEL
from config import RERANK_MODEL as RERANK_MODEL
from config import RERANK_SNIPPET_CHARS as RERANK_SNIPPET_CHARS
from config import RERANK_THRESHOLD as RERANK_THRESHOLD

logger = logging.getLogger(__name__)


def _sync_config() -> None:
    """P10 即時同步：重讀 config 真相，模型名變了清掉舊 CrossEncoder。」"""
    global _ollama_timeout, RERANK_BACKEND, RERANK_BATCH, RERANK_ENABLE
    global RERANK_LLM_MODEL, RERANK_MODEL, RERANK_SNIPPET_CHARS, RERANK_THRESHOLD
    global _cross_model, _cross_model_name, _ollama
    try:
        _m = _config_module
        _ollama_timeout = _m.OLLAMA_TIMEOUT
        RERANK_BACKEND = _m.RERANK_BACKEND
        RERANK_BATCH = _m.RERANK_BATCH
        RERANK_ENABLE = _m.RERANK_ENABLE
        RERANK_LLM_MODEL = _m.RERANK_LLM_MODEL
        RERANK_SNIPPET_CHARS = _m.RERANK_SNIPPET_CHARS
        RERANK_THRESHOLD = _m.RERANK_THRESHOLD
        if RERANK_MODEL != _m.RERANK_MODEL:
            RERANK_MODEL = _m.RERANK_MODEL
            _cross_model = None
            _cross_model_name = ""
        # ollama 超時變了則重建，下次 _get_ollama 會用新逾時
        if _ollama is not None:
            try:
                _ollama = None
            except Exception:
                pass
    except Exception as e:
        logger.warning("重排配置同步失敗，沿用舊快照：%s", e)

_cross_model = None
_cross_model_name = ""
_cross_lock = threading.Lock()

# 優化：延遲建立帶逾時的 ollama client（避免缺套件時 import 就炸）
_ollama = None


def _get_ollama():
    """延遲取得帶逾時的 ollama client，缺套件回 None。」"""
    global _ollama
    if _ollama is not None:
        return _ollama
    try:
        import ollama
        _ollama = ollama.Client(timeout=_ollama_timeout)
    except Exception:  # noqa: BROAD_EXCEPT_OK - 缺套件時回 None，交由上層降級
        _ollama = None
    return _ollama


def _clamp_score(x: object) -> float:
    """把 LLM 分數夾到 0~10，避免模型亂給 100 分。」"""
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return min(10.0, max(0.0, v))


def _apply_threshold(docs: list[dict[str, str]]) -> list[dict[str, str]]:
    """低分過濾：沒設門檻直接回傳，有設就丟掉低分（至少留 1 條避免全空）。"""
    try:
        if RERANK_THRESHOLD == float("-inf"):
            return docs
    except Exception:  # noqa: BROAD_EXCEPT_OK - 分數缺失時不擋路
        return docs
    kept = []
    for d in docs:
        try:
            score = float(d.get("score", float("inf")))
        except (TypeError, ValueError):
            score = float("inf")  # 缺分數或壞分數當保留，避免誤殺
        if score >= RERANK_THRESHOLD:
            kept.append(d)
    return kept if kept else docs[:1]


def _load_cross_model():
    """載入 CrossEncoder，缺套件時回 None（交給上層降級）。"""
    global _cross_model, _cross_model_name
    with _cross_lock:
        if _cross_model is not None and _cross_model_name == RERANK_MODEL:
            return _cross_model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("缺 sentence-transformers，CrossEncoder 不可用（已降級為原順序／LLM 備援）")
            return None
        try:
            _cross_model = CrossEncoder(RERANK_MODEL)
            _cross_model_name = RERANK_MODEL
            return _cross_model
        except Exception as e:
            logger.warning("重排模型載入失敗（%s），降級為原順序", e)
            _cross_model = None
            _cross_model_name = ""
            return None


def is_available() -> bool:
    """輕量檢查：套件＋模型名，不打網路、不載大模型。」"""
    if not RERANK_ENABLE or RERANK_BACKEND == "none":
        return False
    if RERANK_BACKEND in ("auto", "crossencoder"):
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is not None:
            return True
        if RERANK_BACKEND == "crossencoder":
            return False
    if RERANK_BACKEND in ("auto", "llm"):
        if not RERANK_LLM_MODEL.strip():
            return False
        try:
            import ollama  # noqa: F401 - 僅檢查套件存在
            return True
        except ImportError:
            return False
    return False


def _list_ollama_names() -> list[str] | None:
    """列出 Ollama 模型名，失敗回 None。」"""
    try:
        client = _get_ollama()
        if client is None:
            return None
        models = client.list()
    except Exception as e:
        logger.warning("LLM 可用性探測失敗：%s", e)
        return None
    names: list[str] = []
    if isinstance(models, dict):
        items = models.get("models", []) or []
        for m in items:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model")
                if n:
                    names.append(str(n))
    elif isinstance(models, list):
        for m in models:
            if isinstance(m, str):
                names.append(m)
            elif isinstance(m, dict):
                n = m.get("name") or m.get("model")
                if n:
                    names.append(str(n))
    return names


def probe_availability(timeout_s: float = 5.0) -> dict[str, bool]:
    """主動探測兩種評審是否真可用，逾時當不可用｜新手：健康檢查，不是每次問都要做。」"""
    result = {"crossencoder": False, "llm": False}
    import importlib.util
    result["crossencoder"] = importlib.util.find_spec("sentence_transformers") is not None
    # 優化：timeout_s 真的生效，ollama.list 卡住不等到底
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_list_ollama_names)
        try:
            names = fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning("Ollama 探測逾時（%.1fs），視為不可用", timeout_s)
            names = None
    if names is None:
        result["llm"] = False
    elif not names:
        # 舊版回傳形狀不明但連線成功，保守當可用（呼叫失敗時 _llm_scores 會再降級）
        result["llm"] = True
    else:
        want = RERANK_LLM_MODEL.strip()
        # 優化：精確比對（大小寫不敏感），不再 `or True` 恆真
        result["llm"] = any(want.lower() in n.lower() or n.lower() in want.lower() for n in names)
        if not result["llm"]:
            logger.warning("Ollama 有回應但缺模型 %s（現有：%s）", want, names)
    return result


def _rerank_cross(query: str, docs: list[dict[str, str]], top_k: int) -> list[dict[str, str]] | None:
    """用 CrossEncoder 分批打分排序，失敗回 None 讓上層換備援。」"""
    model = _load_cross_model()
    if model is None:
        return None
    try:
        batch = max(1, int(RERANK_BATCH or 8))
        scores: list[float] = []
        for i in range(0, len(docs), batch):
            # 註：按字元截 2000 是省記憶體的近似，中文大致 1 字 ~ 1-2 token
            pairs = [[query, str(d.get("text", "") or "")[:2000]] for d in docs[i:i + batch]]
            try:
                part = model.predict(pairs)
            except Exception as e:
                logger.warning("CrossEncoder 批次 %d 失敗（%s），嘗試備援", i // batch, e)
                return None
            scores.extend(float(s) for s in part)
        if len(scores) != len(docs):
            logger.warning("CrossEncoder 分數數量不符（%d vs %d），降級備援", len(scores), len(docs))
            return None
        ranked = []
        for d, s in zip(docs, scores):
            item = dict(d)
            item["score"] = float(s)  # pyright: ignore[reportArgumentType] - score 欄位實為數字，型別沿用舊宣告
            ranked.append(item)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        logger.warning("CrossEncoder 重排失敗（%s），嘗試備援", e)
        return None


def _llm_scores(query: str, docs: list[dict[str, str]]) -> list[float] | None:
    """用 Ollama 一次幫全部文件打 0~10 分，解析失敗回 None。」"""
    try:
        client = _get_ollama()
        if client is None:
            logger.warning("缺 ollama 套件，LLM 備援不可用（已降級為原順序）")
            return None
    except Exception:
        logger.warning("缺 ollama 套件，LLM 備援不可用（已降級為原順序）")
        return None
    lines = []
    for i, d in enumerate(docs):
        snippet = str(d.get("text", "") or "")[:RERANK_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    prompt = (
        "你是檢索評分員。依與問題的相關程度，為每條文件打 0~10 分（10 最相關）。"
        "只回 JSON 陣列，例如 [8.5, 2.0, 0]，不要解釋。\n"
        f"問題：{query[:500]}\n文件：\n" + "\n".join(lines)
    )
    try:
        msg = client.chat(model=RERANK_LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        raw = msg["message"] if isinstance(msg, dict) else getattr(msg, "message", None)
        text = str(raw.get("content") if isinstance(raw, dict) else getattr(raw, "content", "") or "")
    except Exception as e:
        logger.warning("LLM 重排呼叫失敗（%s），降級為原順序", e)
        return None
    try:
        start, end = text.find("["), text.rfind("]")
        data = json.loads(text[start:end + 1] if start != -1 and end != -1 else text)
        if isinstance(data, list):
            scores = [_clamp_score(x) for x in data]
            if len(scores) == len(docs):
                return scores
    except Exception:
        pass
    nums = [_clamp_score(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) == len(docs) and "[" in text and "]" in text:
        return nums[:len(docs)]
    # P6 強韌：整批解析失敗改逐筆打分，單筆失敗記 0 分不丟整批
    logger.warning("LLM 整批解析失敗，改逐筆備援（%d 筆）", len(docs))
    per_doc: list[float] = []
    for d in docs:
        snippet = str(d.get("text", "") or "")[:RERANK_SNIPPET_CHARS].replace("\n", " ")
        try:
            single = client.chat(
                model=RERANK_LLM_MODEL,
                messages=[{"role": "user", "content": f"你是檢索評分員。依與問題的相關程度，為文件打 0~10 分，只回一個數字。\n問題：{query[:500]}\n文件：{snippet}"}],
            )
            s_raw = single["message"] if isinstance(single, dict) else getattr(single, "message", None)
            s_text = str(s_raw.get("content") if isinstance(s_raw, dict) else getattr(s_raw, "content", "") or "")
            m = re.search(r"\d+(?:\.\d+)?", s_text)
            per_doc.append(_clamp_score(m.group(0)) if m else 0.0)
        except Exception as e:
            logger.warning("LLM 逐筆打分失敗，已記 0 分：%s", e)
            per_doc.append(0.0)
    if len(per_doc) == len(docs):
        return per_doc
    logger.warning("LLM 回傳無法解析為 %d 個分數（原文前 200 字：%r），已降級為原順序", len(docs), text[:200])
    return None


def _rerank_llm(query: str, docs: list[dict[str, str]], top_k: int) -> list[dict[str, str]] | None:
    """LLM 備援排序，失敗回 None。」"""
    scores = _llm_scores(query, docs)
    if scores is None:
        return None
    ranked = []
    for d, s in zip(docs, scores):
        item = dict(d)
        item["score"] = float(s)  # pyright: ignore[reportArgumentType] - 同上
        ranked.append(item)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def rerank(query: str, docs: list[dict[str, str]], top_k: int = 3) -> list[dict[str, str]]:
    """重排主入口：自動選評審，任何失敗都回原順序前 top_k，保證不拋錯。」"""
    _sync_config()
    if not docs:
        return []
    if top_k <= 0:
        return []
    if not RERANK_ENABLE or RERANK_BACKEND == "none":
        return docs[:top_k]
    q = query.strip()[:2000]
    if not q:
        return docs[:top_k]
    result: list[dict[str, str]] | None = None
    if RERANK_BACKEND in ("auto", "crossencoder"):
        result = _rerank_cross(q, docs, top_k)
        if result is not None or RERANK_BACKEND == "crossencoder":
            out = result if result is not None else docs[:top_k]
            return _apply_threshold(out)
    if RERANK_BACKEND in ("auto", "llm"):
        result = _rerank_llm(q, docs, top_k)
        if result is not None:
            return _apply_threshold(result)
    logger.warning("所有重排後援皆失敗，已降級為向量原順序")
    return _apply_threshold(docs[:top_k])
