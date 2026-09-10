"""config.py：全專案唯一的設定真相來源｜新手教學版。

【這支程式在做什麼？（白話版）】
以前每個模組（chat_core、rag_qdrant、reranker）各自讀環境變數，
同一個旋鈕（例如 OLLAMA_TIMEOUT）在三個地方各讀一次。
哪天改了一處忘了另一處，就會出現「同一個設定三種行為」。
現在全部收斂到這裡：想看有什麼旋鈕可轉，看這一份就夠。

【使用方式】
各模組用 `from config import OLLAMA_MODEL` 取值，對外名稱不變，
舊的 `import chat_core`、`chat_core.OLLAMA_MODEL` 寫法照常用。

【注意】
環境變數在 import 時讀取一次，改了要重啟程式才生效。
唯一的例外是 chat_cli 的 DIY_HIST_FILE，它是每次用時現讀。
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """安全讀整數環境變數，寫壞回預設不炸 import｜新手：旋鈕轉壞就當沒轉。」"""
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except (ValueError, AttributeError):
        logger.warning("環境變數 %s 解析失敗，使用預設 %s", name, default)
        return default


def _float_env(name: str, default: float) -> float:
    """安全讀浮點環境變數，同上。」"""
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except (ValueError, AttributeError):
        logger.warning("環境變數 %s 解析失敗，使用預設 %s", name, default)
        return default


def _threshold_env(name: str) -> float:
    """讀分數門檻：沒設回 -inf（= 不設限），寫壞也回 -inf。」"""
    try:
        v = float(str(os.getenv(name, "")).strip() or "nan")
    except ValueError:
        return float("-inf")
    if v != v:  # nan 表示未設
        return float("-inf")
    return v


# --- 聊天核心（chat_core.py 在用）---
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_TIMEOUT: float = _float_env("OLLAMA_TIMEOUT", 120.0)  # HTTP 逾時，避免 Ollama/雲端卡死
SEARCH_MAX_RESULTS: int = _int_env("SEARCH_MAX_RESULTS", 3)
MAX_TOOL_ROUNDS: int = _int_env("MAX_TOOL_ROUNDS", 2)
SEARCH_QUERY_MAX_CHARS: int = _int_env("SEARCH_QUERY_MAX_CHARS", 200)
SEARCH_TIMEOUT: float = _float_env("SEARCH_TIMEOUT", 10.0)
RAG_MAX_RESULTS: int = _int_env("RAG_MAX_RESULTS", 3)
RAG_ENABLE: bool = os.getenv("RAG_ENABLE", "1") != "0"
OLLAMA_RETRIES: int = _int_env("OLLAMA_RETRIES", 1)
SEARCH_CACHE_MAX: int = _int_env("SEARCH_CACHE_MAX", 128)
SEARCH_CACHE_TTL: float = _float_env("SEARCH_CACHE_TTL", 300.0)  # 快取 5 分鐘過期，避免舊新聞殘留
HIST_MAX_CHARS: int = _int_env("HIST_MAX_CHARS", 6000)  # 歷史字數預算，超過從舊的裁
RAG_MAX_CHARS: int = _int_env("RAG_MAX_CHARS", 3000)  # 本地筆記字數預算，防止上下文爆量
# P3 新增：搜尋結果總預算＋單筆截斷＋使用者輸入上限，避免長摘要撐爆上下文
SEARCH_MAX_CHARS: int = _int_env("SEARCH_MAX_CHARS", 3000)
SEARCH_SNIPPET_CHARS: int = _int_env("SEARCH_SNIPPET_CHARS", 500)
SEARCH_TITLE_CHARS: int = _int_env("SEARCH_TITLE_CHARS", 200)
USER_MAX_CHARS: int = _int_env("USER_MAX_CHARS", 2000)
QUERY_REWRITE_LLM: bool = os.getenv("QUERY_REWRITE_LLM", "0") == "1"  # 預設關，開了才多打一次 LLM 改寫

# --- RAG（rag_qdrant.py 在用）---
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "notes")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")  # 只做向量化，不要換成聊天模型
VISION_MODEL: str = os.getenv("VISION_MODEL", "llava:latest")
CHUNK_CHARS: int = _int_env("CHUNK_CHARS", 800)
CHUNK_OVERLAP: int = _int_env("CHUNK_OVERLAP", 100)
RERANK_RECALL: int = _int_env("RERANK_RECALL", 15)
EMBED_BATCH: int = _int_env("EMBED_BATCH", 32)
UPSERT_BATCH: int = _int_env("UPSERT_BATCH", 64)
CHUNK_MAX_TOKENS: int = _int_env("CHUNK_MAX_TOKENS", 0)  # >0 啟用 token 預算，需 pip install tiktoken
QUERY_VEC_CACHE_MAX: int = _int_env("QUERY_VEC_CACHE_MAX", 256)
QUERY_VEC_CACHE_TTL: float = _float_env("QUERY_VEC_CACHE_TTL", 3600.0)  # 查詢向量快取 1 小時過期
# P6 匯入上限：大檔防爆，可配
INGEST_IMAGE_MAX_MB: int = _int_env("INGEST_IMAGE_MAX_MB", 10)
INGEST_PDF_MAX_PAGES: int = _int_env("INGEST_PDF_MAX_PAGES", 200)
INGEST_CSV_MAX_ROWS: int = _int_env("INGEST_CSV_MAX_ROWS", 5000)
# P7 純文字上限：txt／md 整檔讀入防爆
INGEST_TEXT_MAX_CHARS: int = _int_env("INGEST_TEXT_MAX_CHARS", 200000)
# P9 DOCX 上限：段落＋表格列合併防爆
INGEST_DOCX_MAX_PARAS: int = _int_env("INGEST_DOCX_MAX_PARAS", 5000)

# --- 重排（reranker.py 在用）---
RERANK_ENABLE: bool = os.getenv("RERANK_ENABLE", "1") != "0"
RERANK_BACKEND: str = os.getenv("RERANK_BACKEND", "auto").strip().lower()
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_LLM_MODEL: str = os.getenv("RERANK_LLM_MODEL", OLLAMA_MODEL)
RERANK_SNIPPET_CHARS: int = _int_env("RERANK_SNIPPET_CHARS", 300)
RERANK_THRESHOLD: float = _threshold_env("RERANK_THRESHOLD")
RERANK_BATCH: int = _int_env("RERANK_BATCH", 8)  # CrossEncoder 分批，避免大召回 OOM


def refresh() -> dict[str, tuple[Any, Any]]:
    """重讀環境變數並更新本模組全域，免重啟套用新旋鈕，回傳異動表。

    注意：`from config import X` 舊寫法仍是快照，需重新 import 或改用 `import config` 即時讀取。
    """
    global OLLAMA_MODEL, OLLAMA_TIMEOUT, SEARCH_MAX_RESULTS, MAX_TOOL_ROUNDS
    global SEARCH_QUERY_MAX_CHARS, SEARCH_TIMEOUT, RAG_MAX_RESULTS, RAG_ENABLE
    global OLLAMA_RETRIES, SEARCH_CACHE_MAX, SEARCH_CACHE_TTL, HIST_MAX_CHARS
    global RAG_MAX_CHARS, SEARCH_MAX_CHARS, SEARCH_SNIPPET_CHARS, SEARCH_TITLE_CHARS
    global USER_MAX_CHARS, QUERY_REWRITE_LLM
    global QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL, VISION_MODEL
    global CHUNK_CHARS, CHUNK_OVERLAP, RERANK_RECALL, EMBED_BATCH, UPSERT_BATCH
    global CHUNK_MAX_TOKENS, QUERY_VEC_CACHE_MAX, QUERY_VEC_CACHE_TTL
    global INGEST_IMAGE_MAX_MB, INGEST_PDF_MAX_PAGES, INGEST_CSV_MAX_ROWS
    global INGEST_TEXT_MAX_CHARS, INGEST_DOCX_MAX_PARAS
    global RERANK_ENABLE, RERANK_BACKEND, RERANK_MODEL, RERANK_LLM_MODEL
    global RERANK_SNIPPET_CHARS, RERANK_THRESHOLD, RERANK_BATCH
    before = dict(globals())
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    OLLAMA_TIMEOUT = _float_env("OLLAMA_TIMEOUT", 120.0)
    SEARCH_MAX_RESULTS = _int_env("SEARCH_MAX_RESULTS", 3)
    MAX_TOOL_ROUNDS = _int_env("MAX_TOOL_ROUNDS", 2)
    SEARCH_QUERY_MAX_CHARS = _int_env("SEARCH_QUERY_MAX_CHARS", 200)
    SEARCH_TIMEOUT = _float_env("SEARCH_TIMEOUT", 10.0)
    RAG_MAX_RESULTS = _int_env("RAG_MAX_RESULTS", 3)
    RAG_ENABLE = os.getenv("RAG_ENABLE", "1") != "0"
    OLLAMA_RETRIES = _int_env("OLLAMA_RETRIES", 1)
    SEARCH_CACHE_MAX = _int_env("SEARCH_CACHE_MAX", 128)
    SEARCH_CACHE_TTL = _float_env("SEARCH_CACHE_TTL", 300.0)
    HIST_MAX_CHARS = _int_env("HIST_MAX_CHARS", 6000)
    RAG_MAX_CHARS = _int_env("RAG_MAX_CHARS", 3000)
    SEARCH_MAX_CHARS = _int_env("SEARCH_MAX_CHARS", 3000)
    SEARCH_SNIPPET_CHARS = _int_env("SEARCH_SNIPPET_CHARS", 500)
    SEARCH_TITLE_CHARS = _int_env("SEARCH_TITLE_CHARS", 200)
    USER_MAX_CHARS = _int_env("USER_MAX_CHARS", 2000)
    QUERY_REWRITE_LLM = os.getenv("QUERY_REWRITE_LLM", "0") == "1"
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    VISION_MODEL = os.getenv("VISION_MODEL", "llava:latest")
    CHUNK_CHARS = _int_env("CHUNK_CHARS", 800)
    CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 100)
    RERANK_RECALL = _int_env("RERANK_RECALL", 15)
    EMBED_BATCH = _int_env("EMBED_BATCH", 32)
    UPSERT_BATCH = _int_env("UPSERT_BATCH", 64)
    CHUNK_MAX_TOKENS = _int_env("CHUNK_MAX_TOKENS", 0)
    QUERY_VEC_CACHE_MAX = _int_env("QUERY_VEC_CACHE_MAX", 256)
    QUERY_VEC_CACHE_TTL = _float_env("QUERY_VEC_CACHE_TTL", 3600.0)
    INGEST_IMAGE_MAX_MB = _int_env("INGEST_IMAGE_MAX_MB", 10)
    INGEST_PDF_MAX_PAGES = _int_env("INGEST_PDF_MAX_PAGES", 200)
    INGEST_CSV_MAX_ROWS = _int_env("INGEST_CSV_MAX_ROWS", 5000)
    INGEST_TEXT_MAX_CHARS = _int_env("INGEST_TEXT_MAX_CHARS", 200000)
    INGEST_DOCX_MAX_PARAS = _int_env("INGEST_DOCX_MAX_PARAS", 5000)
    RERANK_ENABLE = os.getenv("RERANK_ENABLE", "1") != "0"
    RERANK_BACKEND = os.getenv("RERANK_BACKEND", "auto").strip().lower()
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_LLM_MODEL = os.getenv("RERANK_LLM_MODEL", OLLAMA_MODEL)
    RERANK_SNIPPET_CHARS = _int_env("RERANK_SNIPPET_CHARS", 300)
    RERANK_THRESHOLD = _threshold_env("RERANK_THRESHOLD")
    RERANK_BATCH = _int_env("RERANK_BATCH", 8)
    after = dict(globals())
    diff = {k: (before.get(k), after.get(k)) for k in after if before.get(k) != after.get(k) and not k.startswith("_")}
    # P10 傳染已載入模組的舊快照，避免改 env 還要重啟
    try:
        import sys as _sys

        for _mod_name in ("chat_core", "rag_qdrant", "reranker"):
            _mod = _sys.modules.get(_mod_name)
            _sync = getattr(_mod, "_sync_config", None) if _mod is not None else None
            if callable(_sync):
                try:
                    _sync()
                except Exception as e:
                    logger.warning("同步 %s 失敗：%s", _mod_name, e)
    except Exception as e:
        logger.warning("跨模組同步失敗：%s", e)
    return diff
