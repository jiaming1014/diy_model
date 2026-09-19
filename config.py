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
        logger.warning("環境變數 %s 解析失敗，視為不設限", name)
        return float("-inf")
    if v != v:  # nan 表示未設
        return float("-inf")
    return v


# --- 預設值唯一來源：模組頂層與 refresh() 共用同一批，避免兩處漂移（P15）---
_DEF_OLLAMA_MODEL: str = "gemma4:31b-cloud"
_DEF_OLLAMA_TIMEOUT: float = 120.0
_DEF_SEARCH_MAX_RESULTS: int = 3
_DEF_MAX_TOOL_ROUNDS: int = 2
_DEF_SEARCH_QUERY_MAX_CHARS: int = 200
_DEF_SEARCH_TIMEOUT: float = 10.0
_DEF_SEARCH_REGION: str = "tw-twn"
_DEF_RAG_MAX_RESULTS: int = 3
_DEF_ENABLED: str = "1"
_DEF_DISABLED: str = "0"
_DEF_OLLAMA_RETRIES: int = 1
_DEF_SEARCH_CACHE_MAX: int = 128
_DEF_SEARCH_CACHE_TTL: float = 300.0
_DEF_SEARCH_FAIL_CACHE_TTL: float = 30.0
_DEF_HIST_MAX_CHARS: int = 6000
_DEF_HIST_SUMMARY_MAX_CHARS: int = 800
_DEF_HIST_SUMMARY_MIN_DROPPED: int = 200
_DEF_RAG_MAX_CHARS: int = 3000
_DEF_SEARCH_MAX_CHARS: int = 3000
_DEF_SEARCH_SNIPPET_CHARS: int = 500
_DEF_SEARCH_TITLE_CHARS: int = 200
_DEF_USER_MAX_CHARS: int = 2000
_DEF_QDRANT_URL: str = "http://localhost:6333"
_DEF_QDRANT_COLLECTION: str = "notes"
_DEF_QDRANT_API_KEY: str = ""
_DEF_EMBED_MODEL: str = "nomic-embed-text"
_DEF_VISION_MODEL: str = "llava:latest"
_DEF_CHUNK_CHARS: int = 800
_DEF_CHUNK_OVERLAP: int = 100
_DEF_RERANK_RECALL: int = 15
_DEF_EMBED_BATCH: int = 32
_DEF_UPSERT_BATCH: int = 64
_DEF_CHUNK_MAX_TOKENS: int = 0
_DEF_RAG_QUERY_MAX_CHARS: int = 500
_DEF_QUERY_VEC_CACHE_MAX: int = 256
_DEF_QUERY_VEC_CACHE_TTL: float = 3600.0
_DEF_INGEST_IMAGE_MAX_MB: int = 10
_DEF_INGEST_PDF_MAX_PAGES: int = 200
_DEF_INGEST_CSV_MAX_ROWS: int = 5000
_DEF_INGEST_TEXT_MAX_CHARS: int = 200000
_DEF_INGEST_DOCX_MAX_PARAS: int = 5000
_DEF_RERANK_BACKEND: str = "auto"
_DEF_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
_DEF_RERANK_SNIPPET_CHARS: int = 300
_DEF_RERANK_BATCH: int = 8
_DEF_RERANK_QUERY_MAX_CHARS: int = 500
_DEF_RERANK_DOC_MAX_CHARS: int = 2000
_DEF_WORKSPACE_DIRNAME: str = "AI_Workspace"
_DEF_WORKSPACE_MAX_FILE_CHARS: int = 100000
_DEF_WORKSPACE_PATH_MAX_CHARS: int = 500
_DEF_YOUTUBE_QUERY_MAX_CHARS: int = 100

# --- 聊天核心（chat_core.py 在用）---
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", _DEF_OLLAMA_MODEL)
OLLAMA_TIMEOUT: float = _float_env("OLLAMA_TIMEOUT", _DEF_OLLAMA_TIMEOUT)  # HTTP 逾時，避免 Ollama/雲端卡死
SEARCH_MAX_RESULTS: int = _int_env("SEARCH_MAX_RESULTS", _DEF_SEARCH_MAX_RESULTS)
MAX_TOOL_ROUNDS: int = _int_env("MAX_TOOL_ROUNDS", _DEF_MAX_TOOL_ROUNDS)
SEARCH_QUERY_MAX_CHARS: int = _int_env("SEARCH_QUERY_MAX_CHARS", _DEF_SEARCH_QUERY_MAX_CHARS)
SEARCH_TIMEOUT: float = _float_env("SEARCH_TIMEOUT", _DEF_SEARCH_TIMEOUT)
SEARCH_REGION: str = os.getenv("SEARCH_REGION", _DEF_SEARCH_REGION).strip()  # DuckDuckGo 區域，預設台灣
RAG_MAX_RESULTS: int = _int_env("RAG_MAX_RESULTS", _DEF_RAG_MAX_RESULTS)
RAG_ENABLE: bool = os.getenv("RAG_ENABLE", _DEF_ENABLED) != "0"
OLLAMA_RETRIES: int = _int_env("OLLAMA_RETRIES", _DEF_OLLAMA_RETRIES)
SEARCH_CACHE_MAX: int = _int_env("SEARCH_CACHE_MAX", _DEF_SEARCH_CACHE_MAX)
SEARCH_CACHE_TTL: float = _float_env("SEARCH_CACHE_TTL", _DEF_SEARCH_CACHE_TTL)  # 快取 5 分鐘過期，避免舊新聞殘留
SEARCH_FAIL_CACHE_TTL: float = _float_env("SEARCH_FAIL_CACHE_TTL", _DEF_SEARCH_FAIL_CACHE_TTL)  # P18：搜尋失敗空結果短快取，壞查詢短時間內不再打網路
HIST_MAX_CHARS: int = _int_env("HIST_MAX_CHARS", _DEF_HIST_MAX_CHARS)  # 歷史字數預算，超過從舊的裁
HIST_SUMMARY_ENABLE: bool = os.getenv("HIST_SUMMARY_ENABLE", _DEF_DISABLED) == "1"  # P16：舊訊息改滾動摘要，預設關（會多打模型）
HIST_SUMMARY_MAX_CHARS: int = _int_env("HIST_SUMMARY_MAX_CHARS", _DEF_HIST_SUMMARY_MAX_CHARS)  # 摘要上限字數
HIST_SUMMARY_MIN_DROPPED: int = _int_env("HIST_SUMMARY_MIN_DROPPED", _DEF_HIST_SUMMARY_MIN_DROPPED)  # 丟棄內容達此字數才值得摘要
RAG_MAX_CHARS: int = _int_env("RAG_MAX_CHARS", _DEF_RAG_MAX_CHARS)  # 本地筆記字數預算，防止上下文爆量
# P3 新增：搜尋結果總預算＋單筆截斷＋使用者輸入上限，避免長摘要撐爆上下文
SEARCH_MAX_CHARS: int = _int_env("SEARCH_MAX_CHARS", _DEF_SEARCH_MAX_CHARS)
SEARCH_SNIPPET_CHARS: int = _int_env("SEARCH_SNIPPET_CHARS", _DEF_SEARCH_SNIPPET_CHARS)
SEARCH_TITLE_CHARS: int = _int_env("SEARCH_TITLE_CHARS", _DEF_SEARCH_TITLE_CHARS)
USER_MAX_CHARS: int = _int_env("USER_MAX_CHARS", _DEF_USER_MAX_CHARS)
QUERY_REWRITE_LLM: bool = os.getenv("QUERY_REWRITE_LLM", _DEF_DISABLED) == "1"  # 預設關，開了才多打一次 LLM 改寫
# --- 工作區檔案工具（workspace_write_file／workspace_make_dir 在用）---
WORKSPACE_DIRNAME: str = os.getenv("WORKSPACE_DIRNAME", _DEF_WORKSPACE_DIRNAME).strip() or _DEF_WORKSPACE_DIRNAME
WORKSPACE_MAX_FILE_CHARS: int = _int_env("WORKSPACE_MAX_FILE_CHARS", _DEF_WORKSPACE_MAX_FILE_CHARS)
WORKSPACE_PATH_MAX_CHARS: int = _int_env("WORKSPACE_PATH_MAX_CHARS", _DEF_WORKSPACE_PATH_MAX_CHARS)
YOUTUBE_QUERY_MAX_CHARS: int = _int_env("YOUTUBE_QUERY_MAX_CHARS", _DEF_YOUTUBE_QUERY_MAX_CHARS)

# --- RAG（rag_qdrant.py 在用）---
QDRANT_URL: str = os.getenv("QDRANT_URL", _DEF_QDRANT_URL)
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", _DEF_QDRANT_COLLECTION)
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", _DEF_QDRANT_API_KEY).strip()  # P16：Qdrant Cloud 等需驗證的服務用
EMBED_MODEL: str = os.getenv("EMBED_MODEL", _DEF_EMBED_MODEL)  # 只做向量化，不要換成聊天模型
VISION_MODEL: str = os.getenv("VISION_MODEL", _DEF_VISION_MODEL)
CHUNK_CHARS: int = _int_env("CHUNK_CHARS", _DEF_CHUNK_CHARS)
CHUNK_OVERLAP: int = _int_env("CHUNK_OVERLAP", _DEF_CHUNK_OVERLAP)
RERANK_RECALL: int = _int_env("RERANK_RECALL", _DEF_RERANK_RECALL)
EMBED_BATCH: int = _int_env("EMBED_BATCH", _DEF_EMBED_BATCH)
UPSERT_BATCH: int = _int_env("UPSERT_BATCH", _DEF_UPSERT_BATCH)
CHUNK_MAX_TOKENS: int = _int_env("CHUNK_MAX_TOKENS", _DEF_CHUNK_MAX_TOKENS)  # >0 啟用 token 預算，需 pip install tiktoken
RAG_QUERY_MAX_CHARS: int = _int_env("RAG_QUERY_MAX_CHARS", _DEF_RAG_QUERY_MAX_CHARS)  # P20：檢索查詢截斷，避免長串撐爆嵌入
QUERY_VEC_CACHE_MAX: int = _int_env("QUERY_VEC_CACHE_MAX", _DEF_QUERY_VEC_CACHE_MAX)
QUERY_VEC_CACHE_TTL: float = _float_env("QUERY_VEC_CACHE_TTL", _DEF_QUERY_VEC_CACHE_TTL)  # 查詢向量快取 1 小時過期
# P6 匯入上限：大檔防爆，可配
INGEST_IMAGE_MAX_MB: int = _int_env("INGEST_IMAGE_MAX_MB", _DEF_INGEST_IMAGE_MAX_MB)
INGEST_PDF_MAX_PAGES: int = _int_env("INGEST_PDF_MAX_PAGES", _DEF_INGEST_PDF_MAX_PAGES)
INGEST_CSV_MAX_ROWS: int = _int_env("INGEST_CSV_MAX_ROWS", _DEF_INGEST_CSV_MAX_ROWS)
# P7 純文字上限：txt／md 整檔讀入防爆
INGEST_TEXT_MAX_CHARS: int = _int_env("INGEST_TEXT_MAX_CHARS", _DEF_INGEST_TEXT_MAX_CHARS)
# P9 DOCX 上限：段落＋表格列合併防爆
INGEST_DOCX_MAX_PARAS: int = _int_env("INGEST_DOCX_MAX_PARAS", _DEF_INGEST_DOCX_MAX_PARAS)

# --- 重排（reranker.py 在用）---
RERANK_ENABLE: bool = os.getenv("RERANK_ENABLE", _DEF_ENABLED) != "0"
RERANK_BACKEND: str = os.getenv("RERANK_BACKEND", _DEF_RERANK_BACKEND).strip().lower()
RERANK_MODEL: str = os.getenv("RERANK_MODEL", _DEF_RERANK_MODEL)
RERANK_LLM_MODEL: str = os.getenv("RERANK_LLM_MODEL", OLLAMA_MODEL)
RERANK_SNIPPET_CHARS: int = _int_env("RERANK_SNIPPET_CHARS", _DEF_RERANK_SNIPPET_CHARS)
RERANK_THRESHOLD: float = _threshold_env("RERANK_THRESHOLD")
RERANK_BATCH: int = _int_env("RERANK_BATCH", _DEF_RERANK_BATCH)  # CrossEncoder 分批，避免大召回 OOM
RERANK_QUERY_MAX_CHARS: int = _int_env("RERANK_QUERY_MAX_CHARS", _DEF_RERANK_QUERY_MAX_CHARS)  # P20：打分提示詞的問題截斷
RERANK_DOC_MAX_CHARS: int = _int_env("RERANK_DOC_MAX_CHARS", _DEF_RERANK_DOC_MAX_CHARS)  # P20：CrossEncoder 配對的文件截斷


# --- P18 節流：refresh 扇出對照表，某模組無相關異動時跳過其 _sync_config。
# 保守原則：拿不準就放進集合，多同步一次只花幾微秒，漏同步會拿舊設定。
_SYNC_WATCH: dict[str, frozenset[str]] = {
    "chat_core": frozenset({
        "OLLAMA_MODEL", "OLLAMA_TIMEOUT", "SEARCH_MAX_RESULTS", "MAX_TOOL_ROUNDS",
        "SEARCH_QUERY_MAX_CHARS", "SEARCH_TIMEOUT", "SEARCH_REGION", "RAG_MAX_RESULTS",
        "RAG_ENABLE", "OLLAMA_RETRIES", "SEARCH_CACHE_MAX", "SEARCH_CACHE_TTL",
        "SEARCH_FAIL_CACHE_TTL", "HIST_MAX_CHARS", "HIST_SUMMARY_ENABLE",
        "HIST_SUMMARY_MAX_CHARS", "HIST_SUMMARY_MIN_DROPPED", "RAG_MAX_CHARS",
        "SEARCH_MAX_CHARS", "SEARCH_SNIPPET_CHARS", "SEARCH_TITLE_CHARS",
        "USER_MAX_CHARS", "QUERY_REWRITE_LLM",
        "WORKSPACE_DIRNAME", "WORKSPACE_MAX_FILE_CHARS", "WORKSPACE_PATH_MAX_CHARS",
        "YOUTUBE_QUERY_MAX_CHARS",
    }),
    "rag_qdrant": frozenset({
        "QDRANT_URL", "QDRANT_COLLECTION", "QDRANT_API_KEY", "EMBED_MODEL", "VISION_MODEL",
        "OLLAMA_TIMEOUT", "CHUNK_CHARS", "CHUNK_OVERLAP", "RERANK_RECALL", "EMBED_BATCH",
        "UPSERT_BATCH", "CHUNK_MAX_TOKENS", "QUERY_VEC_CACHE_MAX", "QUERY_VEC_CACHE_TTL",
        "RAG_QUERY_MAX_CHARS",
        "INGEST_IMAGE_MAX_MB", "INGEST_PDF_MAX_PAGES", "INGEST_CSV_MAX_ROWS",
        "INGEST_TEXT_MAX_CHARS", "INGEST_DOCX_MAX_PARAS",
    }),
    "reranker": frozenset({
        "RERANK_ENABLE", "RERANK_BACKEND", "RERANK_MODEL", "RERANK_LLM_MODEL",
        "RERANK_SNIPPET_CHARS", "RERANK_THRESHOLD", "RERANK_BATCH",
        "RERANK_QUERY_MAX_CHARS", "RERANK_DOC_MAX_CHARS",
        "OLLAMA_TIMEOUT", "OLLAMA_MODEL",
    }),
}


def refresh() -> dict[str, tuple[Any, Any]]:
    """重讀環境變數並更新本模組全域，免重啟套用新旋鈕，回傳異動表。

    注意：`from config import X` 舊寫法仍是快照，需重新 import 或改用 `import config` 即時讀取。
    """
    global OLLAMA_MODEL, OLLAMA_TIMEOUT, SEARCH_MAX_RESULTS, MAX_TOOL_ROUNDS
    global SEARCH_QUERY_MAX_CHARS, SEARCH_TIMEOUT, SEARCH_REGION, RAG_MAX_RESULTS, RAG_ENABLE
    global OLLAMA_RETRIES, SEARCH_CACHE_MAX, SEARCH_CACHE_TTL, SEARCH_FAIL_CACHE_TTL, HIST_MAX_CHARS
    global HIST_SUMMARY_ENABLE, HIST_SUMMARY_MAX_CHARS, HIST_SUMMARY_MIN_DROPPED
    global RAG_MAX_CHARS, SEARCH_MAX_CHARS, SEARCH_SNIPPET_CHARS, SEARCH_TITLE_CHARS
    global USER_MAX_CHARS, QUERY_REWRITE_LLM
    global WORKSPACE_DIRNAME, WORKSPACE_MAX_FILE_CHARS, WORKSPACE_PATH_MAX_CHARS
    global YOUTUBE_QUERY_MAX_CHARS
    global QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY, EMBED_MODEL, VISION_MODEL
    global CHUNK_CHARS, CHUNK_OVERLAP, RERANK_RECALL, EMBED_BATCH, UPSERT_BATCH
    global CHUNK_MAX_TOKENS, QUERY_VEC_CACHE_MAX, QUERY_VEC_CACHE_TTL
    global RAG_QUERY_MAX_CHARS
    global INGEST_IMAGE_MAX_MB, INGEST_PDF_MAX_PAGES, INGEST_CSV_MAX_ROWS
    global INGEST_TEXT_MAX_CHARS, INGEST_DOCX_MAX_PARAS
    global RERANK_ENABLE, RERANK_BACKEND, RERANK_MODEL, RERANK_LLM_MODEL
    global RERANK_SNIPPET_CHARS, RERANK_THRESHOLD, RERANK_BATCH
    global RERANK_QUERY_MAX_CHARS, RERANK_DOC_MAX_CHARS
    before = dict(globals())
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", _DEF_OLLAMA_MODEL)
    OLLAMA_TIMEOUT = _float_env("OLLAMA_TIMEOUT", _DEF_OLLAMA_TIMEOUT)
    SEARCH_MAX_RESULTS = _int_env("SEARCH_MAX_RESULTS", _DEF_SEARCH_MAX_RESULTS)
    MAX_TOOL_ROUNDS = _int_env("MAX_TOOL_ROUNDS", _DEF_MAX_TOOL_ROUNDS)
    SEARCH_QUERY_MAX_CHARS = _int_env("SEARCH_QUERY_MAX_CHARS", _DEF_SEARCH_QUERY_MAX_CHARS)
    SEARCH_TIMEOUT = _float_env("SEARCH_TIMEOUT", _DEF_SEARCH_TIMEOUT)
    SEARCH_REGION = os.getenv("SEARCH_REGION", _DEF_SEARCH_REGION).strip()
    RAG_MAX_RESULTS = _int_env("RAG_MAX_RESULTS", _DEF_RAG_MAX_RESULTS)
    RAG_ENABLE = os.getenv("RAG_ENABLE", _DEF_ENABLED) != "0"
    OLLAMA_RETRIES = _int_env("OLLAMA_RETRIES", _DEF_OLLAMA_RETRIES)
    SEARCH_CACHE_MAX = _int_env("SEARCH_CACHE_MAX", _DEF_SEARCH_CACHE_MAX)
    SEARCH_CACHE_TTL = _float_env("SEARCH_CACHE_TTL", _DEF_SEARCH_CACHE_TTL)
    SEARCH_FAIL_CACHE_TTL = _float_env("SEARCH_FAIL_CACHE_TTL", _DEF_SEARCH_FAIL_CACHE_TTL)
    HIST_MAX_CHARS = _int_env("HIST_MAX_CHARS", _DEF_HIST_MAX_CHARS)
    HIST_SUMMARY_ENABLE = os.getenv("HIST_SUMMARY_ENABLE", _DEF_DISABLED) == "1"
    HIST_SUMMARY_MAX_CHARS = _int_env("HIST_SUMMARY_MAX_CHARS", _DEF_HIST_SUMMARY_MAX_CHARS)
    HIST_SUMMARY_MIN_DROPPED = _int_env("HIST_SUMMARY_MIN_DROPPED", _DEF_HIST_SUMMARY_MIN_DROPPED)
    RAG_MAX_CHARS = _int_env("RAG_MAX_CHARS", _DEF_RAG_MAX_CHARS)
    SEARCH_MAX_CHARS = _int_env("SEARCH_MAX_CHARS", _DEF_SEARCH_MAX_CHARS)
    SEARCH_SNIPPET_CHARS = _int_env("SEARCH_SNIPPET_CHARS", _DEF_SEARCH_SNIPPET_CHARS)
    SEARCH_TITLE_CHARS = _int_env("SEARCH_TITLE_CHARS", _DEF_SEARCH_TITLE_CHARS)
    USER_MAX_CHARS = _int_env("USER_MAX_CHARS", _DEF_USER_MAX_CHARS)
    QUERY_REWRITE_LLM = os.getenv("QUERY_REWRITE_LLM", _DEF_DISABLED) == "1"
    WORKSPACE_DIRNAME = os.getenv("WORKSPACE_DIRNAME", _DEF_WORKSPACE_DIRNAME).strip() or _DEF_WORKSPACE_DIRNAME
    WORKSPACE_MAX_FILE_CHARS = _int_env("WORKSPACE_MAX_FILE_CHARS", _DEF_WORKSPACE_MAX_FILE_CHARS)
    WORKSPACE_PATH_MAX_CHARS = _int_env("WORKSPACE_PATH_MAX_CHARS", _DEF_WORKSPACE_PATH_MAX_CHARS)
    YOUTUBE_QUERY_MAX_CHARS = _int_env("YOUTUBE_QUERY_MAX_CHARS", _DEF_YOUTUBE_QUERY_MAX_CHARS)
    QDRANT_URL = os.getenv("QDRANT_URL", _DEF_QDRANT_URL)
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", _DEF_QDRANT_COLLECTION)
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", _DEF_QDRANT_API_KEY).strip()
    EMBED_MODEL = os.getenv("EMBED_MODEL", _DEF_EMBED_MODEL)
    VISION_MODEL = os.getenv("VISION_MODEL", _DEF_VISION_MODEL)
    CHUNK_CHARS = _int_env("CHUNK_CHARS", _DEF_CHUNK_CHARS)
    CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", _DEF_CHUNK_OVERLAP)
    RERANK_RECALL = _int_env("RERANK_RECALL", _DEF_RERANK_RECALL)
    EMBED_BATCH = _int_env("EMBED_BATCH", _DEF_EMBED_BATCH)
    UPSERT_BATCH = _int_env("UPSERT_BATCH", _DEF_UPSERT_BATCH)
    CHUNK_MAX_TOKENS = _int_env("CHUNK_MAX_TOKENS", _DEF_CHUNK_MAX_TOKENS)
    RAG_QUERY_MAX_CHARS = _int_env("RAG_QUERY_MAX_CHARS", _DEF_RAG_QUERY_MAX_CHARS)
    QUERY_VEC_CACHE_MAX = _int_env("QUERY_VEC_CACHE_MAX", _DEF_QUERY_VEC_CACHE_MAX)
    QUERY_VEC_CACHE_TTL = _float_env("QUERY_VEC_CACHE_TTL", _DEF_QUERY_VEC_CACHE_TTL)
    INGEST_IMAGE_MAX_MB = _int_env("INGEST_IMAGE_MAX_MB", _DEF_INGEST_IMAGE_MAX_MB)
    INGEST_PDF_MAX_PAGES = _int_env("INGEST_PDF_MAX_PAGES", _DEF_INGEST_PDF_MAX_PAGES)
    INGEST_CSV_MAX_ROWS = _int_env("INGEST_CSV_MAX_ROWS", _DEF_INGEST_CSV_MAX_ROWS)
    INGEST_TEXT_MAX_CHARS = _int_env("INGEST_TEXT_MAX_CHARS", _DEF_INGEST_TEXT_MAX_CHARS)
    INGEST_DOCX_MAX_PARAS = _int_env("INGEST_DOCX_MAX_PARAS", _DEF_INGEST_DOCX_MAX_PARAS)
    RERANK_ENABLE = os.getenv("RERANK_ENABLE", _DEF_ENABLED) != "0"
    RERANK_BACKEND = os.getenv("RERANK_BACKEND", _DEF_RERANK_BACKEND).strip().lower()
    RERANK_MODEL = os.getenv("RERANK_MODEL", _DEF_RERANK_MODEL)
    RERANK_LLM_MODEL = os.getenv("RERANK_LLM_MODEL", OLLAMA_MODEL)
    RERANK_SNIPPET_CHARS = _int_env("RERANK_SNIPPET_CHARS", _DEF_RERANK_SNIPPET_CHARS)
    RERANK_THRESHOLD = _threshold_env("RERANK_THRESHOLD")
    RERANK_BATCH = _int_env("RERANK_BATCH", _DEF_RERANK_BATCH)
    RERANK_QUERY_MAX_CHARS = _int_env("RERANK_QUERY_MAX_CHARS", _DEF_RERANK_QUERY_MAX_CHARS)
    RERANK_DOC_MAX_CHARS = _int_env("RERANK_DOC_MAX_CHARS", _DEF_RERANK_DOC_MAX_CHARS)
    after = dict(globals())
    diff = {k: (before.get(k), after.get(k)) for k in after if before.get(k) != after.get(k) and not k.startswith("_")}
    # P10 傳染已載入模組的舊快照，避免改 env 還要重啟
    # P18 節流：只傳給本輪有相關異動的模組；diff 為空（無異動）時維持舊行為全傳。
    try:
        import sys as _sys

        for _mod_name in ("chat_core", "rag_qdrant", "reranker"):
            if diff and not (_SYNC_WATCH[_mod_name] & set(diff)):
                continue
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
