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

# --- 重排（reranker.py 在用）---
RERANK_ENABLE: bool = os.getenv("RERANK_ENABLE", "1") != "0"
RERANK_BACKEND: str = os.getenv("RERANK_BACKEND", "auto").strip().lower()
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_LLM_MODEL: str = os.getenv("RERANK_LLM_MODEL", OLLAMA_MODEL)
RERANK_SNIPPET_CHARS: int = _int_env("RERANK_SNIPPET_CHARS", 300)
RERANK_THRESHOLD: float = _threshold_env("RERANK_THRESHOLD")
