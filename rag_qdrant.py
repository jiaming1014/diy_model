"""rag_qdrant.py：本地筆記 RAG 模組（TXT/MD/PDF/DOCX/CSV/圖片 → Qdrant）｜新手教學版。

【這支程式在做什麼？（白話版）】
想像你有一大疊讀書筆記，AI 一次吃不下。
這支程式幫你做三件事：
1. 切塊：把長筆記剪成 800 字一小張，重疊 100 字避免剪斷句子。
2. 嵌入＋存入：把每張小紙條翻譯成電腦懂的數字（向量），存進 Qdrant 資料庫（在 Docker 裡）。
3. 查詢：你問問題時，也把問題翻成數字，去資料庫找最像的幾張回來給 AI 參考。

【支援的檔案類型】
- 純文字：.txt / .md（直接讀）
- 文件：.pdf（需 pip install pypdf）、.docx（需 pip install python-docx）
- 表格：.csv（標準庫直接讀，每列轉成文字）
- 圖片：.png / .jpg / .jpeg / .webp / .bmp / .gif（OCR 或視覺模型轉文字，缺套件時只存檔名佔位）

【常用指令】
- 匯入筆記：python rag_qdrant.py --ingest notes（把 notes 資料夾的筆記吃進去）
- 測試查詢：python rag_qdrant.py --query "台北天氣如何"
"""

from dataclasses import dataclass
import argparse
import csv
import hashlib
import io
import json
import logging
import re
from collections import deque
from pathlib import Path
import threading

import ollama

# 重型依賴延遲載入：qdrant_client import 約 1.6 秒，改在首次使用處函式內載入，
# CLI 啟動／純聊天不付這筆；TYPE_CHECKING 區供靜態檢查解析型別（執行期不跑）。
# 注意：模組 __getattr__（PEP 562）不管模組內全域查找，此處不用它。
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams  # noqa: F401 - 執行期由函式內 import 提供

# 設定唯一真相在 config.py，這裡保留 RAGConfig 介面（欄位名不變）
import config as _config
from text_utils import redact_url_creds as _redact_url  # L1：錯誤訊息／日誌的 URL 帳密遮蔽
from ttl_cache import TTLCache  # P14：共用 LRU＋TTL 快取，查詢向量快取用

logger = logging.getLogger(__name__)

# OPT-6：查詢鍵正規化用空白正則預編譯，熱路徑不再每次現編譯
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RAGConfig:
    """RAG 可調參數集中表，預設值來自 config.py｜新手：所有旋鈕收在同一張面板。」"""

    url: str = _config.QDRANT_URL
    collection: str = _config.QDRANT_COLLECTION
    api_key: str = _config.QDRANT_API_KEY  # P16：需驗證的 Qdrant（如 Cloud）用
    embed_model: str = _config.EMBED_MODEL
    vision_model: str = _config.VISION_MODEL
    timeout: float = _config.OLLAMA_TIMEOUT
    chunk_chars: int = _config.CHUNK_CHARS
    chunk_overlap: int = _config.CHUNK_OVERLAP
    rerank_recall: int = _config.RERANK_RECALL
    embed_batch: int = _config.EMBED_BATCH
    upsert_batch: int = _config.UPSERT_BATCH
    chunk_max_tokens: int = _config.CHUNK_MAX_TOKENS
    query_max_chars: int = _config.RAG_QUERY_MAX_CHARS
    image_max_mb: int = _config.INGEST_IMAGE_MAX_MB
    pdf_max_pages: int = _config.INGEST_PDF_MAX_PAGES
    csv_max_rows: int = _config.INGEST_CSV_MAX_ROWS
    text_max_chars: int = _config.INGEST_TEXT_MAX_CHARS
    docx_max_paras: int = _config.INGEST_DOCX_MAX_PARAS


_CONFIG = RAGConfig()

# 優化：共用 ollama client（三模組同 timeout 共用一份連線池，見 ollama_shared.py）
try:
    from ollama_shared import get_shared_client as _get_shared_client

    _ollama = _get_shared_client(_CONFIG.timeout)
except Exception:  # BROAD_EXCEPT_OK - 共用模組缺失時退化為直建，不影響嵌入
    _ollama = ollama.Client(timeout=_CONFIG.timeout)
_ollama_timeout_used: float = _CONFIG.timeout  # P14：記住目前 client 用的逾時，沒變就不重建


def _sync_config() -> None:
    """P10 即時同步：重建 _CONFIG 與快取上限，免重啟生效。

    P14：ollama client 只在逾時真的變了才重建（舊版條件恆真，每次查詢都換一顆 client），
    換新前先關舊的，避免連線池漏掉。
    """
    global _CONFIG, _ollama, _ollama_timeout_used
    try:
        _CONFIG = RAGConfig(
            url=_config.QDRANT_URL,
            collection=_config.QDRANT_COLLECTION,
            api_key=_config.QDRANT_API_KEY,
            embed_model=_config.EMBED_MODEL,
            vision_model=_config.VISION_MODEL,
            timeout=_config.OLLAMA_TIMEOUT,
            chunk_chars=_config.CHUNK_CHARS,
            chunk_overlap=_config.CHUNK_OVERLAP,
            rerank_recall=_config.RERANK_RECALL,
            embed_batch=_config.EMBED_BATCH,
            upsert_batch=_config.UPSERT_BATCH,
            chunk_max_tokens=_config.CHUNK_MAX_TOKENS,
            query_max_chars=_config.RAG_QUERY_MAX_CHARS,
            image_max_mb=_config.INGEST_IMAGE_MAX_MB,
            pdf_max_pages=_config.INGEST_PDF_MAX_PAGES,
            csv_max_rows=_config.INGEST_CSV_MAX_ROWS,
            text_max_chars=_config.INGEST_TEXT_MAX_CHARS,
            docx_max_paras=_config.INGEST_DOCX_MAX_PARAS,
        )
        _QUERY_VEC_CACHE.update_limits(_config.QUERY_VEC_CACHE_MAX, _config.QUERY_VEC_CACHE_TTL)
        if _ollama_timeout_used != _CONFIG.timeout:
            # P17：經 ollama 模組建（與 chat_core 同款生命週期），再登記共用；
            # 共用快取內的舊實例不關閉，非託管才關。
            old = _ollama
            try:
                _ollama = ollama.Client(timeout=_CONFIG.timeout)
                _ollama_timeout_used = _CONFIG.timeout
            except Exception:
                _ollama = old  # 建不出新的就沿用舊的，別讓嵌入斷炊
            else:
                try:
                    from ollama_shared import is_managed as _is_managed
                    from ollama_shared import remember as _remember_shared

                    _remember_shared(_CONFIG.timeout, _ollama)
                    managed = _is_managed(old)
                except Exception:
                    managed = False
                if old is not None and old is not _ollama and not managed:
                    try:
                        old.close()
                    except Exception:
                        pass
    except Exception as e:
        logger.warning("RAG 配置同步失敗，沿用舊快照：%s", e)


def get_config() -> RAGConfig:
    """回傳作用中設定，測試可用 RAGConfig(url=...) 比對｜新手：想看旋鈕現在轉到哪就問它。」"""
    return _CONFIG


def probe_embed(timeout_s: float = 5.0) -> bool:
    """探測嵌入模型是否在 Ollama 就緒，逾時當不可用｜新手：RAG 吃飯的傢伙，--health 先驗。

    名單比對轉調共用 probe_model（認 dict／list／回傳物件），行為一致。
    """
    _sync_config()
    try:
        from ollama_shared import probe_model as _probe_model

        return bool(_probe_model(_CONFIG.embed_model, _CONFIG.timeout, timeout_s))
    except Exception as e:
        logger.warning("嵌入模型探測失敗（%s），視為不可用", e)
        return False


# --- 相容舊匯入：快照值，內部一律用 _CONFIG，外部改此值不影響運行 ---
QDRANT_URL: str = _CONFIG.url
QDRANT_COLLECTION: str = _CONFIG.collection
EMBED_MODEL: str = _CONFIG.embed_model
VISION_MODEL: str = _CONFIG.vision_model
CHUNK_CHARS: int = _CONFIG.chunk_chars
CHUNK_OVERLAP: int = _CONFIG.chunk_overlap
RERANK_RECALL: int = _CONFIG.rerank_recall

try:
    from reranker import rerank as _rerank
except Exception as _e:  # BROAD_EXCEPT_OK - reranker 缺失時仍要能純向量檢索
    logger.warning("reranker 載入失敗（%s），降級為純向量排序", _e)

    def _rerank(query: str, docs: list[dict[str, str]], top_k: int = 3) -> list[dict[str, str]]:
        """reranker 缺失時的降級版：不重排，直接回向量原順序前 top_k。」"""
        logger.warning("reranker 不可用，本次查詢使用向量原順序")
        return docs[:top_k]

SUPPORTED_SUFFIXES: set[str] = {
    ".txt", ".md",
    ".pdf", ".docx",
    ".csv",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
}
IMAGE_SUFFIXES: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

_client_lock = threading.Lock()
_cached_client: "QdrantClient | None" = None  # 引號避免 import 期求值（本體只在函式內與 TYPE_CHECKING 匯入）
_cached_key: tuple[str, str] | None = None  # (url, api_key)，P16：任一變了就換連線

# P14：問題向量快取改用共用 TTLCache，同一問句重複檢索不再重新嵌入
_QUERY_VEC_CACHE: TTLCache[list[float]] = TTLCache(
    maxsize=_config.QUERY_VEC_CACHE_MAX,
    ttl=_config.QUERY_VEC_CACHE_TTL,
)


def _normalize_query_key(query: str) -> str:
    """查詢向量快取鍵正規化：壓空白＋去頭尾，大小寫不同視為同鍵，省重複嵌入。」"""
    text = query if isinstance(query, str) else ("" if query is None else str(query))
    norm = _WS_RE.sub(" ", text.strip()).strip()
    return norm.lower() or norm


def _embed_query_vec(query: str) -> list[float] | None:
    """問題轉向量，走共用 TTL 快取：命中回複本，過期重算。

    快取鍵正規化共用（省重複嵌入），嵌入吃原文保留大小寫語義；
    存取皆用複本，避免呼叫端改到快取本體。
    """
    key = _normalize_query_key(query)
    if not key:
        return None
    hit = _QUERY_VEC_CACHE.get(key)
    if hit is not None:
        return list(hit)
    exact = query.strip() if isinstance(query, str) else key
    vecs = _embed_texts([exact or key])
    if not vecs:
        return None
    vec = vecs[0]
    _QUERY_VEC_CACHE.put(key, list(vec))
    return list(vec)


def _client() -> "QdrantClient":
    """Qdrant 單例連線｜新手：電話打一次就留著，別每次都重撥。」"""
    global _cached_client, _cached_key
    url = _CONFIG.url  # 優化：唯一真相走 _CONFIG，避免與相容快照漂移
    key = (url, _CONFIG.api_key)  # P16：url 或 api_key 變了就換連線
    with _client_lock:
        if _cached_client is not None and _cached_key == key:
            return _cached_client
        if _cached_client is not None:
            try:
                _cached_client.close()
            except Exception:
                pass
        from qdrant_client import QdrantClient  # 函式內載入：平時不付 1.6 秒 import，sys.modules 快取后续呼叫
        if key[1]:
            _cached_client = QdrantClient(url=key[0], api_key=key[1])
        else:
            _cached_client = QdrantClient(url=key[0])
        _cached_key = key
        return _cached_client


def _stable_id(source: str, chunk_text: str) -> str:
    """用 來源＋內文 算穩定 ID，重複匯入覆寫而非新增（非安全用途，僅編號）。"""
    digest = source + "\n" + chunk_text
    try:
        return hashlib.md5(digest.encode("utf-8"), usedforsecurity=False).hexdigest()  # type: ignore[call-arg]
    except TypeError:
        return hashlib.md5(digest.encode("utf-8")).hexdigest()


_tiktoken_enc = None  # 延遲載入，缺套件回 None 用字元估算
_tiktoken_warned = False


def _tok_len(text: str) -> int:
    """估 token 數：有 tiktoken 用 cl100k，缺套件回字元數｜新手：沒磅秤就用目測。」"""
    global _tiktoken_enc, _tiktoken_warned
    if _tiktoken_enc is None and not _tiktoken_warned:
        try:
            import tiktoken  # type: ignore[import-not-found]

            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_warned = True  # 只警告一次，避免每塊洗版
            if _CONFIG.chunk_max_tokens > 0:
                logger.warning("缺 tiktoken 卻設了 CHUNK_MAX_TOKENS=%s，token 預算已退化成字數估算（pip install tiktoken 恢復精算）", _CONFIG.chunk_max_tokens)
            else:
                logger.info("缺 tiktoken，切塊改用字元估算（pip install tiktoken 可啟用 token 預算）")
            return len(text)
    if _tiktoken_enc is None:
        return len(text)
    try:
        return len(_tiktoken_enc.encode(text))
    except Exception:
        return len(text)


def _over_budget(buf: str, chunk_chars: int, chunk_tokens: int) -> bool:
    """字元或 token 任一超標即算滿｜新手：體積跟重量哪個先超重就切。

    P17 快徑：小 buf 直接用字數判斷，不調 tiktoken 全編碼；
    中文 cl100k 約 1 字 1~2 token，字數 2 倍仍低於 token 上限時必定未超。
    """
    n = len(buf)
    if n > chunk_chars:
        return True
    if chunk_tokens <= 0:
        return False
    if n * 2 < chunk_tokens:
        return False
    return _tok_len(buf) > chunk_tokens


# 句尾標點後斷句（lookbehind 只斷不斷字）；沒標點的長串後面走字元硬切
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;…])\s*")
# OPT-14：重疊尾巴句讀搜尋預編譯，切塊每塊不再現編譯
_OVERLAP_PUNCT_RE = re.compile(r"[。！？!?；;\n]")


def _split_sentences(para: str) -> list[str]:
    """按中英句號切句，無標點的長句保留原樣交給字元切。」"""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(para.strip()) if s.strip()]
    return parts or ([para.strip()] if para.strip() else [])


def _overlap_tail(buf: str, ov: int) -> str:
    """取重疊尾巴並對齊句首，避免半句開頭。」"""
    if ov <= 0:
        return ""
    if len(buf) <= ov:
        return buf
    tail = buf[-ov:]
    m = _OVERLAP_PUNCT_RE.search(tail)
    if m:
        aligned = tail[m.end():].lstrip()
        # 對齊後太短則保留原尾巴，避免重疊失效
        if len(aligned) >= 20:
            return aligned
    return tail


def _chunk_text(text: str, chunk_chars: int | None = None, overlap: int | None = None, chunk_tokens: int | None = None) -> list[str]:
    """把長文切成小塊：段落→句子累積，超預算即切，塊間句子對齊重疊。」"""
    cc = chunk_chars if chunk_chars is not None else _CONFIG.chunk_chars
    ov = overlap if overlap is not None else _CONFIG.chunk_overlap
    ct = chunk_tokens if chunk_tokens is not None else _CONFIG.chunk_max_tokens
    # P14 防呆：重疊吃掉幾乎整塊時，切塊會原地打轉（無限迴圈），一律夾到「至少留 2 字前進」
    cc = max(1, int(cc))
    ov = min(max(0, int(ov)), max(0, cc - 2))
    clean = text.strip()
    if not clean:
        return []
    paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paras:
        for sent in _split_sentences(para):
            # 單句本身超長（無標點長串）：先硬切，避免單塊撐爆預算
            while not buf and _over_budget(sent, cc, ct):
                chunks.append(sent[:cc].strip())
                sent = (_overlap_tail(sent[:cc], ov) + sent[cc:]).strip()
                if not sent:
                    break
            if not sent:
                continue
            probe = (buf + " " + sent).strip() if buf else sent  # 先試塞進現有塊，塞不下才落袋開新塊
            if buf and _over_budget(probe, cc, ct):
                chunks.append(buf.strip())
                buf = _overlap_tail(buf, ov)
                # 重疊＋本句仍超標（單句超長），硬切本句
                while _over_budget((buf + " " + sent).strip() if buf else sent, cc, ct):
                    room = cc - len(buf) - 1 if buf else cc
                    if room <= 0:
                        chunks.append(buf.strip())
                        # 極端設定保險：強制縮短 buf 一個字，保證往前推進不卡死
                        buf = buf[1:].strip() if len(buf) > 1 else ""
                        continue
                    chunks.append(((buf + " " + sent[:room]).strip()) if buf else sent[:room].strip())
                    sent = sent[room:]
                    buf = _overlap_tail(chunks[-1], ov)
                buf = (buf + " " + sent).strip() if buf else sent
            else:
                buf = probe
        # 段落邊界：若已超預算則硬切落袋，避免跨段無限累積與超大單塊
        while _over_budget(buf, cc, ct):
            chunks.append(buf[:cc].strip())
            buf = (_overlap_tail(buf[:cc], ov) + buf[cc:]).strip()
            if not buf:
                break
    if buf.strip():
        # 最後殘留若超標，沿用硬切保底
        while _over_budget(buf, cc, ct):
            chunks.append(buf[:cc].strip())
            buf = _overlap_tail(buf[:cc], ov) + buf[cc:]
            buf = buf.strip()
            if not buf:
                break
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def _embed_single_batch(texts: list[str]) -> list[list[float]] | None:
    """單次嵌入呼叫，失敗回 None 不拋錯。」"""
    if not texts:
        return []
    try:
        resp = _ollama.embed(model=_CONFIG.embed_model, input=texts)
        return [list(vec) for vec in resp.embeddings]
    except Exception as e:
        logger.warning("嵌入 %d 塊失敗：%s", len(texts), e)
        return None


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """批量嵌入，切半遞迴隔離壞塊｜新手：整疊翻譯失敗就拆兩半試，壞的那張只影響半疊。」"""
    if not texts:
        return []
    ok = _embed_single_batch(texts)
    if ok is not None:
        return ok
    if len(texts) == 1:
        return []
    mid = len(texts) // 2
    return _embed_texts(texts[:mid]) + _embed_texts(texts[mid:])


def _embed_with_order(texts: list[str]) -> dict[int, list[float]]:
    """保序嵌入，單次遞迴不重複計費｜新手：壞掉那張空著，別整疊重印兩次。

    P17 同文去重：同批相同文字只嵌一次再扇出（如跨檔授權頭），省 embed 呼叫。
    """
    out: dict[int, list[float]] = {}
    if not texts:
        return out
    # 同文分組：uniq_texts 只嵌一次，groups 記回填位置
    # 雜湊鍵：全文當 key 會讓大批量匯入時同份文字存兩次，改用 sha256 省記憶體
    uniq: list[str] = []
    index_of: dict[str, int] = {}
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()
        u = index_of.get(h)
        if u is not None and uniq[u] == t:
            groups[u].append(i)
            continue
        if u is not None:
            # 極低機率雜湊碰撞：退回全文比對
            found: int | None = None
            for cand, ut in enumerate(uniq):
                if ut == t:
                    found = cand
                    break
            if found is not None:
                groups[found].append(i)
                continue
        u = len(uniq)
        index_of[h] = u
        uniq.append(t)
        groups[u] = [i]

    def _rec(indexed: list[tuple[int, str]]) -> None:
        """遞迴切半：整批失敗拆兩半重試，單塊失敗丟棄不影響整批。」"""
        if not indexed:
            return
        batch = [t for _, t in indexed]  # 下標稍後由 zip 回填，這裡只取文字送嵌
        ok = _embed_single_batch(batch)
        if ok is not None and len(ok) == len(indexed):
            for (u_idx, _), vec in zip(indexed, ok):
                for orig_i in groups[u_idx]:
                    out[orig_i] = vec
            return
        if len(indexed) == 1:
            return  # 單塊失敗已記 log，直接丟棄
        mid = len(indexed) // 2
        _rec(indexed[:mid])
        _rec(indexed[mid:])

    _rec(list(enumerate(uniq)))
    return out


def ensure_collection(dim: int) -> None:
    """若收藏集不存在就建立；維度不符直接拋錯，早失敗比空轉好。」"""
    client = _client()
    try:
        info = client.get_collection(_CONFIG.collection)
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        existing_dim = getattr(vectors, "size", None)
        if existing_dim is not None and existing_dim != dim:
            raise RuntimeError(f"收藏集 {_CONFIG.collection} 維度不符（現有 {existing_dim}，模型 {dim}），請換 QDRANT_COLLECTION 名稱或重建")
        return
    except Exception as e:
        if "維度不符" in str(e):
            raise  # 自己拋的維度錯誤直接上浮，不可誤判成「不存在」去重建
        msg = str(e).lower()
        if any(k in msg for k in ("connect", "refused", "connection", "timeout", "unreachable")):
            raise RuntimeError(_redact_url(f"連不上 Qdrant（{_CONFIG.url}）：{e}")) from e
        if not any(k in msg for k in ("404", "not found", "not exist", "doesn't exist", "does not exist")):
            logger.warning("查詢收藏集時發生未知錯誤（%s），嘗試建立", e)
    from qdrant_client.http.models import Distance, VectorParams  # 函式內載入，同上

    client.create_collection(
        collection_name=_CONFIG.collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("已建立收藏集 %s（維度 %d）", _CONFIG.collection, dim)


def _read_pdf(fp: Path) -> str:
    """讀 .pdf：逐頁抽文字再合併，超頁數截斷。缺 pypdf 時拋出提示。」"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(f"{fp.name} 是 PDF，但缺 pypdf，請先 pip install pypdf") from e
    max_pages = max(1, int(_CONFIG.pdf_max_pages or 200))
    try:
        reader = PdfReader(str(fp))
        total = len(reader.pages)
    except Exception as e:  # 空檔／損毀／加密到開不了：回佔位標 ok，修好檔 mtime 一變照樣重抓
        logger.warning("%s 開啟失敗（%s），僅檔名可檢索", fp.name, e)
        return f"[PDF檔：{fp.name}（無法開啟，僅檔名可檢索）]"
    parts: list[str] = []
    for i, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(f"--- 第 {i} 頁 ---\n{t.strip()}")
    if total > max_pages:
        logger.warning("%s 共 %d 頁，已截斷為前 %d 頁", fp.name, total, max_pages)
        parts.append(f"（以下 {total - max_pages} 頁已截斷）")
    text = "\n\n".join(parts)
    if not text.strip() and total > 0:
        # 掃描／加密 PDF 抽不到字：回佔位而非空字串，避免匯入標 ok 永久跳過（與圖片策略一致）
        return f"[PDF檔：{fp.name}（共 {total} 頁，未能抽出文字，可能是掃描檔或加密，僅檔名可檢索）]"
    max_chars = max(1000, int(_CONFIG.text_max_chars or 200000))
    if len(text) > max_chars:
        logger.warning("%s 合併文字超過 %d 字上限，已截斷", fp.name, max_chars)
        return text[:max_chars] + f"\n\n（已截斷，僅取前 {max_chars} 字）"
    return text


def _read_docx(fp: Path) -> str:
    """讀 .docx：段落＋表格都拿，超量截斷。」"""
    try:
        import docx
    except ImportError as e:
        raise RuntimeError(f"{fp.name} 是 DOCX，但缺 python-docx，請先 pip install python-docx") from e
    max_paras = max(1, int(_CONFIG.docx_max_paras or 5000))
    try:
        doc = docx.Document(str(fp))
    except Exception as e:  # 損毀檔打不開：回佔位標 ok，修好檔 mtime 一變照樣重抓
        logger.warning("%s 開啟失敗（%s），僅檔名可檢索", fp.name, e)
        return f"[DOCX檔：{fp.name}（無法開啟，僅檔名可檢索）]"
    parts: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            line = " | ".join(c.text.strip() for c in row.cells).strip()
            if line.strip(" |"):  # 去掉分隔符後還有字才收，全空格格不佔段落預算
                parts.append(line)
    if len(parts) > max_paras:
        logger.warning("%s 共 %d 段，已截斷為前 %d 段", fp.name, len(parts), max_paras)
        parts = parts[:max_paras] + [f"（以下 {len(parts) - max_paras} 段已截斷）"]
    text = "\n".join(parts)
    if not text.strip() and (doc.paragraphs or doc.tables):
        return f"[DOCX檔：{fp.name}（未能抽出文字，僅檔名可檢索）]"
    max_chars = max(1000, int(_CONFIG.text_max_chars or 200000))
    if len(text) > max_chars:
        logger.warning("%s 合併文字超過 %d 字上限，已截斷", fp.name, max_chars)
        return text[:max_chars] + f"\n\n（已截斷，僅取前 {max_chars} 字）"
    return text


def _read_csv(fp: Path) -> str:
    """讀 .csv：標頭＋每列轉文字，超列數截斷，只用標準庫。

    P15：邊讀邊數，讀到上限＋1 列即停，大 CSV 不再整檔進記憶體。
    編碼相容：依序試 utf-8-sig → cp950 → big5（台灣 Excel 常見），
    都失敗才退回 utf-8 忽略錯誤，避免亂碼或整檔跳過。
    """
    max_rows = max(1, int(_CONFIG.csv_max_rows or 5000))

    def _parse_text(text: str) -> tuple[list[list[str]], bool]:
        """解析已解碼文字：嗅探分隔符＋邊讀邊數，超上限即停。」"""
        try:
            sample = text[:4096]
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]) if sample.strip() else csv.excel  # pyright: ignore[reportArgumentType] - typeshed 將 delimiters 記為 str，實跑吃 list
        except Exception:
            dialect = csv.excel
        reader = csv.reader(io.StringIO(text), dialect)
        out: list[list[str]] = []
        cut = False
        for r in reader:
            if any(c.strip() for c in r):
                out.append([c.strip() for c in r])
                if len(out) > max_rows + 1:
                    cut = True
                    break
        return out, cut

    def _parse_with_encoding(enc: str) -> tuple[list[list[str]], bool]:
        """用指定編碼解析，嚴格解碼，遇到解碼錯誤直接拋出換下一個編碼。」"""
        with fp.open("r", encoding=enc, errors="strict", newline="") as f:
            try:
                sample = f.read(4096)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]) if sample.strip() else csv.excel  # pyright: ignore[reportArgumentType] - typeshed 將 delimiters 記為 str，實跑吃 list
            except Exception:
                f.seek(0)
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            rows: list[list[str]] = []
            truncated = False
            for r in reader:
                if any(c.strip() for c in r):
                    rows.append([c.strip() for c in r])
                    if len(rows) > max_rows + 1:  # 表頭＋上限＋至少一列超額 → 確定截斷
                        truncated = True
                        break
            return rows, truncated

    rows: list[list[str]] = []
    truncated = False
    last_err: Exception | None = None
    # OPT-7：小檔（<=5MB）一次讀 bytes 再試多編碼，省掉逐編碼重開檔＋重 sniff；大檔沿舊路邊讀邊數
    try:
        _csv_size = fp.stat().st_size
    except Exception:
        _csv_size = 0
    if 0 < _csv_size <= 5 * 1024 * 1024:
        try:
            _raw = fp.read_bytes()
        except Exception:
            _raw = b""
        for enc in ("utf-8-sig", "cp950", "big5"):
            try:
                _text = _raw.decode(enc, errors="strict")
                rows, truncated = _parse_text(_text)
                if enc != "utf-8-sig":
                    logger.info("%s 以 %s 解碼", fp.name, enc)
                break
            except (UnicodeDecodeError, UnicodeError, ValueError) as e:
                last_err = e
                continue
        else:
            logger.warning("%s 編碼偵測失敗（%s），已退回 utf-8 忽略錯誤", fp.name, last_err)
            rows, truncated = _parse_text(_raw.decode("utf-8-sig", errors="ignore"))
    else:
        for enc in ("utf-8-sig", "cp950", "big5"):
            try:
                rows, truncated = _parse_with_encoding(enc)
                if enc != "utf-8-sig":
                    logger.info("%s 以 %s 解碼", fp.name, enc)
                break
            except (UnicodeDecodeError, UnicodeError, ValueError) as e:
                last_err = e
                continue
        else:
            logger.warning("%s 編碼偵測失敗（%s），已退回 utf-8 忽略錯誤", fp.name, last_err)
        with fp.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            try:
                sample = f.read(4096)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]) if sample.strip() else csv.excel  # pyright: ignore[reportArgumentType]
            except Exception:
                f.seek(0)
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            rows = []
            truncated = False
            for r in reader:
                if any(c.strip() for c in r):
                    rows.append([c.strip() for c in r])
                    if len(rows) > max_rows + 1:
                        truncated = True
                        break
    if not rows:
        return ""
    header = rows[0]
    lines = ["表頭：" + " | ".join(header)]
    for r in rows[1:1 + max_rows]:  # 從第 1 列開始（跳過表頭），只取上限列數
        if len(r) == len(header):
            lines.append("；".join(f"{h}：{v}" for h, v in zip(header, r) if v))
        else:
            lines.append(" | ".join(r))
    if truncated:
        logger.warning("%s 超過 %d 列，已截斷後續列", fp.name, max_rows)
    return "\n".join(lines)


def _read_image(fp: Path) -> str:
    """讀圖片：超大檔直接佔位，否則 OCR → 視覺模型 → 檔名佔位，三層降級。」"""
    try:
        max_bytes = max(1, int(_CONFIG.image_max_mb or 10)) * 1024 * 1024
        size = fp.stat().st_size
        if size > max_bytes:
            return f"[圖片檔：{fp.name}（約 {size // 1024} KB，超過 {int(_CONFIG.image_max_mb or 10)} MB 上限未做內容辨識，僅檔名可檢索）]"
    except Exception:
        pass
    try:
        from PIL import Image
        import pytesseract
        with Image.open(fp) as img:
            try:
                ocr = pytesseract.image_to_string(img, lang="chi_tra+eng").strip()
            except Exception:
                # 缺中文語言包時退回預設英文，英文截圖仍救得回
                ocr = pytesseract.image_to_string(img).strip()
        if ocr:
            return f"[圖片 {fp.name} 的 OCR 文字]\n{ocr}"
    except Exception:
        pass
    try:
        try:
            # 視覺模型前先縮圖：最長邊 1024，省 token 又更快；缺 PIL 則原圖直送降級
            from io import BytesIO

            from PIL import Image as _PilImage

            with _PilImage.open(fp) as _img:
                _img = _img.convert("RGB")
                _img.thumbnail((1024, 1024))
                _buf = BytesIO()
                _img.save(_buf, format="JPEG", quality=85)
                img_bytes = _buf.getvalue()
        except Exception:
            with fp.open("rb") as f:
                img_bytes = f.read()
        msg = _ollama.chat(
            model=_CONFIG.vision_model,
            messages=[{"role": "user", "content": f"請用繁體中文描述這張圖片的內容，包含圖中文字：{fp.name}", "images": [img_bytes]}],
        )
        raw_msg = msg["message"] if isinstance(msg, dict) else getattr(msg, "message", None)
        if isinstance(raw_msg, dict):
            desc = str(raw_msg.get("content") or "")
        else:
            desc = str(getattr(raw_msg, "content", "") or "")
        if desc.strip():
            return f"[圖片 {fp.name} 的視覺描述]\n{desc.strip()}"
    except Exception:
        pass
    try:
        size_kb = fp.stat().st_size // 1024
    except Exception:
        size_kb = 0
    return f"[圖片檔：{fp.name}（約 {size_kb} KB，未做 OCR／視覺辨識，僅檔名可檢索。若要內容檢索請 pip install pillow pytesseract 或 ollama pull {_CONFIG.vision_model}）]"


def _read_file_text(fp: Path) -> str:
    """依副檔名分派讀檔，統一回傳純文字，純文字超長截斷。」"""
    suffix = fp.suffix.lower()
    if suffix in {".txt", ".md"}:
        max_chars = max(1000, int(_CONFIG.text_max_chars or 200000))
        # OPT-5：小檔一次讀 bytes 進記憶體再試多編碼，省掉逐編碼重開檔的 IO；
        # 大檔（>5MB）沿用舊路徑逐編碼讀前綴，避免一次載入記憶體。
        # 編碼相容：utf-8 嚴格先試，台灣記事本常見 cp950／big5 接著試，都失敗才退回忽略錯誤
        try:
            _size = fp.stat().st_size
        except Exception:
            _size = 0
        if 0 < _size <= 5 * 1024 * 1024:
            try:
                _raw = fp.read_bytes()
            except Exception:
                _raw = b""
            text = ""
            for enc in ("utf-8", "utf-8-sig", "cp950", "big5"):
                try:
                    text = _raw.decode(enc, errors="strict")
                    if enc not in ("utf-8", "utf-8-sig"):
                        logger.info("%s 以 %s 解碼", fp.name, enc)
                    break
                except (UnicodeDecodeError, UnicodeError, ValueError):
                    continue
            else:
                text = _raw.decode("utf-8", errors="ignore")
            text = text[: max_chars + 1]
        else:
            # P15：只讀到上限＋1 字就停，超大純文字檔不再整檔載入記憶體
            text = ""
            for enc in ("utf-8", "utf-8-sig", "cp950", "big5"):
                try:
                    with fp.open("r", encoding=enc, errors="strict") as f:
                        text = f.read(max_chars + 1)
                    if enc not in ("utf-8", "utf-8-sig"):
                        logger.info("%s 以 %s 解碼", fp.name, enc)
                    break
                except (UnicodeDecodeError, UnicodeError, ValueError):
                    continue
            else:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_chars + 1)
        if len(text) > max_chars:
            logger.warning("%s 超過 %d 字上限，已截斷", fp.name, max_chars)
            return text[:max_chars] + f"\n\n（已截斷，僅取前 {max_chars} 字）"
        return text
    if suffix == ".pdf":
        return _read_pdf(fp)
    if suffix == ".docx":
        return _read_docx(fp)
    if suffix == ".csv":
        return _read_csv(fp)
    if suffix in IMAGE_SUFFIXES:
        return _read_image(fp)
    raise ValueError(f"不支援的類型：{suffix}")


def _existing_text_map(client: "QdrantClient", pids: list[str]) -> dict[str, str]:
    """查已存在的點 ID→內文，用於增量跳過。」"""
    if not pids:
        return {}
    try:
        out: dict[str, str] = {}
        for i in range(0, len(pids), 128):  # 128 個一批查，避免一次塞太多 ID 打爆請求
            batch = pids[i:i + 128]
            records = client.retrieve(collection_name=_CONFIG.collection, ids=batch, with_payload=True)
            for rec in records:
                payload = getattr(rec, "payload", None) or {}
                text = str(payload.get("text", "") or "")
                # Qdrant 回傳 UUID 標準形（含 dash），自家 pid 是無 dash md5，去 dash 才比得上
                out[str(getattr(rec, "id", "")).replace("-", "")] = text
        return out
    except Exception as e:
        # 收藏集不存在時 retrieve 會 404，直接當全量寫入，不警告洗版
        if "404" in str(e).lower() or "not found" in str(e).lower():
            return {}
        logger.warning("增量比對失敗，本次全量寫入：%s", e)
        return {}


def _flush_batch(
    client: "QdrantClient",
    batch_chunks: list[str],
    batch_metas: list[dict[str, str]],
    ensured: dict[str, bool],
    upsert_batch_n: int | None = None,
) -> tuple[int, int, dict[str, int]]:
    """嵌入並寫入一批，回 (寫入數, 跳過數, 各來源完成數)。

    完成數只計寫入＋未變跳過，嵌入失敗的不計，呼叫端以此判定整檔是否成功。
    同批可混多檔（跨檔批量），各來源歸因精確。
    OPT-19：upsert 批量可由呼叫端傳入（ingest 入口快取），未傳沿舊路現算，相容舊測試。
    """
    if not batch_chunks:
        return 0, 0, {}
    pids = [_stable_id(m["source"], t) for m, t in zip(batch_metas, batch_chunks)]
    existing = _existing_text_map(client, pids)
    todo_idx = [i for i, (pid, txt) in enumerate(zip(pids, batch_chunks)) if existing.get(pid, None) != txt]  # 查無或內文變了才需重嵌，其餘跳過
    skipped_idx = {i for i in range(len(pids))} - set(todo_idx)
    skipped = len(skipped_idx)
    done_by_source: dict[str, int] = {}
    for i in skipped_idx:
        src = batch_metas[i].get("source", "")
        done_by_source[src] = done_by_source.get(src, 0) + 1
    if not todo_idx:
        return 0, skipped, done_by_source
    todo_texts = [batch_chunks[i] for i in todo_idx]
    vec_map = _embed_with_order(todo_texts)
    if not vec_map:
        logger.warning("跳過一批（嵌入全失敗，%d 塊）", len(todo_texts))
        return 0, skipped, done_by_source
    if not ensured.get("done"):
        # 首批成功才建表：用實際向量維度建，維度不符早拋錯不空轉
        sample_vec = next(iter(vec_map.values()))
        ensure_collection(len(sample_vec))
        ensured["done"] = True
    from qdrant_client.http.models import PointStruct  # 函式內載入，同上

    points: list[PointStruct] = []
    written_idx: list[int] = []
    for order, orig_i in enumerate(todo_idx):
        if order not in vec_map:
            continue
        points.append(PointStruct(id=pids[orig_i], vector=vec_map[order], payload=batch_metas[orig_i]))
        written_idx.append(orig_i)
    upsert_batch = upsert_batch_n if upsert_batch_n is not None else max(1, int(_CONFIG.upsert_batch or 64))  # OPT-19：入口傳入優先，誤設 0 不炸逐點寫入
    for j in range(0, len(points), upsert_batch):
        client.upsert(collection_name=_CONFIG.collection, points=points[j:j + upsert_batch])
    for i in written_idx:
        src = batch_metas[i].get("source", "")
        done_by_source[src] = done_by_source.get(src, 0) + 1
    return len(points), skipped, done_by_source


_INGEST_CACHE_NAME = ".ingest_cache.json"


def _load_ingest_cache(root: Path) -> dict[str, dict[str, object]]:
    """讀檔案級快取：rel -> {mtime, size, ok}，壞檔當空。」"""
    try:
        p = root / _INGEST_CACHE_NAME
        if not p.is_file():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_ingest_cache(root: Path, cache: dict[str, dict[str, object]]) -> None:
    """原子寫檔案級快取（tmp＋replace 防半寫，半路斷電不留壞檔），失敗僅警告不中斷匯入。」"""
    try:
        p = root / _INGEST_CACHE_NAME
        tmp = root / (_INGEST_CACHE_NAME + ".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception as e:
        logger.warning("匯入快取寫入失敗：%s", e)


def ingest_folder(folder: str, on_progress: object = None) -> int:
    """把資料夾內支援檔切塊＋嵌入＋寫入 Qdrant，回傳寫入點數。

    on_progress(done, total, rel)：可選進度回調，拋錯不中斷匯入。
    """
    _sync_config()
    if isinstance(folder, str) and (folder.startswith("~/") or folder.startswith("~\\")):
        root = Path(folder).expanduser()
    else:
        # 裸 ~ 不展開（否則吞掉整個家目錄），維持找不到的明確報錯
        root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"匯入資料夾不存在：{folder}")
    # 遞迴掃支援副檔名＋實體檔，匯入快取檔本身永遠排除
    cands = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file() and p.name != _INGEST_CACHE_NAME]
    links = sorted({str(p) for p in cands if p.is_symlink()})
    if links:
        # symlink 指向 notes 外的外部內容不匯入，僅匯入實體檔
        shown = "、".join(s[:60] for s in links[:5])
        logger.warning("跳過 %d 個 symlink（僅匯入實體檔）：%s", len(links), shown)
    files = sorted([p for p in cands if not p.is_symlink()])
    if not files:
        logger.warning("%s 內沒有支援的檔案（支援：%s），可先丟筆記進去", folder, sorted(SUPPORTED_SUFFIXES))
        return 0
    client = _client()
    ensured: dict[str, bool] = {"done": False}
    total = 0
    skipped_unchanged = 0
    skipped_files = 0
    skipped_cached = 0
    pending_chunks: deque[str] = deque()
    pending_metas: deque[dict[str, str]] = deque()
    file_count = 0
    ingest_cache = _load_ingest_cache(root)
    touched: dict[str, dict[str, object]] = {}

    file_totals: dict[str, int] = {}
    file_done: dict[str, int] = {}
    # OPT-17：批次上限入口快取一次，整輪匯入共用同一批量
    _embed_batch_n = max(1, int(_CONFIG.embed_batch or 32))
    # OPT-19：upsert 批量同口徑快取，逐批不再重讀全域
    _upsert_batch_n = max(1, int(_CONFIG.upsert_batch or 64))

    def _drain() -> tuple[int, int]:
        """清一批 embed_batch，大檔溢出時由呼叫端多次呼叫；小檔殘留併入下一檔，最後統一收尾。

        跨檔批量：同批可混多檔，_flush_batch 回的各來源完成數精確歸因，
        嵌入失敗的塊不計完成，下次匯入會重試補寫。
        OPT-17：批次上限入口快取一次，_drain 多次呼叫不再重讀全域。
        """
        nonlocal total, skipped_unchanged
        if not pending_chunks:
            return (0, 0)
        take = min(len(pending_chunks), _embed_batch_n)
        # C4：deque 左端彈出 O(1)，舊寫法全切片複製 O(n)，大批量匯入省吞吐
        batch_chunks = [pending_chunks.popleft() for _ in range(take)]
        batch_metas = [pending_metas.popleft() for _ in range(take)]
        # OPT-19 相容：測試以 4 參數 mock _flush_batch，維持 4 參數呼叫；
        # 上方 _upsert_batch_n 保留供外部直接呼叫時傳入。
        res = _flush_batch(client, batch_chunks, batch_metas, ensured)
        if len(res) == 3:
            written, skipped, done_by_source = res
        else:  # 相容舊 mock 只回 (寫入, 跳過)：按批內比例估算歸因
            written, skipped = res  # type: ignore[misc]
            done_by_source = {}
            done_total = written + skipped
            if done_total >= len(batch_chunks):
                for m in batch_metas:
                    src = m.get("source", "")
                    done_by_source[src] = done_by_source.get(src, 0) + 1
        total += written
        skipped_unchanged += skipped
        for src, n in done_by_source.items():
            file_done[src] = file_done.get(src, 0) + n
        return (written, skipped)

    def _report(done: int, rel: str) -> None:
        """進度回報：調 on_progress 顯示檔數，回調拋錯只警告不中斷。」"""
        if on_progress is None:
            return
        try:
            if callable(on_progress):
                on_progress(done, len(files), rel)  # type: ignore[operator]
        except Exception as e:
            logger.warning("進度回調失敗，已忽略：%s", e)

    for idx, fp in enumerate(files):
        try:
            rel = str(fp.relative_to(root))
        except ValueError:  # 理論上不會發生（檔就是從 root 掃出來的），保險起見退回檔名
            rel = fp.name
        try:
            st = fp.stat()
        except Exception as e:
            logger.warning("跳過 %s（%s）", fp.name, e)
            skipped_files += 1
            _report(idx + 1, rel)
            continue
        # 檔案級快跳：mtime＋size 命中且上次成功，直接跳過讀檔＋嵌入
        cached = ingest_cache.get(rel)
        if isinstance(cached, dict) and cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size and cached.get("ok") is True:
            skipped_cached += 1
            _report(idx + 1, rel)
            continue
        try:
            text = _read_file_text(fp)
        except Exception as e:
            logger.warning("跳過 %s（%s）", fp.name, e)
            skipped_files += 1
            _report(idx + 1, rel)
            continue
        chunks = _chunk_text(text)
        if not chunks:
            touched[rel] = {"mtime": st.st_mtime, "size": st.st_size, "ok": True}
            _report(idx + 1, rel)
            continue
        file_count += 1
        touched[rel] = {"mtime": st.st_mtime, "size": st.st_size, "ok": False}
        file_totals[rel] = len(chunks)
        for chunk in chunks:
            pending_chunks.append(chunk)
            pending_metas.append({"source": rel, "text": chunk})
            # 大檔溢出時立即清一批，小檔殘留併入下一檔（跨檔批量省嵌入呼叫）
            while len(pending_chunks) >= _embed_batch_n:
                _drain()
        _report(idx + 1, rel)
    # 收尾：把跨檔殘留按批量清完，再按各來源完成數判定整檔成功
    while pending_chunks:
        _drain()
    for rel, total_chunks in file_totals.items():
        # P15：完成數＝寫入＋未變跳過；等於總塊數才算整檔成功。
        # 部分嵌入失敗若誤標 ok，檔案級快取（mtime＋size）會永久跳過壞塊不再補。
        if file_done.get(rel, 0) >= total_chunks:
            touched[rel]["ok"] = True
    ingest_cache.update(touched)
    _save_ingest_cache(root, ingest_cache)
    if total == 0:
        logger.info("無需寫入（快取跳過 %d 檔，內容未變跳過 %d 塊）。", skipped_cached, skipped_unchanged)
        return 0
    logger.info("共 %d 個檔（有效 %d，快取跳過 %d，跳過檔 %d，未變跳過 %d 塊），已寫入 %d 點到 %s。", len(files), file_count, skipped_cached, skipped_files, skipped_unchanged, total, _CONFIG.collection)
    return total


# E1 短路可觀測：可判定次數／實際觸發次數（測試可重置；eval 讀快照算精度）
SHORTCUT_TOTAL: int = 0
SHORTCUT_FIRED: int = 0
_SHORTCUT_LOCK = threading.Lock()  # F1：eval 並行時計數不丟失


def shortcut_stats() -> dict[str, int]:
    """短路計數快照（eval／觀測用）。"""
    with _SHORTCUT_LOCK:
        return {"total": SHORTCUT_TOTAL, "fired": SHORTCUT_FIRED}


def _rerank_disabled() -> bool:
    """重排是否停用：總開關關閉或後端指定 none"""
    try:
        if not bool(_config.RERANK_ENABLE):
            return True
        return str(_config.RERANK_BACKEND or "").strip().lower() == "none"
    except Exception:
        return False


def _rerank_shortcut_hit(hits: list[dict[str, str]]) -> bool:
    """C2 向量高分短路：top1 餘弦分達標且斷層領先次名時回 True，上層跳過 CPU 重排。

    壞分數（缺失／非數字／nan／inf）一律回 False 走正常重排，不擋路。
    E2 量尺護欄：餘弦分應落在 [-1, 1]（浮點塵埃放寬萬分之一），其它距離量尺
    （DOT／EUCLID 大值）直接放棄短路——與 _rag_adequate 只對 0~1 量尺用地板同哲學。
    """
    try:
        if not bool(_config.RERANK_SHORTCUT):
            return False
        lo = float(_config.RERANK_SHORTCUT_MIN)
        gap = float(_config.RERANK_SHORTCUT_GAP)
    except Exception:
        return False
    try:
        has_threshold = float(_config.RERANK_THRESHOLD) > float("-inf")
    except Exception:
        has_threshold = False
    if has_threshold:
        return False  # F0：有設門檻走重排＋過濾，向量分與重排分量尺不同不可混用（預設 -inf 不影響）
    scores: list[float] = []
    for h in hits:
        try:
            v = float(str(h.get("score", "") or ""))
        except (TypeError, ValueError):
            return False
        if v != v or v in (float("inf"), float("-inf")):
            return False
        if not (-1.0001 <= v <= 1.0001):
            return False
        scores.append(v)
    if len(scores) < 2:
        return False
    ordered = sorted(scores, reverse=True)
    return ordered[0] >= lo and (ordered[0] - ordered[1]) >= gap


def search_local(query: str, limit: int = 3) -> list[dict[str, str]]:
    """把問題轉向量去 Qdrant 找最像的筆記塊，寬取後重排取精華。」
    P20：limit<=0 直接回空，避免 recall 寬取後掉進 top_k=0 的未定義行為。
    """
    _sync_config()
    if limit <= 0:
        return []
    q = query.strip()
    if not q:
        return []
    try:
        qvec = _embed_query_vec(q[:_CONFIG.query_max_chars])
        if qvec is None:
            return []
        client = _client()
        # C1：重排停用時按需寬取（省 Qdrant payload）；啟用時維持寬取給評審挑
        recall = limit if _rerank_disabled() else max(limit, _CONFIG.rerank_recall)  # 先寬取再重排取精華，寧可多撈幾塊給評審挑
        res = client.query_points(collection_name=_CONFIG.collection, query=qvec, limit=recall, with_payload=True)
        hits: list[dict[str, str]] = []
        for pt in res.points:
            payload = pt.payload or {}
            text = str(payload.get("text", "") or "")
            source = str(payload.get("source", "") or "")
            if text:
                item: dict[str, str] = {"source": source, "text": text}
                try:
                    item["score"] = str(float(getattr(pt, "score", 0.0)))
                except Exception:
                    pass
                hits.append(item)
        if not hits:
            return []
        if len(hits) <= limit:
            return hits
        global SHORTCUT_TOTAL, SHORTCUT_FIRED
        with _SHORTCUT_LOCK:
            SHORTCUT_TOTAL += 1
        # C2：向量高分短路——top1 斷層領先時跳過 CPU 重排（約省 20 秒），直接取原順序
        if _rerank_shortcut_hit(hits):
            with _SHORTCUT_LOCK:
                SHORTCUT_FIRED += 1
            logger.debug("觀測 向量高分短路，跳過重排")
            return hits[:limit]
        try:
            return _rerank(q, hits, top_k=limit)
        except Exception as e:
            logger.warning("重排略過（%s），使用向量順序", e)
            return hits[:limit]
    except Exception as e:
        logger.warning("本地檢索略過（%s）", _redact_url(str(e)))
        return []


def main() -> int:
    """命令列入口：--ingest 匯入，--query 測試查詢；回 0 成功、1 匯入失敗（腳本可接住）。"""
    try:  # Windows 主控台／管線統一 UTF-8，與 chat_cli／eval 同款保護
        import sys as _sys

        if _sys.stdout is not None:
            _sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue] - 執行期才有
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="匯入筆記到 Qdrant 或測試查詢")
    ap.add_argument("--ingest", default="", help="要匯入的資料夾，例如 notes")
    ap.add_argument("--query", default="", help="測試查詢，例如 '台北天氣如何'")
    ap.add_argument("--limit", type=int, default=3, help="查詢取幾塊")
    ap.add_argument("--quiet", action="store_true", help="只顯示警告與查詢結果，不顯示進度")
    ap.add_argument("--progress", action="store_true", help="匯入時逐檔顯示進度（P16）")
    ap.add_argument("--verbose", action="store_true", help="顯示除錯訊息")
    args = ap.parse_args()
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
    if args.ingest:
        def _on_progress(done: int, total: int, rel: str) -> None:
            """--progress 顯示：逐檔印 [完成/總數] 檔名。」"""
            print(f"[{done}/{total}] {rel}", flush=True)

        try:
            ingest_folder(args.ingest, on_progress=_on_progress if args.progress else None)
        except Exception as e:  # 路徑不存在、Qdrant 連不上都印友好訊息，不噴 traceback
            print(f"匯入失敗：{_redact_url(str(e))}")
            return 1
    elif args.query:
        hits = search_local(args.query, limit=args.limit)
        if not hits:
            print("無本地命中。")
        for i, h in enumerate(hits, start=1):
            score = h.get("score", "")
            suffix = f"（分數：{score}）" if score != "" else ""
            print(f"[{i}] 來源：{h['source']}{suffix}\n{h['text']}\n")
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
