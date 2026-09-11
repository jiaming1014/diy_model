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
import json
import logging
import re
from pathlib import Path
import threading

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# 設定唯一真相在 config.py，這裡保留 RAGConfig 介面（欄位名不變）
import config as _config
from ttl_cache import TTLCache  # P14：共用 LRU＋TTL 快取，查詢向量快取用

logger = logging.getLogger(__name__)


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
    image_max_mb: int = _config.INGEST_IMAGE_MAX_MB
    pdf_max_pages: int = _config.INGEST_PDF_MAX_PAGES
    csv_max_rows: int = _config.INGEST_CSV_MAX_ROWS
    text_max_chars: int = _config.INGEST_TEXT_MAX_CHARS
    docx_max_paras: int = _config.INGEST_DOCX_MAX_PARAS


_CONFIG = RAGConfig()

# 優化：建立帶逾時的共用 client，避免 ollama.embed/chat 卡死
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
            image_max_mb=_config.INGEST_IMAGE_MAX_MB,
            pdf_max_pages=_config.INGEST_PDF_MAX_PAGES,
            csv_max_rows=_config.INGEST_CSV_MAX_ROWS,
            text_max_chars=_config.INGEST_TEXT_MAX_CHARS,
            docx_max_paras=_config.INGEST_DOCX_MAX_PARAS,
        )
        _QUERY_VEC_CACHE.update_limits(_config.QUERY_VEC_CACHE_MAX, _config.QUERY_VEC_CACHE_TTL)
        if _ollama_timeout_used != _CONFIG.timeout:
            old = _ollama
            try:
                _ollama = ollama.Client(timeout=_CONFIG.timeout)
                _ollama_timeout_used = _CONFIG.timeout
            except Exception:
                _ollama = old  # 建不出新的就沿用舊的，別讓嵌入斷炊
            else:
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
    except Exception as e:
        logger.warning("RAG 配置同步失敗，沿用舊快照：%s", e)


def get_config() -> RAGConfig:
    """回傳作用中設定，測試可用 RAGConfig(url=...) 比對｜新手：想看旋鈕現在轉到哪就問它。」"""
    return _CONFIG


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
_cached_client: QdrantClient | None = None
_cached_key: tuple[str, str] | None = None  # (url, api_key)，P16：任一變了就換連線

# P14：問題向量快取改用共用 TTLCache，同一問句重複檢索不再重新嵌入
_QUERY_VEC_CACHE: TTLCache[list[float]] = TTLCache(
    maxsize=_config.QUERY_VEC_CACHE_MAX,
    ttl=_config.QUERY_VEC_CACHE_TTL,
)


def _embed_query_vec(query: str) -> list[float] | None:
    """問題轉向量，走共用 TTL 快取：命中回複本，過期重算。」"""
    hit = _QUERY_VEC_CACHE.get(query)
    if hit is not None:
        return list(hit)
    vecs = _embed_texts([query])
    if not vecs:
        return None
    vec = vecs[0]
    _QUERY_VEC_CACHE.put(query, vec)
    return vec


def _client() -> QdrantClient:
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
            logger.info("缺 tiktoken，切塊改用字元估算（pip install tiktoken 可啟用 token 預算）")
            return len(text)
    if _tiktoken_enc is None:
        return len(text)
    try:
        return len(_tiktoken_enc.encode(text))
    except Exception:
        return len(text)


def _over_budget(buf: str, chunk_chars: int, chunk_tokens: int) -> bool:
    """字元或 token 任一超標即算滿｜新手：體積跟重量哪個先超重就切。」"""
    if len(buf) > chunk_chars:
        return True
    if chunk_tokens > 0 and _tok_len(buf) > chunk_tokens:
        return True
    return False


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;…])\s*")


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
    m = re.search(r"[。！？!?；;\n]", tail)
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
            probe = (buf + " " + sent).strip() if buf else sent
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
    """保序嵌入，單次遞迴不重複計費｜新手：壞掉那張空著，別整疊重印兩次。」"""
    out: dict[int, list[float]] = {}

    def _rec(indexed: list[tuple[int, str]]) -> None:
        if not indexed:
            return
        batch = [t for _, t in indexed]
        ok = _embed_single_batch(batch)
        if ok is not None and len(ok) == len(indexed):
            for (idx, _), vec in zip(indexed, ok):
                out[idx] = vec
            return
        if len(indexed) == 1:
            return  # 單塊失敗已記 log，直接丟棄
        mid = len(indexed) // 2
        _rec(indexed[:mid])
        _rec(indexed[mid:])

    _rec(list(enumerate(texts)))
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
            raise
        msg = str(e).lower()
        if any(k in msg for k in ("connect", "refused", "connection", "timeout", "unreachable")):
            raise RuntimeError(f"連不上 Qdrant（{_CONFIG.url}）：{e}") from e
        if not any(k in msg for k in ("404", "not found", "not exist", "doesn't exist", "does not exist")):
            logger.warning("查詢收藏集時發生未知錯誤（%s），嘗試建立", e)
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
    reader = PdfReader(str(fp))
    parts: list[str] = []
    total = len(reader.pages)
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
    return "\n\n".join(parts)


def _read_docx(fp: Path) -> str:
    """讀 .docx：段落＋表格都拿，超量截斷。」"""
    try:
        import docx
    except ImportError as e:
        raise RuntimeError(f"{fp.name} 是 DOCX，但缺 python-docx，請先 pip install python-docx") from e
    max_paras = max(1, int(_CONFIG.docx_max_paras or 5000))
    doc = docx.Document(str(fp))
    parts: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            line = " | ".join(c.text.strip() for c in row.cells).strip(" |")
            if line.strip(" |"):
                parts.append(line)
    if len(parts) > max_paras:
        logger.warning("%s 共 %d 段，已截斷為前 %d 段", fp.name, len(parts), max_paras)
        parts = parts[:max_paras] + [f"（以下 {len(parts) - max_paras} 段已截斷）"]
    return "\n".join(parts)


def _read_csv(fp: Path) -> str:
    """讀 .csv：標頭＋每列轉文字，超列數截斷，只用標準庫。

    P15：邊讀邊數，讀到上限＋1 列即停，大 CSV 不再整檔進記憶體。
    """
    max_rows = max(1, int(_CONFIG.csv_max_rows or 5000))
    with fp.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
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
    if not rows:
        return ""
    header = rows[0]
    lines = ["表頭：" + " | ".join(header)]
    for r in rows[1:1 + max_rows]:
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
            ocr = pytesseract.image_to_string(img, lang="chi_tra+eng").strip()
        if ocr:
            return f"[圖片 {fp.name} 的 OCR 文字]\n{ocr}"
    except Exception:
        pass
    try:
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
        # P15：只讀到上限＋1 字就停，超大純文字檔不再整檔載入記憶體
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


def _existing_text_map(client: QdrantClient, pids: list[str]) -> dict[str, str]:
    """查已存在的點 ID→內文，用於增量跳過。」"""
    if not pids:
        return {}
    try:
        out: dict[str, str] = {}
        for i in range(0, len(pids), 128):
            batch = pids[i:i + 128]
            records = client.retrieve(collection_name=_CONFIG.collection, ids=batch, with_payload=True)
            for rec in records:
                payload = getattr(rec, "payload", None) or {}
                text = str(payload.get("text", "") or "")
                out[str(getattr(rec, "id", ""))] = text
        return out
    except Exception as e:
        # 收藏集不存在時 retrieve 會 404，直接當全量寫入，不警告洗版
        if "404" in str(e).lower() or "not found" in str(e).lower():
            return {}
        logger.warning("增量比對失敗，本次全量寫入：%s", e)
        return {}


def _flush_batch(client: QdrantClient, batch_chunks: list[str], batch_metas: list[dict[str, str]], ensured: dict[str, bool]) -> tuple[int, int]:
    """嵌入並寫入一批，回 (寫入數, 跳過數)。"""
    if not batch_chunks:
        return 0, 0
    pids = [_stable_id(m["source"], t) for m, t in zip(batch_metas, batch_chunks)]
    existing = _existing_text_map(client, pids)
    todo_idx = [i for i, (pid, txt) in enumerate(zip(pids, batch_chunks)) if existing.get(pid, None) != txt]
    skipped = len(pids) - len(todo_idx)
    if not todo_idx:
        return 0, skipped
    todo_texts = [batch_chunks[i] for i in todo_idx]
    vec_map = _embed_with_order(todo_texts)
    if not vec_map:
        logger.warning("跳過一批（嵌入全失敗，%d 塊）", len(todo_texts))
        return 0, skipped
    if not ensured.get("done"):
        sample_vec = next(iter(vec_map.values()))
        ensure_collection(len(sample_vec))
        ensured["done"] = True
    points: list[PointStruct] = []
    for order, orig_i in enumerate(todo_idx):
        if order not in vec_map:
            continue
        points.append(PointStruct(id=pids[orig_i], vector=vec_map[order], payload=batch_metas[orig_i]))
    for j in range(0, len(points), _CONFIG.upsert_batch):
        client.upsert(collection_name=_CONFIG.collection, points=points[j:j + _CONFIG.upsert_batch])
    return len(points), skipped


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
    """寫檔案級快取，失敗僅警告不中斷匯入。」"""
    try:
        (root / _INGEST_CACHE_NAME).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("匯入快取寫入失敗：%s", e)


def ingest_folder(folder: str, on_progress: object = None) -> int:
    """把資料夾內支援檔切塊＋嵌入＋寫入 Qdrant，回傳寫入點數。

    on_progress(done, total, rel)：可選進度回調，拋錯不中斷匯入。
    """
    _sync_config()
    root = Path(folder)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file() and p.name != _INGEST_CACHE_NAME])
    if not files:
        logger.warning("%s 內沒有支援的檔案（支援：%s），可先丟筆記進去", folder, sorted(SUPPORTED_SUFFIXES))
        return 0
    client = _client()
    ensured: dict[str, bool] = {"done": False}
    total = 0
    skipped_unchanged = 0
    skipped_files = 0
    skipped_cached = 0
    pending_chunks: list[str] = []
    pending_metas: list[dict[str, str]] = []
    file_count = 0
    ingest_cache = _load_ingest_cache(root)
    touched: dict[str, dict[str, object]] = {}

    def _drain() -> tuple[int, int]:
        nonlocal total, skipped_unchanged, pending_chunks, pending_metas
        if not pending_chunks:
            return (0, 0)
        written, skipped = _flush_batch(client, pending_chunks, pending_metas, ensured)
        total += written
        skipped_unchanged += skipped
        pending_chunks = []
        pending_metas = []
        return (written, skipped)

    def _report(done: int, rel: str) -> None:
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
        except ValueError:
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
        file_written = 0
        file_skipped = 0
        for chunk in chunks:
            pending_chunks.append(chunk)
            pending_metas.append({"source": rel, "text": chunk})
            if len(pending_chunks) >= _CONFIG.embed_batch:
                w, s = _drain()
                file_written += w
                file_skipped += s
        # 檔尾清餘批，確保不與下一檔混批，逐檔歸因
        w, s = _drain()
        file_written += w
        file_skipped += s
        # P15：完成數＝寫入＋未變跳過；等於總塊數才算整檔成功。
        # 部分嵌入失敗若誤標 ok，檔案級快取（mtime＋size）會永久跳過壞塊不再補。
        if file_written + file_skipped >= len(chunks):
            touched[rel]["ok"] = True
        _report(idx + 1, rel)
    ingest_cache.update(touched)
    _save_ingest_cache(root, ingest_cache)
    if total == 0:
        logger.info("無需寫入（快取跳過 %d 檔，內容未變跳過 %d 塊）。", skipped_cached, skipped_unchanged)
        return 0
    logger.info("共 %d 個檔（有效 %d，快取跳過 %d，跳過檔 %d，未變跳過 %d 塊），已寫入 %d 點到 %s。", len(files), file_count, skipped_cached, skipped_files, skipped_unchanged, total, _CONFIG.collection)
    return total


def search_local(query: str, limit: int = 3) -> list[dict[str, str]]:
    """把問題轉向量去 Qdrant 找最像的筆記塊，寬取後重排取精華。」"""
    _sync_config()
    q = query.strip()
    if not q:
        return []
    try:
        qvec = _embed_query_vec(q[:500])
        if qvec is None:
            return []
        client = _client()
        recall = max(limit, _CONFIG.rerank_recall) if limit > 0 else _CONFIG.rerank_recall
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
        try:
            return _rerank(q, hits, top_k=limit)
        except Exception as e:
            logger.warning("重排略過（%s），使用向量順序", e)
            return hits[:limit]
    except Exception as e:
        logger.warning("本地檢索略過（%s）", e)
        return []


def main() -> None:
    """命令列入口：--ingest 匯入，--query 測試查詢。」"""
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
            print(f"[{done}/{total}] {rel}", flush=True)

        ingest_folder(args.ingest, on_progress=_on_progress if args.progress else None)
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


if __name__ == "__main__":
    main()
