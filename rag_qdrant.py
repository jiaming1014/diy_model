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
import logging
import os
from pathlib import Path
import threading

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """安全讀整數環境變數，寫壞回預設不炸 import。」"""
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


@dataclass(frozen=True)
class RAGConfig:
    """RAG 可調參數集中表，內部唯一真相｜新手：所有旋鈕收在同一張面板。」"""

    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("QDRANT_COLLECTION", "notes")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    vision_model: str = os.getenv("VISION_MODEL", "llava:latest")
    timeout: float = _float_env("OLLAMA_TIMEOUT", 120.0)  # 優化：HTTP 逾時，避免嵌入/視覺卡死
    chunk_chars: int = _int_env("CHUNK_CHARS", 800)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 100)
    rerank_recall: int = _int_env("RERANK_RECALL", 15)
    embed_batch: int = _int_env("EMBED_BATCH", 32)
    upsert_batch: int = _int_env("UPSERT_BATCH", 64)
    chunk_max_tokens: int = _int_env("CHUNK_MAX_TOKENS", 0)  # 優化：>0 啟用 token 預算，需 pip install tiktoken，0 表示只用字元切


_CONFIG = RAGConfig()

# 優化：建立帶逾時的共用 client，避免 ollama.embed/chat 卡死
_ollama = ollama.Client(timeout=_CONFIG.timeout)


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
except Exception as _e:  # noqa: BROAD_EXCEPT_OK - reranker 缺失時仍要能純向量檢索
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
_cached_url = ""

# 優化：問題向量快取，同一問句重複檢索不再重新嵌入（省時間，回覆品質不變）
_query_vec_cache: dict[str, list[float]] = {}
_query_vec_cache_lock = threading.Lock()
_QUERY_VEC_CACHE_MAX: int = _int_env("QUERY_VEC_CACHE_MAX", 256)


def _embed_query_vec(query: str) -> list[float] | None:
    """問題轉向量，帶快取：同一問句重複查詢直接回之前算的向量。」"""
    with _query_vec_cache_lock:
        hit = _query_vec_cache.get(query)
        if hit is not None:
            return hit
    vecs = _embed_texts([query])
    if not vecs:
        return None
    vec = vecs[0]
    with _query_vec_cache_lock:
        if len(_query_vec_cache) >= _QUERY_VEC_CACHE_MAX:
            # 超量淘汰最舊（dict 保插入序）
            _query_vec_cache.pop(next(iter(_query_vec_cache)), None)
        _query_vec_cache[query] = vec
    return vec


def _client() -> QdrantClient:
    """Qdrant 單例連線｜新手：電話打一次就留著，別每次都重撥。」"""
    global _cached_client, _cached_url
    url = _CONFIG.url  # 優化：唯一真相走 _CONFIG，避免與相容快照漂移
    with _client_lock:
        if _cached_client is not None and _cached_url == url:
            return _cached_client
        if _cached_client is not None:
            try:
                _cached_client.close()
            except Exception:
                pass
        _cached_client = QdrantClient(url=url)
        _cached_url = url
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


def _chunk_text(text: str, chunk_chars: int | None = None, overlap: int | None = None, chunk_tokens: int | None = None) -> list[str]:
    """把長文切成小塊：先按空行分段，再按字數/token 切，塊間留重疊。」"""
    cc = chunk_chars if chunk_chars is not None else _CONFIG.chunk_chars
    ov = overlap if overlap is not None else _CONFIG.chunk_overlap
    ct = chunk_tokens if chunk_tokens is not None else _CONFIG.chunk_max_tokens
    clean = text.strip()
    if not clean:
        return []
    paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        probe = (buf + "\n\n" + para).strip() if buf else para
        if buf and _over_budget(probe, cc, ct):
            chunks.append(buf)
            buf = buf[-ov:] + "\n\n" + para if len(buf) > ov else para
        else:
            buf = probe
        while _over_budget(buf, cc, ct):
            chunks.append(buf[:cc])
            buf = buf[cc - ov:]
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
    """若收藏集不存在就建立；已存在則什麼都不做。」"""
    client = _client()
    try:
        info = client.get_collection(_CONFIG.collection)
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        existing_dim = getattr(vectors, "size", None)
        if existing_dim is not None and existing_dim != dim:
            logger.warning("現有維度與模型維度不符（現有 %s，模型 %s），請換收藏集名稱", existing_dim, dim)
        return
    except Exception as e:
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
    """讀 .pdf：逐頁抽文字再合併。缺 pypdf 時拋出提示。」"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(f"{fp.name} 是 PDF，但缺 pypdf，請先 pip install pypdf") from e
    reader = PdfReader(str(fp))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(f"--- 第 {i} 頁 ---\n{t.strip()}")
    return "\n\n".join(parts)


def _read_docx(fp: Path) -> str:
    """讀 .docx：段落＋表格都拿。」"""
    try:
        import docx
    except ImportError as e:
        raise RuntimeError(f"{fp.name} 是 DOCX，但缺 python-docx，請先 pip install python-docx") from e
    doc = docx.Document(str(fp))
    parts: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            line = " | ".join(c.text.strip() for c in row.cells).strip(" |")
            if line.strip(" |"):
                parts.append(line)
    return "\n".join(parts)


def _read_csv(fp: Path) -> str:
    """讀 .csv：標頭＋每列轉文字，只用標準庫。」"""
    with fp.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        try:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]) if sample.strip() else csv.excel
        except Exception:
            f.seek(0)
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = [[c.strip() for c in r] for r in reader if any(c.strip() for c in r)]
    if not rows:
        return ""
    header = rows[0]
    lines = ["表頭：" + " | ".join(header)]
    truncated = max(0, len(rows) - 1 - 5000)
    for r in rows[1:5001]:
        if len(r) == len(header):
            lines.append("；".join(f"{h}：{v}" for h, v in zip(header, r) if v))
        else:
            lines.append(" | ".join(r))
    if truncated:
        logger.warning("%s 超過 5000 列，已截斷 %d 列", fp.name, truncated)
    return "\n".join(lines)


def _read_image(fp: Path) -> str:
    """讀圖片：OCR → 視覺模型 → 檔名佔位，三層降級。」"""
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
    """依副檔名分派讀檔，統一回傳純文字。」"""
    suffix = fp.suffix.lower()
    if suffix in {".txt", ".md"}:
        return fp.read_text(encoding="utf-8", errors="ignore")
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


def ingest_folder(folder: str) -> int:
    """把資料夾內支援檔切塊＋嵌入＋寫入 Qdrant，回傳寫入點數。」"""
    root = Path(folder)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()])
    if not files:
        logger.warning("%s 內沒有支援的檔案（支援：%s），可先丟筆記進去", folder, sorted(SUPPORTED_SUFFIXES))
        return 0
    client = _client()
    ensured: dict[str, bool] = {"done": False}
    total = 0
    skipped_unchanged = 0
    skipped_files = 0
    pending_chunks: list[str] = []
    pending_metas: list[dict[str, str]] = []
    file_count = 0

    def _drain() -> None:
        nonlocal total, skipped_unchanged, pending_chunks, pending_metas
        if not pending_chunks:
            return
        written, skipped = _flush_batch(client, pending_chunks, pending_metas, ensured)
        total += written
        skipped_unchanged += skipped
        pending_chunks = []
        pending_metas = []

    for fp in files:
        try:
            text = _read_file_text(fp)
        except Exception as e:
            logger.warning("跳過 %s（%s）", fp.name, e)
            skipped_files += 1
            continue
        try:
            rel = str(fp.relative_to(root))
        except ValueError:
            rel = fp.name
        chunks = _chunk_text(text)
        if not chunks:
            continue
        file_count += 1
        for chunk in chunks:
            pending_chunks.append(chunk)
            pending_metas.append({"source": rel, "text": chunk})
            if len(pending_chunks) >= _CONFIG.embed_batch:
                _drain()
    _drain()
    if total == 0:
        logger.info("無需寫入（內容未變跳過 %d 塊）。", skipped_unchanged)
        return 0
    logger.info("共 %d 個檔（有效 %d，跳過檔 %d，未變跳過 %d 塊），已寫入 %d 點到 %s。", len(files), file_count, skipped_files, skipped_unchanged, total, _CONFIG.collection)
    return total


def search_local(query: str, limit: int = 3) -> list[dict[str, str]]:
    """把問題轉向量去 Qdrant 找最像的筆記塊，寬取後重排取精華。」"""
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
    ap.add_argument("--verbose", action="store_true", help="顯示除錯訊息")
    args = ap.parse_args()
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
    if args.ingest:
        ingest_folder(args.ingest)
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
