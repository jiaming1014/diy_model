"""P14 回歸測試：共用 TTL 快取／client 生命週期／切塊防呆／重複搜尋防護。

全部不碰真網路：ollama.Client 僅建構（不呼叫 API），其餘外部依賴以 mock 隔離。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

from ttl_cache import TTLCache


# ------------------------------------------------------------
# 1. 共用 TTLCache：LRU、TTL、調整上限、清空
# ------------------------------------------------------------
class TestTTLCache:
    def test_put_get_roundtrip(self) -> None:
        c: TTLCache[int] = TTLCache(maxsize=2, ttl=100.0)
        c.put("a", 1)
        assert c.get("a") == 1
        assert c.get("missing") is None

    def test_lru_evicts_oldest(self) -> None:
        c: TTLCache[int] = TTLCache(maxsize=2, ttl=100.0)
        c.put("a", 1)
        c.put("b", 2)
        assert c.get("a") == 1  # a 變最近使用
        c.put("c", 3)
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3

    def test_ttl_expiry(self) -> None:
        c: TTLCache[int] = TTLCache(maxsize=2, ttl=10.0)
        c.put("a", 1, now=0.0)
        assert c.get("a", now=5.0) == 1
        assert c.get("a", now=11.0) is None

    def test_update_limits_shrinks_lru(self) -> None:
        c: TTLCache[int] = TTLCache(maxsize=5, ttl=100.0)
        for i in range(5):
            c.put(str(i), i)
        c.update_limits(maxsize=2)
        assert len(c) == 2
        assert c.get("3") == 3
        assert c.get("4") == 4

    def test_clear(self) -> None:
        c: TTLCache[int] = TTLCache(maxsize=2, ttl=100.0)
        c.put("a", 1)
        c.clear()
        assert c.get("a") is None
        assert len(c) == 0


# ------------------------------------------------------------
# 2. client 生命週期：逾時沒變就不重建（舊版每次都換）
# ------------------------------------------------------------
class TestClientLifecycle:
    def test_rag_keeps_client_when_timeout_unchanged(self, monkeypatch) -> None:
        import rag_qdrant as rq

        sentinel = object()
        monkeypatch.setattr(rq, "_ollama", sentinel)
        rq._sync_config()
        assert rq._ollama is sentinel

    def test_rerank_keeps_client_when_timeout_unchanged(self, monkeypatch) -> None:
        import reranker as r

        sentinel = object()
        monkeypatch.setattr(r, "_ollama", sentinel)
        r._sync_config()
        assert r._ollama is sentinel


# ------------------------------------------------------------
# 3. 切塊防呆：overlap >= chunk_chars 不再無限迴圈
# ------------------------------------------------------------
class TestChunkGuard:
    def test_overlap_ge_chunk_no_hang(self) -> None:
        import threading

        import rag_qdrant as rq

        result: dict = {}

        def _run() -> None:
            result["chunks"] = rq._chunk_text("第一句。第二句。" * 200, chunk_chars=100, overlap=100)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=10)
        assert "chunks" in result, "切塊卡死（無限迴圈）"
        assert result["chunks"]
        assert all(len(c) <= 100 for c in result["chunks"])


# ------------------------------------------------------------
# 4. 預補搜決策（P15：串流流程改在模型開跑前預判；
#    舊「工具搜過就跳過補搜」的事後防重複結構已隨流程重構移除）
# ------------------------------------------------------------
class TestPreSearchGuard:
    def test_realtime_without_sources_triggers(self) -> None:
        import chat_core

        assert chat_core._needs_pre_search("台北今天天氣如何？", [], "", chat_core.ChatState()) is True

    def test_inadequate_local_triggers(self) -> None:
        import chat_core

        assert chat_core._needs_pre_search("冷知識問題", [], "", chat_core.ChatState()) is True

    def test_local_adequate_skips(self) -> None:
        import chat_core

        hits = [{"text": "x", "score": "0.9"}]
        assert chat_core._needs_pre_search("這個專案的問題", hits, "", chat_core.ChatState()) is False

    def test_realtime_adequate_with_sources_skips(self) -> None:
        import chat_core

        st = chat_core.ChatState()
        st.last_sources.append({"title": "t", "snippet": "s", "url": "u"})
        hits = [{"text": "x", "score": "0.9"}]
        assert chat_core._needs_pre_search("今天天氣如何？", hits, "", st) is False

    def test_chitchat_never(self) -> None:
        import chat_core

        assert chat_core._needs_pre_search("你好", [], "", chat_core.ChatState()) is False
