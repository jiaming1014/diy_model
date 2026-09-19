"""P18 回歸測試：refresh 扇出節流／失敗短快取／快取統計，不碰真網路。"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

import chat_core
from ttl_cache import TTLCache


class TestRefreshFanout:
    def test_unrelated_modules_skipped(self, monkeypatch) -> None:
        """只改搜尋區：rag／重排無相關異動，跳過同步。」"""
        import config as c

        import rag_qdrant as rq
        import reranker as r

        monkeypatch.setenv("SEARCH_REGION", "xx-yy")
        try:
            with mock.patch.object(rq, "_sync_config") as m_rq:
                with mock.patch.object(r, "_sync_config") as m_r:
                    c.refresh()
                    assert c.SEARCH_REGION == "xx-yy"
                    assert chat_core.SEARCH_REGION == "xx-yy"
                    m_rq.assert_not_called()
                    m_r.assert_not_called()
        finally:
            monkeypatch.undo()
            c.refresh()

    def test_related_module_synced(self, monkeypatch) -> None:
        """改重排批大小：重排模組有相關異動，正常同步。」"""
        import config as c

        import reranker as r

        monkeypatch.setenv("RERANK_BATCH", "5")
        try:
            c.refresh()
            assert r.RERANK_BATCH == 5
        finally:
            monkeypatch.undo()
            c.refresh()


class TestFailCache:
    def test_failed_search_cached_short(self) -> None:
        """兩次全失敗才回空，第二次命中失敗短快取不再打網路。」"""
        chat_core._SEARCH_CACHE.clear()
        q = "fail-cache-unique-p18xyz"
        with mock.patch.object(chat_core, "DDGS", side_effect=Exception("down")) as m:
            with mock.patch.object(chat_core.time, "sleep", return_value=None):
                out1 = chat_core._search_web(q, state=chat_core.ChatState())
                out2 = chat_core._search_web(q, state=chat_core.ChatState())
        assert out1 == [] and out2 == []
        assert m.call_count == 2  # 第二次命中失敗短快取，不再打網路
        chat_core._SEARCH_CACHE.clear()


class TestCacheStats:
    def test_stats_counts(self) -> None:
        """命中統計計數：命中／未中／寫入／淘汰各記一次。」"""
        c: TTLCache[int] = TTLCache(maxsize=2, ttl=10.0)
        c.put("a", 1)
        assert c.get("a") == 1
        assert c.get("z") is None
        c.put("b", 2)
        c.put("c", 3)  # 滿了踢最舊(a)
        assert c.get("a") is None
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 2
        assert s["puts"] == 3
        assert s["evictions"] == 1
        assert s["expirations"] == 0

    def test_per_entry_ttl(self) -> None:
        """單筆 TTL 覆寫預設：短筆過期、長筆仍活。」"""
        c: TTLCache[int] = TTLCache(maxsize=5, ttl=100.0)
        c.put("short", 1, now=0.0, ttl=5.0)
        c.put("long", 2, now=0.0)
        assert c.get("short", now=6.0) is None  # 單筆過期
        assert c.get("long", now=6.0) == 2  # 預設還活著
        assert c.stats()["expirations"] == 1

    def test_clear_resets_stats(self) -> None:
        """清空連計數歸零，下輪統計不污染。」"""
        c: TTLCache[int] = TTLCache(maxsize=2, ttl=10.0)
        c.put("a", 1)
        c.get("a")
        c.clear()
        assert c.stats() == {"hits": 0, "misses": 0, "puts": 0, "evictions": 0, "expirations": 0}


class TestEmptySuccessShortTTL:
    def test_empty_success_uses_short_ttl(self) -> None:
        """成功但 0 筆與失敗同視，短快取避免舊無結果殘留。」"""
        chat_core._SEARCH_CACHE.clear()
        q = "empty-success-unique-p19xyz"
        with mock.patch.object(chat_core, "DDGS") as m:
            inst = m.return_value.__enter__.return_value
            inst.text.return_value = []  # 成功但 0 筆
            out = chat_core._search_web(q, state=chat_core.ChatState())
        assert out == []
        key = f"{chat_core.SEARCH_MAX_RESULTS}::{chat_core.SEARCH_REGION}::{chat_core._clean_query_for_search(q)}"
        # 短 TTL：超過失敗快取秒數即過期（若誤用 300 秒預設則不會過期）
        assert chat_core._SEARCH_CACHE.get(key, now=time.monotonic() + float(chat_core.SEARCH_FAIL_CACHE_TTL) + 1.0) is None
        chat_core._SEARCH_CACHE.clear()


class TestCacheStatsLogged:
    def test_hit_and_store_logged(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """命中與寫入各記一條 debug，數字與 stats() 同源。」"""
        chat_core._SEARCH_CACHE.clear()
        q = "stats-log-unique-p19xyz"
        with mock.patch.object(chat_core, "DDGS") as m:
            inst = m.return_value.__enter__.return_value
            inst.text.return_value = [{"title": "t", "body": "s", "href": "https://e.com"}]
            with caplog.at_level(logging.DEBUG):
                chat_core._search_web(q, state=chat_core.ChatState())  # 未命中→寫入
                chat_core._search_web(q, state=chat_core.ChatState())  # 命中
        assert "搜尋快取寫入" in caplog.text
        assert "搜尋快取命中" in caplog.text
        chat_core._SEARCH_CACHE.clear()
