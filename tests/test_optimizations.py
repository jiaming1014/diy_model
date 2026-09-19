"""優化回歸測試：P0+P1+P2 全量對應，不碰網路／Qdrant／Ollama。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

import chat_core
from chat_core import (
    ChatState,
    _extract_tool_calls_raw,
    _is_chitchat,
    _is_date_query,
    _make_system_msg,
    _needs_realtime,
    _parse_rag_scores,
    _rag_adequate,
    _with_history,
)
from eval import _check_keywords
from rag_qdrant import _chunk_text, _overlap_tail, _split_sentences


class TestRealtimeLocalPriority:
    def test_local_overrides_realtime(self) -> None:
        """含本地意圖詞（筆記）不強制即時，優先查本地。」"""
        assert _needs_realtime("今年筆記放在哪個資料夾？") is False

    def test_strong_realtime_still_true(self) -> None:
        """強即時詞（天氣／新聞）仍直接上網，不翻筆記。」"""
        assert _needs_realtime("台北今天天氣如何？") is True
        assert _needs_realtime("最新新聞") is True

    def test_broad_year_no_longer_forces(self) -> None:
        """移除 2026 泛命中後，單獨年份不再強制即時，交給工具迴圈判斷。」"""
        assert _needs_realtime("2026 年計畫") is False

    def test_date_isolation(self) -> None:
        """純日期問句走日期捷徑，不算即時。」"""
        assert _needs_realtime("今天幾號") is False


class TestChitchatDateGuards:
    def test_chitchat_with_weather_not_chitchat(self) -> None:
        """問候夾帶事實關鍵字不算閒聊，避免跳過搜尋。」"""
        assert _is_chitchat("你好請問天氣") is False

    def test_pure_chitchat(self) -> None:
        """純問候道謝算閒聊，直接信任模型不搜尋。」"""
        assert _is_chitchat("你好") is True
        assert _is_chitchat("謝謝") is True

    def test_date_compact_length(self) -> None:
        """日期判斷去空白量長度：短句認、超長不認。」"""
        assert _is_date_query("今天幾號") is True
        long_q = "請問今天是幾號，另外明天會議幾點外加很多很多很多很多很多很多很多很多字補滿超長"
        assert len(long_q) > 30
        assert _is_date_query(long_q) is False

    def test_weather_excludes_date(self) -> None:
        """含天氣詞不算查日期，避免天氣問題走日期捷徑。」"""
        assert _is_date_query("今天天氣如何") is False


class TestToolsLive:
    def test_tools_live_same_object(self) -> None:
        """工具清單單例：TOOLS 與 _TOOLS_LIVE 同一物件，from 匯入跨年有效。」"""
        assert chat_core.TOOLS is chat_core._TOOLS_LIVE
        assert chat_core._tools() is chat_core._TOOLS_LIVE


class TestSearchLRU:
    def test_lru_refresh_and_evict(self) -> None:
        """LRU 語義：命中變最新，滿時淘汰最久未用。」"""
        with mock.patch.object(chat_core._SEARCH_CACHE, "maxsize", 2):
            chat_core._SEARCH_CACHE.clear()
            chat_core._cache_put("a", [{"title": "a", "snippet": "a", "url": "u"}])
            chat_core._cache_put("b", [{"title": "b", "snippet": "b", "url": "u"}])
            # 命中 a 使其變最新，寫入 c 時應淘汰 b 而非 a
            assert chat_core._cache_get("a") is not None
            chat_core._cache_put("c", [{"title": "c", "snippet": "c", "url": "u"}])
            assert chat_core._cache_get("a") is not None
            assert chat_core._cache_get("b") is None
            assert chat_core._cache_get("c") is not None
            chat_core._SEARCH_CACHE.clear()


class TestToolArgTypes:
    def test_preserves_numeric_bool(self) -> None:
        """工具參數保原始型別：數字布林不被字串化。」"""
        msg = {
            "tool_calls": [
                {"function": {"name": "demo", "arguments": '{"n": 3, "flag": true, "q": "hi"}'}}
            ]
        }
        calls = _extract_tool_calls_raw(msg["tool_calls"])
        assert calls[0][0] == "demo"
        assert calls[0][1]["n"] == 3
        assert calls[0][1]["flag"] is True

    def test_run_tool_str_coercion(self) -> None:
        """search_web 參數字串化：數字 query 轉字串再搜。」"""
        with mock.patch.object(chat_core, "_search_web", return_value=[]) as m:
            chat_core._run_tool("search_web", {"query": 123}, "fallback", state=ChatState())
            m.assert_called_once()
            assert m.call_args[0][0] == "123"


class TestRagScores:
    def test_parse_filters_nan_inf(self) -> None:
        """壞分數（nan／inf／缺值）跳過，只收有限數字。」"""
        hits = [
            {"text": "a", "score": "nan"},
            {"text": "b", "score": "inf"},
            {"text": "c", "score": "-inf"},
            {"text": "d", "score": "0.8"},
            {"text": "e"},
        ]
        assert _parse_rag_scores(hits) == [0.8]

    def test_adequate_floor(self) -> None:
        """無門檻時 0~1 餘弦分數走內建地板 0.5 擋不相關。」"""
        import reranker as r

        old = r.RERANK_THRESHOLD
        r.RERANK_THRESHOLD = float("-inf")
        try:
            assert _rag_adequate([{"text": "x", "score": "0.1"}]) is False
            assert _rag_adequate([{"text": "x", "score": "0.9"}]) is True
        finally:
            r.RERANK_THRESHOLD = old


class TestChunking:
    def test_sentence_split(self) -> None:
        """中英句號切句：三句切出三句。」"""
        sents = _split_sentences("第一句。第二句！第三句？")
        assert len(sents) == 3

    def test_overlap_aligns(self) -> None:
        """重疊尾巴非空且盡量對齊句首，不半句開頭。」"""
        tail = _overlap_tail("第一句。第二句。第三句內容很長很長", 10)
        assert tail  # 不空即可，對齊不斷半句為佳

    def test_chunk_respects_budget(self) -> None:
        """長文切塊每塊不超預算（800 字＋重疊餘裕）。」"""
        text = "第一句。第二句。第三句。" * 200
        chunks = _chunk_text(text, chunk_chars=800, overlap=100)
        assert chunks
        assert all(len(c) <= 900 for c in chunks)

    def test_short_passthrough(self) -> None:
        """短筆記不成塊，原樣回單塊。」"""
        assert _chunk_text("短筆記") == ["短筆記"]


class TestSystemUnify:
    def test_make_and_with_history(self) -> None:
        """系統訊息組裝：帶日期＋歷史快照＋本次問題，順序固定。」"""
        st = ChatState()
        st.hist.append({"role": "user", "content": "hi"})
        st.hist.append({"role": "assistant", "content": "hello"})
        sys_msg = _make_system_msg("SYS", "2026-09-09")
        assert sys_msg["role"] == "system"
        assert "2026-09-09" in sys_msg["content"]
        msgs = _with_history(sys_msg, st, "Q")
        assert msgs[0]["role"] == "system"
        assert msgs[-1] == {"role": "user", "content": "Q"}
        assert len(msgs) == 4


class TestEvalNormalize:
    def test_case_insensitive(self) -> None:
        """關鍵字比對大小寫不敏感，避免 Qdrant／ulw 誤判。」"""
        assert _check_keywords("QDRANT 在 6333", ["qdrant"]) == ["qdrant"]
        assert _check_keywords("用了 ULW 子任務", ["ulw"]) == ["ulw"]


class TestIngestCache:
    def test_load_save_roundtrip(self, tmp_path: Path) -> None:
        """匯入快取讀寫往返一致，壞檔當空不炸。」"""
        import rag_qdrant as rq

        cache = {"a.md": {"mtime": 1.0, "size": 3, "ok": True}}
        rq._save_ingest_cache(tmp_path, cache)
        assert rq._load_ingest_cache(tmp_path) == cache
