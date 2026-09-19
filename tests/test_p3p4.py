"""P3+P4 回歸測試：預算／去重／截斷／清洗／TTL／refresh／MRR，不碰真網路。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

import chat_core
from chat_core import (
    ChatState,
    _clean_query_for_search,
    _format_search_results,
    _normalize_url,
    _truncate_user_msg,
)


class TestUserTruncate:
    def test_short_passthrough(self) -> None:
        """短輸入原樣通過，不截斷不加註記。」"""
        assert _truncate_user_msg("你好") == "你好"

    def test_long_truncated(self) -> None:
        """超長輸入留頭加註記，避免單輪撐爆上下文。」"""
        long_q = "字" * (chat_core.USER_MAX_CHARS + 100)
        out = _truncate_user_msg(long_q)
        assert len(out) <= chat_core.USER_MAX_CHARS + 20
        assert out.endswith("…（過長已截斷）")


class TestQueryClean:
    def test_filler_removed(self) -> None:
        """口語填充詞（請問／？）去除，留檢索關鍵字。」"""
        out = _clean_query_for_search("請問台北天氣如何？")
        assert "請問" not in out
        assert "台北天氣如何" in out

    def test_truncates(self) -> None:
        """清洗後查詢受 SEARCH_QUERY_MAX_CHARS 限制。」"""
        out = _clean_query_for_search("字" * 500)
        assert len(out) <= chat_core.SEARCH_QUERY_MAX_CHARS


class TestUrlNormalize:
    def test_same_url_variants(self) -> None:
        """大小寫＋追蹤參數＋尾斜線視為同一網址（去重用）。」"""
        assert _normalize_url("https://Example.com/a/?x=1") == _normalize_url("https://example.com/a")


class TestSearchDedup:
    def test_duplicate_urls_deduped(self) -> None:
        """同一網址多筆只留第一筆，追蹤參數不算不同來源。」"""
        raw = [
            {"title": "t1", "body": "s1", "href": "https://example.com/a?utm=1"},
            {"title": "t1b", "body": "s1b", "href": "https://EXAMPLE.com/a"},
            {"title": "t2", "body": "s2", "href": "https://example.com/b"},
        ]
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = raw
            chat_core._SEARCH_CACHE.clear()
            out = chat_core._search_web("去重測試P3", state=ChatState())
        assert len(out) == 2
        assert out[0]["url"] == "https://example.com/a?utm=1"
        chat_core._SEARCH_CACHE.clear()

    def test_snippet_truncated(self) -> None:
        """超長標題摘要按單筆上限截斷，不撐爆提示詞。」"""
        raw = [{"title": "t" * 500, "body": "s" * 2000, "href": "https://example.com/c"}]
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = raw
            chat_core._SEARCH_CACHE.clear()
            out = chat_core._search_web("截斷測試P3", state=ChatState())
        assert len(out[0]["title"]) <= chat_core.SEARCH_TITLE_CHARS
        assert len(out[0]["snippet"]) <= chat_core.SEARCH_SNIPPET_CHARS
        chat_core._SEARCH_CACHE.clear()


class TestSearchBudget:
    def test_format_respects_budget(self) -> None:
        """搜尋結果拼提示詞受 SEARCH_MAX_CHARS 總預算限制。」"""
        hits = [
            {"title": f"t{i}", "snippet": "字" * 2000, "url": f"https://e.com/{i}"}
            for i in range(5)
        ]
        out = _format_search_results(hits)
        assert len(out) <= chat_core.SEARCH_MAX_CHARS + 500


class TestRewrite:
    def test_rule_fallback_when_flag_off(self) -> None:
        """改寫開關關閉時走規則清洗，不打模型。」"""
        with mock.patch.object(chat_core, "QUERY_REWRITE_LLM", False):
            assert chat_core._maybe_llm_rewrite("請問台北天氣？") == _clean_query_for_search("請問台北天氣？")


class TestQueryVecTTL:
    def test_hit_returns_copy_and_lru(self) -> None:
        """查詢向量快取命中只嵌一次，二次查詢走快取。」"""
        import rag_qdrant as rq

        rq._QUERY_VEC_CACHE.clear()
        with mock.patch.object(rq, "_embed_texts", return_value=[[0.1, 0.2]]) as m:
            v1 = rq._embed_query_vec("ttl測試")
            v2 = rq._embed_query_vec("ttl測試")
            assert v1 == v2
            assert m.call_count == 1
        rq._QUERY_VEC_CACHE.clear()


class TestConfigRefresh:
    def test_refresh_picks_new_env(self, monkeypatch) -> None:
        """refresh 重讀環境變數並回傳異動表。」"""
        import config as c

        old = c.USER_MAX_CHARS
        monkeypatch.setenv("USER_MAX_CHARS", "1234")
        try:
            diff = c.refresh()
            assert c.USER_MAX_CHARS == 1234
            assert "USER_MAX_CHARS" in diff
        finally:
            # monkeypatch 會還原 env，這裡還原模組快照避免污染後續測試
            c.refresh()


class TestEvalMetrics:
    def test_first_hit_rank(self) -> None:
        """首個全命中排名：命中回 1 起排名，無命中回 None。」"""
        from eval import _first_hit_rank

        hits = [{"text": "無關"}, {"text": "有 帶傘 和 雷陣雨"}]
        assert _first_hit_rank(hits, ["帶傘", "雷陣雨"]) == 2
        assert _first_hit_rank([{"text": "無"}], ["帶傘"]) is None

    def test_retrieval_rr_recall(self) -> None:
        """檢索層指標：第 2 名命中 RR=0.5、關鍵字全中召回 1.0。」"""
        from eval import _eval_retrieval

        fake_hits = [{"source": "a", "text": "帶傘出門"}, {"source": "b", "text": "雷陣雨注意帶傘"}]
        with mock.patch("eval.search_local", return_value=fake_hits):
            r = _eval_retrieval({"q": "天氣", "keywords": ["帶傘", "雷陣雨"]})
            assert r["rank"] == 2
            assert r["rr"] == 0.5
            assert r["recall"] == 1.0
