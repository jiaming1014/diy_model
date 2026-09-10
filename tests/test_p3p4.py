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
        assert _truncate_user_msg("你好") == "你好"

    def test_long_truncated(self) -> None:
        long_q = "字" * (chat_core.USER_MAX_CHARS + 100)
        out = _truncate_user_msg(long_q)
        assert len(out) <= chat_core.USER_MAX_CHARS + 20
        assert out.endswith("…（過長已截斷）")


class TestQueryClean:
    def test_filler_removed(self) -> None:
        out = _clean_query_for_search("請問台北天氣如何？")
        assert "請問" not in out
        assert "台北天氣如何" in out

    def test_truncates(self) -> None:
        out = _clean_query_for_search("字" * 500)
        assert len(out) <= chat_core.SEARCH_QUERY_MAX_CHARS


class TestUrlNormalize:
    def test_same_url_variants(self) -> None:
        assert _normalize_url("https://Example.com/a/?x=1") == _normalize_url("https://example.com/a")


class TestSearchDedup:
    def test_duplicate_urls_deduped(self) -> None:
        raw = [
            {"title": "t1", "body": "s1", "href": "https://example.com/a?utm=1"},
            {"title": "t1b", "body": "s1b", "href": "https://EXAMPLE.com/a"},
            {"title": "t2", "body": "s2", "href": "https://example.com/b"},
        ]
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = raw
            chat_core._search_cache.clear()
            chat_core._search_cache_time.clear()
            out = chat_core._search_web("去重測試P3", state=ChatState())
        assert len(out) == 2
        assert out[0]["url"] == "https://example.com/a?utm=1"
        chat_core._search_cache.clear()
        chat_core._search_cache_time.clear()

    def test_snippet_truncated(self) -> None:
        raw = [{"title": "t" * 500, "body": "s" * 2000, "href": "https://example.com/c"}]
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = raw
            chat_core._search_cache.clear()
            chat_core._search_cache_time.clear()
            out = chat_core._search_web("截斷測試P3", state=ChatState())
        assert len(out[0]["title"]) <= chat_core.SEARCH_TITLE_CHARS
        assert len(out[0]["snippet"]) <= chat_core.SEARCH_SNIPPET_CHARS
        chat_core._search_cache.clear()
        chat_core._search_cache_time.clear()


class TestSearchBudget:
    def test_format_respects_budget(self) -> None:
        hits = [
            {"title": f"t{i}", "snippet": "字" * 2000, "url": f"https://e.com/{i}"}
            for i in range(5)
        ]
        out = _format_search_results(hits)
        assert len(out) <= chat_core.SEARCH_MAX_CHARS + 500


class TestRewrite:
    def test_rule_fallback_when_flag_off(self) -> None:
        with mock.patch.object(chat_core, "QUERY_REWRITE_LLM", False):
            assert chat_core._maybe_llm_rewrite("請問台北天氣？") == _clean_query_for_search("請問台北天氣？")


class TestQueryVecTTL:
    def test_hit_returns_copy_and_lru(self) -> None:
        import rag_qdrant as rq

        rq._query_vec_cache.clear()
        rq._query_vec_cache_time.clear()
        with mock.patch.object(rq, "_embed_texts", return_value=[[0.1, 0.2]]) as m:
            v1 = rq._embed_query_vec("ttl測試")
            v2 = rq._embed_query_vec("ttl測試")
            assert v1 == v2
            assert m.call_count == 1
        rq._query_vec_cache.clear()
        rq._query_vec_cache_time.clear()


class TestConfigRefresh:
    def test_refresh_picks_new_env(self, monkeypatch) -> None:
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
        from eval import _first_hit_rank

        hits = [{"text": "無關"}, {"text": "有 帶傘 和 雷陣雨"}]
        assert _first_hit_rank(hits, ["帶傘", "雷陣雨"]) == 2
        assert _first_hit_rank([{"text": "無"}], ["帶傘"]) is None

    def test_retrieval_rr_recall(self) -> None:
        from eval import _eval_retrieval

        fake_hits = [{"source": "a", "text": "帶傘出門"}, {"source": "b", "text": "雷陣雨注意帶傘"}]
        with mock.patch("eval.search_local", return_value=fake_hits):
            r = _eval_retrieval({"q": "天氣", "keywords": ["帶傘", "雷陣雨"]})
            assert r["rank"] == 2
            assert r["rr"] == 0.5
            assert r["recall"] == 1.0
