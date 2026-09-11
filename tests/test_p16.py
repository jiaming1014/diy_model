"""P16 回歸測試：搜尋快取鍵／region／滾動摘要／Qdrant api_key／ingest 進度／
text_utils 抽取／rewrite 模型，不碰真網路。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

import chat_core


# ------------------------------------------------------------
# 1. 搜尋快取鍵：不同 max_results 不共用快取
# ------------------------------------------------------------
class TestSearchCacheKey:
    def test_different_max_results_not_shared(self) -> None:
        chat_core._SEARCH_CACHE.clear()
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = [{"title": "t", "body": "s", "href": "https://e.com"}]
            chat_core._search_web("cache-key-test", max_results=3, state=chat_core.ChatState())
            chat_core._search_web("cache-key-test", max_results=5, state=chat_core.ChatState())
            assert m_ddgs.call_count == 2
        chat_core._SEARCH_CACHE.clear()

    def test_same_params_hit_cache(self) -> None:
        chat_core._SEARCH_CACHE.clear()
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = [{"title": "t", "body": "s", "href": "https://e.com"}]
            chat_core._search_web("same-key-test", max_results=3, state=chat_core.ChatState())
            chat_core._search_web("same-key-test", max_results=3, state=chat_core.ChatState())
            assert m_ddgs.call_count == 1
        chat_core._SEARCH_CACHE.clear()


# ------------------------------------------------------------
# 2. 搜尋 region 由 config 控制
# ------------------------------------------------------------
class TestSearchRegion:
    def test_region_from_config(self, monkeypatch) -> None:
        chat_core._SEARCH_CACHE.clear()
        # region 的唯一真相在 config：先改真相，_sync_config 會帶進 chat_core 快照
        monkeypatch.setattr(chat_core._config_module, "SEARCH_REGION", "jp-jp")
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = []
            chat_core._search_web("region-test", state=chat_core.ChatState())
        assert inst.text.call_args.kwargs.get("region") == "jp-jp"
        chat_core._SEARCH_CACHE.clear()


# ------------------------------------------------------------
# 3. LLM 查詢改寫沿用本回合 model（P15 改動的補測）
# ------------------------------------------------------------
class TestRewriteModel:
    def test_rewrite_uses_session_model(self) -> None:
        with mock.patch.object(chat_core, "QUERY_REWRITE_LLM", True):
            with mock.patch.object(chat_core, "_call_chat_with_retry", return_value={"message": {"content": "台北天氣"}}) as m:
                out = chat_core._maybe_llm_rewrite("請問台北天氣？", model="my-model")
        assert out == chat_core._clean_query_for_search("台北天氣")
        assert m.call_args[0][1] == "my-model"


# ------------------------------------------------------------
# 4. text_utils 抽取：同名重匯出、config 即時值
# ------------------------------------------------------------
class TestTextUtilsExtraction:
    def test_reexport_same_object(self) -> None:
        import text_utils

        assert chat_core._is_date_query is text_utils._is_date_query
        assert chat_core._needs_realtime is text_utils._needs_realtime
        assert chat_core._clean_query_for_search is text_utils._clean_query_for_search

    def test_truncate_reads_config_live(self, monkeypatch) -> None:
        import config

        monkeypatch.setattr(config, "USER_MAX_CHARS", 5)
        out = chat_core._truncate_user_msg("1234567890")
        assert out.startswith("12345")
        assert out.endswith("…（過長已截斷）")


# ------------------------------------------------------------
# 5. 歷史滾動摘要（opt-in：HIST_SUMMARY_ENABLE）
# ------------------------------------------------------------
class TestRollingSummary:
    def _mk_pairs(self, n: int, size: int = 50) -> chat_core.ChatState:
        st = chat_core.ChatState()
        for i in range(n):
            st.hist.append({"role": "user", "content": f"問題{i}" + "字" * size})
            st.hist.append({"role": "assistant", "content": f"回答{i}" + "字" * size})
        return st

    def test_summary_created_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_ENABLE", True)
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_MIN_DROPPED", 1)
        st = self._mk_pairs(6)  # 6 組 > 4 組上限 → 觸發裁切
        with mock.patch.object(chat_core, "_call_chat_with_retry", return_value={"message": {"content": "這是摘要"}}) as m:
            chat_core._trim_hist(st, model="m")
        assert st.summary == "這是摘要"
        assert m.call_args[0][1] == "m"
        assert len(st.hist) <= 2 * chat_core.backtrace

    def test_disabled_by_default_no_llm(self) -> None:
        st = self._mk_pairs(6)
        with mock.patch.object(chat_core, "_call_chat_with_retry") as m:
            chat_core._trim_hist(st, model="m")
        m.assert_not_called()
        assert st.summary == ""

    def test_failure_keeps_old_summary(self, monkeypatch) -> None:
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_ENABLE", True)
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_MIN_DROPPED", 1)
        st = self._mk_pairs(6)
        st.summary = "舊摘要"
        with mock.patch.object(chat_core, "_call_chat_with_retry", side_effect=RuntimeError("boom")):
            chat_core._trim_hist(st, model="m")
        assert st.summary == "舊摘要"

    def test_small_drop_skipped(self, monkeypatch) -> None:
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_ENABLE", True)
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_MIN_DROPPED", 100000)
        st = self._mk_pairs(6)
        with mock.patch.object(chat_core, "_call_chat_with_retry") as m:
            chat_core._trim_hist(st, model="m")
        m.assert_not_called()

    def test_summary_injected_in_system(self) -> None:
        st = chat_core.ChatState()
        st.summary = "摘要X"
        msgs = chat_core._with_history({"role": "system", "content": "SYS"}, st, "Q")
        assert "摘要X" in msgs[0]["content"]


# ------------------------------------------------------------
# 6. Qdrant api_key 支援
# ------------------------------------------------------------
class TestQdrantApiKey:
    def test_api_key_passed(self, monkeypatch) -> None:
        import dataclasses

        import rag_qdrant as rq

        cfg = dataclasses.replace(rq._CONFIG, api_key="secret")
        monkeypatch.setattr(rq, "_CONFIG", cfg)
        monkeypatch.setattr(rq, "_cached_client", None)
        monkeypatch.setattr(rq, "_cached_key", None)
        with mock.patch.object(rq, "QdrantClient") as m_cls:
            rq._client()
        m_cls.assert_called_once_with(url=cfg.url, api_key="secret")

    def test_no_key_omits_param(self, monkeypatch) -> None:
        import dataclasses

        import rag_qdrant as rq

        cfg = dataclasses.replace(rq._CONFIG, api_key="")
        monkeypatch.setattr(rq, "_CONFIG", cfg)
        monkeypatch.setattr(rq, "_cached_client", None)
        monkeypatch.setattr(rq, "_cached_key", None)
        with mock.patch.object(rq, "QdrantClient") as m_cls:
            rq._client()
        m_cls.assert_called_once_with(url=cfg.url)


# ------------------------------------------------------------
# 7. ingest CLI --progress
# ------------------------------------------------------------
class TestIngestProgressCli:
    def test_progress_flag_passes_callback(self, monkeypatch) -> None:
        import rag_qdrant as rq

        monkeypatch.setattr(sys, "argv", ["rag_qdrant.py", "--ingest", "notes", "--progress"])
        with mock.patch("logging.basicConfig"):
            with mock.patch.object(rq, "ingest_folder") as m:
                rq.main()
        assert m.call_count == 1
        assert callable(m.call_args.kwargs.get("on_progress"))

    def test_without_flag_no_callback(self, monkeypatch) -> None:
        import rag_qdrant as rq

        monkeypatch.setattr(sys, "argv", ["rag_qdrant.py", "--ingest", "notes"])
        with mock.patch("logging.basicConfig"):
            with mock.patch.object(rq, "ingest_folder") as m:
                rq.main()
        assert m.call_args.kwargs.get("on_progress") is None
