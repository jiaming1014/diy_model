"""P16 回歸測試：搜尋快取鍵／region／滾動摘要／Qdrant api_key／ingest 進度／
text_utils 抽取／rewrite 模型，不碰真網路。"""

import sys
from pathlib import Path

from unittest import mock

import chat_core


# ------------------------------------------------------------
# 1. 搜尋快取鍵：不同 max_results 不共用快取
# ------------------------------------------------------------
class TestSearchCacheKey:
    """搜尋快取鍵：不同 max_results 不共用同一格快取。"""
    def test_different_max_results_not_shared(self) -> None:
        """筆數不同不共用同一格快取，各打一次網路。」"""
        chat_core._SEARCH_CACHE.clear()
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = [{"title": "t", "body": "s", "href": "https://e.com"}]
            chat_core._search_web("cache-key-test", max_results=3, state=chat_core.ChatState())
            chat_core._search_web("cache-key-test", max_results=5, state=chat_core.ChatState())
            assert m_ddgs.call_count == 2
        chat_core._SEARCH_CACHE.clear()

    def test_same_params_hit_cache(self) -> None:
        """同查詢同筆數第二次命中快取，不重打網路。」"""
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
    """搜尋 region 由 config 控制，改真相即時帶進快照。"""
    def test_region_from_config(self, monkeypatch) -> None:
        """region 唯一真相在 config，改真相即時帶進快照。」"""
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
    """LLM 查詢改寫沿用本回合 model，不吃 config 預設。"""
    def test_rewrite_uses_session_model(self) -> None:
        """查詢改寫用本回合模型，不吃 config 預設。」"""
        with mock.patch.object(chat_core, "QUERY_REWRITE_LLM", True):
            with mock.patch.object(chat_core, "_call_chat_with_retry", return_value={"message": {"content": "台北天氣"}}) as m:
                out = chat_core._maybe_llm_rewrite("請問台北天氣？", model="my-model")
        assert out == chat_core._clean_query_for_search("台北天氣")
        assert m.call_args[0][1] == "my-model"


# ------------------------------------------------------------
# 4. text_utils 抽取：同名重匯出、config 即時值
# ------------------------------------------------------------
class TestTextUtilsExtraction:
    """text_utils 抽取：同名重匯出同物件、截斷讀 config 即時值。"""
    def test_reexport_same_object(self) -> None:
        """chat_core 重匯出與 text_utils 同一物件，舊寫法不壞。」"""
        import text_utils

        assert chat_core._is_date_query is text_utils._is_date_query
        assert chat_core._needs_realtime is text_utils._needs_realtime
        assert chat_core._clean_query_for_search is text_utils._clean_query_for_search

    def test_truncate_reads_config_live(self, monkeypatch) -> None:
        """截斷讀 config 即時值，改旋鈕免重啟。」"""
        import config

        monkeypatch.setattr(config, "USER_MAX_CHARS", 5)
        out = chat_core._truncate_user_msg("1234567890")
        assert out.startswith("12345")
        assert out.endswith("…（過長已截斷）")


# ------------------------------------------------------------
# 5. 歷史滾動摘要（opt-in：HIST_SUMMARY_ENABLE）
# ------------------------------------------------------------
class TestRollingSummary:
    """歷史滾動摘要（opt-in）：開啟才壓摘要、失敗保留舊摘要、附於系統訊息。"""
    def _mk_pairs(self, n: int, size: int = 50) -> chat_core.ChatState:
        """造 N 組問答歷史，供裁切觸發摘要。」"""
        st = chat_core.ChatState()
        for i in range(n):
            st.hist.append({"role": "user", "content": f"問題{i}" + "字" * size})
            st.hist.append({"role": "assistant", "content": f"回答{i}" + "字" * size})
        return st

    def test_summary_created_when_enabled(self, monkeypatch) -> None:
        """開啟且丟棄量達標 → 調模型壓摘要並沿用本回合 model。」"""
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_ENABLE", True)
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_MIN_DROPPED", 1)
        st = self._mk_pairs(6)  # 6 組 > 4 組上限 → 觸發裁切
        with mock.patch.object(chat_core, "_call_chat_with_retry", return_value={"message": {"content": "這是摘要"}}) as m:
            chat_core._trim_hist(st, model="m")
        assert st.summary == "這是摘要"
        assert m.call_args[0][1] == "m"
        assert len(st.hist) <= 2 * chat_core.backtrace

    def test_disabled_by_default_no_llm(self) -> None:
        """預設關閉：裁切不打模型，摘要保持空。」"""
        st = self._mk_pairs(6)
        with mock.patch.object(chat_core, "_call_chat_with_retry") as m:
            chat_core._trim_hist(st, model="m")
        m.assert_not_called()
        assert st.summary == ""

    def test_failure_keeps_old_summary(self, monkeypatch) -> None:
        """摘要失敗保留舊摘要，不影響對話。」"""
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_ENABLE", True)
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_MIN_DROPPED", 1)
        st = self._mk_pairs(6)
        st.summary = "舊摘要"
        with mock.patch.object(chat_core, "_call_chat_with_retry", side_effect=RuntimeError("boom")):
            chat_core._trim_hist(st, model="m")
        assert st.summary == "舊摘要"

    def test_small_drop_skipped(self, monkeypatch) -> None:
        """丟棄量未達標不值得摘要，不打模型。」"""
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_ENABLE", True)
        monkeypatch.setattr(chat_core, "HIST_SUMMARY_MIN_DROPPED", 100000)
        st = self._mk_pairs(6)
        with mock.patch.object(chat_core, "_call_chat_with_retry") as m:
            chat_core._trim_hist(st, model="m")
        m.assert_not_called()

    def test_summary_injected_in_system(self) -> None:
        """滾動摘要附在系統訊息後，供長對話保留遠期記憶。」"""
        st = chat_core.ChatState()
        st.summary = "摘要X"
        msgs = chat_core._with_history({"role": "system", "content": "SYS"}, st, "Q")
        assert "摘要X" in msgs[0]["content"]


# ------------------------------------------------------------
# 6. Qdrant api_key 支援
# ------------------------------------------------------------
class TestQdrantApiKey:
    """Qdrant api_key 支援：有 key 建連線時帶入、無 key 省略參數。"""
    def test_api_key_passed(self, monkeypatch) -> None:
        """有 api_key 建連線時帶入（Qdrant Cloud 用）。」"""
        import dataclasses

        import qdrant_client

        import rag_qdrant as rq

        cfg = dataclasses.replace(rq._CONFIG, api_key="secret")
        monkeypatch.setattr(rq, "_CONFIG", cfg)
        monkeypatch.setattr(rq, "_cached_client", None)
        monkeypatch.setattr(rq, "_cached_key", None)
        with mock.patch.object(qdrant_client, "QdrantClient") as m_cls:
            rq._client()
        m_cls.assert_called_once_with(url=cfg.url, api_key="secret")

    def test_no_key_omits_param(self, monkeypatch) -> None:
        """無 api_key 不帶參數，本地 Docker 直連（_client 本體走函式內 import）。」"""
        import dataclasses

        import qdrant_client

        import rag_qdrant as rq

        cfg = dataclasses.replace(rq._CONFIG, api_key="")
        monkeypatch.setattr(rq, "_CONFIG", cfg)
        monkeypatch.setattr(rq, "_cached_client", None)
        monkeypatch.setattr(rq, "_cached_key", None)
        with mock.patch.object(qdrant_client, "QdrantClient") as m_cls:
            rq._client()
        m_cls.assert_called_once_with(url=cfg.url)


# ------------------------------------------------------------
# 7. ingest CLI --progress
# ------------------------------------------------------------
class TestIngestProgressCli:
    """ingest CLI --progress：有旗標傳進度回調、無旗標安靜匯入。"""
    def test_progress_flag_passes_callback(self, monkeypatch) -> None:
        """--progress 逐檔顯示，傳 callable 進度回調。」"""
        import rag_qdrant as rq

        monkeypatch.setattr(sys, "argv", ["rag_qdrant.py", "--ingest", "notes", "--progress"])
        with mock.patch("logging.basicConfig"):
            with mock.patch.object(rq, "ingest_folder") as m:
                rq.main()
        assert m.call_count == 1
        assert callable(m.call_args.kwargs.get("on_progress"))

    def test_without_flag_no_callback(self, monkeypatch) -> None:
        """無旗標不傳回調，安靜匯入。」"""
        import rag_qdrant as rq

        monkeypatch.setattr(sys, "argv", ["rag_qdrant.py", "--ingest", "notes"])
        with mock.patch("logging.basicConfig"):
            with mock.patch.object(rq, "ingest_folder") as m:
                rq.main()
        assert m.call_args.kwargs.get("on_progress") is None
