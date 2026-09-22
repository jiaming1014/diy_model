"""P20 測試：limit 護欄、搜尋硬擋、查詢截斷收斂。全部 mock，不碰真網路。」"""

import sys
from pathlib import Path

from unittest import mock

import pytest

import chat_core
import rag_qdrant


# ------------------------------------------------------------
# 1. search_local limit 護欄
# ------------------------------------------------------------
class TestSearchLocalLimit:
    def test_zero_returns_empty_without_io(self) -> None:
        """limit=0 直接回空，連 Qdrant 都不碰。」"""
        with mock.patch.object(rag_qdrant, "_client") as m_client:
            assert rag_qdrant.search_local("問題", limit=0) == []
        m_client.assert_not_called()

    def test_negative_returns_empty(self) -> None:
        """limit<0 同樣回空。」"""
        with mock.patch.object(rag_qdrant, "_client") as m_client:
            assert rag_qdrant.search_local("問題", limit=-1) == []
        m_client.assert_not_called()


# ------------------------------------------------------------
# 2. 搜尋硬擋
# ------------------------------------------------------------
class TestSearchHardBlock:
    def test_blocked_when_disabled(self) -> None:
        """開關關閉時幻覺呼叫也擋下。」"""
        out = chat_core._run_tool(
            "search_web", {"query": "新聞"}, "新聞",
            state=chat_core.ChatState(), allow_web_search=False,
        )
        assert "已關閉" in out

    def test_allowed_by_default(self) -> None:
        """預設照常搜尋（相容舊呼叫）。」"""
        with mock.patch.object(chat_core, "_search_web", return_value=[]):
            out = chat_core._run_tool(
                "search_web", {"query": "新聞"}, "新聞", state=chat_core.ChatState()
            )
        assert "搜尋無結果" in out


# ------------------------------------------------------------
# 3. 查詢截斷收斂
# ------------------------------------------------------------
class TestRagQueryTruncation:
    def test_truncation_uses_config(self) -> None:
        """截斷字數吃 _CONFIG，不再寫死 500。」"""
        import dataclasses

        cfg = dataclasses.replace(rag_qdrant._CONFIG, query_max_chars=10)
        seen: dict = {}

        def fake_embed(q: str):
            seen["q"] = q
            return None  # 回 None → search_local 提早回空

        with mock.patch.object(rag_qdrant, "_sync_config"):
            with mock.patch.object(rag_qdrant, "_CONFIG", cfg):
                with mock.patch.object(rag_qdrant, "_embed_query_vec", side_effect=fake_embed):
                    assert rag_qdrant.search_local("A" * 100, limit=3) == []
        assert seen["q"] == "A" * 10


# ------------------------------------------------------------
# 4. refresh 同步：P19/P20 新鍵全覆蓋
# ------------------------------------------------------------
class TestRefreshSyncNewKeys:
    def test_new_keys_propagate(self, monkeypatch) -> None:
        """新設定改 env → refresh → 各模組快照同步，最後還原。」"""
        import config as c

        monkeypatch.setenv("WORKSPACE_MAX_FILE_CHARS", "12345")
        monkeypatch.setenv("WORKSPACE_PATH_MAX_CHARS", "111")
        monkeypatch.setenv("YOUTUBE_QUERY_MAX_CHARS", "77")
        monkeypatch.setenv("RAG_QUERY_MAX_CHARS", "222")
        try:
            c.refresh()
            assert c.WORKSPACE_MAX_FILE_CHARS == 12345
            assert chat_core.WORKSPACE_MAX_FILE_CHARS == 12345
            assert chat_core.WORKSPACE_PATH_MAX_CHARS == 111
            assert chat_core.YOUTUBE_QUERY_MAX_CHARS == 77
            assert c.RAG_QUERY_MAX_CHARS == 222
            assert rag_qdrant._CONFIG.query_max_chars == 222
        finally:
            c.refresh()


# ------------------------------------------------------------
# 5. P21：重排截斷同步＋/ingest 指令
# ------------------------------------------------------------
class TestRerankTruncSync:
    def test_new_keys_propagate(self, monkeypatch) -> None:
        """改 env → refresh → reranker 快照同步，最後還原。」"""
        import config as c

        import reranker as r

        monkeypatch.setenv("RERANK_QUERY_MAX_CHARS", "123")
        monkeypatch.setenv("RERANK_DOC_MAX_CHARS", "456")
        try:
            c.refresh()
            assert r.RERANK_QUERY_MAX_CHARS == 123
            assert r.RERANK_DOC_MAX_CHARS == 456
        finally:
            monkeypatch.undo()  # 先還原 env 再 refresh，否則快照停在測試值污染後續測試
            c.refresh()

    def test_defaults(self) -> None:
        """預設 500／2000（召回文本僅數百字，DOC 截斷形同虛設，不必砍）。」"""
        import reranker as r

        assert r.RERANK_QUERY_MAX_CHARS == 500
        assert r.RERANK_DOC_MAX_CHARS == 2000


class TestIngestCommand:
    def test_parse_default_workspace(self, tmp_path: Path) -> None:
        """無參數用工作區根。」"""
        import chat_cli

        with mock.patch.object(chat_cli._core, "_workspace_root", return_value=tmp_path):
            assert chat_cli._ingest_target("/ingest") == str(tmp_path)

    def test_parse_explicit_folder(self) -> None:
        """有參數用參數。」"""
        import chat_cli

        assert chat_cli._ingest_target("/ingest notes") == "notes"

    def test_non_command(self) -> None:
        """一般訊息與近似指令不觸發。」"""
        import chat_cli

        assert chat_cli._ingest_target("你好") is None
        assert chat_cli._ingest_target("/ingestx") is None

    def test_run_success(self, tmp_path: Path, capsys) -> None:
        """匯入成功印完成訊息。」"""
        import chat_cli

        with mock.patch.object(chat_cli._core, "_workspace_root", return_value=tmp_path):
            with mock.patch("rag_qdrant.ingest_folder", return_value=7) as m:
                chat_cli._run_ingest_command("/ingest")
        m.assert_called_once()
        out = capsys.readouterr().out
        assert "開始匯入" in out
        assert "匯入完成，共 7 點" in out

    def test_run_failure_keeps_chatting(self, tmp_path: Path, capsys) -> None:
        """匯入失敗印錯不拋，留在對話裡。」"""
        import chat_cli

        with mock.patch.object(chat_cli._core, "_workspace_root", return_value=tmp_path):
            with mock.patch("rag_qdrant.ingest_folder", side_effect=RuntimeError("boom")):
                chat_cli._run_ingest_command("/ingest")
        assert "匯入失敗" in capsys.readouterr().out


# ------------------------------------------------------------
# 6. P21：匯入快取原子寫入
# ------------------------------------------------------------
class TestIngestCacheAtomic:
    def test_no_tmp_leftover(self, tmp_path: Path) -> None:
        """存檔不留 tmp 半檔，讀回一致。」"""
        cache = {"a.md": {"mtime": 1.0, "size": 2, "ok": True}}
        rag_qdrant._save_ingest_cache(tmp_path, cache)
        assert (tmp_path / rag_qdrant._INGEST_CACHE_NAME).is_file()
        assert not (tmp_path / (rag_qdrant._INGEST_CACHE_NAME + ".tmp")).exists()
        assert rag_qdrant._load_ingest_cache(tmp_path) == cache


# ------------------------------------------------------------
# 7. P21：匯入缺目錄不再謊報成功
# ------------------------------------------------------------
class TestIngestMissingDir:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        """不存在的目錄拋 FileNotFoundError，不回 0。」"""
        with pytest.raises(FileNotFoundError):
            rag_qdrant.ingest_folder(str(tmp_path / "不存在"))

    def test_chat_command_reports_failure(self, capsys) -> None:
        """對話內打錯路徑印匯入失敗，不中斷。」"""
        import chat_cli

        chat_cli._run_ingest_command("/ingest /definitely/not/here")
        assert "匯入失敗" in capsys.readouterr().out


# ------------------------------------------------------------
# 8. P21：意圖命中不預搜＋關鍵字衝突給完整清單
# ------------------------------------------------------------
class TestIntentPreSearch:
    def test_music_skips_pre_search(self) -> None:
        """播歌請求不再多打一次無用搜尋。」"""
        st = chat_core.ChatState()
        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
            with mock.patch.object(chat_core, "_search_web") as m_search:
                with mock.patch.object(chat_core, "_stream_chat", return_value=iter([])):
                    list(chat_core.chat_w("sys", "幫我用 YouTube 放歌來聽", search_g=True, model="m", state=st))
        m_search.assert_not_called()

    def test_workspace_skips_pre_search(self) -> None:
        """寫檔請求不預搜，避免污染來源。」"""
        st = chat_core.ChatState()
        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
            with mock.patch.object(chat_core, "_search_web") as m_search:
                with mock.patch.object(chat_core, "_stream_chat", return_value=iter([])):
                    list(chat_core.chat_w("sys", "幫我在工作區寫個讀書計畫", search_g=True, model="m", state=st))
        m_search.assert_not_called()

    def test_general_still_pre_searches(self) -> None:
        """一般問答不受影響，本地不足仍預搜。」"""
        st = chat_core.ChatState()
        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
            with mock.patch.object(chat_core, "_search_web", return_value=[]) as m_search:
                with mock.patch.object(chat_core, "_stream_chat", return_value=iter([])):
                    list(chat_core.chat_w("sys", "冷知識問題", search_g=True, model="m", state=st))
        m_search.assert_called_once()

    def test_conflicting_keywords_yield_all_tools(self) -> None:
        """同時含 YouTube 與工作區詞 → 回 None，工具全給不誤刪。」"""
        q = "幫我在工作區寫個 YouTube 影片腳本"
        assert chat_core._detect_intent(q) is None
        names = {t["function"]["name"] for t in chat_core._tools()}  # type: ignore[index]
        assert "workspace_write_file" in names
        assert "play_youtube_music" in names


# ------------------------------------------------------------
# 9. P22：意圖關鍵字漏判修正＋過寬詞收緊
# ------------------------------------------------------------
class TestIntentKeywordTuning:
    def test_music_verb_noun_caught(self) -> None:
        """口語「播周杰倫的歌」補抓為 music。」"""
        assert chat_core._detect_intent("幫我播周杰倫的歌") == "music"
        assert chat_core._detect_intent("放點輕音樂") == "music"

    def test_workspace_file_phrase_caught(self) -> None:
        """「寫成檔案」「轉成檔案」補抓為 workspace。」"""
        assert chat_core._detect_intent("把這段整理寫成檔案") == "workspace"
        assert chat_core._detect_intent("幫我轉成檔案") == "workspace"

    def test_overbroad_words_removed(self) -> None:
        """「寫一封信」「存到哪」不再誤判工作區。」"""
        assert chat_core._detect_intent("幫我寫一封信") is None
        assert chat_core._detect_intent("這個要存到哪個資料夾") is None

    def test_non_music_not_caught(self) -> None:
        """無音樂名詞的動詞句不誤判。」"""
        assert chat_core._detect_intent("幫我播放新聞") is None
        assert chat_core._detect_intent("放鬆一下") is None


# ------------------------------------------------------------
# 10. P22：重排探測尊重開關＋健康檢查不誤判
# ------------------------------------------------------------
class TestRerankProbeSwitch:
    def test_probe_disabled_returns_false(self, monkeypatch) -> None:
        """RERANK_BACKEND=none 時探測回雙 False，不誤報可用。」"""
        import reranker as r

        monkeypatch.setattr(r, "RERANK_BACKEND", "none")
        assert r.probe_availability() == {"crossencoder": False, "llm": False}

    def test_probe_llm_only_skips_crossencoder(self, monkeypatch) -> None:
        """指定 llm 後端時不探測 CrossEncoder（不誤報）。」"""
        import reranker as r

        monkeypatch.setattr(r, "RERANK_BACKEND", "llm")
        with mock.patch.object(r, "_list_ollama_names", return_value=None):
            out = r.probe_availability()
        assert out["crossencoder"] is False

    def test_health_disabled_prints_and_passes(self, monkeypatch, capsys) -> None:
        """停用重排時 --health 印已停用且回 0，不誤判失敗。」"""
        import types

        import chat_cli
        import reranker as r

        monkeypatch.setattr(r, "RERANK_BACKEND", "none")
        coll = types.SimpleNamespace(collections=[types.SimpleNamespace(name="notes")])
        with mock.patch.object(rag_qdrant, "_client") as mc:
            mc.return_value.get_collections.return_value = coll
            with mock.patch.object(rag_qdrant, "probe_embed", return_value=True):
                with mock.patch("ollama_shared.probe_model", return_value=True):
                    rc = chat_cli._run_health_check("m")
        assert rc == 0
        assert "已停用" in capsys.readouterr().out


# ------------------------------------------------------------
# 11. 窄語境動作限定：遊戲／聽說只認動作，不認單純提及
# ------------------------------------------------------------
class TestWorkspaceNarrow:
    def test_game_bare_mention_not_workspace(self) -> None:
        """「遊戲工作區在哪」是打聽位置，走一般問答。」"""
        assert chat_core._detect_intent("遊戲工作區在哪") is None

    def test_game_action_still_workspace(self) -> None:
        """窄語境＋動作（新增檔案）照走寫檔。」"""
        assert chat_core._detect_intent("在遊戲mydocs新增檔案") == "workspace"

    def test_hearsay_bare_not_workspace(self) -> None:
        """「聽說有工作區這東西」是打聽，走一般問答。」"""
        assert chat_core._detect_intent("聽說有工作區這東西") is None

    def test_hearsay_action_still_workspace(self) -> None:
        """聽說＋動作照走寫檔。」"""
        assert chat_core._detect_intent("聽說工作區可以新增檔案") == "workspace"
