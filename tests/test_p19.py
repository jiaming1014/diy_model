"""P19 測試：意圖分流、工具子集、工作區沙盒、--no-search 拆分。

全部 mock 外部 I/O（Ollama/Qdrant/瀏覽器），不碰真網路真檔案；
工作區路徑測試把根目錄 mock 到 pytest 的 tmp_path。
"""

import sys
from pathlib import Path

from unittest import mock

import chat_core


# ------------------------------------------------------------
# 1. 意圖偵測
# ------------------------------------------------------------
class TestDetectIntent:
    def test_music(self) -> None:
        """播歌請求判為 music。」"""
        assert chat_core._detect_intent("幫我在 YouTube 播周杰倫的歌") == "music"
        assert chat_core._detect_intent("放首輕音樂來聽") == "music"

    def test_workspace(self) -> None:
        """寫檔／建資料夾判為 workspace。」"""
        assert chat_core._detect_intent("幫我在工作區寫個讀書計畫") == "workspace"
        assert chat_core._detect_intent("建立資料夾 報告/2026") == "workspace"

    def test_general_is_none(self) -> None:
        """一般問答回 None，上層走完整工具清單。」"""
        assert chat_core._detect_intent("台北今天天氣如何？") is None
        assert chat_core._detect_intent("這個專案用什麼嵌入模型？") is None


# ------------------------------------------------------------
# 2. 工具子集
# ------------------------------------------------------------
class TestToolSubsets:
    def test_default_has_all_five(self) -> None:
        """無參數呼叫維持 5 個工具（含搜尋）。」"""
        names = {t["function"]["name"] for t in chat_core._tools()}  # type: ignore[index]
        assert names == {"get_today", "search_web", "play_youtube_music", "workspace_write_file", "workspace_make_dir"}

    def test_music_subset(self) -> None:
        """音樂意圖只給播歌＋搜尋＋日期（時間指涉用）。」"""
        names = {t["function"]["name"] for t in chat_core._tools(intent="music")}  # type: ignore[index]
        assert names == {"play_youtube_music", "search_web", "get_today"}

    def test_workspace_subset(self) -> None:
        """工作區意圖不給播歌工具。」"""
        names = {t["function"]["name"] for t in chat_core._tools(intent="workspace")}  # type: ignore[index]
        assert "play_youtube_music" not in names
        assert {"workspace_write_file", "workspace_make_dir", "search_web"} <= names

    def test_no_search_drops_search_web(self) -> None:
        """allow_search=False 剔掉 search_web，其他保留。」"""
        names = {t["function"]["name"] for t in chat_core._tools(allow_search=False)}  # type: ignore[index]
        assert "search_web" not in names
        assert {"get_today", "workspace_write_file"} <= names


# ------------------------------------------------------------
# 3. 工作區沙盒（根目錄 mock 到 tmp_path）
# ------------------------------------------------------------
class TestWorkspaceSandbox:
    def test_traversal_blocked(self, tmp_path: Path) -> None:
        """../ 穿越擋下，不寫檔。」"""
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            out = chat_core._run_tool("workspace_write_file", {"path": "../evil.txt", "content": "x"}, "test")
        assert "穿越" in out or "工作區內" in out
        assert not (tmp_path.parent / "evil.txt").exists()

    def test_absolute_blocked(self, tmp_path: Path) -> None:
        """絕對路徑擋下（用跨平台都算絕對的路徑，Windows C:/ 在 Linux 不算絕對）。」"""
        abs_path = str(tmp_path / "hack")
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            out = chat_core._run_tool("workspace_make_dir", {"path": abs_path}, "test")
        assert "相對路徑" in out

    def test_write_and_mkdir_roundtrip(self, tmp_path: Path) -> None:
        """正常相對路徑可寫可建，落在沙盒內。」"""
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            assert "已在" in chat_core._run_tool("workspace_make_dir", {"path": "報告/2026"}, "test")
            out = chat_core._run_tool(
                "workspace_write_file", {"path": "報告/草稿.md", "content": "hello"}, "test"
            )
        assert "已寫入" in out
        assert (tmp_path / "報告" / "草稿.md").read_text(encoding="utf-8") == "hello"


# ------------------------------------------------------------
# 4. 意圖連動：RAG 跳過＋工具子集進提示
# ------------------------------------------------------------
class TestIntentFlow:
    def test_workspace_skips_rag(self) -> None:
        """寫檔請求不查 RAG（省一次嵌入＋Qdrant）。」"""
        st = chat_core.ChatState()
        with mock.patch.object(chat_core, "_retrieve_rag") as m_rag:
            with mock.patch.object(chat_core, "_tool_flow", return_value=iter(["ok"])):
                out = list(chat_core.chat_w("sys", "幫我在工作區寫個讀書計畫", search_g=True, model="m", state=st))
        m_rag.assert_not_called()
        assert "".join(out) == "ok"

    def test_workspace_limits_tools(self) -> None:
        """寫檔輪只送工作區子集給模型。」"""
        st = chat_core.ChatState()
        seen: dict = {}

        def fake_stream(messages, model, tools=None, calls_out=None):
            seen["tools"] = tools
            return iter([])

        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
            with mock.patch.object(chat_core, "_search_web", return_value=[]):
                with mock.patch.object(chat_core, "_stream_chat", side_effect=fake_stream):
                    list(chat_core.chat_w("sys", "幫我在工作區寫個讀書計畫", search_g=True, model="m", state=st))
        names = {t["function"]["name"] for t in seen["tools"]}  # type: ignore[index]
        assert "play_youtube_music" not in names
        assert {"workspace_write_file", "workspace_make_dir"} <= names


# ------------------------------------------------------------
# 5. --no-search 拆分：只關上網，本機工具保留
# ------------------------------------------------------------
class TestWebSearchSplit:
    def test_no_web_search_skips_presearch(self) -> None:
        """web_search=False 不預補搜，工具清單無 search_web。」"""
        st = chat_core.ChatState()
        seen: dict = {}

        def fake_stream(messages, model, tools=None, calls_out=None):
            seen["tools"] = tools
            return iter([])

        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
            with mock.patch.object(chat_core, "_search_web") as m_search:
                with mock.patch.object(chat_core, "_stream_chat", side_effect=fake_stream):
                    list(chat_core.chat_w("sys", "冷知識問題", search_g=True, web_search=False, model="m", state=st))
        m_search.assert_not_called()
        names = {t["function"]["name"] for t in seen["tools"]}  # type: ignore[index]
        assert "search_web" not in names
        assert "workspace_write_file" in names


# ------------------------------------------------------------
# 6. YouTube 無瀏覽器降級
# ------------------------------------------------------------
class TestYoutubeFallback:
    def test_browser_no_response(self) -> None:
        """webbrowser.open 回 False 給手動連結，不謊報成功。」"""
        with mock.patch("webbrowser.open", return_value=False):
            out = chat_core._run_tool("play_youtube_music", {"query": "測試歌"}, "test")
        assert "手動" in out
        assert "youtube.com" in out

    def test_browser_ok(self) -> None:
        """打得開就報成功。」"""
        with mock.patch("webbrowser.open", return_value=True):
            out = chat_core._run_tool("play_youtube_music", {"query": "測試歌"}, "test")
        assert "已在瀏覽器" in out


# ------------------------------------------------------------
# 7. P20：可執行檔防護＋匯入提示＋YT 上限＋health 工作區
# ------------------------------------------------------------
class TestP20Hardening:
    def test_blocked_ext(self, tmp_path: Path) -> None:
        """exe 擋下且不落地。」"""
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            out = chat_core._run_tool("workspace_write_file", {"path": "run.exe", "content": "x"}, "test")
        assert "可執行" in out
        assert not (tmp_path / "run.exe").exists()

    def test_blocked_ext_case_insensitive(self, tmp_path: Path) -> None:
        """大小寫繞不過（.PS1 照擋）。」"""
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            out = chat_core._run_tool("workspace_write_file", {"path": "evil.PS1", "content": "x"}, "test")
        assert "可執行" in out
        assert not (tmp_path / "evil.PS1").exists()

    def test_write_hint_mentions_ingest(self, tmp_path: Path) -> None:
        """寫檔成功訊息帶匯入提示，銜接 RAG。」"""
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            out = chat_core._run_tool("workspace_write_file", {"path": "a.md", "content": "hi"}, "test")
        assert "已寫入" in out
        assert "rag_qdrant" in out

    def test_youtube_limit_from_config(self) -> None:
        """超長關鍵字按設定截斷，不寫死。」"""
        long_q = "啊" * (chat_core.YOUTUBE_QUERY_MAX_CHARS + 50)
        with mock.patch("webbrowser.open", return_value=True):
            out = chat_core._run_tool("play_youtube_music", {"query": long_q}, "test")
        assert ("啊" * chat_core.YOUTUBE_QUERY_MAX_CHARS) in out
        assert ("啊" * (chat_core.YOUTUBE_QUERY_MAX_CHARS + 1)) not in out

    def test_health_reports_workspace(self, tmp_path: Path, capsys) -> None:
        """--health 報工作區可用。」"""
        import types

        import chat_cli
        import rag_qdrant

        coll = types.SimpleNamespace(collections=[types.SimpleNamespace(name="notes")])
        with mock.patch("reranker.probe_availability", return_value={"crossencoder": True, "llm": True}):
            with mock.patch.object(rag_qdrant, "_client") as m_client:
                with mock.patch.object(chat_cli._core, "_workspace_root", return_value=tmp_path):
                    m_client.return_value.get_collections.return_value = coll
                    # 嵌入／聊天探測不碰真 Ollama，否則無服務的 CI 必紅
                    with mock.patch.object(rag_qdrant, "probe_embed", return_value=True):
                        with mock.patch("ollama_shared.probe_model", return_value=True):
                            rc = chat_cli._run_health_check("m")
        assert rc == 0
        assert "工作區：可用" in capsys.readouterr().out
        assert not (tmp_path / ".health_probe").exists()


# ------------------------------------------------------------
# 5. 工作區路徑邊界：磁碟機前綴與歷史路徑回退
# ------------------------------------------------------------
class TestWorkspacePathEdge:
    def test_drive_letter_blocked(self, tmp_path: Path) -> None:
        """C:foo 磁碟機相對路徑擋下（只在 Windows 有意義）。」"""
        import os

        import pytest

        if os.name != "nt":
            pytest.skip("磁碟機前綴只在 Windows 有意義")
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path):
            out = chat_core._run_tool("workspace_make_dir", {"path": "C:evil"}, "test")
        assert "磁碟機" in out

    def test_hist_path_empty_falls_back(self, monkeypatch) -> None:
        """DIY_HIST_FILE 空字串回預設，不寫怪檔。」"""
        import chat_cli

        monkeypatch.setenv("DIY_HIST_FILE", "")
        assert chat_cli._hist_path() == chat_cli._DEFAULT_HIST

    def test_hist_path_tilde_expands(self, monkeypatch) -> None:
        """~/ 開頭展開為家目錄。」"""
        import chat_cli

        monkeypatch.setenv("DIY_HIST_FILE", "~/diy_hist.json")
        assert str(chat_cli._hist_path()) == str(Path.home() / "diy_hist.json")
