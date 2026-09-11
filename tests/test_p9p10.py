"""P9+P10 回歸測試：DOCX上限／逐檔ok／搜尋重試／跨模組同步，不碰真網路。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock


class TestDocxLimit:
    def test_truncates_paras(self) -> None:
        import dataclasses

        import rag_qdrant as rq

        fake_doc = mock.Mock()
        fake_doc.paragraphs = [mock.Mock(text=f"p{i}") for i in range(10)]
        fake_doc.tables = []
        with mock.patch.dict("sys.modules", {"docx": mock.Mock(Document=mock.Mock(return_value=fake_doc))}):
            cfg = dataclasses.replace(rq._CONFIG, docx_max_paras=3)
            with mock.patch.object(rq, "_CONFIG", cfg):
                out = rq._read_docx(Path("x.docx"))
        assert "已截斷" in out
        assert out.count("\n") <= 5


class TestPerFileOk:
    def test_failed_file_not_marked_ok(self, tmp_path: Path) -> None:
        import rag_qdrant as rq

        (tmp_path / "ok.md").write_text("hello world hello", encoding="utf-8")
        (tmp_path / "bad.md").write_text("bad content here", encoding="utf-8")

        def _fake_flush(client, chunks, metas, ensured):
            if any(m.get("source") == "bad.md" for m in metas):
                return (0, 0)  # 嵌入全失敗
            return (len(chunks), 0)

        with mock.patch.object(rq, "_client"):
            with mock.patch.object(rq, "_flush_batch", side_effect=_fake_flush):
                rq.ingest_folder(str(tmp_path))
        import json

        cache = json.loads((tmp_path / rq._INGEST_CACHE_NAME).read_text(encoding="utf-8"))
        assert cache["ok.md"]["ok"] is True
        assert cache["bad.md"]["ok"] is False


class TestSearchRetry:
    def test_retry_succeeds_second(self) -> None:
        import chat_core

        chat_core._SEARCH_CACHE.clear()
        good = mock.MagicMock()
        good.__enter__.return_value.text.return_value = [{"title": "t", "body": "s", "href": "https://e.com"}]
        with mock.patch.object(chat_core, "DDGS", side_effect=[Exception("boom"), good]) as m:
            with mock.patch.object(chat_core.time, "sleep", return_value=None):
                out = chat_core._search_web("重試測試P9", state=chat_core.ChatState())
        assert len(out) == 1
        assert m.call_count == 2
        chat_core._SEARCH_CACHE.clear()


class TestCrossModuleSync:
    def test_refresh_propagates(self, monkeypatch) -> None:
        import config as c

        import chat_core
        import rag_qdrant as rq
        import reranker as r

        monkeypatch.setenv("USER_MAX_CHARS", "4321")
        monkeypatch.setenv("RERANK_BATCH", "4")
        monkeypatch.setenv("QUERY_VEC_CACHE_MAX", "11")
        try:
            c.refresh()
            assert c.USER_MAX_CHARS == 4321
            assert chat_core.USER_MAX_CHARS == 4321
            assert r.RERANK_BATCH == 4
            assert rq._QUERY_VEC_CACHE.maxsize == 11
        finally:
            c.refresh()
