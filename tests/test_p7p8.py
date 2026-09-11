"""P7+P8 回歸測試：文字上限／工具去重／維度早失敗／首字延遲／進度回調。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

import chat_core


class TestTextLimit:
    def test_txt_truncated(self, tmp_path: Path) -> None:
        import dataclasses

        import rag_qdrant as rq

        fp = tmp_path / "big.md"
        fp.write_text("字" * 5000, encoding="utf-8")
        cfg = dataclasses.replace(rq._CONFIG, text_max_chars=1000)
        with mock.patch.object(rq, "_CONFIG", cfg):
            out = rq._read_file_text(fp)
        assert len(out) < 5000
        assert "已截斷" in out


class TestToolDedup:
    def test_same_call_runs_once(self) -> None:
        """同一輪內同名同參重複出現，只執行一次（P15 串流工具輪版本）。"""
        calls: list = []

        def fake_chat(**kw):
            calls.append(kw)
            if len(calls) == 1:
                tc = {"function": {"name": "search_web", "arguments": '{"query": "同"}'}}
                return iter([{"message": {"content": "", "tool_calls": [tc, dict(tc)]}}])
            return iter([{"message": {"content": "done"}}])

        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[{"text": "x", "score": "0.9"}]):
            with mock.patch.object(chat_core, "_run_tool", return_value="結果") as m_tool:
                with mock.patch.object(chat_core._ollama, "chat", side_effect=fake_chat):
                    out = list(chat_core.chat_w("sys", "問題", search_g=True, model="m", state=chat_core.ChatState()))
        assert m_tool.call_count == 1
        assert "done" in "".join(out)

    def test_key_stable(self) -> None:
        assert chat_core._tool_call_key("a", {"x": 1}) == chat_core._tool_call_key("a", {"x": 1})
        assert chat_core._tool_call_key("a", {"x": 1}) != chat_core._tool_call_key("a", {"x": 2})


class TestDimFailFast:
    def test_mismatch_raises(self) -> None:
        import rag_qdrant as rq

        fake_info = mock.Mock()
        fake_info.config.params.vectors.size = 100
        fake_client = mock.Mock()
        fake_client.get_collection.return_value = fake_info
        with mock.patch.object(rq, "_client", return_value=fake_client):
            try:
                rq.ensure_collection(999)
            except RuntimeError as e:
                assert "維度不符" in str(e)
                return
            raise AssertionError("應拋維度不符")


class TestFirstToken:
    def test_logs_first_token(self, caplog) -> None:
        import logging

        fake_stream = iter([{"message": {"content": "嗨"}}])
        with mock.patch.object(chat_core._ollama, "chat", return_value=fake_stream):
            with caplog.at_level(logging.DEBUG, logger="chat_core"):
                out = list(chat_core._stream_reply([{"role": "user", "content": "hi"}], model="m"))
        assert out == ["嗨"]
        assert any("首字延遲" in r.message for r in caplog.records)


class TestIngestProgress:
    def test_callback_called(self, tmp_path: Path) -> None:
        import rag_qdrant as rq

        (tmp_path / "a.md").write_text("hello", encoding="utf-8")
        (tmp_path / "b.md").write_text("world", encoding="utf-8")
        calls: list[tuple] = []
        with mock.patch.object(rq, "_client"):
            with mock.patch.object(rq, "_flush_batch", return_value=(1, 0)):
                rq.ingest_folder(str(tmp_path), on_progress=lambda d, t, r: calls.append((d, t, r)))
        assert len(calls) == 2
        assert calls[0][1] == 2
