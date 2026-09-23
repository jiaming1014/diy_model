"""P11+P12 回歸測試：雙預算／雙指標／串流重試，不碰真網路。"""

from pathlib import Path

from unittest import mock

import chat_core


class TestDualBudget:
    """雙預算：token 尺有效，字數＋token 任一超標即裁。"""
    def test_tokens_fallback_len(self) -> None:
        """token 計算有正值：缺 tiktoken 退化字數也不回 0。」"""
        n = chat_core._content_tokens("hello")
        assert n > 0

    def test_trim_uses_both(self) -> None:
        """字數＋token 雙預算任一超標即裁，不無限成長。」"""
        buf = [
            {"role": "user", "content": "字" * 5000},
            {"role": "assistant", "content": "答" * 5000},
        ]
        chat_core._trim_hist(buf)
        total = sum(len(m.get("content", "")) for m in buf)
        assert total <= chat_core.HIST_MAX_CHARS or len(buf) == 0


class TestEvalDual:
    """eval 雙指標：引用標註偵測與模型層「有依據需引用」判定。"""
    def test_citation_detect(self) -> None:
        """引用標註偵測：[來源i]／[筆記i] 算有引用，其餘不算。」"""
        from eval import _has_citation

        assert _has_citation("根據 [來源1] 回答") is True
        assert _has_citation("根據 [筆記2] 回答") is True
        assert _has_citation("沒有引用") is False

    def test_model_requires_cite_when_evidence(self) -> None:
        """有外部依據（RAG）卻無引用 → 模型層評分不過。」"""
        from eval import _eval_model

        with mock.patch.object(chat_core, "chat_w", return_value=iter(["帶傘"])):
            with mock.patch.object(chat_core, "get_last_rag", return_value=[{"text": "x"}]):
                with mock.patch.object(chat_core, "get_last_sources", return_value=[]):
                    r = _eval_model({"q": "天氣", "keywords": ["帶傘"]}, "m")
        assert r["pass"] is False  # 有 RAG 卻無引用
        assert r["cited"] is False


class TestStreamRetry:
    """串流建立失敗重試：第一次炸、第二次成功即回內容。"""
    def test_retry_succeeds(self) -> None:
        """串流建立失敗重試一次：第一次炸、第二次串流成功即回內容。」"""
        good = iter([{"message": {"content": "hi"}}])
        with mock.patch.object(chat_core._ollama, "chat", side_effect=[Exception("boom"), good]):
            with mock.patch.object(chat_core.time, "sleep", return_value=None):
                out = list(chat_core._stream_reply([{"role": "user", "content": "hi"}], model="m"))
        assert out == ["hi"]
