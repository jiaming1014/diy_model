"""diy_model 整合測試：mock 外部依賴（Ollama/Qdrant/網路），驗證 chat_w 主流程與快取。

【測什麼？】
1. chat_w 日期捷徑：短句含日期關鍵字 → 直接回今天，不碰模型
2. chat_w 傳統降級路徑：search_g=False → 走 _stream_reply 串流
3. chat_w 工具路徑：模型回 tool_calls → 執行工具 → 再串流
4. _search_web 快取：同一 query 二次查詢不重打網路
5. rerank 降級：CrossEncoder 缺套件時回原順序

所有測試都 mock 掉外部 I/O，不碰 Ollama/Qdrant/DuckDuckGo。
"""

import sys
from pathlib import Path

import pytest
from unittest import mock

import chat_core
import rag_qdrant
import reranker


# ------------------------------------------------------------
# 1. chat_w 日期捷徑
# ------------------------------------------------------------
class TestChatDateShortcut:
    def test_date_query_skips_model(self) -> None:
        """「今天幾號」走日期捷徑，完全不呼叫模型。"""
        with mock.patch.object(chat_core, "_ollama") as m:
            replies = list(chat_core.chat_w("sys", "今天幾號", search_g=True, model="gemma4:31b-cloud"))
        assert any("今天是" in r or "星期" in r for r in replies)
        m.chat.assert_not_called()


# ------------------------------------------------------------
# 2. chat_w 傳統降級路徑（search_g=False）
# ------------------------------------------------------------
class TestChatLegacy:
    def test_legacy_streams_reply(self) -> None:
        """search_g=False → 走 _handle_legacy_flow，串流回覆。"""
        fake = iter(["你", "好", "！"])
        with mock.patch.object(chat_core, "_stream_reply", return_value=fake) as m:
            with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
                replies = list(chat_core.chat_w("sys", "你好", search_g=False, model="gemma4:31b-cloud"))
        m.assert_called_once()
        assert "".join(replies) == "你好！"


# ------------------------------------------------------------
# 3. chat_w 工具路徑（模型呼叫 search_web 工具）
# ------------------------------------------------------------
class TestChatToolFlow:
    def test_tool_loop_executes_tool(self) -> None:
        """模型回 tool_calls → 執行工具 → 再串流作答（P15 串流工具輪）。"""
        import chat_core as c

        calls: list = []

        def fake_chat(**kw):
            """假模型：首輪回工具呼叫，次輪回答案片。」"""
            calls.append(kw)
            if len(calls) == 1:
                tc = {"function": {"name": "search_web", "arguments": '{"query": "測試"}'}}
                return iter([{"message": {"content": "", "tool_calls": [tc]}}])
            return iter([{"message": {"content": "結果"}}, {"message": {"content": "OK"}}])

        st = c.ChatState()
        with mock.patch.object(c, "_retrieve_rag", return_value=[{"text": "x", "score": "0.9"}]):
            with mock.patch.object(c, "_search_web", return_value=[]) as m_search:
                with mock.patch.object(c._ollama, "chat", side_effect=fake_chat):
                    replies = list(c.chat_w("sys", "問題", search_g=True, model="m", state=st))
        assert "".join(replies) == "結果OK"
        # 本地筆記夠力 → 不預補搜；只發生工具自己那一次搜尋
        assert m_search.call_count == 1


# ------------------------------------------------------------
# 4. _search_web 快取
# ------------------------------------------------------------
class TestSearchCache:
    def test_cached_query_skips_network(self) -> None:
        """同一 query 二次查詢，第二次不重打網路（快取命中）。"""
        results = [{"title": "t", "snippet": "s", "url": "u"}]
        # 第一次真實呼叫 ddgs（mock），第二次應命中快取不再呼叫
        with mock.patch.object(chat_core, "DDGS") as m_ddgs:
            inst = m_ddgs.return_value.__enter__.return_value
            inst.text.return_value = [{"title": "t", "body": "s", "href": "u"}]
            first = chat_core._search_web("快取測試", state=chat_core.ChatState())
            second = chat_core._search_web("快取測試", state=chat_core.ChatState())
        assert first == results
        assert second == results
        # DDGS 應只被實例化一次（快取命中）
        assert m_ddgs.call_count == 1


# ------------------------------------------------------------
# 5. rerank 降級：缺 CrossEncoder 套件 → 回原順序
# ------------------------------------------------------------
class TestRerankFallback:
    def test_no_cross_encoder_returns_original_order(self) -> None:
        """CrossEncoder 不可用時，rerank 回原順序（不拋錯）。"""
        docs = [{"text": "a", "source": "s1"}, {"text": "b", "source": "s2"}]
        # 強制 crossencoder 後援失敗 → 走 LLM 備援（也 mock 成失敗）→ 回原順序
        with mock.patch.object(reranker, "_rerank_cross", return_value=None):
            with mock.patch.object(reranker, "_rerank_llm", return_value=None):
                out = reranker.rerank("q", docs, top_k=3)
        assert out == docs


# ------------------------------------------------------------
# 6. _resolve_model helper
# ------------------------------------------------------------
class TestResolveModel:
    def test_default_uses_ollama_model(self) -> None:
        """未指定模型走 OLLAMA_MODEL 預設。」"""
        assert chat_core._resolve_model(None) == chat_core.OLLAMA_MODEL

    def test_explicit_overrides(self) -> None:
        """指定模型優先於預設。」"""
        assert chat_core._resolve_model("custom") == "custom"
