"""P15 回歸測試：單次生成串流工具流程／預補搜／legacy 帶筆記／client 生命週期／
有界讀取／整檔 ok 判定／--health，不碰真網路。"""

import sys
from pathlib import Path

from unittest import mock

import chat_core
import rag_qdrant as rq


# ------------------------------------------------------------
# 1. 優化主軸：模型沒叫工具的那一輪，串流內容就是答案（不再二次生成）
# ------------------------------------------------------------
class TestSingleCallAnswer:
    def test_no_tool_answer_single_model_call(self) -> None:
        """沒叫工具的那一輪串流即答案，只生成一次不重生。」"""
        st = chat_core.ChatState()
        calls: list = []

        def fake_chat(**kw):
            """假模型：直接回兩片答案，不帶工具呼叫。」"""
            calls.append(kw)
            return iter([{"message": {"content": "答"}}, {"message": {"content": "案"}}])

        with mock.patch.object(chat_core, "_retrieve_rag", return_value=[]):
            with mock.patch.object(chat_core, "_search_web", return_value=[]):
                with mock.patch.object(chat_core._ollama, "chat", side_effect=fake_chat):
                    replies = list(chat_core.chat_w("sys", "隨意問題", search_g=True, model="m", state=st))
        assert "".join(replies) == "答案"
        assert len(calls) == 1  # 舊版偵測輪丟棄後會再生一次，P15 起只生成一次


# ------------------------------------------------------------
# 2. 工具輪：一輪工具＋一輪作答＝兩次生成，且作答輪看得到工具結果
# ------------------------------------------------------------
class TestToolRoundStream:
    def test_tool_then_answer(self) -> None:
        """一輪工具＋一輪作答＝兩次生成，作答輪看得到工具結果。」"""
        st = chat_core.ChatState()
        calls: list = []

        def fake_chat(**kw):
            """假模型：首輪回工具呼叫，次輪回答案。」"""
            calls.append(kw)
            if len(calls) == 1:
                tc = {"function": {"name": "search_web", "arguments": '{"query": "q"}'}}
                return iter([{"message": {"content": "", "tool_calls": [tc]}}])
            return iter([{"message": {"content": "完成"}}])

        hits = [{"source": "s", "text": "本地", "score": "0.9"}]
        with mock.patch.object(chat_core, "_retrieve_rag", return_value=hits):
            with mock.patch.object(chat_core, "_search_web", return_value=[]) as m_search:
                with mock.patch.object(chat_core._ollama, "chat", side_effect=fake_chat):
                    replies = list(chat_core.chat_w("sys", "問題", search_g=True, model="m", state=st))
        assert "".join(replies) == "完成"
        assert len(calls) == 2
        assert calls[0].get("tools")  # 工具輪帶工具
        assert any(m.get("role") == "tool" for m in calls[1]["messages"])  # 作答輪帶工具結果
        assert m_search.call_count == 1


# ------------------------------------------------------------
# 3. 預補搜：模型開跑前先搜一次（本地不夠先鋪事實、realtime 保證新鮮）
# ------------------------------------------------------------
class TestPreSearchFlow:
    def test_presearch_before_model_call(self) -> None:
        """預補搜在模型開跑前：順序 search→model。」"""
        st = chat_core.ChatState()
        order: list[str] = []

        def fake_search(query, max_results=None, state=None):
            """假搜尋：記順序回空，不碰網路。」"""
            order.append("search")
            return []

        def fake_chat(**kw):
            """假模型：記順序回單片答案。」"""
            order.append("model")
            return iter([{"message": {"content": "回"}}])

        with mock.patch.object(chat_core, "_search_web", side_effect=fake_search):
            with mock.patch.object(chat_core._ollama, "chat", side_effect=fake_chat):
                out = list(chat_core.chat_w("sys", "台北今天天氣如何？", search_g=True, model="m", state=st))
        assert "".join(out) == "回"
        assert order == ["search", "model"]


# ------------------------------------------------------------
# 4. legacy（--no-search）要真的吃本地筆記，不再白檢索
# ------------------------------------------------------------
class TestLegacyRagBlock:
    def test_no_search_includes_rag_block(self) -> None:
        """--no-search 照吃本地筆記：提示詞含筆記內容。」"""
        st = chat_core.ChatState()
        hits = [{"source": "a.md", "text": "筆記內容", "score": "0.9"}]
        captured: list = []

        def fake_stream(messages, model=None):
            """假串流：收提示詞供斷言，回單片答案。」"""
            captured.append(messages)
            return iter(["好"])

        with mock.patch.object(chat_core, "_retrieve_rag", return_value=hits):
            with mock.patch.object(chat_core, "_stream_reply", side_effect=fake_stream):
                replies = list(chat_core.chat_w("sys", "筆記的問題", search_g=False, model="m", state=st))
        assert "".join(replies) == "好"
        user_msgs = [m for m in captured[0] if m.get("role") == "user"]
        assert "筆記內容" in user_msgs[-1]["content"]


# ------------------------------------------------------------
# 5. 配置同步：換 Ollama client 前先關舊的（與 rag/reranker 同款）
# ------------------------------------------------------------
class TestClientLifecycle:
    def test_sync_config_closes_replaced_client(self, monkeypatch) -> None:
        """換逾時重建 client：舊的關閉、新的上位，共用快取快照還原。」"""
        import ollama_shared

        snapshot = dict(ollama_shared._clients)
        try:
            old_client = mock.Mock()
            monkeypatch.setattr(chat_core, "_ollama", old_client)
            monkeypatch.setattr(chat_core, "OLLAMA_TIMEOUT", -1.0)  # 與 config 快照不同 → 觸發重建
            new_client = mock.Mock()
            monkeypatch.setattr(chat_core.ollama, "Client", mock.Mock(return_value=new_client))
            chat_core._sync_config()
            old_client.close.assert_called_once()
            assert chat_core._ollama is new_client
        finally:
            ollama_shared._clients.clear()
            ollama_shared._clients.update(snapshot)


# ------------------------------------------------------------
# 6. 匯入：部分塊嵌入失敗不得標整檔 ok，否則永遠不會補壞塊
# ------------------------------------------------------------
class TestIngestPartialOk:
    def test_partial_embed_failure_not_ok(self, tmp_path: Path) -> None:
        """部分嵌入失敗不標整檔 ok，壞塊下次重試補寫。」"""
        (tmp_path / "full.md").write_text("hello world", encoding="utf-8")
        (tmp_path / "partial.md").write_text("A" * 900 + "\n\n" + "B" * 900, encoding="utf-8")

        def fake_flush(client, chunks, metas, ensured):
            """假寫入：partial.md 只成功一部分，其餘全成功（相容跨檔混批）。"""
            done: dict[str, int] = {}
            written = 0
            for m in metas:
                src = m.get("source", "")
                if src == "partial.md":
                    # 該檔只算 1 塊完成，其餘視為嵌入失敗不計完成
                    if done.get(src, 0) < 1:
                        done[src] = done.get(src, 0) + 1
                        written += 1
                else:
                    done[src] = done.get(src, 0) + 1
                    written += 1
            return (written, 0, done)

        with mock.patch.object(rq, "_client"):
            with mock.patch.object(rq, "_flush_batch", side_effect=fake_flush):
                rq.ingest_folder(str(tmp_path))
        import json

        cache = json.loads((tmp_path / rq._INGEST_CACHE_NAME).read_text(encoding="utf-8"))
        assert cache["full.md"]["ok"] is True
        assert cache["partial.md"]["ok"] is False

class TestExistingIdNormalization:
    def test_dashed_ids_match(self) -> None:
        """Qdrant 回傳加 dash 的 UUID 也比對得上，未變塊零寫入零嵌入（曾全量重嵌）。"""
        import types

        chunks = ["第一塊", "第二塊"]
        metas = [{"source": "a.md", "text": t} for t in chunks]
        pids = [rq._stable_id(m["source"], t) for m, t in zip(metas, chunks)]
        assert "-" not in pids[0]  # 自家是無 dash md5，庫裡是加 dash 形

        def _dashed(h: str) -> str:
            """32 字 hex 補成 UUID 標準形，模擬 Qdrant 回傳。」"""
            return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

        recs = [types.SimpleNamespace(id=_dashed(pid), payload={"source": m["source"], "text": m["text"]}) for pid, m in zip(pids, metas)]
        fake_client = types.SimpleNamespace(retrieve=lambda **kw: recs)
        ensured: dict[str, bool] = {}
        written, skipped, done = rq._flush_batch(fake_client, chunks, metas, ensured)
        assert (written, skipped) == (0, 2)
        assert done == {"a.md": 2}


# ------------------------------------------------------------
# 7. 有界讀取：超過上限的內容不再載入記憶體
# ------------------------------------------------------------
class TestBoundedReads:
    def test_txt_does_not_read_beyond_limit(self, tmp_path: Path) -> None:
        """純文字只讀到上限＋1 字就停，尾部標記讀不到。」"""
        import dataclasses

        fp = tmp_path / "big.md"
        fp.write_text("A" * 1000 + "TAIL_MARKER" + "B" * 5000, encoding="utf-8")
        cfg = dataclasses.replace(rq._CONFIG, text_max_chars=1000)
        with mock.patch.object(rq, "_CONFIG", cfg):
            out = rq._read_file_text(fp)
        assert "TAIL_MARKER" not in out
        assert "已截斷" in out

    def test_csv_stops_at_limit(self, tmp_path: Path) -> None:
        """CSV 讀到上限＋1 列即停，大檔不整檔進記憶體。」"""
        import dataclasses

        fp = tmp_path / "big.csv"
        fp.write_text("h1,h2\n" + "\n".join(f"a{i},b{i}" for i in range(10000)), encoding="utf-8")
        cfg = dataclasses.replace(rq._CONFIG, csv_max_rows=5)
        with mock.patch.object(rq, "_CONFIG", cfg):
            out = rq._read_csv(fp)
        assert out.count("\n") == 5  # 表頭＋上限 5 列


# ------------------------------------------------------------
# 8. --health 健康檢查：全通回 0、有缺回 1
# ------------------------------------------------------------
class TestHealthCheck:
    def test_all_ok(self) -> None:
        """全通回 0：重排雙可用＋Qdrant 有收藏集。」"""
        import types

        import chat_cli

        coll = types.SimpleNamespace(collections=[types.SimpleNamespace(name="notes")])
        with mock.patch("reranker.probe_availability", return_value={"crossencoder": True, "llm": True}):
            with mock.patch.object(rq, "_client") as m_client:
                m_client.return_value.get_collections.return_value = coll
                # 嵌入／聊天探測不碰真 Ollama，否則無服務的 CI 必紅
                with mock.patch.object(rq, "probe_embed", return_value=True):
                    with mock.patch("ollama_shared.probe_model", return_value=True):
                        rc = chat_cli._run_health_check("m")
        assert rc == 0

    def test_failures_return_nonzero(self) -> None:
        """有缺回 1：重排全不可用＋Qdrant 連不上。」"""
        import chat_cli

        with mock.patch("reranker.probe_availability", return_value={"crossencoder": False, "llm": False}):
            with mock.patch.object(rq, "_client", side_effect=RuntimeError("boom")):
                with mock.patch.object(rq, "probe_embed", return_value=False):
                    with mock.patch("ollama_shared.probe_model", return_value=False):
                        rc = chat_cli._run_health_check("m")
        assert rc == 1
