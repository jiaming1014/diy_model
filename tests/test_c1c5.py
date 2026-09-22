"""test_c1c5.py：C1–C5＋D3 回歸測試（全 mock，不碰網路）。"""
from types import SimpleNamespace
from unittest import mock

import chat_core
import config
import rag_qdrant
from chat_core import ChatState


def _mock_client_with_points(points):
    m = mock.MagicMock()
    m.query_points.return_value = SimpleNamespace(points=points)
    return m


class TestC1OnDemandRecall:
    """C1：重排停用時按需寬取，啟用時維持寬取。"""

    def _search(self, limit=3):
        client = _mock_client_with_points([])
        with mock.patch.object(rag_qdrant, "_client", return_value=client), \
             mock.patch.object(rag_qdrant, "_embed_query_vec", return_value=[0.1] * 8):
            out = rag_qdrant.search_local("寬取測試唯一串", limit=limit)
        return out, client.query_points.call_args.kwargs["limit"]

    def test_backend_none_uses_limit(self):
        with mock.patch.object(config, "RERANK_BACKEND", "none"), \
             mock.patch.object(config, "RERANK_ENABLE", True):
            out, recall = self._search()
        assert out == [] and recall == 3

    def test_switch_off_uses_limit(self):
        with mock.patch.object(config, "RERANK_BACKEND", "auto"), \
             mock.patch.object(config, "RERANK_ENABLE", False):
            out, recall = self._search()
        assert out == [] and recall == 3

    def test_enabled_keeps_wide_recall(self):
        with mock.patch.object(config, "RERANK_BACKEND", "auto"), \
             mock.patch.object(config, "RERANK_ENABLE", True):
            out, recall = self._search()
        assert out == [] and recall == max(3, rag_qdrant._CONFIG.rerank_recall)


class TestC2Shortcut:
    """C2：向量高分短路。"""

    def _hits(self, scores):
        return [
            {"source": "s", "text": f"t{i}", **({"score": str(s)} if s is not None else {})}
            for i, s in enumerate(scores)
        ]

    def _patch_thresholds(self):
        return (mock.patch.object(config, "RERANK_SHORTCUT", True),
                mock.patch.object(config, "RERANK_SHORTCUT_MIN", 0.85),
                mock.patch.object(config, "RERANK_SHORTCUT_GAP", 0.15))

    def test_triggers_on_clear_lead(self):
        p1, p2, p3 = self._patch_thresholds()
        with p1, p2, p3:
            assert rag_qdrant._rerank_shortcut_hit(self._hits([0.92, 0.70, 0.50])) is True

    def test_boundary_inclusive(self):
        p1, p2, p3 = self._patch_thresholds()
        with p1, p2, p3:
            assert rag_qdrant._rerank_shortcut_hit(self._hits([0.85, 0.70])) is True

    def test_low_top_no_trigger(self):
        p1, p2, p3 = self._patch_thresholds()
        with p1, p2, p3:
            assert rag_qdrant._rerank_shortcut_hit(self._hits([0.80, 0.50])) is False

    def test_small_gap_no_trigger(self):
        p1, p2, p3 = self._patch_thresholds()
        with p1, p2, p3:
            assert rag_qdrant._rerank_shortcut_hit(self._hits([0.90, 0.85])) is False

    def test_bad_scores_fall_through(self):
        p1, p2, p3 = self._patch_thresholds()
        with p1, p2, p3:
            assert rag_qdrant._rerank_shortcut_hit(self._hits([None, 0.5])) is False
            assert rag_qdrant._rerank_shortcut_hit(self._hits(["nan", 0.5])) is False
            assert rag_qdrant._rerank_shortcut_hit(self._hits([0.9])) is False

    def test_switch_off(self):
        with mock.patch.object(config, "RERANK_SHORTCUT", False):
            assert rag_qdrant._rerank_shortcut_hit(self._hits([0.95, 0.5])) is False

    def test_search_local_skips_rerank(self):
        pts = [SimpleNamespace(payload={"source": "s", "text": f"t{i}"}, score=s)
               for i, s in enumerate([0.95, 0.70, 0.60, 0.50, 0.40])]
        client = _mock_client_with_points(pts)
        p1, p2, p3 = self._patch_thresholds()
        with p1, p2, p3, \
             mock.patch.object(rag_qdrant, "_client", return_value=client), \
             mock.patch.object(rag_qdrant, "_embed_query_vec", return_value=[0.1] * 8), \
             mock.patch.object(rag_qdrant, "_rerank", side_effect=AssertionError("短路不該進重排")):
            out = rag_qdrant.search_local("短路測試唯一串", limit=3)
        assert [h["text"] for h in out] == ["t0", "t1", "t2"]


class TestC3LazyTokens:
    """C3：字數不到一半跳過 tiktoken，超一半才算。"""

    def test_small_history_skips_tiktoken(self):
        st = ChatState(hist=[{"role": "user", "content": "x" * 100},
                             {"role": "assistant", "content": "y" * 100}])
        with mock.patch.object(chat_core, "HIST_MAX_CHARS", 6000), \
             mock.patch.object(chat_core, "_content_tokens",
                               side_effect=AssertionError("字數未過半不該算 token")):
            chat_core._trim_hist(st)
        assert len(st.hist) == 2

    def test_large_history_computes_tokens(self):
        st = ChatState(hist=[{"role": "user", "content": "y" * 4000},
                             {"role": "assistant", "content": "z" * 100}])
        with mock.patch.object(chat_core, "HIST_MAX_CHARS", 6000), \
             mock.patch.object(chat_core, "_content_tokens",
                               wraps=chat_core._content_tokens) as m:
            chat_core._trim_hist(st)
        assert m.call_count > 0
        assert len(st.hist) == 2  # 4100 字未超 6000，不裁


class TestC4DequeDrain:
    """C4：deque 分批與舊切片等價。"""

    def test_batching_identical(self, tmp_path):
        (tmp_path / "a.txt").write_text("甲乙丙丁戊" * 40 + "\n\n" + "一二三四五" * 40, encoding="utf-8")
        sizes = []

        def fake_flush(client, chunks, metas, ensured):
            sizes.append(len(chunks))
            return (len(chunks), 0, {})

        with mock.patch.object(config, "CHUNK_CHARS", 20), \
             mock.patch.object(config, "CHUNK_OVERLAP", 0), \
             mock.patch.object(config, "EMBED_BATCH", 4), \
             mock.patch.object(rag_qdrant, "_client", return_value=mock.MagicMock()), \
             mock.patch.object(rag_qdrant, "_flush_batch", side_effect=fake_flush):
            total = rag_qdrant.ingest_folder(str(tmp_path))
        assert total == sum(sizes) > 0
        assert all(s <= 4 for s in sizes)
        assert len(sizes) > 1  # 確實發生多次 drain


class TestC5ConfigurableRetries:
    """C5：重試次數可配，預設行為不變。"""

    def _run_fail(self, retries, tag):
        chat_core._SEARCH_CACHE.clear()
        with mock.patch.object(config, "SEARCH_RETRIES", retries), \
             mock.patch.object(chat_core, "DDGS", side_effect=Exception("down")) as m, \
             mock.patch.object(chat_core.time, "sleep", return_value=None):
            out = chat_core._search_web(f"重試-{tag}-唯一串", state=ChatState())
        return out, m.call_count

    def test_zero_retries_single_try(self):
        out, n = self._run_fail(0, "zero")
        assert out == [] and n == 1

    def test_two_retries_three_tries(self):
        out, n = self._run_fail(2, "two")
        assert out == [] and n == 3


class TestD3SmallBundle:
    """D3：日期捷徑不洗來源＋resolve 快取。"""

    def test_date_query_keeps_sources(self):
        st = ChatState()
        st.last_sources.append({"title": "t", "snippet": "s", "url": "u"})
        st.last_rag.append({"source": "s", "text": "t"})
        list(chat_core.chat_w("系統", "今天幾號", state=st))
        assert len(st.last_sources) == 1 and len(st.last_rag) == 1

    def test_resolved_root_cache(self, tmp_path):
        (tmp_path / "子").mkdir()
        resolved = tmp_path.resolve()
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path), \
             mock.patch.object(chat_core, "_workspace_root_resolved", (tmp_path, resolved)):
            ok, err = chat_core._resolve_workspace_path("子/檔.md")
            bad, err2 = chat_core._resolve_workspace_path("../逃.md")
        assert err is None and ok is not None and ok.parent.name == "子"
        assert bad is None and "穿越" in (err2 or "")

    def test_stale_resolved_cache_ignored(self, tmp_path):
        """舊快照配不同根時自動現算，不誤判穿越（多根／mock 场景）。"""
        (tmp_path / "子").mkdir()
        other = tmp_path / "別處"
        with mock.patch.object(chat_core, "_workspace_root", return_value=tmp_path), \
             mock.patch.object(chat_core, "_workspace_root_resolved",
                               (other, other / "x")):
            ok, err = chat_core._resolve_workspace_path("子/檔.md")
        assert err is None and ok is not None and ok.parent.name == "子"


class TestE1E2Observability:
    """E1 計數器＋E2 量尺護欄。"""

    def _score_hits(self, scores):
        return [{"source": "s", "text": f"t{i}", "score": str(s)} for i, s in enumerate(scores)]

    def test_out_of_range_abstains(self):
        p1 = mock.patch.object(config, "RERANK_SHORTCUT", True)
        p2 = mock.patch.object(config, "RERANK_SHORTCUT_MIN", 0.78)
        p3 = mock.patch.object(config, "RERANK_SHORTCUT_GAP", 0.06)
        with p1, p2, p3:
            assert rag_qdrant._rerank_shortcut_hit(self._score_hits([5.2, 3.1])) is False
            assert rag_qdrant._rerank_shortcut_hit(self._score_hits([2.5, 3.0])) is False
            assert rag_qdrant._rerank_shortcut_hit(self._score_hits([0.90, 0.70])) is True

    def test_counters_track_firing(self):
        pts = [SimpleNamespace(payload={"source": "s", "text": f"t{i}"}, score=s)
               for i, s in enumerate([0.95, 0.70, 0.60, 0.50, 0.40])]
        client = _mock_client_with_points(pts)
        old_total, old_fired = rag_qdrant.SHORTCUT_TOTAL, rag_qdrant.SHORTCUT_FIRED
        try:
            with mock.patch.object(config, "RERANK_SHORTCUT", True), \
                 mock.patch.object(config, "RERANK_SHORTCUT_MIN", 0.78), \
                 mock.patch.object(config, "RERANK_SHORTCUT_GAP", 0.06), \
                 mock.patch.object(rag_qdrant, "_client", return_value=client), \
                 mock.patch.object(rag_qdrant, "_embed_query_vec", return_value=[0.1] * 8), \
                 mock.patch.object(rag_qdrant, "_rerank", side_effect=AssertionError("短路不該進重排")):
                out = rag_qdrant.search_local("計數測試唯一串", limit=3)
            assert len(out) == 3
            stats = rag_qdrant.shortcut_stats()
            assert stats["total"] == old_total + 1 and stats["fired"] == old_fired + 1
        finally:
            rag_qdrant.SHORTCUT_TOTAL, rag_qdrant.SHORTCUT_FIRED = old_total, old_fired


class TestF0F1:
    """F0 門檻護欄＋F1 評估並行保序。"""

    def test_threshold_set_skips_shortcut(self):
        hits = [{"source": "s", "text": "t0", "score": "0.95"},
                {"source": "s", "text": "t1", "score": "0.70"}]
        base = (mock.patch.object(config, "RERANK_SHORTCUT", True),
                mock.patch.object(config, "RERANK_SHORTCUT_MIN", 0.78),
                mock.patch.object(config, "RERANK_SHORTCUT_GAP", 0.06))
        with base[0], base[1], base[2], mock.patch.object(config, "RERANK_THRESHOLD", 0.5):
            assert rag_qdrant._rerank_shortcut_hit(hits) is False
        with base[0], base[1], base[2], mock.patch.object(config, "RERANK_THRESHOLD", float("-inf")):
            assert rag_qdrant._rerank_shortcut_hit(hits) is True

    def test_model_layer_parallel_order(self):
        import eval as _eval

        calls = []

        def fake_retrieval(case):
            return {"q": case["q"], "pass": True, "found": [], "missing": [],
                    "hits": [], "rank": 1, "rr": 1.0, "recall": 1.0,
                    "shortcut": False, "shortcut_precise": None}

        def fake_model(case, model):
            calls.append(case["q"])
            return {"q": case["q"], "pass": True, "found": [], "missing": [],
                    "used_rag": False, "used_web": False, "cited": False,
                    "cite_pass": True, "reply_head": "ok"}

        with mock.patch.object(_eval, "_eval_retrieval", side_effect=fake_retrieval), \
             mock.patch.object(_eval, "_eval_model", side_effect=fake_model):
            rc = _eval.main(["--with-model", "dummy-model"])
        assert rc == 0
        assert calls == [c["q"] for c in _eval.CASES if c.get("model")]
