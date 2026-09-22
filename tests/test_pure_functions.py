"""diy_model 純函式測試：不碰網路、不碰 Qdrant、不碰 Ollama。

【測什麼？】
1. chat_core._trim_hist：組數上限＋字數預算裁切
2. rag_qdrant._stable_id：同內容同 ID、異內容異 ID
3. reranker._clamp_score：分數夾 0~10、壞值回 0
4. reranker._apply_threshold：門檻過濾、至少留 1 條
"""

from pathlib import Path

# sys.path 由 tests/conftest.py 集中設定（專案根），故可直接匯入模組
from chat_core import ChatMessage, _format_rag_results, _trim_hist, backtrace  # noqa: E402
from eval import _check_keywords  # noqa: E402
from rag_qdrant import _stable_id  # noqa: E402
from reranker import _apply_threshold, _clamp_score  # noqa: E402


# ------------------------------------------------------------
# 1. _trim_hist：組數與字數預算
# ------------------------------------------------------------
class TestTrimHist:
    def test_trims_to_pair_limit(self) -> None:
        """超過 2*backtrace 則時裁到上限且成對。"""
        buf: list[ChatMessage] = []
        for i in range(backtrace + 3):  # 7 組 > 上限 4 組
            buf.append({"role": "user", "content": f"q{i}"})
            buf.append({"role": "assistant", "content": f"a{i}"})
        _trim_hist(buf)
        assert len(buf) == 2 * backtrace
        # 成對檢查：偶數索引必為 user
        assert all(buf[i]["role"] == "user" for i in range(0, len(buf), 2))
        # 保留最新
        assert buf[-1]["content"] == f"a{backtrace + 2}"

    def test_char_budget_trims_old(self) -> None:
        """字數預算超標時從舊的裁。"""
        big = "長" * 500
        buf: list[ChatMessage] = []
        for i in range(6):
            buf.append({"role": "user", "content": f"{big}{i}"})
            buf.append({"role": "assistant", "content": f"{big}答{i}"})
        _trim_hist(buf)  # 預設 HIST_MAX_CHARS=6000，6 組 * 1004 字 ≈ 6024 > 6000
        total = sum(len(m.get("content", "")) for m in buf)
        assert total <= 6000 or len(buf) == 0

    def test_empty_buf(self) -> None:
        """空歷史不炸。"""
        _trim_hist([])


# ------------------------------------------------------------
# 2. _stable_id：穩定雜湊
# ------------------------------------------------------------
class TestStableId:
    def test_same_content_same_id(self) -> None:
        """同來源同內容 ID 穩定，重複匯入覆寫不新增。」"""
        assert _stable_id("a.md", "hello") == _stable_id("a.md", "hello")

    def test_diff_content_diff_id(self) -> None:
        """內容不同 ID 不同。」"""
        assert _stable_id("a.md", "hello") != _stable_id("a.md", "world")

    def test_diff_source_diff_id(self) -> None:
        """來源不同 ID 不同。」"""
        assert _stable_id("a.md", "hello") != _stable_id("b.md", "hello")

    def test_is_32hex(self) -> None:
        """ID 為 32 字小寫十六進位（MD5 格式）。」"""
        v = _stable_id("s", "t")
        assert len(v) == 32
        assert all(c in "0123456789abcdef" for c in v)


# ------------------------------------------------------------
# 3. _clamp_score：夾取
# ------------------------------------------------------------
class TestClampScore:
    def test_within_range(self) -> None:
        """範圍內分數原樣通過。」"""
        assert _clamp_score(5.5) == 5.5

    def test_clamps_high(self) -> None:
        """超 10 夾到 10，避免模型亂給高分。」"""
        assert _clamp_score(100.0) == 10.0

    def test_clamps_negative(self) -> None:
        """負分夾到 0。」"""
        assert _clamp_score(-3.0) == 0.0

    def test_bad_value_zero(self) -> None:
        """壞值（字串／None）回 0 不拋錯。」"""
        assert _clamp_score("abc") == 0.0
        assert _clamp_score(None) == 0.0

    def test_numeric_string(self) -> None:
        """數字字串去空白轉浮點。」"""
        assert _clamp_score(" 7.2 ") == 7.2


# ------------------------------------------------------------
# 4. _apply_threshold：門檻過濾
# ------------------------------------------------------------
class TestApplyThreshold:
    def test_no_threshold_keeps_all(self) -> None:
        """沒設門檻（-inf）全保留。」"""
        import reranker as r
        docs = [{"text": "a", "score": "1"}, {"text": "b", "score": "9"}]
        old = r.RERANK_THRESHOLD
        r.RERANK_THRESHOLD = float("-inf")
        try:
            assert _apply_threshold(docs) == docs
        finally:
            r.RERANK_THRESHOLD = old

    def test_filters_low_scores(self) -> None:
        """低於門檻丟掉，只留高分。」"""
        import reranker as r
        docs = [{"text": "a", "score": "1"}, {"text": "b", "score": "9"}]
        old = r.RERANK_THRESHOLD
        r.RERANK_THRESHOLD = 5.0
        try:
            kept = _apply_threshold(docs)
            assert len(kept) == 1
            assert kept[0]["text"] == "b"
        finally:
            r.RERANK_THRESHOLD = old

    def test_keeps_at_least_one(self) -> None:
        """全被濾掉也留第 1 名，避免空結果。」"""
        import reranker as r
        docs = [{"text": "a", "score": "0.1"}]
        old = r.RERANK_THRESHOLD
        r.RERANK_THRESHOLD = 5.0
        try:
            kept = _apply_threshold(docs)
            assert len(kept) == 1  # 全被濾掉也留第 1 名
        finally:
            r.RERANK_THRESHOLD = old

    def test_missing_score_kept(self) -> None:
        """缺分數當保留，避免誤殺無分數命中。」"""
        import reranker as r
        docs = [{"text": "no-score"}]
        old = r.RERANK_THRESHOLD
        r.RERANK_THRESHOLD = 5.0
        try:
            assert _apply_threshold(docs) == docs  # 缺分數當 inf 保留
        finally:
            r.RERANK_THRESHOLD = old


# ------------------------------------------------------------
# 5. _format_rag_results：RAG 字數預算與截斷
# ------------------------------------------------------------
class TestFormatRagResults:
    def test_empty_hits_returns_empty(self) -> None:
        """無命中回空字串。"""
        assert _format_rag_results([]) == ""

    def test_short_hit_passthrough(self) -> None:
        """短筆記完整保留，含來源與標頭。"""
        out = _format_rag_results([{"source": "a.md", "text": "台北九月偏熱"}])
        assert "a.md" in out
        assert "台北九月偏熱" in out

    def test_overlong_single_hit_truncated(self) -> None:
        """單筆超長筆記被截斷，總長受 RAG_MAX_CHARS 限制。"""
        import chat_core as c
        big = "長" * 10000
        out = _format_rag_results([{"source": "b.md", "text": big}])
        assert out  # 有內容
        assert len(out) < len(big)  # 被截斷
        assert len(out) <= c.RAG_MAX_CHARS + 200  # 預算＋標頭餘裕

    def test_total_budget_not_exceeded(self) -> None:
        """多筆合計受總預算限制，不會無限拼接。"""
        import chat_core as c
        hits = [{"source": f"s{i}.md", "text": "字" * 900} for i in range(10)]
        out = _format_rag_results(hits)
        assert len(out) <= c.RAG_MAX_CHARS + len(hits) * 50

    def test_budget_complete_early_stop(self) -> None:
        """預算用完即停止拼接，不拋錯且仍有輸出。"""
        import chat_core as c
        hits = [{"source": f"s{i}.md", "text": "字" * 1500} for i in range(50)]
        out = _format_rag_results(hits)
        assert out
        assert len(out) <= c.RAG_MAX_CHARS + 500


# ------------------------------------------------------------
# 6. _check_keywords：eval 評分函式
# ------------------------------------------------------------
class TestCheckKeywords:
    def test_all_hit(self) -> None:
        """關鍵字全中回全部。」"""
        assert _check_keywords("帶傘雷陣雨", ["帶傘", "雷陣雨"]) == ["帶傘", "雷陣雨"]

    def test_partial_hit(self) -> None:
        """只中部分回命中的子集。」"""
        assert _check_keywords("只有帶傘", ["帶傘", "雷陣雨"]) == ["帶傘"]

    def test_no_hit(self) -> None:
        """全不中回空串列。」"""
        assert _check_keywords("無關文字", ["帶傘"]) == []

    def test_empty_keywords(self) -> None:
        """空考題回空，不誤判。」"""
        assert _check_keywords("任何文字", []) == []


# ------------------------------------------------------------
# 7. config 一致性：三模組吃同一個真相來源
# ------------------------------------------------------------
class TestConfigConsistency:
    def test_timeout_single_source(self) -> None:
        """OLLAMA_TIMEOUT 三處一致（config.py 唯一真相）。"""
        import config
        import chat_core as c
        import rag_qdrant as rq
        import reranker as r
        assert config.OLLAMA_TIMEOUT == c.OLLAMA_TIMEOUT == rq._CONFIG.timeout == r._ollama_timeout

    def test_model_single_source(self) -> None:
        """OLLAMA_MODEL 兩處一致。」"""
        import config
        import chat_core as c
        assert config.OLLAMA_MODEL == c.OLLAMA_MODEL

    def test_embed_model_untouched(self) -> None:
        """嵌入模型維持 nomic-embed-text，不被聊天模型牽連。"""
        import config
        import rag_qdrant as rq
        assert config.EMBED_MODEL == "nomic-embed-text" == rq._CONFIG.embed_model
