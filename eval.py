# -*- coding: utf-8 -*-
"""eval.py：RAG 品質評估｜新手教學版。

【這支程式在做什麼？（白話版）】
換模型（llama/gemma/qwen）後，怎麼知道「誰真的會吃本地筆記」？
靠肉眼每次重測太累，這支就是自動考卷：固定題目＋標準答案關鍵字，
跑完直接給分數。

【前置】
需 Qdrant 可連線、已匯入筆記、ollama 有嵌入模型（＋模型層要聊天模型），否則檢索全空。
注意：考題是針對範例筆記出的標準答案，用私人筆記跑低分屬正常，不代表 RAG 壞掉。
【兩種模式】
- python eval.py：只測「檢索層」（search_local 有沒有撈到），快、不花模型錢
- python eval.py --with-model llama3.2:1b：加測「模型層」
  （chat_w 的回覆有沒有採納筆記），慢、會打真實 Ollama

【結束碼】
檢索全過回 0，有掛掉回 1，方便以後接 CI。
"""

import argparse
import logging
import re
import sys

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue] - Windows 下重設 stdout 編碼
except Exception:
    pass

import chat_core
import config as _config
from chat_core import DEFAULT_SYS_MSG  # P17：系統提示唯一真相在 chat_core
from rag_qdrant import search_local
from rag_qdrant import shortcut_stats as _shortcut_stats

logger = logging.getLogger(__name__)

# OPT-8：引用標註正則預編譯，模型層逐回覆不再現編譯
_CITE_RE = re.compile(r"[(［\[] *(來源|筆記) *\d+ *[)］\]]")


def _norm_keywords(keywords: list[str]) -> list[tuple[str, str]]:
    """預正規化關鍵字：回 [(原文, 小寫去空白版)]，每 case 算一次，命中比對不再重複 lower/strip。」"""
    out: list[tuple[str, str]] = []
    for kw in keywords:
        k = kw.lower().strip() if isinstance(kw, str) else ""
        if k:
            out.append((kw, k))
    return out

SYS_MSG = DEFAULT_SYS_MSG  # 相容舊名：值與 chat_core.DEFAULT_SYS_MSG 同一內容

# 固定考題：問題＋回覆裡必須出現的關鍵字
# model=True 才跑模型層（需觸發 RAG；含「天氣」等即時關鍵字的題目會被
# chat_core 的 realtime 過濾器跳過 RAG，屬設計行為，不列入模型評分）
CASES: list[dict] = [
    {
        "q": "台北九月天氣如何？出門要帶什麼？",
        "keywords": ["帶傘", "雷陣雨"],
        "model": False,  # 含「天氣」→ 走即時路徑，檢索層仍要撈到
    },
    {
        "q": "這個專案的嵌入模型要用哪一個？",
        "keywords": ["nomic-embed-text"],
        "model": True,
    },
    {
        "q": "向量存在哪裡？",
        "keywords": ["Qdrant", "6333"],
        "model": True,
    },
    {
        "q": "筆記要放在哪個資料夾？",
        "keywords": ["notes"],
        "model": True,
    },
    {
        "q": "這個專案可以用子任務嗎？",
        "keywords": ["ulw"],
        "model": True,
    },
]


def _check_keywords(text: str, keywords: list[str]) -> list[str]:
    """回傳命中的關鍵字（純函式，好測試）｜新手：改考卷先對答案。

    正規化大小寫＋去空白，避免 Qdrant／ulw 大小寫誤判。
    """
    return _check_precomputed(text.lower(), _norm_keywords(keywords))


def _check_precomputed(norm_text: str, norm_kws: list[tuple[str, str]]) -> list[str]:
    """已正規化文本＋關鍵字表的比對核心，_check_keywords 與 _first_hit_rank 共用。」"""
    return [kw for kw, k in norm_kws if k in norm_text]


def _first_hit_rank(hits: list[dict], keywords: list[str]) -> int | None:
    """首個全命中關鍵字的排名（1 起），無則回 None，用於 MRR。

    OPT-12：關鍵字表每 case 正規化一次，不再每命中重算。
    """
    norm_kws = _norm_keywords(keywords)
    if len(norm_kws) != len(keywords):
        return None  # 含空關鍵字時沿舊語意永不命中（_check_keywords 恆少一）
    for i, h in enumerate(hits, start=1):
        text = str(h.get("text", "") or "").lower()
        if len(_check_precomputed(text, norm_kws)) == len(norm_kws):
            return i
    return None


def _eval_retrieval(case: dict) -> dict:
    """檢索層：search_local 撈到的筆記有沒有含關鍵字，附 MRR 與召回率。

    E1：順手記錄短路是否觸發＋首命中是否含關鍵字（短路精度），供閾值調整看數據。
    """
    before = _shortcut_stats()
    hits = search_local(case["q"], limit=3)
    after = _shortcut_stats()
    fired = after["fired"] > before["fired"]
    blob = "\n".join(h.get("text", "") for h in hits)
    found = _check_keywords(blob, case["keywords"])
    rank = _first_hit_rank(hits, case["keywords"])
    rr = (1.0 / rank) if rank else 0.0  # MRR 取倒數排名：首筆命中得 1 分，第 2 名得 0.5，沒命中 0 分
    recall = (len(found) / len(case["keywords"])) if case["keywords"] else 1.0
    return {
        "q": case["q"],
        "pass": len(found) == len(case["keywords"]),
        "found": found,
        "missing": [k for k in case["keywords"] if k not in found],
        "hits": [(h.get("source", ""), h.get("text", "")[:40]) for h in hits],
        "rank": rank,
        "rr": rr,
        "recall": recall,
        "shortcut": fired,
        "shortcut_precise": (rank == 1) if fired else None,
    }


def _has_citation(reply: str) -> bool:
    """回覆有無引用標註 [來源i]／[筆記i]，純函式好測試。」"""
    return bool(_CITE_RE.search(reply))


def _eval_model(case: dict, model: str) -> dict:
    """模型層：關鍵字＋引用雙指標（打真實 Ollama，較慢）。」"""
    state = chat_core.ChatState()  # 獨立狀態，不污染歷史檔
    reply = "".join(chat_core.chat_w(SYS_MSG, case["q"], search_g=True, model=model, state=state))
    found = _check_keywords(reply, case["keywords"])
    rag = chat_core.get_last_rag(state)
    src = chat_core.get_last_sources(state)
    cited = _has_citation(reply)
    kw_pass = len(found) == len(case["keywords"])
    # 有外部依據（RAG／網路）時要求引用，無依據的純知識題只看關鍵字
    needs_cite = len(rag) > 0 or len(src) > 0
    cite_pass = (cited or not needs_cite)
    return {
        "q": case["q"],
        "pass": kw_pass and cite_pass,
        "found": found,
        "missing": [k for k in case["keywords"] if k not in found],
        "used_rag": len(rag) > 0,
        "used_web": len(src) > 0,
        "cited": cited,
        "cite_pass": cite_pass,
        "reply_head": reply[:120],
    }


def main(argv: list[str] | None = None) -> int:
    """跑完全部考題，印成績單，回傳結束碼。」"""
    ap = argparse.ArgumentParser(description="RAG 品質評估")
    ap.add_argument("--with-model", default="", help="加測模型層，例如 llama3.2:1b")
    ap.add_argument("--verbose", action="store_true", help="顯示詳細命中")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, force=True)
    with_model = (args.with_model or "").strip()  # 空白字串視為沒給，不打模型直接出檢索成績

    model_label = with_model if with_model else f"{_config.OLLAMA_MODEL}（檢索層未用）"
    print(f"評估模型設定：MODEL={model_label}，EMBED_MODEL={_config.EMBED_MODEL}")
    print("=" * 60)

    results = [_eval_retrieval(c) for c in CASES]
    ok = sum(1 for r in results if r["pass"])
    for r in results:
        mark = "✅" if r["pass"] else "❌"
        print(f"{mark} 檢索｜{r['q']}")
        print(f"   命中：{r['found']} 缺失：{r['missing']} 排名：{r['rank']} RR：{r['rr']:.2f} 召回：{r['recall']:.2f}")
        if args.verbose:
            for src, head in r["hits"]:
                print(f"   - {src}｜{head}")
    mrr = sum(r["rr"] for r in results) / len(results) if results else 0.0
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
    print(f"檢索：{ok}/{len(results)} 過 MRR：{mrr:.2f} 平均召回：{avg_recall:.2f}")
    fired = [r for r in results if r.get("shortcut")]
    precise = [r for r in fired if r.get("shortcut_precise")]
    print(f"短路：觸發 {len(fired)}/{len(results)}，首命中精確 {len(precise)}/{len(fired) if fired else 0}（調閾值看這行）")

    if with_model:
        print("=" * 60)
        m_cases = [c for c in CASES if c.get("model")]
        # F1：各例獨立 ChatState，線程池並行省數分鐘等待；map 保序，成績單順序不變
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(4, len(m_cases)))) as ex:
            m_results = list(ex.map(lambda c: _eval_model(c, with_model), m_cases))
        m_ok = sum(1 for r in m_results if r["pass"])
        for r in m_results:
            mark = "✅" if r["pass"] else "❌"
            path = f"RAG={'有' if r['used_rag'] else '無'}/網路={'有' if r['used_web'] else '無'}/引用={'有' if r['cited'] else '無'}"
            print(f"{mark} 模型｜{r['q']}（{path}）")
            print(f"   命中：{r['found']} 缺失：{r['missing']} 引用過：{r['cite_pass']}")
            print(f"   回覆前120字：{r['reply_head']}")
        print(f"模型：{m_ok}/{len(m_results)} 過")
        return 0 if (ok == len(results) and m_ok == len(m_results)) else 1
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
