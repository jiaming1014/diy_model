# -*- coding: utf-8 -*-
"""eval.py：RAG 品質評估｜新手教學版。

【這支程式在做什麼？（白話版）】
換模型（llama/gemma/qwen）後，怎麼知道「誰真的會吃本地筆記」？
靠肉眼每次重測太累，這支就是自動考卷：固定題目＋標準答案關鍵字，
跑完直接給分數。

【兩種模式】
- python eval.py：只測「檢索層」（search_local 有沒有撈到），快、不花模型錢
- python eval.py --with-model gemma4:31b-cloud：加測「模型層」
  （chat_w 的回覆有沒有採納筆記），慢、會打真實 Ollama

【結束碼】
檢索全過回 0，有掛掉回 1，方便以後接 CI。
"""

import argparse
import logging
import sys

sys.stdout.reconfigure(encoding="utf-8")

import chat_core
import config as _config
from rag_qdrant import search_local

logger = logging.getLogger(__name__)

SYS_MSG = "請透過所提供的資料回答使用者問題，並一律使用繁體中文（台灣用語）回答"

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
    """回傳命中的關鍵字（純函式，好測試）｜新手：改考卷先對答案。」"""
    return [kw for kw in keywords if kw in text]


def _eval_retrieval(case: dict) -> dict:
    """檢索層：search_local 撈到的筆記有沒有含關鍵字。」"""
    hits = search_local(case["q"], limit=3)
    blob = "\n".join(h.get("text", "") for h in hits)
    found = _check_keywords(blob, case["keywords"])
    return {
        "q": case["q"],
        "pass": len(found) == len(case["keywords"]),
        "found": found,
        "missing": [k for k in case["keywords"] if k not in found],
        "hits": [(h.get("source", ""), h.get("text", "")[:40]) for h in hits],
    }


def _eval_model(case: dict, model: str) -> dict:
    """模型層：chat_w 回覆有沒有採納筆記（打真實 Ollama，較慢）。」"""
    state = chat_core.ChatState()  # 獨立狀態，不污染歷史檔
    reply = "".join(chat_core.chat_w(SYS_MSG, case["q"], search_g=True, model=model, state=state))
    found = _check_keywords(reply, case["keywords"])
    rag = chat_core.get_last_rag(state)
    src = chat_core.get_last_sources(state)
    return {
        "q": case["q"],
        "pass": len(found) == len(case["keywords"]),
        "found": found,
        "missing": [k for k in case["keywords"] if k not in found],
        "used_rag": len(rag) > 0,
        "used_web": len(src) > 0,
        "reply_head": reply[:120],
    }


def main(argv: list[str] | None = None) -> int:
    """跑完全部考題，印成績單，回傳結束碼。」"""
    ap = argparse.ArgumentParser(description="RAG 品質評估")
    ap.add_argument("--with-model", default="", help="加測模型層，例如 gemma4:31b-cloud")
    ap.add_argument("--verbose", action="store_true", help="顯示詳細命中")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, force=True)

    print(f"評估模型設定：OLLAMA_MODEL={_config.OLLAMA_MODEL}，EMBED_MODEL={_config.EMBED_MODEL}")
    print("=" * 60)

    results = [_eval_retrieval(c) for c in CASES]
    ok = sum(1 for r in results if r["pass"])
    for r in results:
        mark = "✅" if r["pass"] else "❌"
        print(f"{mark} 檢索｜{r['q']}")
        print(f"   命中：{r['found']} 缺失：{r['missing']}")
        if args.verbose:
            for src, head in r["hits"]:
                print(f"   - {src}｜{head}")
    print(f"檢索：{ok}/{len(results)} 過")

    if args.with_model:
        print("=" * 60)
        m_cases = [c for c in CASES if c.get("model")]
        m_results = [_eval_model(c, args.with_model) for c in m_cases]
        m_ok = sum(1 for r in m_results if r["pass"])
        for r in m_results:
            mark = "✅" if r["pass"] else "❌"
            path = f"RAG={'有' if r['used_rag'] else '無'}/網路={'有' if r['used_web'] else '無'}"
            print(f"{mark} 模型｜{r['q']}（{path}）")
            print(f"   命中：{r['found']} 缺失：{r['missing']}")
            print(f"   回覆前120字：{r['reply_head']}")
        print(f"模型：{m_ok}/{len(m_results)} 過")
        return 0 if (ok == len(results) and m_ok == len(m_results)) else 1
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
