"""chat_cli.py（原 ch8_30.py）：命令列聊天機器人的主程式（入口）｜新手教學版。

【這支程式在做什麼？（白話版）】
這就是你跟 AI 聊天時看到的那個黑框框：
你打一句，它回一句，一直循環，直到你說掰掰。

【怎麼離開？】
輸入 q、quit、exit、離開、結束、掰掰、再見，或按 Ctrl+C / Ctrl+D 都可以優雅離開。
空行會視為「還沒想好」，直接等下一句，不會關店。

【啟動參數】
- python chat_cli.py：直接開聊（自動載入上次歷史）
- python chat_cli.py --model llama3.2:1b：指定模型
- python chat_cli.py --no-search：關掉工具搜尋（純問模型）
- python chat_cli.py --no-history：不載入／不存歷史
"""

import argparse
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from chat_core import OLLAMA_MODEL as _DEFAULT_MODEL
from chat_core import ChatState, chat_w, get_last_rag, get_last_sources
import chat_core as _core

sys_msg = '請透過所提供的資料回答使用者問題，並一律使用繁體中文（台灣用語）回答'
MODEL = _DEFAULT_MODEL
_QUIT_CMDS = {"q", "quit", "exit", "離開", "結束", "掰掰", "再見"}
_DEFAULT_HIST = Path.home() / ".diy_model_hist.json"  # 相容快照，運行請走 _hist_path()
_hist_lock = threading.Lock()  # 存檔鎖，同進程多執行緒不互踩


def _hist_path() -> Path:
    """每次讀環境變數，改 DIY_HIST_FILE 不用重啟｜新手：地址每次出門現查，不抄舊紙條。」"""
    return Path(os.getenv("DIY_HIST_FILE", str(_DEFAULT_HIST)))

# 相容舊匯入：保留名稱，值為啟動時快照
_HIST_FILE = _hist_path()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析啟動參數｜新手：開店前先看客人有沒有特別交代。」"""
    ap = argparse.ArgumentParser(description="本地 Ollama 聊天 CLI")
    ap.add_argument("--model", default=MODEL, help=f"使用的 Ollama 模型（預設 {MODEL}）")
    ap.add_argument("--no-search", action="store_true", help="關掉工具搜尋，純問模型")
    ap.add_argument("--verbose", action="store_true", help="顯示降級等除錯訊息")
    ap.add_argument("--no-history", action="store_true", help="不載入／不存歷史")
    return ap.parse_args(argv)


def _hist_keep_n() -> int:
    """歷史保留則數與 chat_core.backtrace 對齊，避免兩處寫死脫鉤。」"""
    try:
        return max(2, 2 * int(_core.backtrace))
    except Exception:
        return 8


def _content_key(role: object, content: object) -> tuple:
    """去重鍵只存雜湊，不存全文，避免長問答撐大集合。」"""
    text = content if isinstance(content, str) else str(content or "")
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return (role, len(text), digest)


def _trim_tail_to_budget(msgs: list[dict]) -> list[dict]:
    """按 HIST_MAX_CHARS 字數＋token 雙預算從舊裁剪，與 chat_core._trim_hist 同尺。」"""
    try:
        budget = int(_core.HIST_MAX_CHARS)
    except Exception:
        return msgs
    try:
        tok_len = _core._content_tokens
    except Exception:
        tok_len = lambda s: len(s or "")  # noqa: E731 - 退化字數
    total_chars = sum(len(str(m.get("content", "") or "")) for m in msgs)
    total_toks = sum(tok_len(str(m.get("content", "") or "")) for m in msgs)
    i = 0
    while i < len(msgs) and (total_chars > budget or total_toks > budget):
        total_chars -= len(str(msgs[i].get("content", "") or ""))
        total_toks -= tok_len(str(msgs[i].get("content", "") or ""))
        i += 1
    # 保持 user→assistant 成對：若從 assistant 開始多裁一則
    tail = msgs[i:]
    if len(tail) % 2 == 1 and tail and tail[0].get("role") == "assistant":
        tail = tail[1:]
    return tail


def _load_history(state: ChatState, path: Path | None = None) -> None:
    """載入上次存的問答，壞檔當無歷史｜新手：開店先把上次的小抄拿出來。」"""
    p = path if path is not None else _hist_path()
    try:
        if not p.is_file():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for m in data[-_hist_keep_n():]:
                if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                    state.hist.append({"role": m["role"], "content": m["content"]})
    except Exception as e:
        logging.getLogger(__name__).warning("歷史載入失敗，已從空開始：%s", e)


def _save_history(state: ChatState, path: Path | None = None) -> None:
    """原子合併存檔：讀現有+本次去重留最後 N 組＋字數預算，tmp+replace 防半寫｜新手：關店先對帳再鎖門。」"""
    p = path if path is not None else _hist_path()
    with _hist_lock:
        try:
            merged: list[dict] = []
            if p.is_file():
                try:
                    old = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(old, list):
                        merged.extend([m for m in old if isinstance(m, dict)])
                except Exception:
                    pass  # 舊檔壞掉當空的，直接覆寫
            merged.extend([{"role": m["role"], "content": m["content"]} for m in state.hist])
            # 去重保序：同 role+雜湊只留最後一次
            seen: set[tuple] = set()
            dedup: list[dict] = []
            for m in reversed(merged):
                key = _content_key(m.get("role"), m.get("content"))
                if key not in seen:
                    seen.add(key)
                    dedup.append(m)
            dedup.reverse()
            tail = _trim_tail_to_budget(dedup[-_hist_keep_n():])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(tail, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, p)  # 原子替換，多開同時寫不留半檔
        except Exception as e:
            logging.getLogger(__name__).warning("歷史存檔失敗：%s", e)


def _print_rag(state: ChatState) -> None:
    """顯示本地筆記來源，有命中才印標題。」"""
    rag_hits = get_last_rag(state)
    if not rag_hits:
        return
    print("本地筆記來源：", flush=True)
    for i, h in enumerate(rag_hits, start=1):
        print(f"  [筆記{i}] {h.get('source', '')}", flush=True)
    print('', flush=True)


def _print_sources(state: ChatState) -> None:
    """顯示網路參考來源，有來源才印，編號與提示詞 [來源i] 對齊。」"""
    sources = get_last_sources(state)
    if not sources:
        return
    print("參考來源：", flush=True)
    for i, src in enumerate(sources, start=1):
        title = src.get("title", "")
        url = src.get("url", "")
        print(f"  [來源{i}] {title} - {url}", flush=True)
    print('', flush=True)


def main(argv: list[str] | None = None) -> None:
    """主迴圈：不斷問「你說：」，再即時印出「小助理：」的串流回覆。」"""
    args = parse_args(argv)
    # 優化：force=True 讓重複 basicConfig 生效，避免第二入口的設定被吃掉
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    model = args.model
    search_g = not args.no_search
    # 優化：獨立會話狀態，不再共用全域；--no-history 用完即丟
    state = ChatState()
    if not args.no_history:
        _load_history(state)
        # 相容舊寫法：把載入的歷史同步給全域，舊外掛讀 hist 不會空
        _core.hist.extend([m for m in state.hist if m not in _core.hist])
    print(f"小助理已啟動（模型：{model}，搜尋：{'開' if search_g else '關'}），輸入 q 可離開。", flush=True)
    # P12 體感：有 prompt_toolkit 走歷史上下鍵＋持久歷史，缺套件退化 input
    _prompt = None
    try:
        from prompt_toolkit import PromptSession as _Session  # type: ignore[import-not-found]
        from prompt_toolkit.history import FileHistory as _FileHistory  # type: ignore[import-not-found]

        _prompt = _Session(history=_FileHistory(str(_hist_path().with_suffix(".prompt_hist"))))
    except Exception:
        _prompt = None
    def _ask() -> str:
        if _prompt is not None:
            try:
                return _prompt.prompt("你說：")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                return input("你說：")
        return input("你說：")
    try:
        while True:
            try:
                msg = _ask()
            except (EOFError, KeyboardInterrupt):
                print("\n再見！", flush=True)
                break
            text = msg.strip()
            if not text:
                continue  # 空行等下一句，不關店
            if text.lower() in _QUIT_CMDS:
                print("再見！", flush=True)
                break
            print("小助理：", end="", flush=True)
            try:
                for reply in chat_w(sys_msg, text, search_g=search_g, model=model, state=state):
                    print(reply, end="", flush=True)
            except KeyboardInterrupt:
                print("\n[已中斷本次回答]", flush=True)
                continue
            print('\n', flush=True)
            _print_rag(state)
            _print_sources(state)
    finally:
        if not args.no_history:
            _save_history(state)


if __name__ == "__main__":
    main()
