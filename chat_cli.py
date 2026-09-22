"""chat_cli.py（原 ch8_30.py）：命令列聊天機器人的主程式（入口）｜新手教學版。

【這支程式在做什麼？（白話版）】
這就是你跟 AI 聊天時看到的那個黑框框：
你打一句，它回一句，一直循環，直到你說掰掰。

【怎麼離開？】
輸入 q、quit、exit、離開、結束、退出、掰掰、再見，或按 Ctrl+C / Ctrl+D 都可以優雅離開。
空行會視為「還沒想好」，直接等下一句，不會關店。

【對話內指令】
- /ingest：就地匯入工作區筆記到 Qdrant（不用跳出對話）
- /ingest 資料夾：匯入指定資料夾，例如 /ingest notes

【啟動參數】
- python chat_cli.py：直接開聊（自動載入上次歷史）
- python chat_cli.py --model llama3.2:1b：指定模型
- python chat_cli.py --no-search：只關上網搜尋（本機寫檔、播音樂照常）
- python chat_cli.py --no-history：不載入／不存歷史
- python chat_cli.py --health：檢查重排、Qdrant、嵌入／聊天模型與工作區後結束（P15）
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import cast

from chat_core import OLLAMA_MODEL as _DEFAULT_MODEL
from chat_core import ChatMessage, ChatState, chat_w, get_last_rag, get_last_sources
from chat_core import DEFAULT_SYS_MSG  # P17：系統提示唯一真相在 chat_core
import chat_core as _core

sys_msg = DEFAULT_SYS_MSG  # 相容舊匯入：值與 chat_core.DEFAULT_SYS_MSG 同一內容
MODEL = _DEFAULT_MODEL
_QUIT_CMDS = {"q", "quit", "exit", "離開", "結束", "退出", "掰掰", "再見"}
_DEFAULT_HIST = Path.home() / ".diy_model_hist.json"  # 相容快照，運行請走 _hist_path()
_hist_lock = threading.Lock()  # 存檔鎖，同進程多執行緒不互踩


def _hist_path() -> Path:
    """每次讀環境變數，改 DIY_HIST_FILE 不用重啟；空字串視為沒設，回退預設｜新手：地址每次出門現查，不抄舊紙條。」"""
    raw = (os.getenv("DIY_HIST_FILE", "") or "").strip()
    if not raw or raw == "~":
        return _DEFAULT_HIST
    p = Path(raw)
    return p.expanduser() if raw.startswith("~/") or raw.startswith("~\\") else p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析啟動參數｜新手：開店前先看客人有沒有特別交代。」"""
    ap = argparse.ArgumentParser(description="本地 Ollama 聊天 CLI")
    ap.add_argument("--model", default=MODEL, help=f"使用的 Ollama 模型（預設 {MODEL}）")
    ap.add_argument("--no-search", action="store_true", help="只關上網搜尋（本機寫檔、播音樂、查日期照常）")
    ap.add_argument("--verbose", action="store_true", help="顯示降級等除錯訊息")
    ap.add_argument("--no-history", action="store_true", help="不載入／不存歷史")
    ap.add_argument("--health", action="store_true", help="檢查重排、Qdrant、嵌入／聊天模型與工作區可寫後結束（不做對話）")
    return ap.parse_args(argv)


def _hist_keep_n() -> int:
    """歷史保留則數與 chat_core.backtrace 對齊，避免兩處寫死脫鉤。」"""
    try:
        return max(2, 2 * int(_core.backtrace))
    except Exception:
        return 8


def _content_key(role: object, content: object) -> tuple:
    """去重鍵只存雜湊，不存全文，避免長問答撐大集合。」"""
    text = content if isinstance(content, str) else ("" if content is None else str(content))
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return (role, len(text), digest)


def _trim_tail_to_budget(msgs: list[dict]) -> list[dict]:
    """按 HIST_MAX_CHARS 雙預算從舊裁剪，轉調 chat_core._trim_hist 保單一真相（回新清單，不動原輸入）。"""
    buf = [dict(m) for m in msgs]  # 淺拷貝一份再裁，原輸入保持不動
    try:
        _core._trim_hist(cast(list[ChatMessage], buf))
    except Exception:
        return msgs
    return buf


_HIST_FILE_MAX_BYTES: int = 1_000_000  # 歷史檔讀取上限，被外部撐大時從空開始不硬讀


def _load_history(state: ChatState, path: Path | None = None) -> None:
    """載入上次存的問答，壞檔／巨檔當無歷史｜新手：開店先把上次的小抄拿出來。」"""
    p = path if path is not None else _hist_path()
    try:
        if not p.is_file():
            return
        if p.stat().st_size > _HIST_FILE_MAX_BYTES:
            logging.getLogger(__name__).warning("歷史檔過大（%d 位元組），已從空開始：%s", p.stat().st_size, p)
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
                    if p.stat().st_size > _HIST_FILE_MAX_BYTES:
                        logging.getLogger(__name__).warning("歷史舊檔過大，跳過合併直接覆寫：%s", p)
                    else:
                        old = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(old, list):
                            merged.extend([m for m in old if isinstance(m, dict) and isinstance(m.get("content"), str) and isinstance(m.get("role"), str)])
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
            tail = _trim_tail_to_budget(dedup[-_hist_keep_n():])  # 取最後 N 組（最新的留，舊的裁）
            for m in tail:  # 模型回覆若夾代理字會炸存檔致永遠存不了，先消毒（正常字串無影響）
                c = m.get("content", "")
                if isinstance(c, str):
                    m["content"] = c.encode("utf-8", "ignore").decode("utf-8")
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


def _ingest_target(cmd: str) -> str | None:
    """解析 /ingest 指令：非指令回 None；有參數用參數，否則用工作區根。」"""
    s = cmd.strip()
    if s != "/ingest" and not s.startswith("/ingest "):
        return None
    arg = s[len("/ingest"):].strip().strip("\"'")  # 去掉前後引號，貼上含空格路徑也不怕
    if arg:
        return arg
    return str(_core._workspace_root())


def _run_ingest_command(cmd: str) -> None:
    """執行 /ingest：就地匯入筆記，任何失敗印訊息不中斷對話。」"""
    try:
        target = _ingest_target(cmd)
        if target is None:
            return
        import rag_qdrant  # 延遲匯入：平時對話不付 Qdrant 啟動成本

        print(f"開始匯入：{target}", flush=True)
        n = rag_qdrant.ingest_folder(
            target,
            # 進度回調簽名 (已完成數, 總數, 檔名)：逐檔即時印，匯入大資料夾才知道卡在哪
            on_progress=lambda done, total, rel: print(f"[{done}/{total}] {rel}", flush=True),
        )
    except Exception as e:
        print(f"匯入失敗：{e}", flush=True)
        return
    print(f"匯入完成，共 {n} 點。", flush=True)


def _run_health_check(model: str) -> int:
    """--health：檢查重排後端、Qdrant 連線、嵌入／聊天模型與工作區可寫，回 0 全通、1 有缺（P15）。

        四路 I/O 探測並行（各帶約 5 秒逾時），輸出保持固定順序；工作區會建目錄＋寫探針檔（用完即刪），其餘唯讀。
        不進對話迴圈。
    """
    ok = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as _pool:
        # 開關判斷是便宜本地讀，主線先算；慢 I/O 四路並行，Ollama 全掛時約 5 秒收斂
        try:
            import reranker as _rr

            _rerank_off = (not _rr.RERANK_ENABLE) or _rr.RERANK_BACKEND == "none"
        except Exception:
            _rerank_off = False

        def _do_rerank() -> dict:
            """重排探測（函式內現 import，測試 mock 點不變）。"""
            from reranker import probe_availability

            return probe_availability()

        _f_rerank = None if _rerank_off else _pool.submit(_do_rerank)
        try:
            import rag_qdrant

            try:
                rag_qdrant._sync_config()
            except Exception:
                pass
            _rag_import_ok = True
        except Exception as e:
            print(f"Qdrant：不可用（{e}）", flush=True)
            ok = False
            _rag_import_ok = False

        def _do_qdrant():
            """Qdrant 收藏集列舉（含連線）。"""
            return rag_qdrant._client().get_collections()

        def _do_embed() -> bool:
            """嵌入模型探測。"""
            return rag_qdrant.probe_embed()

        def _do_chat() -> bool:
            """聊天模型探測。"""
            from ollama_shared import probe_model

            return probe_model(model, _ollama_timeout)

        try:
            import config as _cfg

            rag_on = bool(_cfg.RAG_ENABLE)
            _ollama_timeout = float(_cfg.OLLAMA_TIMEOUT)
        except Exception:
            rag_on, _ollama_timeout = True, 300.0
        _f_qdrant = _pool.submit(_do_qdrant) if _rag_import_ok else None
        _f_embed = None if not rag_on else _pool.submit(_do_embed)
        _f_chat = _pool.submit(_do_chat)

        # 依固定順序取結果印出（輸出順序不變）
        if _f_rerank is None:
            # P22：刻意停用不算故障，印狀態但不判定失敗
            print("重排：已停用（RERANK_ENABLE=0 或 RERANK_BACKEND=none）", flush=True)
        else:
            try:
                _avail_rerank = _f_rerank.result()
                print(f"重排 CrossEncoder：{'可用' if _avail_rerank.get('crossencoder') else '不可用'}", flush=True)
                print(f"重排 LLM 備援：{'可用' if _avail_rerank.get('llm') else '不可用'}", flush=True)
                if not _avail_rerank.get("crossencoder") and not _avail_rerank.get("llm"):
                    ok = False
            except Exception as e:
                print(f"重排探測失敗：{e}", flush=True)
                ok = False
        if _f_qdrant is not None:
            try:
                collections = _f_qdrant.result()
                names = [getattr(c, "name", str(c)) for c in getattr(collections, "collections", [])]  # mock 或新版欄位名變了也不炸
                url = rag_qdrant.get_config().url
                print(f"Qdrant：可用（{url}，收藏集：{', '.join(names) if names else '無'}）", flush=True)
            except Exception as e:
                print(f"Qdrant：不可用（{e}）", flush=True)
                ok = False
        if _f_embed is None:
            # P22 慣例：刻意停用不算故障，印狀態但不判定失敗
            print("嵌入模型：已停用（RAG_ENABLE=0）", flush=True)
        else:
            try:
                embed_name = rag_qdrant.get_config().embed_model
                _avail_embed = _f_embed.result()
                print(f"嵌入模型：{'可用' if _avail_embed else '不可用'}（{embed_name}）", flush=True)
                if not _avail_embed:
                    ok = False
            except Exception as e:
                print(f"嵌入模型探測失敗：{e}", flush=True)
                ok = False
        try:
            chat_ok = _f_chat.result()
            print(f"聊天模型：{'可用' if chat_ok else '不可用'}（{model}）", flush=True)
            if not chat_ok:
                ok = False
        except Exception as e:
            print(f"聊天模型探測失敗：{e}，啟動後實際對話時驗證", flush=True)
            ok = False
    try:
        root = _core._workspace_root()
        probe = root / ".health_probe"
        try:
            # 寫得進又刪得掉才算可用，只讀目錄存在不算數
            probe.write_text("ok", encoding="utf-8")
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
        print(f"工作區：可用（{root}）", flush=True)
    except Exception as e:
        print(f"工作區：不可用（{e}）", flush=True)
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    """主迴圈：不斷問「你說：」，再即時印出「小助理：」的串流回覆。」"""
    args = parse_args(argv)
    # P16：Windows 主控台／管線輸出統一走 UTF-8，避免中文變亂碼
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue] - 執行期才有
        except Exception:
            pass
    # 優化：force=True 讓重複 basicConfig 生效，避免第二入口的設定被吃掉
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if args.health:
        raise SystemExit(_run_health_check(args.model))
    model = args.model
    web_search = not args.no_search
    # 優化：獨立會話狀態，不再共用全域；--no-history 用完即丟
    state = ChatState()
    if not args.no_history:
        _load_history(state)
        # 相容舊寫法：把載入的歷史同步給全域，舊外掛讀 hist 不會空（集合比對取代 O(n²) 逐個比 dict）
        _seen = {_content_key(m.get("role"), m.get("content")) for m in _core.hist}
        for m in state.hist:
            if _content_key(m.get("role"), m.get("content")) not in _seen:
                _core.hist.append(m)
                _seen.add(_content_key(m.get("role"), m.get("content")))
    print(f"小助理已啟動（模型：{model}，上網搜尋：{'開' if web_search else '關'}），輸入 q 可離開，/ingest 可匯入筆記。", flush=True)
    # P12 體感：有 prompt_toolkit 走歷史上下鍵＋持久歷史，缺套件退化 input
    _prompt = None
    try:
        from prompt_toolkit import PromptSession as _Session  # type: ignore[import-not-found]
        from prompt_toolkit.history import FileHistory as _FileHistory  # type: ignore[import-not-found]

        _prompt = _Session(history=_FileHistory(str(_hist_path().with_suffix(".prompt_hist"))))
    except Exception:
        _prompt = None
    def _ask() -> str:
        """讀一句輸入：有 prompt_toolkit 走上下鍵歷史，缺套件退化 input。」"""
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
            if text == "/ingest" or text.startswith("/ingest "):
                _run_ingest_command(text)
                continue
            print("小助理：", end="", flush=True)  # 先印前綴再一片片串流，像直播字幕
            try:
                for reply in chat_w(sys_msg, text, search_g=True, web_search=web_search, model=model, state=state):
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
