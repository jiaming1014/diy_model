# diy_model

![CI](https://github.com/jiaming1014/diy_model/actions/workflows/ci.yml/badge.svg)

本地 Ollama 聊天 CLI＋RAG（Qdrant）＋網路搜尋的新手教學專案。
一句話：**你打一句，它回一句；會查本地筆記、會上網查資料、會記住上下文。**

## 功能特色

- **串流對話**：即時逐字輸出，支援工具呼叫（查日期／上網搜尋）
- **本地筆記 RAG**：TXT／MD／PDF／DOCX／CSV／圖片 → 切塊、嵌入、存 Qdrant、檢索
- **重排品質**：CrossEncoder（bge-reranker-v2-m3）優先、Ollama LLM 備援，缺套件自動降級
- **智慧分流**：問日期走捷徑；即時資訊直接搜網；本地可答就不打模型（P15 起單次生成）
- **降級保證**：任何外部依賴（Ollama／Qdrant／搜尋）掛掉都有備案，不會整支炸
- **快取**：搜尋結果＋查詢向量共用 LRU＋TTL 快取
- **132 個測試**：全 mock、不碰網路，CI 可跑

## 系統需求

- Python 3.12+
- [Ollama](https://ollama.com/)（本機或 Cloud）
- [Qdrant](https://qdrant.tech/)（Docker 一鍵起）

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
ollama pull nomic-embed-text   # 嵌入模型（RAG 必需）
# 聊天模型依你選用：本機如 llama3.2:1b／qwen3:8b；預設為 Ollama Cloud 的 gemma4:31b-cloud
```

## 安裝

```bash
pip install -r requirements.txt            # 核心：聊天＋檢索＋測試
pip install -r requirements-optional.txt   # 可選：PDF／DOCX／OCR／重排／tiktoken／prompt_toolkit
```

## 快速開始

```bash
# 1) 把筆記放進 notes/，再匯入 Qdrant
python rag_qdrant.py --ingest notes --progress

# 2) 開始聊天（自動載入上次歷史）
python chat_cli.py

# 3) 指定模型／關掉搜尋／不存歷史
python chat_cli.py --model llama3.2:1b
python chat_cli.py --no-search
python chat_cli.py --no-history

# 4) 健康檢查（重排後端＋Qdrant 連線）
python chat_cli.py --health

# 5) 測試查詢本地筆記
python rag_qdrant.py --query "台北天氣如何" --limit 3

# 6) RAG 品質評估（--with-model 會打真實模型）
python eval.py
python eval.py --with-model gemma4:31b-cloud
```

## 主要環境變數

完整清單（含預設值）都在 `config.py`；常用如下（改完可用 `config.refresh()` 免重啟，或重新 import）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | 聊天模型 |
| `OLLAMA_TIMEOUT` | `120` | Ollama HTTP 逾時（秒） |
| `SEARCH_MAX_RESULTS` | `3` | 每次搜網取幾筆 |
| `SEARCH_REGION` | `tw-twn` | DDGS 搜尋區域 |
| `SEARCH_CACHE_TTL` | `300` | 搜尋快取秒數 |
| `HIST_MAX_CHARS` | `6000` | 歷史預算（字數＋token 雙尺） |
| `HIST_SUMMARY_ENABLE` | `0` | 開 `1` 後，被裁的舊訊息改壓成滾動摘要 |
| `MAX_TOOL_ROUNDS` | `2` | 工具呼叫最多幾輪 |
| `RAG_ENABLE` | `1` | 設 `0` 關閉本地檢索 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 位址 |
| `QDRANT_COLLECTION` | `notes` | 收藏集名稱 |
| `QDRANT_API_KEY` | （空） | Qdrant Cloud 等需驗證時使用 |
| `EMBED_MODEL` | `nomic-embed-text` | 嵌入模型 |
| `RERANK_BACKEND` | `auto` | `auto`／`crossencoder`／`llm`／`none` |
| `QUERY_REWRITE_LLM` | `0` | 開 `1` 多打一次模型改寫檢索關鍵字 |
| `DIY_HIST_FILE` | `~/.diy_model_hist.json` | 歷史檔位置（每次讀取） |

## 測試與靜態檢查

```bash
python -m pytest -q     # 132 passed，不碰真網路
python -m ruff check .  # 靜態檢查（E/F）
python -m pyright       # 型別檢查
```

## 專案結構

```
diy_model/
├── chat_cli.py       # 命令列入口（歷史、--health、prompt_toolkit）
├── chat_core.py      # 聊天核心：串流工具流程、RAG 注入、降級
├── text_utils.py     # 純文字規則：日期／閒聊／即時判斷、查詢清洗
├── rag_qdrant.py     # 筆記匯入＋向量檢索（Qdrant）
├── reranker.py       # 重排（CrossEncoder／Ollama LLM）
├── ttl_cache.py      # 共用 LRU＋TTL 快取
├── config.py         # 全域設定唯一真相
├── eval.py           # RAG 品質評估（檢索層 MRR／模型層引用）
├── notes/            # 你的筆記放這裡
└── tests/            # 132 個測試
```

## 流程速覽

```
使用者輸入
  ├─ 日期捷徑（今天幾號）──────────────► 直接回答
  ├─ 需要時先補一次網路搜尋（本地不足／即時需求）
  ├─ 模型邊串流邊決定工具（get_today／search_web）
  │     └─ 沒叫工具的那一輪＝最終答案（不重打模型）
  └─ 工具流程失敗 → 傳統降級路徑（純問模型，附本地筆記）
```
