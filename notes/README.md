# 筆記資料夾使用說明

- 把 `.txt` / `.md` 丟進這個資料夾即可。
- 檔名會當成來源名稱，存進 Qdrant 的 `source` 欄位。
- 新增或修改後，執行一次匯入：
  `python rag_qdrant.py --ingest notes`
- 之後直接 `python chat_cli.py` 發問，會先查本地筆記，再查網路。
