"""ttl_cache.py：執行緒安全的 LRU＋TTL 快取（全專案共用）｜新手教學版。

【這支程式在做什麼？（白話版）】
有些東西「算一次不便宜、放久了又會過期」——像搜尋結果、問題向量。
這種東西適合放進一個小抽屜：
- 常用的放前面（LRU：最久沒用的先丟）。
- 太舊的直接丟（TTL：放超過幾秒就當過期）。

這支程式就是那個抽屜，只有兩個主要動作：
- get(key)：拿東西。沒放過或放太久回 None；命中會把它移到最前面。
- put(key, value)：放東西。放滿淘汰最久沒用的，順手清掉過期的。

【為什麼要共用？】
以前 chat_core（搜尋結果）和 rag_qdrant（查詢向量）各自維護
「鍵→值」＋「鍵→時間」兩份字典再加一把鎖，同一套邏輯抄兩次，
改一處容易漏另一處。現在兩邊都改用這一個 TTLCache。

【注意】
- 執行緒安全：內部有一把鎖，多執行緒同時 get／put 不會打架。
- 值不可為 None（None 在 API 裡代表「沒命中」）。
- ttl 單位是秒；查詢向量與搜尋結果都走 time.monotonic()，不怕系統調時間。
"""

from collections import OrderedDict
import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """LRU＋TTL 快取｜新手：容量滿了踢最久沒用的，過期了直接不算數。」"""

    def __init__(self, maxsize: int, ttl: float) -> None:
        # 一個字典同時記「值」與「寫入時間」，取代舊版兩份字典對齊的寫法
        self._data: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.maxsize = max(1, int(maxsize))
        self.ttl = float(ttl)

    def get(self, key: str, now: float | None = None) -> T | None:
        """取值：未命中或過期回 None；命中刷新順序（最近使用放尾端）。"""
        t = time.monotonic() if now is None else now
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, ts = item
            if t - ts > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: T, now: float | None = None) -> None:
        """寫入：先清過期的，再放新值，超過容量淘汰最久未用。"""
        t = time.monotonic() if now is None else now
        with self._lock:
            expired = [k for k, (_, ts) in self._data.items() if t - ts > self.ttl]
            for k in expired:
                del self._data[k]
            self._data[key] = (value, t)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """清空全部（測試或換設定時用）。"""
        with self._lock:
            self._data.clear()

    def update_limits(self, maxsize: int | None = None, ttl: float | None = None) -> None:
        """調整容量與過期秒數（config.refresh 用）；容量縮小立刻淘汰到符合。"""
        with self._lock:
            if maxsize is not None:
                self.maxsize = max(1, int(maxsize))
            if ttl is not None:
                self.ttl = float(ttl)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
