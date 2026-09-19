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
    """LRU＋TTL 快取｜新手：容量滿了踢最久沒用的，過期了直接不算數。

    P18：put 可帶單筆 ttl（失敗空結果短快取用，不帶沿用預設）；
    內建命中統計 stats()，優化是否有效看數字不憑感覺。
    """

    def __init__(self, maxsize: int, ttl: float) -> None:
        """初始化：maxsize 為容量上限，ttl 為預設過期秒數（單筆可覆寫）。」"""
        # 一個字典同時記「值、寫入時間、單筆TTL」，取代舊版兩份字典對齊的寫法
        self._data: OrderedDict[str, tuple[T, float, float | None]] = OrderedDict()
        self._lock = threading.Lock()  # 一把鎖保護 _data 與計數
        self.maxsize = max(1, int(maxsize))  # 至少留 1 格，避免 0 容量鬼打牆
        self.ttl = float(ttl)  # 預設過期秒數（單筆可覆寫）
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0  # 容量淘汰（LRU 踢人）
        self._expirations = 0  # 過期丟棄

    def _effective_ttl(self, entry_ttl: float | None) -> float:
        """單筆 TTL 優先，沒設回預設。」"""
        return entry_ttl if entry_ttl is not None else self.ttl

    def get(self, key: str, now: float | None = None) -> T | None:
        """取值：未命中或過期回 None；命中刷新順序（最近使用放尾端）。"""
        t = time.monotonic() if now is None else now  # now 可注入，測試免等真實時間
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return None
            value, ts, entry_ttl = item
            if t - ts > self._effective_ttl(entry_ttl):  # 放太久＝過期，當場刪掉
                del self._data[key]
                self._misses += 1
                self._expirations += 1
                return None
            self._data.move_to_end(key)  # 命中即刷新，排到「最近使用」端
            self._hits += 1
            return value

    def put(self, key: str, value: T, now: float | None = None, ttl: float | None = None) -> None:
        """寫入：同 key 直接覆寫；未滿不掃過期，只有滿時才清過期＋淘汰最久未用。

        P17 懶清：舊版每次 put 全掃 O(n)，小快取無感、放大明顯；改為滿時才掃。
        P18 單筆 ttl：失敗空結果短快取用（秒），不帶沿用預設。
        """
        t = time.monotonic() if now is None else now
        entry_ttl = float(ttl) if ttl is not None else None  # None＝沿用預設 TTL
        with self._lock:
            self._puts += 1
            if key in self._data:  # 同 key：直接覆寫並刷新順序
                self._data[key] = (value, t, entry_ttl)
                self._data.move_to_end(key)
                return
            if len(self._data) < self.maxsize:  # 還有空間：直接放，不掃過期（懶清）
                self._data[key] = (value, t, entry_ttl)
                self._data.move_to_end(key)
                return
            expired = [k for k, (_, ts, e_ttl) in self._data.items() if t - ts > self._effective_ttl(e_ttl)]
            self._expirations += len(expired)
            for k in expired:  # 滿了才掃一輪，順手清過期
                del self._data[k]
            self._data[key] = (value, t, entry_ttl)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:  # 仍超量：踢最久未用（LRU 前端）
                self._data.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """清空全部（測試或換設定時用），計數一併歸零。」"""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._puts = 0
            self._evictions = 0
            self._expirations = 0

    def stats(self) -> dict[str, int]:
        """回傳命中統計複本：hits／misses／puts／evictions／expirations。」"""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "puts": self._puts,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }

    def update_limits(self, maxsize: int | None = None, ttl: float | None = None) -> None:
        """調整容量與過期秒數（config.refresh 用）；容量縮小立刻淘汰到符合。"""
        with self._lock:
            if maxsize is not None:
                self.maxsize = max(1, int(maxsize))
            if ttl is not None:
                self.ttl = float(ttl)
            while len(self._data) > self.maxsize:  # 縮容量時立刻淘汰到符合
                self._data.popitem(last=False)

    def __len__(self) -> int:
        """目前快取筆數（含尚未清出的過期項）。」"""
        with self._lock:
            return len(self._data)
