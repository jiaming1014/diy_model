"""pytest 共用啟動：把專案根加入 sys.path。

原本 15 個測試檔各自複製同一行 sys.path.insert 樣板，容易漏寫、也難維護；
集中在這裡後，新測試檔直接 import 專案模組即可。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
