# data_loader.py
"""
加载并预处理 Rec-Tmall 三个 CSV 的简化 loader。
功能：
- 读取 CSV（自动检测关键列名）
- 建立 user_id / item_id -> 索引映射
- 将 log 按时间排序并返回一个 generator：按时间流式产出交互 (user_idx, item_idx, action_type, timestamp)
- 提供 build_initial_memory()：返回热门 item 列表（用于 cold-start）
注意：不同样本文件列名可能不同，本模块会自动尝试常见列名。
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from config import PRODUCT_CSV, LOG_CSV, REVIEW_CSV, CSV_ENCODING, ALTERNATIVE_ENCODINGS


COMMON_USER = ["user_id", "uid", "user" , "rater_uid"]
COMMON_ITEM = ["item_id", "iid", "item"]
COMMON_TIME = ["time", "timestamp", "date", "vtime"]
COMMON_ACTION = ["action", "behavior", "behavior_type", "type"]

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_read_csv(path, encodings=('gb18030','gbk','utf-8','latin1'), nrows=None):
    """
    尝试多种编码读取 CSV，遇到坏行直接跳过。
    兼容 pandas 1.3 及以上版本（包括 2.x）。
    """
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                engine="python",      # 建议显式指定
                on_bad_lines='skip',  # ✅ 新版参数
                nrows=nrows
            )
            print(f"Loaded {path} with encoding '{enc}'")
            return df
        except UnicodeDecodeError:
            print(f"Failed {path} with encoding '{enc}' (UnicodeDecodeError)")
            continue
        except Exception as e:
            print(f"Other error with {enc}: {e}")
            continue
    print(f"[WARN] All encodings failed for {path}, return empty DataFrame")
    return pd.DataFrame()


class RecTmallLoader:
    def __init__(self, product_csv, log_csv, review_csv=None):
        self.product_csv = product_csv
        self.log_csv = log_csv
        self.review_csv = review_csv

        self.products = None
        self.logs = None
        self.reviews = None

        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}

    def load(self, n_rows=None):
        # load product & log (sample files small)
        # self.products = pd.read_csv(self.product_csv, nrows=n_rows, encoding=CSV_ENCODING)
        # self.logs     = pd.read_csv(self.log_csv, nrows=n_rows, encoding=CSV_ENCODING)
        self.products = safe_read_csv(self.product_csv, nrows=n_rows)
        self.logs     = safe_read_csv(self.log_csv, nrows=n_rows)
        self.reviews = safe_read_csv(self.review_csv, nrows=n_rows)
        # if self.review_csv and os.path.exists(self.review_csv):
            # self.reviews = pd.read_csv(self.review_csv, nrows=n_rows, encoding=CSV_ENCODING)
        # detect columns
        ucol = _find_col(self.logs, COMMON_USER)
        icol = _find_col(self.logs, COMMON_ITEM)
        tcol = _find_col(self.logs, COMMON_TIME)
        acol = _find_col(self.logs, COMMON_ACTION)
        assert ucol and icol and tcol, f"无法找到 user/item/time 列，当前列：{self.logs.columns.tolist()}"
        # normalize names
        self.logs = self.logs.rename(columns={ucol:"user", icol:"item", tcol:"time"})
        if acol:
            self.logs = self.logs.rename(columns={acol:"action"})
        else:
            self.logs["action"] = 1  # 若无 action, 认为每行为一次交互

        # try convert time to datetime (some dataset time is int)
        try:
            self.logs["time"] = pd.to_datetime(self.logs["time"])
        except:
            # if numeric timestamp
            self.logs["time"] = pd.to_datetime(self.logs["time"], unit="s", errors="coerce")

        # filter NAs and sort
        self.logs = self.logs.dropna(subset=["user","item","time"])
        self.logs = self.logs.sort_values("time").reset_index(drop=True)

        # build id maps
        users = pd.unique(self.logs["user"].values)
        items = pd.unique(self.logs["item"].values)
        self.user2idx = {u:i for i,u in enumerate(users)}
        self.item2idx = {it:i for i,it in enumerate(items)}
        self.idx2item = {i:it for it,i in self.item2idx.items()}

        # map to index columns
        self.logs["user_idx"] = self.logs["user"].map(self.user2idx)
        self.logs["item_idx"] = self.logs["item"].map(self.item2idx)

    def stream_interactions(self):
        """
        按 time 顺序逐条返回交互：(user_idx, item_idx, action_value, timestamp)
        action_value: 原样返回（数值或字符串）；训练时由 agent 做映射到 reward
        """
        for _, row in self.logs.iterrows():
            yield int(row["user_idx"]), int(row["item_idx"]), row.get("action", 1), row["time"]

    def build_initial_popular_items(self, topn=100):
        # 返回按出现次数排序的 topn item idx
        counts = self.logs["item_idx"].value_counts()
        top = counts.head(topn).index.tolist()
        return top
