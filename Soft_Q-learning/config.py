# config.py

import os

# 项目中 data 文件夹路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PRODUCT_CSV = os.path.join(DATA_DIR, "sam_tianchi_2014001_rec_tmall_product.csv")
LOG_CSV     = os.path.join(DATA_DIR, "sam_tianchi_2014002_rec_tmall_log.csv")
REVIEW_CSV  = os.path.join(DATA_DIR, "sam_tianchi_2014003_rec_tmall_review.csv")

# 推荐系统 Soft Q 参数
EMBED_DIM = 64
STATE_DIM = EMBED_DIM * 2
TOPK = 32
ALPHA = 0.5
GAMMA = 0.95
LR = 5e-4
BATCH_SIZE = 128
REPLAY_SIZE = 200000
UPDATE_AFTER = 1000
UPDATES_PER_STEP = 1
POLYAK = 0.995
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# 状态 history 长度
MAX_HISTORY_LEN = 10


CSV_ENCODING = "gb18030"
ALTERNATIVE_ENCODINGS = ["gb18030", "gbk", "latin1", "cp1252", "utf-8"]