# check_data_files.py

import os
from config import PRODUCT_CSV, LOG_CSV, REVIEW_CSV

def get_basename(path):
    return os.path.basename(path)

def check_file(path):
    exists = os.path.exists(path)
    return exists

def list_data_dir(data_dir):
    try:
        files = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"Directory not found: {data_dir}")
        return []
    print(f"Files in directory '{data_dir}':")
    for f in files:
        print("  ", f)
    return files

def main():
    # 从 config 导出的期望路径
    exp_prod = PRODUCT_CSV
    exp_log = LOG_CSV
    exp_rev = REVIEW_CSV

    print("Expected file paths from config:")
    print("  PRODUCT_CSV:", exp_prod)
    print("  LOG_CSV:",     exp_log)
    print("  REVIEW_CSV:",  exp_rev)

    # 基本比较 basenames
    print("\nFile basenames expected:")
    print("  PRODUCT file:", get_basename(exp_prod))
    print("  LOG file:    ", get_basename(exp_log))
    print("  REVIEW file: ", get_basename(exp_rev))

    # 检查每个是否存在
    for key, path in [("PRODUCT_CSV", exp_prod), ("LOG_CSV", exp_log), ("REVIEW_CSV", exp_rev)]:
        exists = check_file(path)
        print(f"Check {key}: {'FOUND' if exists else 'MISSING'} at path: {path}")

    # 列出 data 目录
    # 假设 DATA_DIR 在 config 中定义
    from config import DATA_DIR
    print("\n---- Directory listing ----")
    list_data_dir(DATA_DIR)

if __name__ == "__main__":
    # main()
    import pandas as pd
    print(pd.__version__)

