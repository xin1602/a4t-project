"""
資料前處理與特徵工程模組 - 獨立版本
無外部依賴（除標準 ML 庫）

包含：
- 資料載入
- 基礎特徵計算（120+ 個）
- 資料品質檢查
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any

# ==================== 資料品質檢查 ====================

INVALID_IP_HASHES: Set[str] = {
    'cfcd208495d565ef66e7dff9f98764da',  # "0" 的 MD5
    '', 'null', '00000000000000000000000000000000',
}

def is_valid_ip(ip_hash: Optional[str]) -> bool:
    """檢查 IP 是否有效"""
    return ip_hash and ip_hash not in INVALID_IP_HASHES

def parse_date(date_str: str) -> datetime:
    """解析日期字串"""
    try:
        if ':' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return datetime(2025, 1, 1)

def parse_hour(created_at: str) -> int:
    """解析小時"""
    try:
        return int(created_at[11:13])
    except:
        return 12

def is_night_hour(hour: int) -> int:
    """判斷是否為夜間（23:00-06:00）"""
    return 1 if (hour >= 23 or hour <= 6) else 0

# ==================== 資料載入 ====================

def load_data(base_dir: str):
    """載入所有 JSONL 資料"""
    print("載入資料...")
    
    # 載入標籤
    train_labels = {}
    with open(f'{base_dir}/train_label.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            train_labels[d['user_id']] = d['status']
    
    # 載入使用者資訊
    user_info = {}
    with open(f'{base_dir}/user_info.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            user_info[d['user_id']] = d
    
    # 載入交易資料
    def load_jsonl(filename):
        data = []
        with open(f'{base_dir}/{filename}', 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    twd_txs = load_jsonl('twd_transfer.jsonl')
    crypto_txs = load_jsonl('crypto_transfer.jsonl')
    trading_txs = load_jsonl('usdt_twd_trading.jsonl')
    swap_txs = load_jsonl('usdt_swap.jsonl')
    
    print(f"  使用者: {len(train_labels):,}, 詐欺: {sum(train_labels.values()):,}")
    print(f"  TWD 交易: {len(twd_txs):,}")
    print(f"  加密貨幣交易: {len(crypto_txs):,}")
    print(f"  交易訂單: {len(trading_txs):,}")
    print(f"  交換訂單: {len(swap_txs):,}")
    
    return train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs

# ==================== 特徵計算輔助函數 ====================

def calculate_hhi(counts: List[int]) -> float:
    """計算 HHI 集中度指標"""
    if not counts or sum(counts) == 0:
        return 0
    total = sum(counts)
    return sum((c / total) ** 2 for c in counts)

def get_concentration(amounts: List[float]) -> float:
    """計算 Top-2 集中度"""
    if not amounts or sum(amounts) == 0:
        return 0
    amounts = sorted(amounts, reverse=True)
    total = sum(amounts)
    return sum(amounts[:2]) / total if total > 0 else 0

if __name__ == "__main__":
    # 測試資料載入
    base_dir = 'D:/lxh/github/a4t-project'
    train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(base_dir)
    print("\n資料載入成功！")
