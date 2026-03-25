"""
共享特徵工程模組 - 可重用版本
包含所有訓練腳本需要的特徵計算函數

使用方式：
    from shared_features import (
        load_data, 
        compute_base_features,
        compute_graph_features,
        compute_temporal_features
    )
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from tqdm import tqdm

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

def compute_time_delta_hours(start: Optional[str], end: Optional[str]) -> float:
    """計算兩個時間點之間的小時數差"""
    if not start or not end:
        return -1.0
    try:
        s = parse_date(start)
        e = parse_date(end)
        delta_sec = abs((e - s).total_seconds())
        delta_hours = delta_sec / 3600
        if delta_hours < 1/3600:
            return 0.0
        return delta_hours
    except:
        return -1.0

# ==================== 資料載入 ====================

def load_data(base_dir: str) -> Tuple[Dict, Dict, List, List, List, List]:
    """
    載入所有 JSONL 資料
    
    Returns:
        train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs
    """
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

def crypto_amount_to_twd(tx: Dict) -> float:
    """正確的加密貨幣金額計算（轉換為 TWD）"""
    return tx.get('ori_samount', 0) * tx.get('twd_srate', 1) * 1e-16

# ==================== 基礎特徵計算 ====================

def compute_base_features(
    user_ids: List[str],
    user_info: Dict,
    twd_txs: List[Dict],
    crypto_txs: List[Dict],
    trading_txs: List[Dict],
    swap_txs: List[Dict],
    verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    計算基礎特徵（120+ 個）
    
    Returns:
        X: 特徵矩陣 (n_users, n_features)
        feature_names: 特徵名稱列表
    """
    if verbose:
        print("計算基礎特徵...")
    
    # 按使用者分組交易
    twd_by_user = defaultdict(list)
    crypto_by_user = defaultdict(list)
    trading_by_user = defaultdict(list)
    swap_by_user = defaultdict(list)
    
    for tx in twd_txs:
        twd_by_user[tx['user_id']].append(tx)
    for tx in crypto_txs:
        crypto_by_user[tx['user_id']].append(tx)
    for tx in trading_txs:
        trading_by_user[tx['user_id']].append(tx)
    for tx in swap_txs:
        swap_by_user[tx['user_id']].append(tx)
    
    features_list = []
    iterator = tqdm(user_ids, desc="  計算特徵") if verbose else user_ids
    
    for uid in iterator:
        info = user_info.get(uid, {})
        user_twd = twd_by_user.get(uid, [])
        user_crypto = crypto_by_user.get(uid, [])
        user_trading = trading_by_user.get(uid, [])
        user_swap = swap_by_user.get(uid, [])
        
        feat = _compute_user_features(
            uid, info, user_twd, user_crypto, user_trading, user_swap
        )
        features_list.append(feat)
    
    # 轉換為矩陣
    feature_names = sorted(features_list[0].keys())
    X = np.array([[d.get(k, 0) for k in feature_names] for d in features_list])
    
    if verbose:
        print(f"  完成：{len(feature_names)} 個特徵")
    
    return X, feature_names

def _compute_user_features(
    user_id: str,
    user_info: Dict,
    twd_txs: List[Dict],
    crypto_txs: List[Dict],
    trading_txs: List[Dict],
    swap_txs: List[Dict]
) -> Dict[str, Any]:
    """計算單個使用者的所有特徵"""
    f = {}
    
    # 1. 人口統計特徵
    f['sex'] = user_info.get('sex', 0)
    f['age'] = user_info.get('age', 0)
    f['career'] = user_info.get('career', 0)
    f['income_source'] = user_info.get('income_source', 0)
    f['user_source'] = user_info.get('user_source', 0)
    f['has_kyc_level1'] = 1 if user_info.get('level1_finished_at') else 0
    f['has_kyc_level2'] = 1 if user_info.get('level2_finished_at') else 0
    
    # 2. 交易筆數
    twd_count = len(twd_txs)
    crypto_count = len(crypto_txs)
    trading_count = len(trading_txs)
    swap_count = len(swap_txs)
    total_count = twd_count + crypto_count + trading_count + swap_count
    
    f['twd_tx_count'] = twd_count
    f['crypto_tx_count'] = crypto_count
    f['trading_tx_count'] = trading_count
    f['swap_tx_count'] = swap_count
    f['total_tx_count'] = total_count
    f['has_crypto'] = 1 if crypto_count > 0 else 0
    
    # 3. 時序模式
    crypto_hours = [parse_hour(tx.get('created_at', '')) for tx in crypto_txs]
    crypto_night_count = sum(is_night_hour(h) for h in crypto_hours)
    f['crypto_night_tx_count'] = crypto_night_count
    f['crypto_night_tx_ratio'] = crypto_night_count / crypto_count if crypto_count > 0 else 0
    
    if crypto_hours:
        hour_dist = Counter(crypto_hours)
        f['crypto_hour_diversity'] = len(hour_dist)
        f['crypto_peak_hour'] = hour_dist.most_common(1)[0][0]
        f['crypto_peak_hour_count'] = hour_dist.most_common(1)[0][1]
    else:
        f['crypto_hour_diversity'] = 0
        f['crypto_peak_hour'] = 12
        f['crypto_peak_hour_count'] = 0
    
    twd_hours = [parse_hour(tx.get('created_at', '')) for tx in twd_txs]
    twd_night_count = sum(is_night_hour(h) for h in twd_hours)
    f['twd_night_tx_count'] = twd_night_count
    f['twd_night_tx_ratio'] = twd_night_count / twd_count if twd_count > 0 else 0
    f['total_night_tx_ratio'] = (crypto_night_count + twd_night_count) / total_count if total_count > 0 else 0
    
    # 4. 金額統計
    twd_amounts = [tx.get('ori_samount', 0) / 1e8 for tx in twd_txs]
    f['twd_total_amount'] = sum(twd_amounts)
    f['twd_avg_amount'] = np.mean(twd_amounts) if twd_amounts else 0
    f['twd_max_amount'] = max(twd_amounts) if twd_amounts else 0
    f['twd_amount_std'] = np.std(twd_amounts) if len(twd_amounts) > 1 else 0
    
    crypto_amounts_twd = [crypto_amount_to_twd(tx) for tx in crypto_txs]
    crypto_total = sum(crypto_amounts_twd)
    f['crypto_total_amount'] = crypto_total
    f['crypto_avg_amount'] = np.mean(crypto_amounts_twd) if crypto_amounts_twd else 0
    f['crypto_max_amount'] = max(crypto_amounts_twd) if crypto_amounts_twd else 0
    f['crypto_amount_std'] = np.std(crypto_amounts_twd) if len(crypto_amounts_twd) > 1 else 0
    
    trading_amounts = [tx.get('trade_samount', 0) for tx in trading_txs]
    f['trading_total_amount'] = sum(trading_amounts)
    f['trading_avg_amount'] = np.mean(trading_amounts) if trading_amounts else 0
    f['trading_max_amount'] = max(trading_amounts) if trading_amounts else 0
    
    swap_amounts_twd = [tx.get('twd_samount', 0) for tx in swap_txs]
    f['swap_total_twd_amount'] = sum(swap_amounts_twd)
    f['swap_avg_twd_amount'] = np.mean(swap_amounts_twd) if swap_amounts_twd else 0
    f['swap_max_twd_amount'] = max(swap_amounts_twd) if swap_amounts_twd else 0
    
    # 5. 流入流出特徵
    crypto_inflow = [tx for tx in crypto_txs if tx.get('kind') == 1]
    crypto_outflow = [tx for tx in crypto_txs if tx.get('kind') == 0]
    inflow_amounts = [crypto_amount_to_twd(tx) for tx in crypto_inflow]
    outflow_amounts = [crypto_amount_to_twd(tx) for tx in crypto_outflow]
    
    f['crypto_inflow_count'] = len(crypto_inflow)
    f['crypto_outflow_count'] = len(crypto_outflow)
    f['crypto_inflow_sum'] = sum(inflow_amounts)
    f['crypto_outflow_sum'] = sum(outflow_amounts)
    f['crypto_net_flow'] = f['crypto_inflow_sum'] - f['crypto_outflow_sum']
    
    flow_total = f['crypto_inflow_sum'] + f['crypto_outflow_sum'] + 1
    f['crypto_inflow_ratio'] = f['crypto_inflow_sum'] / flow_total
    f['crypto_outflow_ratio'] = f['crypto_outflow_sum'] / flow_total
    
    twd_inflow = [tx for tx in twd_txs if tx.get('kind') == 1]
    twd_outflow = [tx for tx in twd_txs if tx.get('kind') == 0]
    f['twd_inflow_count'] = len(twd_inflow)
    f['twd_outflow_count'] = len(twd_outflow)
    f['twd_inflow_sum'] = sum(tx.get('ori_samount', 0) for tx in twd_inflow)
    f['twd_outflow_sum'] = sum(tx.get('ori_samount', 0) for tx in twd_outflow)
    f['twd_net_flow'] = f['twd_inflow_sum'] - f['twd_outflow_sum']
    
    # 6. 對手方集中度
    to_wallets = [tx.get('to_wallet_hash', '') for tx in crypto_txs if tx.get('to_wallet_hash')]
    from_wallets = [tx.get('from_wallet_hash', '') for tx in crypto_txs if tx.get('from_wallet_hash')]
    
    f['crypto_to_wallet_hhi'] = calculate_hhi(list(Counter(to_wallets).values()))
    f['crypto_from_wallet_hhi'] = calculate_hhi(list(Counter(from_wallets).values()))
    f['crypto_to_wallet_diversity'] = len(set(to_wallets))
    f['crypto_from_wallet_diversity'] = len(set(from_wallets))
    
    to_wallet_counts = list(Counter(to_wallets).values())
    f['crypto_to_wallet_max_share'] = max(to_wallet_counts) / sum(to_wallet_counts) if to_wallet_counts else 0
    
    # 7. 行為穩定性
    f['crypto_amount_cv'] = np.std(crypto_amounts_twd) / (np.mean(crypto_amounts_twd) + 1e-6) if len(crypto_amounts_twd) > 1 else 0
    
    # 8. 活動標記
    if total_count == 0:
        f['activity_level'] = 0
    elif total_count <= 2:
        f['activity_level'] = 1
    elif total_count <= 5:
        f['activity_level'] = 2
    elif total_count <= 10:
        f['activity_level'] = 3
    else:
        f['activity_level'] = 4
    
    f['is_high_activity'] = 1 if total_count >= 5 else 0
    f['is_very_high_activity'] = 1 if total_count >= 10 else 0
    
    # 9. 交易類型多樣性
    tx_types = sum([twd_count > 0, crypto_count > 0, trading_count > 0, swap_count > 0])
    f['tx_type_count'] = tx_types
    f['is_multi_type_user'] = 1 if tx_types >= 3 else 0
    
    # 10. 跨類型比率
    f['swap_to_crypto_ratio'] = swap_count / crypto_count if crypto_count > 0 else 0
    f['twd_to_crypto_ratio'] = twd_count / crypto_count if crypto_count > 0 else 0
    f['trading_to_total_ratio'] = trading_count / total_count if total_count > 0 else 0
    twd_total = sum(twd_amounts)
    f['crypto_to_fiat_ratio'] = crypto_total / (twd_total + crypto_total) if (twd_total + crypto_total) > 0 else 0
    
    # 11. 外部 vs 內部
    external_crypto = sum(1 for tx in crypto_txs if tx.get('relation_user_id') is None)
    internal_crypto = crypto_count - external_crypto
    f['crypto_external_count'] = external_crypto
    f['crypto_internal_count'] = internal_crypto
    f['crypto_external_ratio'] = external_crypto / crypto_count if crypto_count else 0
    
    # 12. 交互特徵
    wallets = set()
    for tx in crypto_txs:
        wallets.add(tx.get('from_wallet_hash', ''))
        wallets.add(tx.get('to_wallet_hash', ''))
    wallets.discard('')
    
    f['high_activity_with_crypto'] = 1 if (total_count >= 5 and crypto_count > 0) else 0
    f['multi_type_with_crypto'] = 1 if (tx_types >= 3 and crypto_count > 0) else 0
    f['high_freq_trading'] = 1 if trading_count >= 5 else 0
    f['diverse_wallet_with_high_amount'] = 1 if (len(wallets) >= 5 and crypto_total > 10000) else 0
    
    return f

# ==================== 圖特徵計算 ====================

def compute_graph_features(
    user_ids: List[str],
    crypto_txs: List[Dict],
    twd_txs: List[Dict],
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], nx.Graph]:
    """
    計算圖特徵
    
    Returns:
        X_graph: 圖特徵矩陣
        graph_feature_names: 特徵名稱
        G: NetworkX 圖
    """
    if verbose:
        print("計算圖特徵...")
    
    # 建立圖
    G = nx.Graph()
    G.add_nodes_from(user_ids)
    user_set = set(user_ids)
    
    # 添加內部轉帳邊
    for tx in crypto_txs:
        if tx.get('sub_kind') == 1:
            u = tx.get('user_id')
            v = tx.get('relation_user_id')
            if u in user_set and v in user_set and u != v:
                G.add_edge(u, v)
    
    # 添加共享 IP 邊
    ip_to_users = defaultdict(set)
    for tx in twd_txs + crypto_txs:
        uid = tx.get('user_id')
        if uid in user_set:
            ip = tx.get('source_ip_hash')
            if is_valid_ip(ip):
                ip_to_users[ip].add(uid)
    
    for ip, users in ip_to_users.items():
        users = list(users)
        if 2 <= len(users) <= 20:
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    G.add_edge(users[i], users[j])
    
    if verbose:
        print(f"  圖結構: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
    
    # 計算圖特徵
    features_list = []
    iterator = tqdm(user_ids, desc="  計算圖特徵") if verbose else user_ids
    
    for uid in iterator:
        feat = {}
        
        # 度特徵
        feat['graph_degree'] = G.degree(uid) if uid in G else 0
        
        # 聚類係數
        feat['graph_clustering'] = nx.clustering(G, uid) if uid in G else 0
        
        # 連通分量大小
        if uid in G:
            component = nx.node_connected_component(G, uid)
            feat['graph_component_size'] = len(component)
        else:
            feat['graph_component_size'] = 1
        
        features_list.append(feat)
    
    # 轉換為矩陣
    graph_feature_names = sorted(features_list[0].keys())
    X_graph = np.array([[d.get(k, 0) for k in graph_feature_names] for d in features_list])
    
    if verbose:
        print(f"  完成：{len(graph_feature_names)} 個圖特徵")
    
    return X_graph, graph_feature_names, G

# ==================== 社群檢測特徵 ====================

def compute_community_features(
    user_ids: List[str],
    G: nx.Graph,
    train_labels: Optional[Dict[str, int]] = None,
    train_indices: Optional[np.ndarray] = None,
    verbose: bool = True,
    global_fraud_rate: float = 0.03
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    計算社群檢測特徵（無 data leakage）
    
    只使用圖結構特徵，不使用任何標籤資訊：
    - 方法1：結構特徵（社群大小、平均度數、密度）
    - 方法3：全局先驗概率
    
    Args:
        user_ids: 所有使用者 ID
        G: NetworkX 圖
        train_labels: 標籤字典（已忽略，僅保留參數相容性）
        train_indices: 訓練集索引（已忽略，僅保留參數相容性）
        verbose: 是否顯示進度
        global_fraud_rate: 全域詐欺率先驗（預設 0.03）
    
    Returns:
        X_community: 社群特徵矩陣
        community_feature_names: 特徵名稱
        partition: 社群分配字典
    """
    if verbose:
        print("計算社群檢測特徵（無 data leakage）...")
    
    # Louvain 社群檢測
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
    except ImportError:
        # Fallback
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        partition = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                partition[node] = comm_id
    
    # 計算社群結構特徵（不涉及標籤）
    community_sizes = Counter(partition.values())
    
    # 計算每個社群的度數統計
    community_degrees = defaultdict(list)
    for uid in user_ids:
        comm_id = partition.get(uid, -1)
        degree = G.degree(uid)
        community_degrees[comm_id].append(degree)
    
    # 計算社群內部邊數
    community_internal_edges = defaultdict(int)
    for u, v in G.edges():
        if u in partition and v in partition and partition[u] == partition[v]:
            community_internal_edges[partition[u]] += 1
    
    if verbose:
        print(f"  檢測到 {len(community_sizes)} 個社群")
    
    # 計算特徵（方法1 + 方法3）
    features_list = []
    for uid in user_ids:
        feat = {}
        
        comm_id = partition.get(uid, -1)
        size = community_sizes.get(comm_id, 1)
        
        # 方法1：結構特徵
        feat['community_id'] = comm_id
        feat['community_size'] = size
        feat['log_community_size'] = np.log1p(size)
        
        # 平均度數
        degrees = community_degrees.get(comm_id, [0])
        feat['community_avg_degree'] = np.mean(degrees)
        feat['community_max_degree'] = np.max(degrees) if degrees else 0
        feat['community_std_degree'] = np.std(degrees) if len(degrees) > 1 else 0
        
        # 社群密度
        max_edges = size * (size - 1) / 2
        internal_edges = community_internal_edges.get(comm_id, 0)
        feat['community_density'] = internal_edges / max(max_edges, 1)
        
        # 方法3：全局先驗概率
        feat['community_fraud_rate'] = global_fraud_rate
        
        features_list.append(feat)
    
    # 轉換為矩陣
    community_feature_names = sorted(features_list[0].keys())
    X_community = np.array([[d.get(k, 0) for k in community_feature_names] for d in features_list])
    
    if verbose:
        print(f"  完成：{len(community_feature_names)} 個社群特徵")
    
    return X_community, community_feature_names, partition

# ==================== 完整特徵計算 ====================

def compute_all_features(
    user_ids: List[str],
    train_labels: Optional[Dict[str, int]],
    user_info: Dict,
    twd_txs: List[Dict],
    crypto_txs: List[Dict],
    trading_txs: List[Dict],
    swap_txs: List[Dict],
    include_graph: bool = True,
    include_community: bool = True,
    train_indices: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], Optional[nx.Graph], Optional[Dict]]:
    """
    計算所有特徵（一站式函數，防止 data leakage）
    
    Args:
        train_indices: 訓練集索引（用於 CV 時只使用訓練集標籤）
    
    Returns:
        X: 完整特徵矩陣
        feature_names: 特徵名稱列表
        G: NetworkX 圖（如果 include_graph=True）
        partition: 社群分配（如果 include_community=True）
    """
    # 1. 基礎特徵
    X_base, base_names = compute_base_features(
        user_ids, user_info, twd_txs, crypto_txs, trading_txs, swap_txs, verbose
    )
    
    feature_matrices = [X_base]
    all_feature_names = base_names.copy()
    G = None
    partition = None
    
    # 2. 圖特徵
    if include_graph:
        X_graph, graph_names, G = compute_graph_features(
            user_ids, crypto_txs, twd_txs, verbose
        )
        feature_matrices.append(X_graph)
        all_feature_names.extend(graph_names)
        
        # 3. 社群特徵（防止 data leakage）
        if include_community:
            X_community, community_names, partition = compute_community_features(
                user_ids, G, train_labels, train_indices, verbose
            )
            feature_matrices.append(X_community)
            all_feature_names.extend(community_names)
    
    # 合併所有特徵
    X = np.hstack(feature_matrices)
    
    if verbose:
        print(f"\n總計：{X.shape[1]} 個特徵")
    
    return X, all_feature_names, G, partition

# ==================== 主函數（用於測試）====================

if __name__ == "__main__":
    # 測試資料載入和特徵計算
    BASE_DIR = 'D:/lxh/github/a4t-project'
    
    # 載入資料
    train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(BASE_DIR)
    user_ids = list(train_labels.keys())
    
    # 計算所有特徵（不使用標籤，防止 leakage）
    X, feature_names, G, partition = compute_all_features(
        user_ids, None, user_info,
        twd_txs, crypto_txs, trading_txs, swap_txs,
        include_graph=True,
        include_community=False  # 需要在 CV 內計算
    )
    
    print(f"\n特徵矩陣形狀: {X.shape}")
    print(f"特徵名稱範例: {feature_names[:10]}")
    print("\n測試完成！")
