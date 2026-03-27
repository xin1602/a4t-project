"""
GraphSAGE V13 GPU Complete 訓練腳本 - 使用共享特徵模組
包含 GNNExplainer 解釋器

使用方式：
    python train_graphsage_v13_gpu_complete.py

輸出：
    graphsage_v13_gpu_complete_results.pkl - 包含 OOF 預測、評估指標和解釋器結果
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')

import sys
import os

# Ensure src/ and project root are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from shared_features import (
    load_data,
    compute_all_features,
    parse_date
)
from config import get_data_dir, get_output_dir

# ==================== 配置 ====================

BASE_DIR = get_data_dir()
OUTPUT_DIR = get_output_dir()

print("="*70)
print("GraphSAGE V13 GPU Complete - 時序圖特徵 + Walk-Forward 驗證")
print("包含 GNNExplainer 解釋器")
print("="*70)

# ==================== 載入資料 ====================

train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(BASE_DIR)
user_ids = list(train_labels.keys())

print(f"\n使用者數: {len(user_ids):,}")
print(f"詐欺數: {sum(train_labels.values()):,}")

# ==================== 計算特徵（不含社群特徵）====================

print("\n計算基礎特徵和圖結構...")

X_base, feature_names, G, _ = compute_all_features(
    user_ids=user_ids,
    train_labels=None,  # 不使用標籤
    user_info=user_info,
    twd_txs=twd_txs,
    crypto_txs=crypto_txs,
    trading_txs=trading_txs,
    swap_txs=swap_txs,
    include_graph=True,
    include_community=False,  # 稍後在每個 fold 內計算
    verbose=True
)

y = np.array([train_labels[uid] for uid in user_ids])

# ==================== 計算額外特徵 (v11 的 vj_ 和 fp_ 特徵) ====================

print("\n計算額外特徵 (vj_ + fp_)...")

def crypto_amount_to_twd(tx):
    return tx.get('ori_samount', 0) * tx.get('twd_srate', 1) * 1e-16

# ==================== 圖結構時序特徵計算函數 ====================

def compute_transaction_velocity(crypto_txs, twd_txs, user_ids):
    """計算交易速度 (transactions per hour) over 1h, 6h, 24h windows"""
    all_txs = []

    for tx in crypto_txs:
        if tx['user_id'] in user_ids:
            all_txs.append({
                'user_id': tx['user_id'],
                'created_at': parse_date(tx.get('created_at', ''))
            })

    for tx in twd_txs:
        if tx['user_id'] in user_ids:
            all_txs.append({
                'user_id': tx['user_id'],
                'created_at': parse_date(tx.get('created_at', ''))
            })

    if not all_txs:
        return {uid: {'velocity_1h': 0, 'velocity_6h': 0, 'velocity_24h': 0, 'velocity_trend': 0} for uid in user_ids}

    tx_by_user = defaultdict(list)
    for tx in all_txs:
        tx_by_user[tx['user_id']].append(tx['created_at'])

    velocity_features = {}

    for uid in user_ids:
        user_txs = sorted(tx_by_user.get(uid, []))

        if len(user_txs) == 0:
            velocity_features[uid] = {
                'velocity_1h': 0,
                'velocity_6h': 0,
                'velocity_24h': 0,
                'velocity_trend': 0
            }
            continue

        last_tx = user_txs[-1]

        # Count transactions in time windows
        count_1h = sum(1 for t in user_txs if (last_tx - t).total_seconds() <= 3600)
        count_6h = sum(1 for t in user_txs if (last_tx - t).total_seconds() <= 21600)
        count_24h = sum(1 for t in user_txs if (last_tx - t).total_seconds() <= 86400)

        # Velocity = count / hours
        velocity_1h = count_1h / 1.0 if count_1h > 0 else 0
        velocity_6h = count_6h / 6.0 if count_6h > 0 else 0
        velocity_24h = count_24h / 24.0 if count_24h > 0 else 0

        # Velocity trend
        velocity_trend = velocity_1h - velocity_24h

        velocity_features[uid] = {
            'velocity_1h': velocity_1h,
            'velocity_6h': velocity_6h,
            'velocity_24h': velocity_24h,
            'velocity_trend': velocity_trend
        }

    return velocity_features

def compute_burst_detection(crypto_txs, twd_txs, user_ids, threshold_multiplier=3.0):
    """檢測交易爆發模式 - 短時間內的交易激增"""
    all_txs = []

    for tx in crypto_txs:
        if tx['user_id'] in user_ids:
            all_txs.append({
                'user_id': tx['user_id'],
                'created_at': parse_date(tx.get('created_at', ''))
            })

    for tx in twd_txs:
        if tx['user_id'] in user_ids:
            all_txs.append({
                'user_id': tx['user_id'],
                'created_at': parse_date(tx.get('created_at', ''))
            })

    if not all_txs:
        return {uid: {'burst_count': 0, 'max_burst_intensity': 0, 'burst_ratio': 0, 'time_since_last_burst': 999999} for uid in user_ids}

    tx_by_user = defaultdict(list)
    for tx in all_txs:
        tx_by_user[tx['user_id']].append(tx['created_at'])

    burst_features = {}
    burst_window_hours = 1

    for uid in user_ids:
        user_txs = sorted(tx_by_user.get(uid, []))

        if len(user_txs) < 2:
            burst_features[uid] = {
                'burst_count': 0,
                'max_burst_intensity': 0,
                'burst_ratio': 0,
                'time_since_last_burst': 999999
            }
            continue

        # Calculate average transaction rate
        time_span_hours = (user_txs[-1] - user_txs[0]).total_seconds() / 3600
        avg_rate = len(user_txs) / max(time_span_hours, 1)
        burst_threshold = avg_rate * threshold_multiplier

        # Sliding window to detect bursts
        burst_periods = []
        txs_in_bursts = 0

        for i, tx_time in enumerate(user_txs):
            window_end = tx_time + timedelta(hours=burst_window_hours)
            txs_in_window = sum(1 for t in user_txs if tx_time <= t < window_end)

            if txs_in_window >= burst_threshold and txs_in_window >= 3:
                burst_periods.append({
                    'start': tx_time,
                    'intensity': txs_in_window
                })
                txs_in_bursts += txs_in_window

        # Deduplicate overlapping burst periods
        unique_bursts = []
        if burst_periods:
            burst_periods.sort(key=lambda x: x['start'])
            current_burst = burst_periods[0]

            for burst in burst_periods[1:]:
                if (burst['start'] - current_burst['start']).total_seconds() / 3600 > burst_window_hours:
                    unique_bursts.append(current_burst)
                    current_burst = burst
                else:
                    current_burst['intensity'] = max(current_burst['intensity'], burst['intensity'])

            unique_bursts.append(current_burst)

        burst_count = len(unique_bursts)
        max_burst_intensity = max([b['intensity'] for b in unique_bursts]) if unique_bursts else 0
        burst_ratio = min(txs_in_bursts / len(user_txs), 1.0) if len(user_txs) > 0 else 0

        if unique_bursts:
            last_burst_time = max([b['start'] for b in unique_bursts])
            time_since_last_burst = (user_txs[-1] - last_burst_time).total_seconds() / 3600
        else:
            time_since_last_burst = 999999

        burst_features[uid] = {
            'burst_count': burst_count,
            'max_burst_intensity': max_burst_intensity,
            'burst_ratio': burst_ratio,
            'time_since_last_burst': min(time_since_last_burst, 999999)
        }

    return burst_features

def compute_time_decay_features(crypto_txs, user_ids, decay_rate=0.1):
    """計算時間衰減權重的圖特徵 - 近期連接權重更高"""
    # Only internal transfers
    internal_txs = [tx for tx in crypto_txs
                    if tx.get('sub_kind') == 1
                    and tx['user_id'] in user_ids
                    and tx.get('relation_user_id') in user_ids]

    if not internal_txs:
        return {uid: {'weighted_degree': 0, 'weighted_clustering': 0, 'weighted_centrality': 0} for uid in user_ids}

    # Get the latest timestamp as reference
    latest_time = max(parse_date(tx.get('created_at', '')) for tx in internal_txs)

    # Build weighted graph
    weighted_edges = defaultdict(lambda: {'weight': 0, 'neighbors': set()})

    for tx in internal_txs:
        u = tx['user_id']
        v = tx.get('relation_user_id')
        if not v:
            continue

        tx_time = parse_date(tx.get('created_at', ''))
        hours_ago = (latest_time - tx_time).total_seconds() / 3600
        decay_weight = np.exp(-decay_rate * hours_ago / 24)

        weighted_edges[u]['weight'] += decay_weight
        weighted_edges[u]['neighbors'].add(v)
        weighted_edges[v]['weight'] += decay_weight
        weighted_edges[v]['neighbors'].add(u)

    # Compute weighted features
    decay_features = {}

    for uid in user_ids:
        if uid in weighted_edges:
            weighted_degree = weighted_edges[uid]['weight']
            neighbor_count = len(weighted_edges[uid]['neighbors'])

            # Weighted clustering coefficient (simplified)
            neighbors = list(weighted_edges[uid]['neighbors'])
            if len(neighbors) >= 2:
                neighbor_connections = 0
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if n2 in weighted_edges.get(n1, {}).get('neighbors', set()):
                            neighbor_connections += 1

                max_possible = len(neighbors) * (len(neighbors) - 1) / 2
                weighted_clustering = neighbor_connections / max_possible if max_possible > 0 else 0
            else:
                weighted_clustering = 0

            weighted_centrality = weighted_degree / max(neighbor_count, 1)
        else:
            weighted_degree = 0
            weighted_clustering = 0
            weighted_centrality = 0

        decay_features[uid] = {
            'weighted_degree': weighted_degree,
            'weighted_clustering': weighted_clustering,
            'weighted_centrality': weighted_centrality
        }

    return decay_features

def compute_temporal_clustering(crypto_txs, user_ids, snapshot_windows=[7, 14, 30]):
    """計算時序聚類特徵 - 用戶連接隨時間的變化"""
    import networkx as nx

    # Only internal transfers
    internal_txs = [tx for tx in crypto_txs
                    if tx.get('sub_kind') == 1
                    and tx['user_id'] in user_ids
                    and tx.get('relation_user_id') in user_ids]

    if not internal_txs:
        return {uid: {'clustering_volatility': 0, 'neighbor_turnover_rate': 0} for uid in user_ids}

    latest_time = max(parse_date(tx.get('created_at', '')) for tx in internal_txs)

    # Create graph snapshots at different time windows
    snapshots = []
    for days_back in snapshot_windows:
        cutoff_time = latest_time - timedelta(days=days_back)
        snapshot_txs = [tx for tx in internal_txs if parse_date(tx.get('created_at', '')) >= cutoff_time]

        G = nx.Graph()
        for tx in snapshot_txs:
            u = tx['user_id']
            v = tx.get('relation_user_id')
            if v:
                G.add_edge(u, v)

        snapshots.append(G)

    # Compute temporal features
    temporal_features = {}

    for uid in user_ids:
        clustering_coeffs = []
        neighbor_sets = []

        for G in snapshots:
            if uid in G:
                cc = nx.clustering(G, uid)
                clustering_coeffs.append(cc)
                neighbors = set(G.neighbors(uid))
                neighbor_sets.append(neighbors)
            else:
                clustering_coeffs.append(0)
                neighbor_sets.append(set())

        # Clustering volatility
        if len(clustering_coeffs) > 1:
            clustering_volatility = np.std(clustering_coeffs)
        else:
            clustering_volatility = 0

        # Neighbor turnover rate
        if len(neighbor_sets) >= 2:
            turnovers = []
            for i in range(len(neighbor_sets) - 1):
                set1 = neighbor_sets[i]
                set2 = neighbor_sets[i + 1]

                if len(set1) == 0 and len(set2) == 0:
                    turnovers.append(0)
                else:
                    union = set1.union(set2)
                    intersection = set1.intersection(set2)
                    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                    turnover = 1 - jaccard
                    turnovers.append(turnover)

            neighbor_turnover_rate = np.mean(turnovers) if turnovers else 0
        else:
            neighbor_turnover_rate = 0

        temporal_features[uid] = {
            'clustering_volatility': clustering_volatility,
            'neighbor_turnover_rate': neighbor_turnover_rate
        }

    return temporal_features

def compute_neighbor_aggregation(crypto_txs, user_info, twd_txs, user_ids):
    """計算一階鄰居聚合特徵"""
    internal_txs = [tx for tx in crypto_txs
                    if tx.get('sub_kind') == 1
                    and tx['user_id'] in user_ids
                    and tx.get('relation_user_id') in user_ids]

    if not internal_txs:
        return {uid: {'neighbor_count': 0, 'avg_neighbor_deposit': 0, 'max_neighbor_deposit': 0,
                      'min_neighbor_account_age': 0, 'avg_neighbor_account_age': 0, 'total_tx_volume': 0}
                for uid in user_ids}

    # Compute user base features
    user_deposits = {}
    for tx in twd_txs:
        if tx.get('kind') == 0 and tx['user_id'] in user_ids:
            user_deposits[tx['user_id']] = user_deposits.get(tx['user_id'], 0) + tx.get('ori_samount', 0) / 1e8

    user_ages = {}
    for uid in user_ids:
        info = user_info.get(uid, {})
        confirmed_at = info.get('confirmed_at', '')
        if confirmed_at:
            age = (datetime.now() - parse_date(confirmed_at)).days
            user_ages[uid] = age
        else:
            user_ages[uid] = 0

    # Build neighbor relationships
    neighbors_by_user = defaultdict(list)
    for tx in internal_txs:
        u = tx['user_id']
        v = tx.get('relation_user_id')
        if v:
            neighbors_by_user[u].append({
                'neighbor_id': v,
                'amount': tx.get('ori_samount', 0) * tx.get('twd_srate', 1) * 1e-16
            })

    # Compute neighbor features
    neighbor_features = {}

    for uid in user_ids:
        neighbors = neighbors_by_user.get(uid, [])

        if not neighbors:
            neighbor_features[uid] = {
                'neighbor_count': 0,
                'avg_neighbor_deposit': 0,
                'max_neighbor_deposit': 0,
                'min_neighbor_account_age': 0,
                'avg_neighbor_account_age': 0,
                'total_tx_volume': 0
            }
            continue

        neighbor_ids = list(set(n['neighbor_id'] for n in neighbors))
        neighbor_deposits = [user_deposits.get(nid, 0) for nid in neighbor_ids]
        neighbor_ages = [user_ages.get(nid, 0) for nid in neighbor_ids]
        tx_volumes = [n['amount'] for n in neighbors]

        neighbor_features[uid] = {
            'neighbor_count': len(neighbor_ids),
            'avg_neighbor_deposit': np.mean(neighbor_deposits) if neighbor_deposits else 0,
            'max_neighbor_deposit': max(neighbor_deposits) if neighbor_deposits else 0,
            'min_neighbor_account_age': min(neighbor_ages) if neighbor_ages else 0,
            'avg_neighbor_account_age': np.mean(neighbor_ages) if neighbor_ages else 0,
            'total_tx_volume': sum(tx_volumes)
        }

    return neighbor_features

def compute_cluster_features(crypto_txs, user_ids, max_wallet_freq=500):
    """找出共用錢包的連通分量"""
    import networkx as nx

    withdrawals = [tx for tx in crypto_txs
                   if tx.get('kind') == 1 and tx['user_id'] in user_ids]

    if not withdrawals:
        return {uid: {'wallet_cluster_size': 0, 'is_large_cluster': 0, 'is_medium_cluster': 0} for uid in user_ids}

    # Count wallet frequencies
    wallet_counts = defaultdict(int)
    for tx in withdrawals:
        wallet = tx.get('to_wallet_hash')
        if wallet:
            wallet_counts[wallet] += 1

    # Filter safe wallets
    safe_wallets = {w for w, c in wallet_counts.items() if c <= max_wallet_freq}

    # Build graph
    G = nx.Graph()
    for tx in withdrawals:
        wallet = tx.get('to_wallet_hash')
        if wallet in safe_wallets:
            G.add_edge(tx['user_id'], wallet)

    # Find connected components
    components = list(nx.connected_components(G))

    cluster_features = {}
    for uid in user_ids:
        cluster_features[uid] = {'wallet_cluster_size': 0, 'is_large_cluster': 0, 'is_medium_cluster': 0}

    for comp in components:
        users_in_cluster = [n for n in comp if n in user_ids]
        cluster_size = len(users_in_cluster)
        for u in users_in_cluster:
            cluster_features[u] = {
                'wallet_cluster_size': cluster_size,
                'is_large_cluster': 1 if cluster_size >= 5 else 0,
                'is_medium_cluster': 1 if 3 <= cluster_size < 5 else 0
            }

    return cluster_features

def compute_centrality_features(crypto_txs, user_ids):
    """計算 PageRank 和中心度"""
    import networkx as nx

    internal_txs = [tx for tx in crypto_txs
                    if tx.get('sub_kind') == 1
                    and tx['user_id'] in user_ids
                    and tx.get('relation_user_id') in user_ids]

    if not internal_txs:
        return {uid: {'pagerank_score': 0, 'in_degree_ratio': 0, 'out_degree_ratio': 0,
                      'pagerank_log': 0, 'is_water_house': 0} for uid in user_ids}

    # Build directed graph
    DG = nx.DiGraph()

    for tx in internal_txs:
        u = tx['user_id']
        v = tx.get('relation_user_id')
        if not v:
            continue

        w = np.log1p(abs(tx.get('ori_samount', 0) * tx.get('twd_srate', 1) * 1e-16) + 1)

        if DG.has_edge(u, v):
            DG[u][v]['weight'] += w
        else:
            DG.add_edge(u, v, weight=w)

    # Compute centrality metrics
    pagerank_scores = nx.pagerank(DG, weight='weight', alpha=0.85)
    in_degree_centrality = nx.in_degree_centrality(DG)
    out_degree_centrality = nx.out_degree_centrality(DG)

    # Compute threshold for water house
    pr_values = list(pagerank_scores.values())
    pr_threshold = np.quantile(pr_values, 0.95) if pr_values else 0

    centrality_features = {}

    for uid in user_ids:
        pr = pagerank_scores.get(uid, 0)
        in_deg = in_degree_centrality.get(uid, 0)
        out_deg = out_degree_centrality.get(uid, 0)

        centrality_features[uid] = {
            'pagerank_score': pr,
            'in_degree_ratio': in_deg,
            'out_degree_ratio': out_deg,
            'pagerank_log': np.log1p(pr * 1000),
            'is_water_house': 1 if pr > pr_threshold else 0
        }

    return centrality_features

# ==================== 計算圖結構時序特徵 ====================

print("\n計算圖結構時序特徵...")

print("  [1/7] Transaction Velocity...")
velocity_dict = compute_transaction_velocity(crypto_txs, twd_txs, user_ids)

print("  [2/7] Burst Detection...")
burst_dict = compute_burst_detection(crypto_txs, twd_txs, user_ids)

print("  [3/7] Time-Decay Weighted...")
decay_dict = compute_time_decay_features(crypto_txs, user_ids)

print("  [4/7] Temporal Clustering...")
temporal_dict = compute_temporal_clustering(crypto_txs, user_ids)

print("  [5/7] Neighbor Aggregation...")
neighbor_dict = compute_neighbor_aggregation(crypto_txs, user_info, twd_txs, user_ids)

print("  [6/7] Cluster Features...")
cluster_dict = compute_cluster_features(crypto_txs, user_ids)

print("  [7/7] Centrality Features...")
centrality_dict = compute_centrality_features(crypto_txs, user_ids)

# 合併所有圖結構時序特徵
graph_temporal_features = []
for uid in user_ids:
    feats = {}
    feats.update(velocity_dict[uid])
    feats.update(burst_dict[uid])
    feats.update(decay_dict[uid])
    feats.update(temporal_dict[uid])
    feats.update(neighbor_dict[uid])
    feats.update(cluster_dict[uid])
    feats.update(centrality_dict[uid])
    graph_temporal_features.append(feats)

graph_temporal_feature_names = sorted(graph_temporal_features[0].keys())
X_graph_temporal = np.array([[d.get(k, 0) for k in graph_temporal_feature_names] for d in graph_temporal_features])

print(f"  圖結構時序特徵數: {len(graph_temporal_feature_names)}")

# ==================== 計算 vj_/fp_ 特徵 ====================

print("\n計算 vj_/fp_ 特徵...")

# 分組交易
from collections import defaultdict
twd_by_user = defaultdict(list)
crypto_by_user = defaultdict(list)
trading_by_user = defaultdict(list)
swap_by_user = defaultdict(list)

for tx in twd_txs:
    if tx['user_id'] in train_labels:
        twd_by_user[tx['user_id']].append(tx)
for tx in crypto_txs:
    if tx['user_id'] in train_labels:
        crypto_by_user[tx['user_id']].append(tx)
for tx in trading_txs:
    if tx['user_id'] in train_labels:
        trading_by_user[tx['user_id']].append(tx)
for tx in swap_txs:
    if tx['user_id'] in train_labels:
        swap_by_user[tx['user_id']].append(tx)

# 計算所有額外特徵
extra_features = []

for uid in tqdm(user_ids, desc="  計算 vj_/fp_ 特徵"):
    info = user_info.get(uid, {})
    twd_list = twd_by_user.get(uid, [])
    crypto_list = crypto_by_user.get(uid, [])
    trading_list = trading_by_user.get(uid, [])
    swap_list = swap_by_user.get(uid, [])

    feats = {}

    # Victim journey 特徵
    confirmed_at = info.get('confirmed_at', '')
    all_txs = twd_list + crypto_list + trading_list + swap_list

    if all_txs and confirmed_at:
        first_tx_time = min(parse_date(tx.get('created_at', '')) for tx in all_txs)
        reg_time = parse_date(confirmed_at)
        reg_to_first_tx_hours = abs((first_tx_time - reg_time).total_seconds() / 3600)
        feats['vj_reg_to_first_tx_hours'] = reg_to_first_tx_hours
        feats['vj_first_tx_within_24h'] = 1 if reg_to_first_tx_hours <= 24 else 0
        feats['vj_first_tx_within_1h'] = 1 if reg_to_first_tx_hours <= 1 else 0
    else:
        feats['vj_reg_to_first_tx_hours'] = -1
        feats['vj_first_tx_within_24h'] = 0
        feats['vj_first_tx_within_1h'] = 0

    twd_deposits = [tx for tx in twd_list if tx.get('kind') == 0]
    if twd_deposits and crypto_list:
        first_deposit_time = min(parse_date(tx.get('created_at', '')) for tx in twd_deposits)
        external_withdrawals = [tx for tx in crypto_list
                                if (tx.get('sub_kind') == 0 or tx.get('sub_kind') is None)
                                and tx.get('kind') == 1]
        if external_withdrawals:
            first_withdraw_time = min(parse_date(tx.get('created_at', '')) for tx in external_withdrawals)
            deposit_to_withdraw_hours = abs((first_withdraw_time - first_deposit_time).total_seconds() / 3600)
            feats['vj_deposit_to_withdraw_hours'] = deposit_to_withdraw_hours
            feats['vj_withdraw_within_1h'] = 1 if deposit_to_withdraw_hours <= 1 else 0
            feats['vj_withdraw_within_24h'] = 1 if deposit_to_withdraw_hours <= 24 else 0
        else:
            feats['vj_deposit_to_withdraw_hours'] = -1
            feats['vj_withdraw_within_1h'] = 0
            feats['vj_withdraw_within_24h'] = 0
    else:
        feats['vj_deposit_to_withdraw_hours'] = -1
        feats['vj_withdraw_within_1h'] = 0
        feats['vj_withdraw_within_24h'] = 0

    twd_deposit_total = sum(tx.get('ori_samount', 0) / 1e8 for tx in twd_deposits)
    crypto_withdrawals = [tx for tx in crypto_list
                          if (tx.get('sub_kind') == 0 or tx.get('sub_kind') is None)
                          and tx.get('kind') == 1]
    crypto_withdraw_total = sum(crypto_amount_to_twd(tx) for tx in crypto_withdrawals)

    if twd_deposit_total > 0:
        feats['vj_fiat_to_crypto_ratio'] = crypto_withdraw_total / (twd_deposit_total + crypto_withdraw_total + 1e-6)
        feats['vj_high_conversion_ratio'] = 1 if crypto_withdraw_total / (twd_deposit_total + 1e-6) > 0.8 else 0
    else:
        feats['vj_fiat_to_crypto_ratio'] = 0
        feats['vj_high_conversion_ratio'] = 0

    has_trading = len(trading_list) > 0
    has_swap = len(swap_list) > 0
    total_tx = len(all_txs)

    if total_tx > 0:
        feats['vj_trading_tx_ratio'] = len(trading_list) / total_tx
        feats['vj_swap_tx_ratio'] = len(swap_list) / total_tx
        feats['vj_no_trading'] = 1 if (not has_trading and not has_swap) else 0
    else:
        feats['vj_trading_tx_ratio'] = 0
        feats['vj_swap_tx_ratio'] = 0
        feats['vj_no_trading'] = 0

    if all_txs:
        tx_times = [parse_date(tx.get('created_at', '')) for tx in all_txs]
        account_span_hours = abs(max(tx_times) - min(tx_times)).total_seconds() / 3600
        feats['vj_account_span_hours'] = account_span_hours
        feats['vj_short_lifespan_24h'] = 1 if account_span_hours <= 24 else 0
        feats['vj_short_lifespan_1week'] = 1 if account_span_hours <= 168 else 0
    else:
        feats['vj_account_span_hours'] = 0
        feats['vj_short_lifespan_24h'] = 0
        feats['vj_short_lifespan_1week'] = 0

    inflow = twd_deposit_total
    outflow = crypto_withdraw_total

    if inflow > 0:
        feats['vj_one_way_flow_ratio'] = outflow / (inflow + outflow + 1e-6)
        feats['vj_all_money_out'] = 1 if outflow >= inflow * 0.9 else 0
    else:
        feats['vj_one_way_flow_ratio'] = 0
        feats['vj_all_money_out'] = 0

    external_wallets = set()
    for tx in crypto_list:
        if (tx.get('sub_kind') == 0 or tx.get('sub_kind') is None):
            if tx.get('kind') == 1:
                wallet = tx.get('to_wallet_hash')
                if wallet:
                    external_wallets.add(wallet)
            elif tx.get('kind') == 0:
                wallet = tx.get('from_wallet_hash')
                if wallet:
                    external_wallets.add(wallet)

    feats['vj_external_wallet_count'] = len(external_wallets)
    feats['vj_single_external_wallet'] = 1 if len(external_wallets) <= 1 else 0
    feats['vj_few_external_wallets'] = 1 if len(external_wallets) <= 3 else 0

    if crypto_withdrawals:
        round_count = 0
        for tx in crypto_withdrawals:
            amount_twd = crypto_amount_to_twd(tx)
            if amount_twd > 1000 and abs(amount_twd / 1000 - round(amount_twd / 1000)) < 0.01:
                round_count += 1
        feats['vj_round_withdrawal_ratio'] = round_count / len(crypto_withdrawals) if crypto_withdrawals else 0
    else:
        feats['vj_round_withdrawal_ratio'] = 0

    feats['vj_total_tx_count'] = total_tx
    feats['vj_low_tx_count'] = 1 if total_tx <= 5 else 0

    # KYC 特徵
    lvl1_at = info.get('level1_finished_at')
    if confirmed_at and lvl1_at:
        try:
            from feature_pool_system.pro_aml_features_v3 import compute_time_delta_hours
            email_to_lvl1 = compute_time_delta_hours(confirmed_at, lvl1_at)
            feats['vj_kyc_email_to_lvl1_hours'] = email_to_lvl1
            feats['vj_fast_kyc'] = 1 if email_to_lvl1 < 1 else 0
        except:
            feats['vj_kyc_email_to_lvl1_hours'] = -1
            feats['vj_fast_kyc'] = 0
    else:
        feats['vj_kyc_email_to_lvl1_hours'] = -1
        feats['vj_fast_kyc'] = 0

    # IP 多樣性
    ip_set = set()
    for tx in twd_list:
        ip = tx.get('source_ip_hash')
        if ip and ip not in {'cfcd208495d565ef66e7dff9f98764da', '', 'null', '00000000000000000000000000000000'}:
            ip_set.add(ip)
    for tx in crypto_list:
        ip = tx.get('source_ip_hash')
        if ip and ip not in {'cfcd208495d565ef66e7dff9f98764da', '', 'null', '00000000000000000000000000000000'}:
            ip_set.add(ip)

    feats['vj_ip_diversity'] = len(ip_set)
    feats['vj_single_ip'] = 1 if len(ip_set) <= 1 else 0

    # Fraud pattern 特徵
    if all_txs:
        all_txs_sorted = sorted(all_txs, key=lambda x: parse_date(x.get('created_at', '')))
        tx_times = [parse_date(tx.get('created_at', '')) for tx in all_txs_sorted]
        min_time = min(tx_times)
        max_time = max(tx_times)
        time_span_hours = max((max_time - min_time).total_seconds() / 3600, 1)

        feats['fp_tx_velocity_1h'] = min(len(all_txs) / time_span_hours, 100)
        feats['fp_tx_velocity_6h'] = min(len(all_txs) / max(time_span_hours / 6, 1), 100)
        feats['fp_tx_velocity_24h'] = min(len(all_txs) / max(time_span_hours / 24, 1), 100)

        total_amount = sum(
            (tx.get('ori_samount', 0) / 1e8 if 'twd' in str(tx.get('currency', 'twd')).lower()
             else crypto_amount_to_twd(tx))
            for tx in all_txs
        )
        feats['fp_amount_velocity_1h'] = min(total_amount / max(time_span_hours, 1), 100000)
        feats['fp_amount_velocity_6h'] = min(total_amount / max(time_span_hours / 6, 1), 100000)
        feats['fp_amount_velocity_24h'] = min(total_amount / max(time_span_hours / 24, 1), 100000)
    else:
        feats['fp_tx_velocity_1h'] = 0
        feats['fp_tx_velocity_6h'] = 0
        feats['fp_tx_velocity_24h'] = 0
        feats['fp_amount_velocity_1h'] = 0
        feats['fp_amount_velocity_6h'] = 0
        feats['fp_amount_velocity_24h'] = 0

    if crypto_withdrawals:
        withdraw_amounts = [crypto_amount_to_twd(tx) for tx in crypto_withdrawals]
        max_withdraw = max(withdraw_amounts)
        total_withdraw = sum(withdraw_amounts)
        feats['fp_max_withdrawal_ratio'] = max_withdraw / max(total_withdraw, 1e-6)
    else:
        feats['fp_max_withdrawal_ratio'] = 0

    # 連續提領
    consecutive_count = 0
    max_consecutive = 0
    if all_txs:
        all_txs_sorted = sorted(all_txs, key=lambda x: parse_date(x.get('created_at', '')))
        for tx in all_txs_sorted:
            is_withdraw = tx in crypto_list and tx.get('kind') == 1
            if is_withdraw:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
    feats['fp_consecutive_withdrawals'] = min(max_consecutive, 10) / 10

    # 小額/大額模式
    amounts = []
    if all_txs:
        all_txs_sorted = sorted(all_txs, key=lambda x: parse_date(x.get('created_at', '')))[:10]
        amounts = [
            (tx.get('ori_samount', 0) / 1e8 if 'twd' in str(tx.get('currency', 'twd')).lower()
             else crypto_amount_to_twd(tx))
            for tx in all_txs_sorted
        ]
    if len(amounts) >= 3:
        small_count = sum(1 for a in amounts if a < 1000)
        large_count = sum(1 for a in amounts if a > 5000)
        feats['fp_small_large_pattern'] = 1 if (small_count > 0 and large_count > 0) else 0
    else:
        feats['fp_small_large_pattern'] = 0

    # 整數金額比例
    round_count = 0
    for tx in crypto_withdrawals:
        amt = crypto_amount_to_twd(tx)
        if amt > 100 and abs(amt / 1000 - round(amt / 1000)) < 0.01:
            round_count += 1
    feats['fp_round_number_ratio'] = round_count / max(len(crypto_withdrawals), 1)

    # 夜間/週末活動
    if all_txs:
        night_txs = [tx for tx in all_txs if 0 <= parse_date(tx.get('created_at', '')).hour < 6]
        feats['fp_night_activity_ratio'] = len(night_txs) / len(all_txs)

        weekend_txs = [tx for tx in all_txs if parse_date(tx.get('created_at', '')).weekday() >= 5]
        feats['fp_weekend_activity_ratio'] = len(weekend_txs) / len(all_txs)

        tx_times = [parse_date(tx.get('created_at', '')) for tx in all_txs]
        if len(tx_times) >= 3:
            intervals = [(tx_times[i+1] - tx_times[i]).total_seconds() / 60 for i in range(len(tx_times)-1)]
            short_intervals = sum(1 for i in intervals if i < 5)
            feats['fp_tx_burst_score'] = short_intervals / max(len(intervals), 1)
        else:
            feats['fp_tx_burst_score'] = 0
    else:
        feats['fp_night_activity_ratio'] = 0
        feats['fp_weekend_activity_ratio'] = 0
        feats['fp_tx_burst_score'] = 0

    # 每日多錢包
    wallets_by_day = defaultdict(set)
    for tx in crypto_list:
        day = parse_date(tx.get('created_at', '')).date()
        wallet = tx.get('to_wallet_hash') or tx.get('from_wallet_hash')
        if wallet:
            wallets_by_day[day].add(wallet)
    max_wallets_day = max((len(wallets) for wallets in wallets_by_day.values()), default=0)
    feats['fp_multiple_wallets_same_day'] = min(max_wallets_day, 5) / 5

    # IP 變換
    ips_by_day = defaultdict(set)
    for tx in all_txs:
        ip = tx.get('source_ip_hash')
        day = parse_date(tx.get('created_at', '')).date()
        if ip and ip not in {'cfcd208495d565ef66e7dff9f98764da', '', 'null', '00000000000000000000000000000000'}:
            ips_by_day[day].add(ip)
    max_ips_day = max((len(ips) for ips in ips_by_day.values()), default=0)
    feats['fp_ip_switch_count'] = min(max_ips_day, 5) / 5

    # 金額異常
    if len(amounts) >= 5:
        median_amt = np.median(amounts)
        spike_count = sum(1 for a in amounts if a > median_amt * 10)
        feats['fp_amount_spike'] = spike_count / len(amounts)
    else:
        feats['fp_amount_spike'] = 0

    # 提領立即性
    if twd_deposits and crypto_withdrawals:
        deposit_time = parse_date(twd_deposits[0].get('created_at', ''))
        withdraw_time = parse_date(crypto_withdrawals[0].get('created_at', ''))
        immediacy = (withdraw_time - deposit_time).total_seconds() / 3600
        feats['fp_withdrawal_immediacy'] = 1 if immediacy <= 1 else (1 / max(immediacy, 1))
    else:
        feats['fp_withdrawal_immediacy'] = 0

    # 快速轉換
    if twd_deposits and crypto_withdrawals:
        total_deposit = sum(tx.get('ori_samount', 0) / 1e8 for tx in twd_deposits)
        total_withdraw = sum(crypto_amount_to_twd(tx) for tx in crypto_withdrawals)
        if total_deposit > 0:
            feats['fp_rapid_conversion'] = total_withdraw / total_deposit
        else:
            feats['fp_rapid_conversion'] = 0
    else:
        feats['fp_rapid_conversion'] = 0

    extra_features.append(feats)

# 轉換為矩陣
extra_feature_names = sorted(extra_features[0].keys())
X_extra = np.array([[d.get(k, 0) for k in extra_feature_names] for d in extra_features])

print(f"  額外特徵數: {len(extra_feature_names)}")

# 合併基礎特徵、圖結構時序特徵和額外特徵
X_base = np.hstack([X_base, X_graph_temporal, X_extra])
feature_names = feature_names + graph_temporal_feature_names + extra_feature_names

print(f"  總特徵數: {len(feature_names)}")
print(f"    - 基礎特徵: {len(feature_names) - len(graph_temporal_feature_names) - len(extra_feature_names)}")
print(f"    - 圖結構時序特徵: {len(graph_temporal_feature_names)}")
print(f"    - vj_/fp_ 特徵: {len(extra_feature_names)}")

# ==================== 計算時序排序 ====================

print("\n計算時序排序（t_obs）...")

# OPTIMIZED: Pre-group transactions by user_id once, then lookup
# This is O(n_transactions) instead of O(n_users * n_transactions)
from collections import defaultdict

# Create set once instead of recreating for every transaction
user_id_set = set(user_ids)

twd_times_by_user = defaultdict(list)
crypto_times_by_user = defaultdict(list)

for tx in twd_txs:
    if tx['user_id'] in user_id_set:
        twd_times_by_user[tx['user_id']].append(parse_date(tx.get('created_at', '')))
for tx in crypto_txs:
    if tx['user_id'] in user_id_set:
        crypto_times_by_user[tx['user_id']].append(parse_date(tx.get('created_at', '')))

user_t_obs = {}
for uid in tqdm(user_ids, desc="  計算 t_obs"):
    times = twd_times_by_user.get(uid, []) + crypto_times_by_user.get(uid, [])
    user_t_obs[uid] = max(times) if times else datetime(2025, 1, 1)

# 按時間排序
temporal_indices = sorted(range(len(user_ids)), key=lambda i: user_t_obs[user_ids[i]])
user_ids_sorted = [user_ids[i] for i in temporal_indices]
X_base_sorted = X_base[temporal_indices]
y_sorted = y[temporal_indices]

print(f"  時間範圍: {min(user_t_obs.values())} 到 {max(user_t_obs.values())}")

# ==================== 時序圖重建函數 ====================

def build_temporal_graph(crypto_txs, twd_txs, cutoff_time, user_ids_sorted):
    """
    根據截止時間建立時序圖

    Args:
        crypto_txs: 所有加密貨幣交易
        twd_txs: 所有 TWD 交易
        cutoff_time: 截止時間（驗證集開始時間）
        user_ids_sorted: 排序後的使用者 ID 列表

    Returns:
        G: NetworkX 圖（只包含截止時間之前的邊）
        edge_index: PyTorch Geometric 格式的邊索引
    """
    from collections import defaultdict

    G = nx.Graph()
    user_set = set(user_ids_sorted)
    G.add_nodes_from(user_ids_sorted)

    # 篩選截止時間之前的交易
    valid_crypto = [tx for tx in crypto_txs
                   if parse_date(tx.get('created_at', '')) <= cutoff_time]
    valid_twd = [tx for tx in twd_txs
                if parse_date(tx.get('created_at', '')) <= cutoff_time]

    # 添加內部轉帳邊（加密貨幣）
    for tx in valid_crypto:
        if tx.get('sub_kind') == 1:
            u = tx.get('user_id')
            v = tx.get('relation_user_id')
            if u in user_set and v in user_set and u != v:
                G.add_edge(u, v)

    # 添加共享 IP 邊（TWD + 加密貨幣）
    ip_to_users = defaultdict(set)
    for tx in valid_twd + valid_crypto:
        uid = tx.get('user_id')
        if uid in user_set:
            ip = tx.get('source_ip_hash')
            if ip and ip not in {'cfcd208495d565ef66e7dff9f98764da', '', 'null', '00000000000000000000000000000000'}:
                ip_to_users[ip].add(uid)

    for ip, users in ip_to_users.items():
        users = list(users)
        if 2 <= len(users) <= 20:
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    G.add_edge(users[i], users[j])

    # 轉換為 PyTorch Geometric 格式
    edge_list = []
    user_to_idx = {uid: i for i, uid in enumerate(user_ids_sorted)}

    for u, v in G.edges():
        if u in user_to_idx and v in user_to_idx:
            edge_list.append([user_to_idx[u], user_to_idx[v]])
            edge_list.append([user_to_idx[v], user_to_idx[u]])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Self-loops
        edge_index = torch.stack([
            torch.arange(len(user_ids_sorted)),
            torch.arange(len(user_ids_sorted))
        ]).long()

    return G, edge_index


# ==================== GraphSAGE 模型 ====================

class SAGE_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=4, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Use mean aggregation for better heterogeneous feature mixing
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = nn.Dropout(dropout)
        # Skip connection from input projection
        self.final = nn.Linear(hidden_channels * 2, 1)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        # Input projection with residual
        x_input = self.input_bn(F.gelu(self.input_proj(x)))

        x_hidden = x_input
        for i, conv in enumerate(self.convs):
            x_new = conv(x_hidden, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection (match dimensions if needed)
            if x_new.size(-1) == x_hidden.size(-1):
                x_hidden = x_hidden + x_new  # Residual
            else:
                x_hidden = x_new

        # Concatenate input features and final hidden state (JK-like)
        x_out = torch.cat([x_input, x_hidden], dim=-1)

        return self.final(x_out).squeeze()

# ==================== Walk-Forward 驗證 ====================

print("\n" + "="*70)
print("Walk-Forward 驗證 (TimeSeriesSplit)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

tscv = TimeSeriesSplit(n_splits=5)
oof_predictions = np.zeros(len(y_sorted))
fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_base_sorted), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5")
    print(f"{'='*60}")

    # 計算驗證集開始時間（第一個驗證樣本的 t_obs）
    val_start_time = user_t_obs[user_ids_sorted[val_idx[0]]]
    print(f"  驗證集開始時間: {val_start_time}")

    # 建立時序圖（只包含 val_start_time 之前的邊）
    print("  建立時序圖...")
    G_fold, edge_index = build_temporal_graph(
        crypto_txs, twd_txs, val_start_time, user_ids_sorted
    )
    print(f"    圖邊數: {edge_index.shape[1]:,}")

    # 計算社群特徵（只在訓練集節點上，確保驗證集不參與社群檢測）
    print("  計算社群特徵（只在訓練集節點上）...")
    from shared_features import compute_community_features

    # 只對訓練集節點進行社群檢測
    train_nodes = [user_ids_sorted[i] for i in train_idx]
    G_train = G_fold.subgraph(train_nodes)

    X_community, community_names, _ = compute_community_features(
        user_ids=train_nodes,
        G=G_train,
        train_labels=train_labels,
        train_indices=np.array([i for i in range(len(train_idx))]),
        verbose=False
    )

    # 建立完整的社群特徵矩陣（訓練集有值，驗證集為 0）
    X_community_full = np.zeros((len(user_ids_sorted), X_community.shape[1]))
    X_community_full[train_idx] = X_community

    # 合併特徵
    X_sorted = np.hstack([X_base_sorted, X_community_full])
    all_feature_names = feature_names + community_names

    X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
    y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

    print(f"  訓練集: {len(train_idx):,} | 驗證集: {len(val_idx):,}")
    print(f"  訓練詐欺率: {y_train.mean():.4f} | 驗證詐欺率: {y_val.mean():.4f}")
    print(f"  特徵數: {X_sorted.shape[1]}")

    # 標準化（只在訓練集上 fit，防止 data leakage）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Inductive Learning: 建立兩個版本的特徵矩陣
    # 1. 訓練版本：驗證集節點特徵設為 0（訓練時使用）
    X_train_only = np.zeros_like(X_sorted)
    X_train_only[train_idx] = X_train_scaled

    # 2. 完整版本：包含所有節點特徵（驗證時使用）
    X_full = np.zeros_like(X_sorted)
    X_full[train_idx] = X_train_scaled
    X_full[val_idx] = X_val_scaled

    # 轉換為 PyTorch
    X_train_only_t = torch.FloatTensor(X_train_only).to(device)
    X_full_t = torch.FloatTensor(X_full).to(device)
    y_t = torch.FloatTensor(y_sorted).to(device)
    edge_index_t = edge_index.to(device)

    train_idx_t = torch.LongTensor(train_idx).to(device)
    val_idx_t = torch.LongTensor(val_idx).to(device)

    # 訓練
    model = SAGE_Model(
        in_channels=X_full.shape[1],
        hidden_channels=384,
        num_layers=4,
        dropout=0.25
    ).to(device)

    # More aggressive class weights for imbalanced data (fraud ~3%)
    fraud_rate = y_train.sum() / len(y_train)
    # Use sqrt scaling for more aggressive weighting
    pos_weight = torch.tensor([(1 - fraud_rate) / (fraud_rate + 1e-6) ** 0.5]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # OneCycleLR for faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.02,
        epochs=150,
        steps_per_epoch=1,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Early stopping 變數
    best_prauc = 0
    patience_counter = 0
    patience = 15  # 15 個 epoch 沒有改善則停止
    best_model_state = None

    for epoch in range(150):
        model.train()
        optimizer.zero_grad()

        # Inductive Learning: 訓練時使用 X_train_only_t（驗證集特徵為 0）
        output = model(X_train_only_t, edge_index_t)

        # 只取訓練集節點的輸出來計算 loss
        train_output = output[train_idx_t]
        train_labels_tensor = y_t[train_idx_t]

        loss = criterion(train_output, train_labels_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 驗證時使用 X_full_t（包含驗證集特徵）
                val_output = model(X_full_t, edge_index_t)
                val_probs = torch.sigmoid(val_output[val_idx_t]).cpu().numpy()

                pr_auc = average_precision_score(y_val, val_probs)
                roc_auc = roc_auc_score(y_val, val_probs)

                # Early stopping 邏輯：監控驗證集 PR-AUC
                if pr_auc > best_prauc:
                    best_prauc = pr_auc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 最終驗證
    model.eval()
    with torch.no_grad():
        final_output = model(X_full_t, edge_index_t)
        val_probs = torch.sigmoid(final_output[val_idx_t]).cpu().numpy()

        pr_auc = average_precision_score(y_val, val_probs)
        roc_auc = roc_auc_score(y_val, val_probs)

        # Find optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_threshold_idx]
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5

    # 保存 OOF 預測
    oof_predictions[val_idx] = val_probs

    fold_results.append({
        'fold': fold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'best_f1': best_f1,
        'best_threshold': best_threshold
    })

    print(f"  Fold {fold} 結果:")
    print(f"    PR-AUC:  {pr_auc:.4f}")
    print(f"    ROC-AUC: {roc_auc:.4f}")
    print(f"    F1:      {best_f1:.4f} (threshold={best_threshold:.4f})")

    # 保存最後一個 fold 的模型和數據，用於解釋器
    if fold == 5:
        best_model_state_for_explain = best_model_state.copy() if best_model_state is not None else model.state_dict().copy()
        X_full_for_explain = X_full.copy()  # numpy array use .copy()
        edge_index_for_explain = edge_index.clone()  # PyTorch tensor use .clone()

    # GPU 記憶體清理
    del model, optimizer, scheduler, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

mean_pr = np.mean([r['pr_auc'] for r in fold_results])
mean_roc = np.mean([r['roc_auc'] for r in fold_results])
mean_f1 = np.mean([r['best_f1'] for r in fold_results])

print(f"\n平均指標:")
print(f"  PR-AUC:  {mean_pr:.4f}")
print(f"  ROC-AUC: {mean_roc:.4f}")
print(f"  F1:      {mean_f1:.4f}")

# 整體 OOF 評估
overall_pr = average_precision_score(y_sorted, oof_predictions)
overall_roc = roc_auc_score(y_sorted, oof_predictions)

print(f"\n整體 OOF 結果:")
print(f"  PR-AUC:  {overall_pr:.4f}")
print(f"  ROC-AUC: {overall_roc:.4f}")

# ==================== 模型解釋 (GNNExplainer + Integrated Gradients) ====================

print("\n" + "="*70)
print("生成模型解釋 (GNNExplainer + Integrated Gradients)")
print("="*70)

# 儲存結果
results = {
    'model': 'GraphSAGE_V13_GPU_Complete',
    'fold_results': fold_results,
    'oof_predictions': oof_predictions,
    'y_true': y_sorted,
    'user_ids': user_ids_sorted,
    'feature_names': all_feature_names,  # 使用最後一個 fold 的特徵名稱
    'mean_pr_auc': mean_pr,
    'mean_roc_auc': mean_roc,
    'mean_f1': mean_f1,
    'overall_pr_auc': overall_pr,
    'overall_roc_auc': overall_roc,
    'gpu_used': torch.cuda.is_available(),
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
}

# 只在最後一個 fold 的模型上生成解釋
# 找出高風險用戶（優先使用 ensemble 排名，fallback 到 graphsage oof）
top_k = 50

# Try to load ensemble results for better ranking
try:
    ensemble_path = os.path.join(OUTPUT_DIR, 'ensemble_results.pkl')
    if os.path.exists(ensemble_path):
        with open(ensemble_path, 'rb') as f:
            ens_data = pickle.load(f)
        ens_uids = ens_data.get('user_ids', [])
        ens_preds = ens_data.get('oof_predictions', np.array([]))
        # Map ensemble user_ids to sorted indices
        uid_to_sorted_idx = {uid: i for i, uid in enumerate(user_ids_sorted)}
        ens_top_indices = np.argsort(ens_preds)[-top_k:][::-1]
        high_risk_indices = []
        for eidx in ens_top_indices:
            euid = ens_uids[eidx]
            if euid in uid_to_sorted_idx:
                high_risk_indices.append(uid_to_sorted_idx[euid])
        high_risk_indices = np.array(high_risk_indices[:top_k])
        print(f"\n使用 ensemble 排名選取 TOP-{len(high_risk_indices)} 高風險用戶")
    else:
        high_risk_indices = np.argsort(oof_predictions)[-top_k:][::-1]
        print(f"\n使用 graphsage oof 排名選取 TOP-{top_k} 高風險用戶")
except Exception as e:
    high_risk_indices = np.argsort(oof_predictions)[-top_k:][::-1]
    print(f"\nFallback 到 graphsage oof 排名: {e}")

print(f"\n為 TOP-{top_k} 高風險用戶生成解釋...")

# 重新創建最後一個 fold 的模型，用於解釋器
print("  重新載入模型...")
model_for_explain = SAGE_Model(
    in_channels=X_full_for_explain.shape[1],
    hidden_channels=384,
    num_layers=4,
    dropout=0.25
).to(device)
model_for_explain.load_state_dict(best_model_state_for_explain)
model_for_explain.eval()

X_full_t = torch.FloatTensor(X_full_for_explain).to(device)
edge_index_t = edge_index_for_explain.to(device)

# 初始化 GNNExplainer (GPU 優化)
try:
    from torch_geometric.explain import Explainer, GNNExplainer

    # 確保模型在正確的設備上
    model_device = device if torch.cuda.is_available() else torch.device('cpu')
    model_explain = model_for_explain.to(model_device).eval()

    gnn_explainer = Explainer(
        model=model_explain,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    gnn_explanations = {}

    # 批次處理以節省記憶體
    batch_size = 5
    for batch_start in range(0, len(high_risk_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(high_risk_indices))
        batch_indices = high_risk_indices[batch_start:batch_end]

        for idx in batch_indices:
            uid = user_ids_sorted[idx]
            try:
                explanation = gnn_explainer(
                    x=X_full_t,
                    edge_index=edge_index_t,
                    index=idx,
                )
                gnn_explanations[uid] = {
                    'edge_mask': explanation.edge_mask.detach().cpu().numpy(),
                    'node_mask': explanation.node_mask.detach().cpu().numpy() if explanation.node_mask is not None else None,
                }
            except Exception as e:
                gnn_explanations[uid] = {'edge_mask': None, 'node_mask': None, 'error': str(e)}

        # 批次後清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results['gnn_explanations'] = gnn_explanations
    print(f"  GNNExplainer: 完成 {len(gnn_explanations)} 個用戶")

except ImportError:
    print("  GNNExplainer: torch_geometric.explain 未支援，跳過")
except Exception as e:
    print(f"  GNNExplainer 錯誤: {e}")

print("\n解釋生成完成！")

output_path = os.path.join(OUTPUT_DIR, 'graphsage_v13_gpu_complete_results.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n結果已儲存: {output_path}")
print("="*70)
