"""
GraphSAGE V13 訓練腳本 - 使用共享特徵模組

使用方式：
    python train_graphsage.py

輸出：
    graphsage_v13_results.pkl - 包含 OOF 預測和評估指標
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
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# 使用共享特徵模組
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
print("GraphSAGE V13 - 時序圖特徵 + Walk-Forward 驗證")
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

# ==================== 計算時序排序 ====================

print("\n計算時序排序（t_obs）...")

def get_last_activity_time(user_id):
    """獲取用戶最後活動時間"""
    times = []
    for tx in twd_txs:
        if tx['user_id'] == user_id:
            times.append(parse_date(tx.get('created_at', '')))
    for tx in crypto_txs:
        if tx['user_id'] == user_id:
            times.append(parse_date(tx.get('created_at', '')))
    return max(times) if times else datetime(2025, 1, 1)

user_t_obs = {uid: get_last_activity_time(uid) for uid in tqdm(user_ids, desc="  計算 t_obs")}

# 按時間排序
temporal_indices = sorted(range(len(user_ids)), key=lambda i: user_t_obs[user_ids[i]])
user_ids_sorted = [user_ids[i] for i in temporal_indices]
X_base_sorted = X_base[temporal_indices]
y_sorted = y[temporal_indices]

print(f"  時間範圍: {min(user_t_obs.values())} 到 {max(user_t_obs.values())}")

# ==================== GraphSAGE 模型 ====================

class SAGE_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='add'))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='add'))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_channels * num_layers + in_channels, 1)
    
    def forward(self, x, edge_index):
        x_in = x
        layer_outputs = []
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.elu(x)
                x = self.dropout(x)
            layer_outputs.append(x)
        
        x_jk = torch.cat(layer_outputs, dim=1)
        x_final = torch.cat([x_jk, x_in], dim=1)
        return self.final(x_final).squeeze()

# ==================== 建立圖結構 ====================

print("\n建立圖結構...")

# 從 NetworkX 圖轉換為 PyTorch Geometric 格式
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

print(f"  圖邊數: {edge_index.shape[1]:,}")

# ==================== Walk-Forward 驗證 ====================

print("\n" + "="*70)
print("Walk-Forward 驗證 (TimeSeriesSplit)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

tscv = TimeSeriesSplit(n_splits=5)
oof_predictions = np.zeros(len(y_sorted))
fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_base_sorted), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5")
    print(f"{'='*60}")
    
    # 計算社群特徵（僅使用訓練集標籤）
    print("  計算社群特徵（僅使用訓練集標籤）...")
    from shared_features import compute_community_features
    
    # 將排序後的索引映射回原始索引
    train_idx_original = [temporal_indices[i] for i in train_idx]
    
    X_community, community_names, _ = compute_community_features(
        user_ids=user_ids,
        G=G,
        train_labels=train_labels,
        train_indices=np.array(train_idx_original),
        verbose=False
    )
    
    # 按時間排序社群特徵
    X_community_sorted = X_community[temporal_indices]
    
    # 合併特徵
    X_sorted = np.hstack([X_base_sorted, X_community_sorted])
    all_feature_names = feature_names + community_names
    
    X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
    y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]
    
    print(f"  訓練集: {len(train_idx):,} | 驗證集: {len(val_idx):,}")
    print(f"  訓練詐欺率: {y_train.mean():.4f} | 驗證詐欺率: {y_val.mean():.4f}")
    print(f"  特徵數: {X_sorted.shape[1]}")
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sorted)
    
    # 轉換為 PyTorch
    X_t = torch.FloatTensor(X_scaled).to(device)
    y_t = torch.FloatTensor(y_sorted).to(device)
    edge_index_t = edge_index.to(device)
    
    train_idx_t = torch.LongTensor(train_idx).to(device)
    val_idx_t = torch.LongTensor(val_idx).to(device)
    
    # 訓練
    model = SAGE_Model(
        in_channels=X_scaled.shape[1],
        hidden_channels=128,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    
    best_pr_auc = 0
    patience = 30
    no_improve = 0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_t, edge_index_t)
        train_output = output[train_idx_t]
        train_labels = y_t[train_idx_t]
        
        loss = criterion(train_output, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_t, edge_index_t)
                val_probs = torch.sigmoid(val_output[val_idx_t]).cpu().numpy()
                
                pr_auc = average_precision_score(y_val, val_probs)
                roc_auc = roc_auc_score(y_val, val_probs)
                
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    no_improve = 0
                else:
                    no_improve += 1
            
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            if no_improve >= patience:
                print(f"  Early stopping")
                break
            
            model.train()
    
    # 最終評估
    model.eval()
    with torch.no_grad():
        final_output = model(X_t, edge_index_t)
        val_probs = torch.sigmoid(final_output[val_idx_t]).cpu().numpy()
        
        pr_auc = average_precision_score(y_val, val_probs)
        roc_auc = roc_auc_score(y_val, val_probs)
        
        # 尋找最佳閾值
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_threshold_idx]
        
        oof_predictions[val_idx] = val_probs
    
    print(f"\nFold {fold} 結果:")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Best F1: {best_f1:.4f} (閾值: {best_threshold:.4f})")
    
    fold_results.append({
        'fold': fold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'best_f1': best_f1,
        'threshold': best_threshold,
    })

# ==================== 最終結果 ====================

print("\n" + "="*70)
print("GraphSAGE V13 最終結果")
print("="*70)

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

# 儲存結果
results = {
    'model': 'GraphSAGE_V13',
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
    'model_state_dict': model.state_dict(),   # last fold model weights (for ExplainerService)
    'scaler': scaler,                          # last fold StandardScaler (for ExplainerService)
    'edge_index': edge_index.cpu(),            # graph edge_index on CPU (for ExplainerService)
    'X_scaled': X_scaled,                      # last fold full scaled feature matrix, shape: (n_users, n_features)
}

output_path = str(Path(OUTPUT_DIR) / 'graphsage_v13_results.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n結果已儲存: {output_path}")
print("="*70)
