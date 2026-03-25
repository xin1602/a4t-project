"""
Ensemble Stacking 訓練腳本 - 使用共享特徵模組

結合 GraphSAGE + LightGBM 的預測結果

使用方式：
    # 先訓練基礎模型
    python train_lgbm.py
    python train_graphsage.py
    
    # 再訓練 Ensemble
    python train_ensemble.py

輸出：
    ensemble_results.pkl - 包含 OOF 預測和評估指標
"""

import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# 使用共享特徵模組
from shared_features import load_data, parse_date
from config import get_data_dir, get_output_dir

# ==================== 配置 ====================

BASE_DIR = get_data_dir()
OUTPUT_DIR = get_output_dir()

print("="*70)
print("Ensemble Stacking - GraphSAGE V13 + LightGBM V12")
print("="*70)

# ==================== 載入資料 ====================

train_labels, user_info, twd_txs, crypto_txs, _, _ = load_data(BASE_DIR)
user_ids = list(train_labels.keys())
y = np.array([train_labels[uid] for uid in user_ids])

print(f"\n使用者數: {len(user_ids):,}")
print(f"詐欺數: {sum(y):,}")

# ==================== 載入基礎模型結果 ====================

print("\n載入基礎模型結果...")

try:
    with open(str(Path(OUTPUT_DIR) / 'lgbm_v12_results.pkl'), 'rb') as f:
        lgbm_results = pickle.load(f)
    lgbm_oof = lgbm_results['oof_predictions']
    print(f"  ✓ LightGBM OOF: {len(lgbm_oof)}")
except FileNotFoundError:
    print("  ❌ 找不到 lgbm_v12_results.pkl")
    print("  請先執行: python train_lgbm.py")
    exit(1)

try:
    with open(str(Path(OUTPUT_DIR) / 'graphsage_v13_results.pkl'), 'rb') as f:
        gs_results = pickle.load(f)
    gs_oof = gs_results['oof_predictions']
    gs_user_ids = gs_results['user_ids']
    print(f"  ✓ GraphSAGE OOF: {len(gs_oof)}")
except FileNotFoundError:
    print("  ❌ 找不到 graphsage_v13_results.pkl")
    print("  請先執行: python train_graphsage.py")
    exit(1)

# ==================== 計算時序排序 ====================

print("\n計算時序排序...")

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
temporal_data = []
for i, uid in enumerate(user_ids):
    temporal_data.append({
        'user_id': uid,
        't_obs': user_t_obs[uid],
        'label': y[i],
        'lgbm_pred': lgbm_oof[i],
        'gs_pred': gs_oof[i] if uid in gs_user_ids else 0,
    })

temporal_data.sort(key=lambda x: x['t_obs'])

# 提取排序後的數據
user_ids_sorted = [d['user_id'] for d in temporal_data]
y_sorted = np.array([d['label'] for d in temporal_data])
lgbm_sorted = np.array([d['lgbm_pred'] for d in temporal_data])
gs_sorted = np.array([d['gs_pred'] for d in temporal_data])

print(f"  時間範圍: {min(user_t_obs.values())} 到 {max(user_t_obs.values())}")

# ==================== 創建 Meta-Features ====================

print("\n創建 Meta-Features...")

meta_features = np.column_stack([
    lgbm_sorted,
    gs_sorted,
    lgbm_sorted * gs_sorted,  # 交互
    np.abs(lgbm_sorted - gs_sorted),  # 分歧度
    np.maximum(lgbm_sorted, gs_sorted),  # 最大值
    np.minimum(lgbm_sorted, gs_sorted),  # 最小值
    (lgbm_sorted + gs_sorted) / 2,  # 平均值
])

meta_feature_names = [
    'lgbm_pred',
    'gs_pred',
    'lgbm_gs_interaction',
    'prediction_disagreement',
    'max_pred',
    'min_pred',
    'avg_pred',
]

print(f"  Meta-Features 形狀: {meta_features.shape}")

# ==================== Walk-Forward 驗證 ====================

print("\n" + "="*70)
print("訓練 Stacking Ensemble (Walk-Forward 驗證)")
print("="*70)

# LightGBM Meta-learner 參數
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

tscv = TimeSeriesSplit(n_splits=5)
oof_predictions = np.zeros(len(y_sorted))
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(meta_features), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5")
    print(f"{'='*60}")
    
    X_tr, X_val = meta_features[train_idx], meta_features[val_idx]
    y_tr, y_val = y_sorted[train_idx], y_sorted[val_idx]
    
    print(f"訓練集: {len(train_idx):,} | 驗證集: {len(val_idx):,}")
    print(f"訓練詐欺率: {y_tr.mean():.4f} | 驗證詐欺率: {y_val.mean():.4f}")
    
    # 訓練 LightGBM Meta-learner
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    # 預測
    y_pred_proba = model.predict(X_val)
    oof_predictions[val_idx] = y_pred_proba
    
    # 評估
    roc = roc_auc_score(y_val, y_pred_proba)
    precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_pred_proba)
    pr = auc(recall_arr, precision_arr)
    
    # 尋找最佳 F1
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.05, 0.5, 0.01):
        y_pred = (y_pred_proba >= th).astype(int)
        current_f1 = f1_score(y_val, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_th = th
    
    fold_metrics.append({
        'fold': fold,
        'roc_auc': roc,
        'pr_auc': pr,
        'f1': best_f1,
        'best_th': best_th,
    })
    
    print(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}, F1: {best_f1:.4f} (閾值={best_th:.2f})")

# ==================== 最終結果 ====================

print(f"\n{'='*70}")
print("Ensemble Stacking 最終結果")
print(f"{'='*70}")

mean_roc = np.mean([m['roc_auc'] for m in fold_metrics])
mean_pr = np.mean([m['pr_auc'] for m in fold_metrics])
mean_f1 = np.mean([m['f1'] for m in fold_metrics])

print(f"\n平均指標:")
print(f"  ROC-AUC: {mean_roc:.4f}")
print(f"  PR-AUC:  {mean_pr:.4f}")
print(f"  F1:      {mean_f1:.4f}")

# 整體 OOF 評估
overall_roc = roc_auc_score(y_sorted, oof_predictions)
precision_arr, recall_arr, _ = precision_recall_curve(y_sorted, oof_predictions)
overall_pr = auc(recall_arr, precision_arr)

# 尋找最佳 F1
best_f1_overall = 0
best_th_overall = 0.5
for th in np.arange(0.05, 0.5, 0.01):
    y_pred = (oof_predictions >= th).astype(int)
    current_f1 = f1_score(y_sorted, y_pred, zero_division=0)
    if current_f1 > best_f1_overall:
        best_f1_overall = current_f1
        best_th_overall = th

print(f"\n整體 OOF 結果:")
print(f"  ROC-AUC: {overall_roc:.4f}")
print(f"  PR-AUC:  {overall_pr:.4f}")
print(f"  F1:      {best_f1_overall:.4f} (閾值: {best_th_overall:.2f})")

# 與基礎模型比較
print(f"\n與基礎模型比較:")
lgbm_f1 = lgbm_results.get('mean_f1', 0)
gs_f1 = gs_results.get('mean_f1', 0)
print(f"  LightGBM:  F1={lgbm_f1:.4f}")
print(f"  GraphSAGE: F1={gs_f1:.4f}")
print(f"  Ensemble:  F1={mean_f1:.4f}")

improvement = mean_f1 - max(lgbm_f1, gs_f1)
print(f"\n改進: {improvement:+.4f}")

# 儲存結果
results = {
    'model': 'Ensemble_Stacking',
    'base_models': ['LightGBM_V12', 'GraphSAGE_V13'],
    'fold_metrics': fold_metrics,
    'oof_predictions': oof_predictions,
    'y_true': y_sorted,
    'user_ids': user_ids_sorted,
    'meta_feature_names': meta_feature_names,
    'mean_roc_auc': mean_roc,
    'mean_pr_auc': mean_pr,
    'mean_f1': mean_f1,
    'overall_roc_auc': overall_roc,
    'overall_pr_auc': overall_pr,
    'overall_f1': best_f1_overall,
}

output_path = str(Path(OUTPUT_DIR) / 'ensemble_results.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n結果已儲存: {output_path}")
print("="*70)
