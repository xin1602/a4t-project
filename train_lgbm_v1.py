"""
LightGBM V12 訓練腳本 - 獨立版本
基於 train_lightgbm_v12.py，包含社群檢測特徵

使用方式：
    python train_lgbm.py

輸出：
    lgbm_v12_results.pkl - 包含 OOF 預測和評估指標
"""

import json
import numpy as np
import os
import pickle
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# 使用共享特徵模組
from shared_features import (
    load_data,
    compute_all_features
)
from config import get_data_dir, get_output_dir

# ==================== 配置 ====================

BASE_DIR = get_data_dir()
OUTPUT_DIR = get_output_dir()

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'max_depth': 10,
    'learning_rate': 0.02,
    'n_estimators': 1000,
    'min_child_samples': 15,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

print("="*70)
print("LightGBM V12 - 社群檢測特徵")
print("="*70)

# ==================== 載入資料 ====================

train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(BASE_DIR)
user_ids = list(train_labels.keys())

print(f"\n使用者數: {len(user_ids):,}")
print(f"詐欺數: {sum(train_labels.values()):,}")

# ==================== 計算基礎特徵 + 圖結構 ====================

print("\n計算基礎特徵和圖結構...")

# 先計算不涉及標籤的特徵
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

print(f"基礎特徵矩陣形狀: {X_base.shape}")
print(f"正樣本比例: {y.mean():.4f}")

# ==================== 交叉驗證訓練 ====================

print("\n" + "="*70)
print("開始 5-Fold 交叉驗證訓練")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(y))
oof_shap_values = None  # initialized on first fold when full feature count is known
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, y), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5")
    print(f"{'='*60}")
    
    # 在訓練集上計算社群特徵（防止 data leakage）
    print("  計算社群特徵（僅使用訓練集標籤）...")
    from shared_features import compute_community_features
    X_community, community_names, _ = compute_community_features(
        user_ids=user_ids,
        G=G,
        train_labels=train_labels,
        train_indices=train_idx,  # 只使用訓練集標籤
        verbose=False
    )
    
    # 合併特徵
    X = np.hstack([X_base, X_community])
    all_feature_names = feature_names + community_names
    
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    print(f"  訓練集: {X_tr.shape}, 驗證集: {X_val.shape}")
    
    # 訓練 LightGBM
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    # 預測
    y_pred_proba = model.predict(X_val)
    oof_predictions[val_idx] = y_pred_proba

    # SHAP values for validation set (TreeExplainer is fast for LightGBM)
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)
    # For binary classification, shap_values returns [class0, class1] or single array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class 1 (fraud)
    if oof_shap_values is None:
        oof_shap_values = np.zeros((len(y), shap_vals.shape[1]))
    oof_shap_values[val_idx] = shap_vals
    
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
        'roc_auc': roc,
        'pr_auc': pr,
        'f1': best_f1,
        'best_th': best_th
    })
    
    print(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}, F1: {best_f1:.4f} (閾值={best_th:.2f})")

# ==================== 最終結果 ====================

print(f"\n{'='*70}")
print("LightGBM V12 最終結果")
print(f"{'='*70}")

mean_roc = np.mean([m['roc_auc'] for m in fold_metrics])
std_roc = np.std([m['roc_auc'] for m in fold_metrics])
mean_pr = np.mean([m['pr_auc'] for m in fold_metrics])
std_pr = np.std([m['pr_auc'] for m in fold_metrics])
mean_f1 = np.mean([m['f1'] for m in fold_metrics])
std_f1 = np.std([m['f1'] for m in fold_metrics])

print(f"\n平均指標:")
print(f"  ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f}")
print(f"  PR-AUC:  {mean_pr:.4f} ± {std_pr:.4f}")
print(f"  F1:      {mean_f1:.4f} ± {std_f1:.4f}")

# 儲存結果
results = {
    'version': 'V12',
    'fold_metrics': fold_metrics,
    'oof_predictions': oof_predictions,
    'oof_shap_values': oof_shap_values,
    'user_ids': user_ids,
    'y_true': y,
    'feature_names': all_feature_names,  # 使用最後一個 fold 的特徵名稱
    'mean_roc_auc': mean_roc,
    'mean_pr_auc': mean_pr,
    'mean_f1': mean_f1,
}

output_path = str(Path(OUTPUT_DIR) / 'lgbm_v12_results.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n結果已儲存: {output_path}")
print("="*70)
