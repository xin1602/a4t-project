"""
LightGBM V13 訓練腳本 - 無圖特徵版本
移除所有圖相關特徵，避免 data leakage

修正：
- 完全移除圖特徵（graph_degree, graph_clustering, graph_component_size）
- 完全移除社群特徵（community_id, community_size 等）
- 只使用基礎交易特徵（69個）
- 保存 fold models 供後續 ensemble 使用
- 導出 feature importance
- Hyperparameter tuning via Optuna

使用方式：
    python train_lgbm.py

輸出：
    lgbm_v13_no_graph_results.pkl - 包含 OOF 預測、fold models 和評估指標
    lgbm_v13_feature_importance.csv - 特徵重要性排名
    lgbm_v13_best_params.json - 最佳超參數
"""

import json
import numpy as np
import os
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import optuna
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
print("LightGBM V13 - 無圖特徵版本")
print("="*70)
print("修正：完全移除圖特徵，避免 data leakage")
print("  - 移除所有圖特徵（degree, clustering, component_size）")
print("  - 移除所有社群特徵（community_id, size, density 等）")
print("  - 只使用基礎交易特徵（69個）")
print("="*70)

# ==================== 載入資料 ====================

train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(BASE_DIR)
user_ids = list(train_labels.keys())

print(f"\n使用者數: {len(user_ids):,}")
print(f"詐欺數: {sum(train_labels.values()):,}")

# ==================== 計算基礎特徵（無圖）====================

print("\n計算基礎特徵（不含圖特徵）...")

# 只計算基礎交易特徵，不使用圖
X, feature_names, _, _ = compute_all_features(
    user_ids=user_ids,
    train_labels=None,  # 不使用標籤
    user_info=user_info,
    twd_txs=twd_txs,
    crypto_txs=crypto_txs,
    trading_txs=trading_txs,
    swap_txs=swap_txs,
    include_graph=False,  # 不使用圖特徵
    include_community=False,  # 不使用社群特徵
    verbose=True
)

y = np.array([train_labels[uid] for uid in user_ids])

print(f"正樣本比例: {y.mean():.4f}")
print(f"特徵矩陣形狀: {X.shape}")

# ==================== Hyperparameter Tuning with Optuna ====================

print("\n" + "="*70)
print("Hyperparameter Tuning with Optuna")
print("="*70)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }

    skf_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf_tune.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

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

        y_pred_proba = model.predict(X_val)

        # Find best F1
        best_f1 = 0
        for th in np.arange(0.05, 0.5, 0.02):
            y_pred = (y_pred_proba >= th).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        f1_scores.append(best_f1)

    return np.mean(f1_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest F1: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# ==================== 交叉驗證訓練（使用最佳參數） ====================

print("\n" + "="*70)
print("開始 5-Fold 交叉驗證訓練（無圖特徵 + 調優參數）")
print("="*70)

LGBM_PARAMS.update(best_params)
LGBM_PARAMS['n_estimators'] = 2000  # 用更多 trees

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(y))
oof_shap_values = None
fold_metrics = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/5")
    print(f"{'='*60}")
    
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

    fold_models.append(model)

    # 預測
    y_pred_proba = model.predict(X_val)
    oof_predictions[val_idx] = y_pred_proba

    # SHAP values for validation set
    import shap
    shap_explainer = shap.TreeExplainer(model)
    shap_vals = shap_explainer.shap_values(X_val)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
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
print("LightGBM V13 最終結果（無圖特徵）")
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

# OOF 預測分布
print(f"\nOOF 預測分布:")
print(f"  Mean: {oof_predictions.mean():.4f}")
print(f"  Std:  {oof_predictions.std():.4f}")
print(f"  Min:  {oof_predictions.min():.4f}")
print(f"  Max:  {oof_predictions.max():.4f}")

# Feature Importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': fold_models[0].feature_importance()
}).sort_values('importance', ascending=False)
importance_df.to_csv(str(Path(OUTPUT_DIR) / 'lgbm_v13_feature_importance.csv'), index=False)
print(f"\nFeature Importance Top 10:")
print(importance_df.head(10).to_string(index=False))

# 儲存結果
results = {
    'version': 'V13_no_graph_tuned',
    'best_params': best_params,
    'fold_metrics': fold_metrics,
    'fold_models': fold_models,
    'oof_predictions': oof_predictions,
    'oof_shap_values': oof_shap_values,
    'user_ids': user_ids,
    'y_true': y,
    'feature_names': feature_names,
    'mean_roc_auc': mean_roc,
    'mean_pr_auc': mean_pr,
    'mean_f1': mean_f1,
}

output_path = str(Path(OUTPUT_DIR) / 'lgbm_v13_no_graph_results.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(results, f)

# 儲存最佳參數
best_params_path = str(Path(OUTPUT_DIR) / 'lgbm_v13_best_params.json')
with open(best_params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n結果已儲存: {output_path}")
print("="*70)
