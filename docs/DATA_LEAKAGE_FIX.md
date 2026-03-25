# Data Leakage 修正說明

## 問題描述

在原始的訓練腳本中，`community_fraud_rate` 特徵存在嚴重的 data leakage 問題。

### 問題根源

在交叉驗證時，社群詐欺率是用全部訓練集的標籤計算的，包括驗證集的標籤。

### 影響

- 驗證指標虛高（ROC-AUC 0.9989, F1 0.9135）
- 無法反映真實的泛化能力
- 生產環境效能會顯著下降

## 修正方案

### 1. 修改 shared_features.py

增加 `train_indices` 參數，只使用訓練集標籤計算社群詐欺率。

### 2. 修改訓練腳本

在每個 fold 內重新計算社群特徵。

## 修正後的預期變化

| 指標 | 修正前 | 預期修正後 |
|------|--------|-----------|
| ROC-AUC | 0.9989 | 0.97-0.98 |
| PR-AUC | 0.9748 | 0.85-0.90 |
| F1 | 0.9135 | 0.75-0.85 |

## 驗證方法

```bash
cd a4t-project
python train_lgbm.py
python train_graphsage.py
python train_ensemble.py
```
