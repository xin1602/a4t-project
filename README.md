# A4T 獨立訓練腳本

這個目錄包含完整的獨立訓練腳本，使用**共享特徵模組**避免代碼重複。

## ⭐ 核心設計

所有重複的特徵工程代碼已提取到 `shared_features.py`，各訓練腳本直接導入使用：

```python
from shared_features import (
    load_data,              # 資料載入
    compute_all_features,   # 完整特徵計算（120+ 特徵）
    compute_base_features,  # 基礎特徵
    compute_graph_features, # 圖特徵
    compute_community_features  # 社群特徵
)
```

## 📁 檔案結構

```
a4t/
├── shared_features.py      # 🔧 共享特徵工程模組（600 行，可重用）
├── train_lgbm.py           # LightGBM 訓練（200 行）
├── train_graphsage.py      # GraphSAGE 訓練（250 行）
├── train_ensemble.py       # Ensemble 訓練（200 行）
├── example_simple.py       # 簡化示例（200 行）
├── test_setup.py           # 環境測試
├── create_standalone.py    # 生成完全獨立版本
└── 文檔檔案...
```

## 🚀 快速開始

### 方法 1：使用共享模組（推薦）

```bash
# 1. 測試環境
python a4t-project/test_setup.py

# 2. 訓練 LightGBM（使用共享特徵）
python a4t-project/train_lgbm.py
# 輸出: lgbm_v12_results.pkl

# 3. 訓練 GraphSAGE（使用共享特徵）
python a4t-project/train_graphsage.py
# 輸出: graphsage_v13_results.pkl

# 4. 訓練 Ensemble（結合上述兩個模型）
python a4t-project/train_ensemble.py
# 輸出: ensemble_results.pkl
```

### 方法 2：生成完全獨立版本

```bash
# 自動生成單檔案版本（包含所有代碼）
python a4t-project/create_standalone.py --all

# 執行生成的獨立腳本
python a4t-project/train_lgbm_standalone.py
python a4t-project/train_graphsage_standalone.py
python a4t-project/train_ensemble_standalone.py
```

### 方法 3：快速學習（簡化版）

```bash
# 執行簡化示例（15 個特徵，5 分鐘）
python a4t-project/example_simple.py
```

## 📊 特徵說明

### shared_features.py 提供的功能

1. **資料載入** - `load_data(base_dir)`
   - 載入所有 JSONL 檔案
   - 返回標籤、使用者資訊、交易資料

2. **基礎特徵** - `compute_base_features(...)`
   - 人口統計（7 個）
   - 交易筆數（6 個）
   - 時序模式（10 個）
   - 金額統計（14 個）
   - 流入流出（12 個）
   - 對手方集中度（5 個）
   - 行為穩定性（1 個）
   - 活動標記（5 個）
   - 交易類型多樣性（2 個）
   - 跨類型比率（4 個）
   - 外部 vs 內部（3 個）
   - 交互特徵（4 個）
   - **總計：73 個基礎特徵**

3. **圖特徵** - `compute_graph_features(...)`
   - 度特徵
   - 聚類係數
   - 連通分量大小
   - **總計：3 個圖特徵**

4. **社群特徵** - `compute_community_features(...)`
   - Louvain 社群檢測
   - 社群大小
   - 社群詐欺率
   - 社群詐欺數量
   - **總計：4 個社群特徵**

5. **一站式函數** - `compute_all_features(...)`
   - 自動計算所有特徵
   - 返回特徵矩陣 + 特徵名稱 + 圖結構
   - **總計：80+ 個特徵**

## 💡 優勢

### 相比原始檔案

| 特性 | 原始檔案 | 共享模組版本 |
|------|---------|-------------|
| 代碼重複 | 高（每個腳本都有特徵代碼） | 無（共享模組） |
| 維護性 | 低（需同步更新多個檔案） | 高（只需更新一個模組） |
| 可讀性 | 中（單檔案 600-1500 行） | 高（訓練腳本 200 行） |
| 測試性 | 難（特徵與訓練耦合） | 易（特徵模組可獨立測試） |
| 總行數 | ~3000 行（分散） | ~1250 行（集中） |

### 相比完全獨立版本

| 特性 | 完全獨立版本 | 共享模組版本 |
|------|-------------|-------------|
| 檔案數 | 3 個（每個 2000+ 行） | 4 個（共享 + 3 個訓練） |
| 代碼重複 | 極高 | 無 |
| 部署便利性 | 高（單檔案） | 中（需 2 個檔案） |
| 維護成本 | 極高 | 低 |

## 📝 使用範例

### 基本使用

```python
from shared_features import load_data, compute_all_features

# 載入資料
train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(BASE_DIR)
user_ids = list(train_labels.keys())

# 計算所有特徵
X, feature_names, G = compute_all_features(
    user_ids=user_ids,
    train_labels=train_labels,
    user_info=user_info,
    twd_txs=twd_txs,
    crypto_txs=crypto_txs,
    trading_txs=trading_txs,
    swap_txs=swap_txs,
    include_graph=True,
    include_community=True
)

# X: (n_users, n_features) 特徵矩陣
# feature_names: 特徵名稱列表
# G: NetworkX 圖結構
```

### 自定義特徵組合

```python
# 只計算基礎特徵
X_base, base_names = compute_base_features(
    user_ids, user_info, twd_txs, crypto_txs, trading_txs, swap_txs
)

# 只計算圖特徵
X_graph, graph_names, G = compute_graph_features(
    user_ids, crypto_txs, twd_txs
)

# 手動組合
import numpy as np
X = np.hstack([X_base, X_graph])
feature_names = base_names + graph_names
```

## 🔧 依賴套件

```bash
pip install numpy pandas scikit-learn lightgbm torch torch-geometric networkx tqdm
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 📂 資料路徑

預設資料路徑：`D:/lxh/github/a4t-project`

修改方式：
1. 在各腳本開頭修改 `BASE_DIR` 變數
2. 或在 `shared_features.py` 中設置預設值

## 🎯 訓練流程

```
1. test_setup.py
   ↓ 檢查環境
   
2. train_lgbm.py
   ↓ 訓練 LightGBM
   ↓ 輸出: lgbm_v12_results.pkl
   
3. train_graphsage.py
   ↓ 訓練 GraphSAGE
   ↓ 輸出: graphsage_v13_results.pkl
   
4. train_ensemble.py
   ↓ 訓練 Ensemble
   ↓ 輸出: ensemble_results.pkl
```

## 📚 相關文檔

- **SUMMARY.md** - 完整總結與對比
- **INTEGRATION_GUIDE.md** - 詳細整合指南
- **INDEX.md** - 檔案索引
- **test_setup.py** - 環境測試腳本

## ❓ 常見問題

**Q: 為什麼使用共享模組而不是完全獨立？**
A: 共享模組避免了代碼重複，更易維護。如需完全獨立版本，可使用 `create_standalone.py` 生成。

**Q: 如何添加新特徵？**
A: 在 `shared_features.py` 的 `_compute_user_features()` 函數中添加，所有訓練腳本自動使用。

**Q: 效能如何？**
A: 與原始檔案完全相同（使用相同的特徵計算邏輯）。

**Q: 可以只使用部分特徵嗎？**
A: 可以，使用 `compute_base_features()` 或 `compute_graph_features()` 單獨計算。

## 🚀 下一步

1. 執行 `python a4t-project/test_setup.py` 檢查環境
2. 執行 `python a4t-project/example_simple.py` 快速體驗
3. 執行 `python a4t-project/train_lgbm.py` 開始訓練
4. 閱讀 `SUMMARY.md` 了解更多細節
