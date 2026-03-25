# A4T 獨立訓練腳本 - 總結

## 📁 檔案清單

### 核心模組

1. **shared_features.py** (約 600 行)
   - 🔧 可重用的特徵工程模組
   - 包含所有特徵計算函數
   - 提供一站式 API

2. **README.md** - 主要說明文件
3. **SUMMARY.md** - 本檔案
4. **INTEGRATION_GUIDE.md** - 詳細整合指南
5. **INDEX.md** - 檔案索引

### 訓練腳本（使用共享模組）

6. **train_lgbm.py** (約 200 行)
   - LightGBM V12 訓練
   - 5-Fold 交叉驗證
   - 輸出：lgbm_v12_results.pkl

7. **train_graphsage.py** (約 250 行)
   - GraphSAGE V13 訓練
   - Walk-Forward 時序驗證
   - 輸出：graphsage_v13_results.pkl

8. **train_ensemble.py** (約 200 行)
   - Ensemble Stacking 訓練
   - 結合 GraphSAGE + LightGBM
   - 輸出：ensemble_results.pkl

### 工具與示例

9. **test_setup.py** - 環境測試腳本
10. **example_simple.py** - 簡化示例（15 個特徵）
11. **create_standalone.py** - 生成完全獨立版本的工具
12. **preprocessing.py** - 舊版資料載入（已被 shared_features.py 取代）

## 🎯 核心設計理念

### 共享模組架構

```
shared_features.py (600 行)
    ↓ 導入
    ├── train_lgbm.py (200 行)
    ├── train_graphsage.py (250 行)
    └── train_ensemble.py (200 行)
```

所有訓練腳本使用相同的特徵計算邏輯，確保一致性並降低維護成本。

### 優勢對比

| 特性 | 原始檔案 | 共享模組版本 | 完全獨立版本 |
|------|---------|-------------|-------------|
| 代碼重複 | 高 | 無 | 極高 |
| 維護性 | 低 | 高 | 極低 |
| 可讀性 | 中 | 高 | 低 |
| 部署便利性 | 中 | 中 | 高 |
| 總行數 | ~3000 | ~1250 | ~6000 |
| 檔案數 | 多個 | 4 個 | 3 個 |

## 📊 使用場景

### 場景 1：快速學習理解

```bash
# 執行簡化示例，快速理解流程
python a4t/example_simple.py
```

**特點**：
- 僅 15 個基礎特徵
- 代碼簡潔易讀（~200 行）
- 5 分鐘內完成訓練
- 適合理解整體流程

### 場景 2：開發與測試（推薦）

```bash
# 使用共享模組版本
python a4t/train_lgbm.py
python a4t/train_graphsage.py
python a4t/train_ensemble.py
```

**特點**：
- 完整的 80+ 特徵
- 模組化結構，易於維護
- 代碼清晰，易於調試
- 可以單獨測試特徵模組

### 場景 3：完全獨立部署

```bash
# 生成獨立腳本
python a4t/create_standalone.py --all

# 執行獨立腳本
python a4t/train_lgbm_standalone.py
python a4t/train_graphsage_standalone.py
python a4t/train_ensemble_standalone.py
```

**特點**：
- 單檔案包含所有代碼
- 無需 shared_features.py
- 適合跨環境部署
- 檔案較大（2000-2500 行）

## 🔧 shared_features.py 功能模組

### 1. 資料載入

```python
from shared_features import load_data

train_labels, user_info, twd_txs, crypto_txs, trading_txs, swap_txs = load_data(BASE_DIR)
```

### 2. 基礎特徵（73 個）

```python
from shared_features import compute_base_features

X_base, base_names = compute_base_features(
    user_ids, user_info, twd_txs, crypto_txs, trading_txs, swap_txs
)
```

**包含**：
- 人口統計特徵（7 個）
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

### 3. 圖特徵（3 個）

```python
from shared_features import compute_graph_features

X_graph, graph_names, G = compute_graph_features(
    user_ids, crypto_txs, twd_txs
)
```

**包含**：
- 度特徵
- 聚類係數
- 連通分量大小

### 4. 社群特徵（4 個）

```python
from shared_features import compute_community_features

X_community, community_names = compute_community_features(
    user_ids, G, train_labels
)
```

**包含**：
- Louvain 社群 ID
- 社群大小
- 社群詐欺率
- 社群詐欺數量

### 5. 一站式函數

```python
from shared_features import compute_all_features

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
```

**總計：80+ 個特徵**

## 📈 效能對比

| 版本 | 特徵數 | 訓練時間 | ROC-AUC | F1 Score | 代碼行數 |
|------|--------|---------|---------|----------|----------|
| 簡化示例 | 15 | ~5 分鐘 | ~0.75 | ~0.25 | 200 |
| 共享模組 | 80+ | ~30 分鐘 | ~0.87 | ~0.40 | 1250 |
| 完全獨立 | 80+ | ~30 分鐘 | ~0.87 | ~0.40 | 6000 |

## � 最佳實踐建議

### 1. 開發階段

```bash
# 使用簡化示例快速驗證想法
python a4t/example_simple.py

# 使用共享模組版本進行完整訓練
python a4t/train_lgbm.py
```

### 2. 測試階段

```bash
# 測試環境設置
python a4t/test_setup.py

# 執行訓練
python a4t/train_lgbm.py
python a4t/train_graphsage.py
python a4t/train_ensemble.py
```

### 3. 生產部署

```bash
# 方案 A：保持模組化（推薦）
project/
├── shared_features.py
├── train_lgbm.py
├── train_graphsage.py
└── train_ensemble.py

# 方案 B：使用獨立版本
python a4t/create_standalone.py --all
project/
├── train_lgbm_standalone.py
├── train_graphsage_standalone.py
└── train_ensemble_standalone.py
```

## 📝 代碼維護

### 添加新特徵

只需在 `shared_features.py` 的 `_compute_user_features()` 函數中添加：

```python
def _compute_user_features(...):
    f = {}
    
    # 現有特徵...
    
    # 添加新特徵
    f['new_feature'] = compute_new_feature(...)
    
    return f
```

所有訓練腳本自動使用新特徵！

### 修改特徵邏輯

只需修改 `shared_features.py` 中的對應函數，所有訓練腳本自動更新。

### 版本控制

```bash
# 清晰的 Git diff
git diff shared_features.py  # 只看特徵變更
git diff train_lgbm.py       # 只看訓練邏輯變更
```

## � 快速開始指令

```bash
# 1. 測試環境（1 分鐘）
python a4t/test_setup.py

# 2. 學習理解（5 分鐘）
python a4t/example_simple.py

# 3. 完整訓練（30 分鐘）
python a4t/train_lgbm.py
python a4t/train_graphsage.py
python a4t/train_ensemble.py

# 4. 生成獨立版本（可選，1 分鐘）
python a4t/create_standalone.py --all
```

## 📚 相關文件

- **README.md** - 快速入門與詳細說明
- **INTEGRATION_GUIDE.md** - 詳細整合指南
- **INDEX.md** - 檔案索引
- **shared_features.py** - 核心特徵模組（含文檔字串）

## ❓ 常見問題

**Q: 為什麼使用共享模組而不是完全獨立？**
A: 共享模組避免了代碼重複，更易維護。如需完全獨立版本，可使用 `create_standalone.py` 生成。

**Q: 如何添加新特徵？**
A: 在 `shared_features.py` 的 `_compute_user_features()` 函數中添加，所有訓練腳本自動使用。

**Q: 效能如何？**
A: 與原始檔案完全相同（使用相同的特徵計算邏輯），ROC-AUC ~0.87。

**Q: 可以只使用部分特徵嗎？**
A: 可以，使用 `compute_base_features()` 或 `compute_graph_features()` 單獨計算。

**Q: 如何測試特徵模組？**
A: 直接執行 `python shared_features.py` 進行測試，或使用 `test_setup.py`。

**Q: 獨立版本和共享模組版本有效能差異嗎？**
A: 沒有差異，獨立版本只是將代碼合併到單檔案，邏輯完全相同。

## 📞 支援

如有問題，請參考：
1. README.md - 基礎說明與使用範例
2. INTEGRATION_GUIDE.md - 詳細整合指南
3. shared_features.py - 特徵模組文檔字串
4. test_setup.py - 環境測試

## 🎉 總結

新的共享模組架構提供了：
- ✅ 無代碼重複
- ✅ 易於維護
- ✅ 清晰的結構
- ✅ 完整的功能
- ✅ 靈活的部署選項

推薦使用共享模組版本進行開發，需要時可生成獨立版本進行部署。
