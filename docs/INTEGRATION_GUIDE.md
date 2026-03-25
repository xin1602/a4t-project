# 整合指南：如何創建完全獨立的訓練腳本

本指南說明如何將原始訓練腳本轉換為完全獨立的版本。

## 問題分析

原始檔案的依賴關係：

```
train_lightgbm_v12.py
├── feature_pool_system/base_features.py (500 行)
├── feature_pool_system/pro_aml_features_v3.py (800 行)
└── 總計需要內嵌 ~1300 行代碼

train_graphsage_v13.py  
├── feature_pool_system/base_features_gat.py (300 行)
├── feature_pool_system/base_features.py (500 行)
├── feature_pool_system/pro_aml_features_v3.py (800 行)
└── 總計需要內嵌 ~1600 行代碼

train_ensemble_stacking_temporal.py
├── 依賴上述兩個模型的輸出
└── 相對獨立，但需要預訓練模型
```

## 方案 A：直接使用原始檔案（推薦）

原始檔案已經相對獨立，只需確保 `feature_pool_system/` 目錄存在：

```bash
# 項目結構
project/
├── feature_pool_system/
│   ├── __init__.py
│   ├── base_features.py
│   ├── base_features_gat.py
│   └── pro_aml_features_v3.py
├── train_lightgbm_v12.py
├── train_graphsage_v13.py
└── train_ensemble_stacking_temporal.py

# 直接執行
python train_lightgbm_v12.py
```

## 方案 B：創建單檔案版本

### 步驟 1：提取特徵計算函數

從 `feature_pool_system/base_features.py` 複製：

```python
# 複製這些函數到你的獨立腳本
def get_base_features_v2(user_id, user_info, twd_txs, crypto_txs, trading_txs, swap_txs):
    # ... 500 行代碼 ...
    pass

def parse_date(created_at):
    # ... 輔助函數 ...
    pass

def calculate_hhi(counts):
    # ... 輔助函數 ...
    pass
```

### 步驟 2：提取資料品質函數

從 `feature_pool_system/pro_aml_features_v3.py` 複製：

```python
# 複製這些常數和函數
INVALID_IP_HASHES = {
    'cfcd208495d565ef66e7dff9f98764da',
    '', 'null', '00000000000000000000000000000000',
}

def is_valid_ip(ip_hash):
    # ... 實現 ...
    pass

def compute_time_delta_hours(start, end):
    # ... 實現 ...
    pass
```

### 步驟 3：整合到訓練腳本

```python
# train_lgbm_standalone.py

"""
完全獨立的 LightGBM 訓練腳本
包含所有必要的特徵工程代碼
"""

import json
import numpy as np
# ... 其他導入 ...

# ==================== 第一部分：輔助函數 ====================
# 從 pro_aml_features_v3.py 複製
INVALID_IP_HASHES = {...}
def is_valid_ip(ip_hash): ...
def parse_date(date_str): ...

# ==================== 第二部分：特徵計算 ====================
# 從 base_features.py 複製
def get_base_features_v2(...): ...
def calculate_hhi(...): ...

# ==================== 第三部分：訓練邏輯 ====================
# 原始 train_lightgbm_v12.py 的主要邏輯
def main():
    # 載入資料
    train_labels, user_info, ... = load_data(BASE_DIR)
    
    # 計算特徵
    features = []
    for uid in user_ids:
        feat = get_base_features_v2(...)
        features.append(feat)
    
    # 訓練模型
    # ... LightGBM 訓練代碼 ...

if __name__ == "__main__":
    main()
```

## 方案 C：模組化方法（平衡）

創建一個共享的特徵模組：

```
a4t/
├── shared_features.py      # 所有特徵計算函數（1500 行）
├── train_lgbm.py           # LightGBM 訓練（300 行）
├── train_graphsage.py      # GraphSAGE 訓練（500 行）
└── train_ensemble.py       # Ensemble 訓練（200 行）
```

`shared_features.py` 內容：
```python
"""
共享特徵工程模組
包含所有模型需要的特徵計算函數
"""

# 從 feature_pool_system/ 複製所有必要函數
# 這樣每個訓練腳本只需 import shared_features
```

訓練腳本：
```python
from shared_features import get_base_features_v2, is_valid_ip, ...

# 其餘訓練邏輯
```

## 實際操作示例

### 創建 LightGBM 獨立版本

```bash
# 1. 創建新檔案
touch train_lgbm_standalone.py

# 2. 複製特徵代碼
cat feature_pool_system/base_features.py >> train_lgbm_standalone.py
cat feature_pool_system/pro_aml_features_v3.py >> train_lgbm_standalone.py

# 3. 複製訓練邏輯
cat train_lightgbm_v12.py >> train_lgbm_standalone.py

# 4. 手動調整 import 語句（移除外部導入）
```

### 創建 GraphSAGE 獨立版本

```bash
# 類似步驟，但需要額外複製
cat feature_pool_system/base_features_gat.py >> train_graphsage_standalone.py
```

## 推薦方案

根據使用場景選擇：

| 場景 | 推薦方案 | 原因 |
|------|---------|------|
| 生產部署 | 方案 A（原始檔案） | 最穩定，易維護 |
| 代碼審查 | 方案 C（模組化） | 結構清晰 |
| 單次執行 | 方案 B（單檔案） | 完全獨立 |
| 學習理解 | 方案 C（模組化） | 易於閱讀 |

## 注意事項

1. **代碼重複**：單檔案版本會有大量重複代碼
2. **維護成本**：修改特徵需要同步更新多個檔案
3. **檔案大小**：單檔案版本可能超過 2000 行
4. **測試困難**：大檔案難以進行單元測試

## 結論

**最佳實踐**：保持原始的模組化結構，使用相對導入：

```python
# 在訓練腳本中
import sys
sys.path.insert(0, '.')  # 確保可以找到 feature_pool_system

from feature_pool_system.base_features import get_base_features_v2
from feature_pool_system.pro_aml_features_v3 import is_valid_ip
```

這樣既保持了代碼的可維護性，又確保了功能的完整性。
