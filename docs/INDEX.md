# A4T 目錄索引

快速導航到所需文件。

## 📖 文檔

| 檔案 | 用途 | 適合對象 |
|------|------|---------|
| [README.md](README.md) | 快速入門指南 | 所有人 |
| [SUMMARY.md](SUMMARY.md) | 完整總結與對比 | 決策者 |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | 詳細整合指南 | 開發者 |
| [INDEX.md](INDEX.md) | 本檔案 | 導航用 |

## 🛠️ 工具腳本

| 檔案 | 功能 | 執行時間 |
|------|------|---------|
| [test_setup.py](test_setup.py) | 環境測試 | < 1 分鐘 |
| [create_standalone.py](create_standalone.py) | 生成獨立腳本 | < 1 分鐘 |
| [example_simple.py](example_simple.py) | 簡化示例 | ~5 分鐘 |

## 📦 架構腳本

| 檔案 | 說明 | 狀態 |
|------|------|------|
| [preprocessing.py](preprocessing.py) | 資料載入與基礎函數 | 架構版本 |
| [train_lgbm.py](train_lgbm.py) | LightGBM 訓練架構 | 需補充特徵 |

## 🚀 生成的獨立腳本

執行 `python create_standalone.py --all` 後生成：

| 檔案 | 大小 | 特徵數 | 完整度 |
|------|------|--------|--------|
| train_lgbm_standalone.py | ~2000 行 | 120+ | 100% |
| train_graphsage_standalone.py | ~2500 行 | 140+ | 100% |
| train_ensemble_standalone.py | ~500 行 | Meta | 100% |

## 📂 原始檔案位置

完整的原始訓練腳本位於專案根目錄：

```
../train_lightgbm_v12.py          # LightGBM V12（600 行）
../train_graphsage_v13.py         # GraphSAGE V13（1500 行）
../train_ensemble_stacking_temporal.py  # Ensemble（300 行）

../feature_pool_system/
├── base_features.py              # 基礎特徵（500 行）
├── base_features_gat.py          # GAT 特徵（300 行）
└── pro_aml_features_v3.py        # AML 特徵（800 行）
```

## 🎯 使用流程圖

```
開始
  │
  ├─→ 學習理解？
  │     └─→ example_simple.py (5 分鐘)
  │
  ├─→ 環境測試？
  │     └─→ test_setup.py (1 分鐘)
  │
  ├─→ 完整訓練？
  │     ├─→ 模組化版本（推薦）
  │     │     └─→ train_lightgbm_v12.py (30 分鐘)
  │     │
  │     └─→ 獨立版本
  │           ├─→ create_standalone.py --all (1 分鐘)
  │           └─→ train_lgbm_standalone.py (30 分鐘)
  │
  └─→ 自定義開發？
        └─→ train_lgbm.py + 手動補充特徵
```

## 📊 效能對比表

| 版本 | 檔案數 | 總行數 | 特徵數 | ROC-AUC | 維護性 |
|------|--------|--------|--------|---------|--------|
| 簡化示例 | 1 | 200 | 15 | ~0.75 | ⭐⭐⭐⭐⭐ |
| 架構版本 | 2 | 500 | 50 | ~0.80 | ⭐⭐⭐⭐ |
| 模組化版本 | 4 | 2000 | 120+ | ~0.87 | ⭐⭐⭐⭐⭐ |
| 獨立版本 | 1 | 2000 | 120+ | ~0.87 | ⭐⭐⭐ |

## 🔗 快速連結

### 立即開始

```bash
# 1. 測試環境
python a4t/test_setup.py

# 2. 快速體驗
python a4t/example_simple.py

# 3. 完整訓練
python train_lightgbm_v12.py
```

### 生成獨立版本

```bash
# 生成所有獨立腳本
python a4t/create_standalone.py --all

# 或單獨生成
python a4t/create_standalone.py --model lgbm
python a4t/create_standalone.py --model graphsage
python a4t/create_standalone.py --model ensemble
```

### 查看文檔

```bash
# 快速入門
cat a4t/README.md

# 完整總結
cat a4t/SUMMARY.md

# 整合指南
cat a4t/INTEGRATION_GUIDE.md
```

## 💡 推薦路徑

### 新手入門

1. 閱讀 [README.md](README.md)
2. 執行 `python a4t/test_setup.py`
3. 執行 `python a4t/example_simple.py`
4. 閱讀 [SUMMARY.md](SUMMARY.md)

### 開發者

1. 閱讀 [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
2. 執行 `python train_lightgbm_v12.py`
3. 根據需求修改特徵
4. 參考 `feature_pool_system/` 模組

### 部署工程師

1. 執行 `python a4t/test_setup.py`
2. 執行 `python a4t/create_standalone.py --all`
3. 測試生成的獨立腳本
4. 部署到目標環境

## 📞 問題排查

| 問題 | 解決方案 | 參考文件 |
|------|---------|---------|
| 環境問題 | 執行 test_setup.py | README.md |
| 找不到資料 | 修改 BASE_DIR | README.md |
| 缺少套件 | pip install ... | README.md |
| 效能不佳 | 使用完整版本 | SUMMARY.md |
| 代碼太長 | 使用模組化版本 | INTEGRATION_GUIDE.md |
| 需要獨立 | 執行 create_standalone.py | INTEGRATION_GUIDE.md |

## 📈 版本歷史

- **V1.0** - 初始架構版本
- **V1.1** - 添加 create_standalone.py 工具
- **V1.2** - 添加 example_simple.py 示例
- **V1.3** - 添加完整文檔

## 🎓 學習資源

### 理解特徵工程

- `feature_pool_system/base_features.py` - 120+ 基礎特徵實現
- `feature_pool_system/pro_aml_features_v3.py` - AML 專業特徵
- `example_simple.py` - 簡化示例

### 理解模型訓練

- `train_lightgbm_v12.py` - LightGBM 完整實現
- `train_graphsage_v13.py` - GraphSAGE 完整實現
- `train_ensemble_stacking_temporal.py` - Ensemble 實現

### 理解評估指標

- ROC-AUC：模型整體區分能力
- PR-AUC：不平衡資料集的精確度
- F1 Score：精確率和召回率的調和平均

## 🏆 最佳實踐

1. **開發階段**：使用模組化版本（原始檔案）
2. **測試階段**：使用獨立版本（create_standalone.py）
3. **生產部署**：根據環境選擇模組化或獨立版本
4. **學習理解**：從 example_simple.py 開始

---

**最後更新**: 2024
**維護者**: A4T Team
**授權**: Internal Use Only
