# 專案結構

```
/
├── config/
│   ├── __init__.py             # 路徑解析邏輯（優先 ./data，fallback 到 settings_local.py）
│   ├── settings_local.py       # 本地路徑設定（git ignore，不上傳）
│   └── settings_local.example.py  # 範本，複製後填入本地路徑
├── data/                       # JSONL 原始資料（git ignore）
│   ├── train_label.jsonl
│   ├── predict_label.jsonl
│   ├── user_info.jsonl
│   ├── twd_transfer.jsonl
│   ├── crypto_transfer.jsonl
│   ├── usdt_twd_trading.jsonl
│   └── usdt_swap.jsonl
├── outputs/                    # 訓練結果 .pkl（git ignore）
├── src/
│   ├── shared_features.py      # 核心特徵工程模組（唯一來源，勿複製）
│   └── preprocessing.py        # 獨立前處理（舊版，功能有限）
├── docs/
├── train_lgbm.py               # LightGBM 訓練（步驟 1）
├── train_graphsage.py          # GraphSAGE 訓練（步驟 2）
├── train_ensemble.py           # Stacking 集成（步驟 3，依賴步驟 1、2）
├── requirements.txt
└── README.md
```

## 路徑解析規則

`config/__init__.py` 的 `get_data_dir()` 依序嘗試：
1. `./data/` 資料夾（存在 `train_label.jsonl` 即採用）
2. `config/settings_local.py` 中的 `DATA_DIR`
3. 兩者皆失敗則拋出明確錯誤訊息

## 主要慣例

- `src/shared_features.py` 是所有特徵工程的唯一來源，訓練腳本只 import，不複製邏輯
- 新增特徵請在 `src/shared_features.py` 的 `_compute_user_features()` 中加入
- 特徵矩陣格式固定為 `np.ndarray`，shape `(n_users, n_features)`，搭配平行的 `feature_names: List[str]`
- 社群特徵（community features）必須在每個 CV fold 內部計算，只使用 `train_indices`，嚴禁在 split 前全局計算
- 結果以 `.pkl` dict 儲存，固定 key：`oof_predictions`、`y_true`、`feature_names`、各 fold 指標
- 訓練腳本透過 `sys.path.insert` 將 `src/` 加入路徑，不需安裝套件
