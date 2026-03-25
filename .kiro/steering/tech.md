# 技術棧

## 程式語言
- Python 3.x

## 核心套件
- `lightgbm` — 梯度提升（gradient boosting），處理表格式特徵
- `torch` + `torch-geometric` — GraphSAGE 圖神經網路
- `networkx` — 圖結構建立與分析
- `scikit-learn` — 交叉驗證（cross-validation）、評估指標、資料前處理
- `numpy`、`pandas` — 資料操作
- `tqdm` — 進度條
- `community`（python-louvain）— Louvain 社群偵測（fallback：`networkx.algorithms.community`）

## 資料格式
- 輸入：JSONL 格式（每行一筆 JSON）
- 輸出：訓練結果以 `.pkl` 儲存（透過 `pickle`）

## 環境安裝

```bash
pip install -r requirements.txt
# 或使用 uv（推薦）
uv venv .venv
uv pip install -r requirements.txt
```

## 常用指令

```bash
# 執行訓練（依序執行）
uv run python train_lgbm.py          # 輸出：outputs/lgbm_v12_results.pkl
uv run python train_graphsage.py     # 輸出：outputs/graphsage_v13_results.pkl
uv run python train_ensemble.py      # 輸出：outputs/ensemble_results.pkl（需先完成上兩步）
```

## 交叉驗證策略

- LightGBM：`StratifiedKFold(n_splits=5)`
- GraphSAGE + Ensemble：`TimeSeriesSplit(n_splits=5)`（walk-forward 時序驗證）
