# A4T — 加密貨幣詐欺偵測系統

A4T 是一套針對加密貨幣交易所的詐欺偵測系統，透過分析使用者在多種交易類型中的行為模式來識別詐欺帳號，並提供視覺化 Dashboard 供分析師使用。

## Demo

https://github.com/user-attachments/assets/1a52d183-1e65-48d1-9ce9-21c2c787b069

---

## 系統架構

```
前端（Vite + React）  ──→  後端（FastAPI）  ──→  模型（LightGBM + GraphSAGE + Ensemble）
     port 3000                port 8001                   outputs/*.pkl
```

### 模型架構

| 模型 | 說明 |
|------|------|
| LightGBM | 處理 120+ 個表格式行為特徵 |
| GraphSAGE | 圖神經網路，捕捉使用者間的交易關聯 |
| Stacking Ensemble | 結合上述兩個模型的預測結果 |

### 涵蓋交易類型

- TWD 轉帳（`twd_transfer`）
- 加密貨幣轉帳（`crypto_transfer`）
- USDT/TWD 交易（`usdt_twd_trading`）
- USDT 兌換（`usdt_swap`）

---

## 快速開始

### 前置條件

- Python 3.8+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js + npm

### 1. 安裝後端依賴

```bash
uv pip install -r backend/requirements-dashboard.txt
```

### 2. 設定環境變數

複製範本並填入 OpenRouter API 金鑰：

```bash
cp .env.example .env
# 編輯 .env，填入 OPENROUTER_API_KEY
```

### 3. 確認模型檔案存在

後端啟動時需要以下三個 pkl 檔案：

```
outputs/lgbm_v13_no_graph_results.pkl
outputs/graphsage_v13_gpu_complete_results.pkl
outputs/ensemble_results.pkl
```

若尚未訓練，請依序執行：

```bash
uv run python train_lgbm.py
uv run python train_graphsage.py
uv run python train_ensemble.py
uv run python scripts/compute_overview.py
```

### 4. 啟動後端

```bash
uv run uvicorn backend.app.main:app --port 8001 --reload
```

### 5. 啟動前端

```bash
cd frontend
npm install
npm run dev
```

開啟瀏覽器訪問：**http://127.0.0.1:3000**

---

## 專案結構

```
a4t-project/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI 入口
│       ├── routers/             # API 路由
│       └── services/            # 業務邏輯（模型推論、LLM、圖分析）
├── frontend/
│   └── src/
│       ├── pages/               # 頁面元件
│       ├── components/          # 共用元件
│       └── api/client.ts        # API 呼叫層
├── src/
│   └── shared_features.py       # 核心特徵工程模組
├── data/                        # 原始 JSONL 資料（git ignore）
├── outputs/                     # 訓練產出 pkl（git ignore）
├── docs/
│   ├── LOCAL_DEV.md             # 本地開發啟動指南
│   └── assets/                  # 靜態資源（demo 影片等）
├── .env.example                 # 環境變數範本
├── train_lgbm.py                # LightGBM 訓練（步驟 1）
├── train_graphsage.py           # GraphSAGE 訓練（步驟 2）
└── train_ensemble.py            # Stacking 集成訓練（步驟 3）
```

---

## 文件

| 文件 | 說明 |
|------|------|
| [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md) | 本地開發啟動指南（含常見問題排查） |
| [docs/dashboard_design.md](docs/dashboard_design.md) | Dashboard 設計文件 |
| [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) | AWS 部署指南 |
