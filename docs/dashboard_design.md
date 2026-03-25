# A4T 詐欺偵測儀表板 — 設計文件 v2.0

> 本文件涵蓋：使用者情境、UI/UX 設計、可解釋性架構、LLM 串接設計、新用戶預測流程、Live Demo 腳本，以及可實現 vs. 目前缺少的功能清單。
>
> 模擬資料段落均以 `【模擬資料】` 標註。

---

## 一、使用者情境（User Scenarios）

### 角色定義

| 角色 | 主要任務 | 技術熟悉度 |
|------|---------|-----------|
| 風控分析師 | 審查高風險帳號、決定凍結/放行 | 中 |
| 合規人員 | AML 報告、監管溝通 | 低 |
| 資安工程師 | 模型監控、閾值調整、新資料推論 | 高 |
| 客服人員 | 處理申訴、解釋凍結原因 | 低 |

---

### 情境 A：風控分析師的早晨例行審查

**觸發點**：早上 9 點登入儀表板

1. 首頁顯示昨日新增高風險用戶 12 筆，待審查 8 筆
2. 分析師點開風險分數最高的用戶（user_id: U-00234）
3. 用戶查詢頁顯示：風險分數 0.91、SHAP Top 5 特徵、交易時間熱力圖
4. 分析師發現該用戶 87% 的加密貨幣交易集中在凌晨 2–4 點
5. 點擊「圖網絡」，以 U-00234 為 root 展開 1-hop 關聯圖
6. 發現 3 個鄰居節點也被預測為高風險（橘色），其中 1 個已確認詐欺（紅色）
7. 圖網絡下方自動顯示 LLM 生成的群聚摘要（無需提問，頁面載入即觸發）：

   > **節點角色**：U-00234 的 `graph_degree=8`，遠高於全體用戶中位數（2），在此群聚中扮演「資金分發者」角色（hub）；鄰居 U-00891 的 `crypto_inflow_sum` 佔群聚總流入的 63%，為主要「資金接收者」（authority）。
   >
   > **群聚特性**：此 1-hop 群聚共 9 個節點，其中 4 個（44%）被預測為高風險，遠高於全體詐欺率（約 3–5%）。社群密度 0.34，屬中高密度群聚，節點間存在多條直接交易路徑，符合「環形洗錢」結構特徵。

8. 在 LLM 問答框輸入：「這個群聚的資金流向模式是什麼？」
9. LLM 回答（含參考特徵與統計比較，詳見第四章 4.4 節）
10. 分析師標記案件狀態為「已凍結」，系統記錄稽核軌跡

---

### 情境 B：客服人員處理申訴

**觸發點**：用戶來電申訴帳號被凍結

1. 客服搜尋 user_id，進入用戶查詢頁
2. 頁面顯示「白話文診斷書」（由 LLM 生成）：「您的帳號因以下異常行為被標記為高風險：(1) 多筆交易集中在深夜時段 (2) 資金快速流入後立即轉出...」
3. 客服直接複製診斷書內容向用戶說明，無需理解技術術語

---

### 情境 C：資安工程師上傳新資料批次預測

**觸發點**：每日 00:00 排程或手動觸發

資料來源：直接從 BitoPro 官方 API 抓取
- Swagger UI：https://aws-event-docs.bitopro.com/
- API 端點：https://aws-event-api.bitopro.com/
- 預測目標：`predict_label.jsonl` 中的 user_id 清單（無標籤，需模型推論）

1. 系統透過 API 自動拉取當日新增用戶的交易記錄（`twd_transfer`、`crypto_transfer`、`usdt_twd_trading`、`usdt_swap`）
2. 後端執行特徵工程（呼叫 `shared_features._compute_user_features()`）+ 模型推論（約 30 秒）
3. 完成後右上角出現通知：「批次預測完成，發現 5 位高風險用戶」
4. 點擊通知跳轉案件管理頁，新增用戶已自動進入待審查列表
5. 亦支援手動上傳 CSV（欄位對應 `feature_names`）作為 API 不可用時的 fallback

> 【模擬資料】：`predict_label.jsonl` 目前只有 user_id，無 status 欄位，批次預測結果的準確率無法驗證，Demo 時以模擬高風險用戶清單呈現。

---

### 情境 D：新用戶單筆即時評估

**觸發點**：新用戶完成 KYC，資安工程師需要快速評估

1. 進入「即時預測 > 單筆預測」
2. 填入用戶基本資訊與交易摘要（或貼上 JSON）
3. 點擊「立即評估」，2 秒內回傳風險分數與 SHAP 解釋
4. 若分數 > 閾值，系統詢問是否加入案件管理列表


---

## 二、UI/UX 設計

### 頁面架構

```
A4T 儀表板
├── 首頁（風險總覽）
│   ├── KPI 卡片：今日高風險數 / 詐欺率 / 待處理案件數
│   ├── 詐欺率趨勢折線圖（30 天）
│   ├── 風險分數分布直方圖（全體用戶）
│   └── Top 10 高風險用戶快速連結
│
├── 用戶查詢
│   ├── 搜尋列（user_id）
│   ├── 風險分數儀表板 + 等級標籤（低/中/高/極高）
│   ├── 關鍵數字卡片（總交易量、法幣金額、加密金額、夜間交易比例）
│   ├── SHAP 特徵貢獻橫向長條圖（Top 10，正向紅色 / 負向藍色）
│   ├── GNNExplainer 關鍵子圖（高風險用戶專屬，顯示最可疑的交易路徑）
│   ├── 交易時間熱力圖（24 小時 × 7 天）
│   ├── 白話文風險診斷書（LLM 生成）
│   ├── 資金流向網絡圖（以此用戶為 root，1-hop 預設）
│   └── LLM 問答框
│
├── 圖網絡分析（獨立頁面，支援更大範圍探索）
│   ├── 搜尋列（user_id 作為 root）
│   ├── 互動式圖（縮放 / 拖曳 / 節點固定）
│   ├── 1-hop / 2-hop 切換
│   ├── 節點 tooltip（user_id、風險分數、黑名單狀態、Top 3 特徵）
│   ├── 邊的方向與粗細（資金流向與金額）
│   ├── 圖例（紅=確認詐欺 / 橘=高風險 / 黃=待審查 / 灰=正常）
│   └── LLM 問答框（群聚層級）
│
├── 即時預測
│   ├── 單筆預測（表單輸入 or JSON 貼上）
│   └── 批次預測（CSV 上傳 + 結果下載）
│
├── 案件管理
│   ├── 高風險用戶列表（可排序 / 篩選 / 批次操作）
│   ├── 案件狀態標記（待審查 / 已凍結 / 誤判放行）
│   └── 匯出（CSV / PDF 含 SHAP 圖）
│
└── 模型監控
    ├── 當前模型資訊（版本、訓練日期、CV 指標）
    ├── 閾值調整滑桿 + 即時 Precision / Recall / F1 預覽
    ├── 全局 SHAP 特徵重要性圖（Top 20）
    └── 實驗版本比較表
```

---

### 圖網絡節點設計

每個節點顯示以下資訊（inline 標籤 + hover tooltip）：

```
┌─────────────────────────────┐
│  ● U-00234  [高風險]        │  ← 節點顏色：橘色
│  風險分數：0.91             │
│  加密轉帳總額：$142,000     │
│  夜間交易比例：87%          │
│  社群大小：12               │
│  → 點擊進入用戶查詢頁       │
└─────────────────────────────┘
```

節點大小：依 `graph_degree`（連接度）決定，度數越高節點越大。
邊的粗細：依交易金額決定；邊的方向：箭頭指向資金接收方。

---

### 設計原則

1. 可解釋優先：每個風險判斷都必須有人類可讀的理由，不能只顯示分數
2. 行動導向：每個頁面都有明確的下一步動作（標記 / 凍結 / 匯出）
3. 漸進揭露：客服看白話文診斷書，分析師可展開 SHAP 技術細節
4. 稽核可追溯：所有操作留存記錄，支援監管查核
5. 效能優先：查詢單一用戶 < 1 秒，圖網絡 1-hop < 2 秒，總覽頁 < 3 秒


---

## 三、可解釋性設計（XAI Architecture）

本系統採用多層次可解釋性策略，針對不同模型與不同解釋需求選用最適合的方法。

### 3.1 可解釋性方法對照表

| 方法 | 適用模型 | 解釋層次 | 解釋對象 | 目前可實現 |
|------|---------|---------|---------|-----------|
| SHAP | LightGBM | 特徵層 | 帳戶靜態行為（金額、頻率、夜間比例） | ✅ 可實現 |
| SHAP on Aggregated Embedding | GraphSAGE | 特徵層 | GraphSAGE 聚合後的宏觀行為特徵 | ✅ 可實現 |
| GNNExplainer | GraphSAGE | 拓撲層 | 關鍵交易對象與資金流向路徑 | ⚠️ 需額外實作 |
| Integrated Gradients (IG) | GraphSAGE | 梯度層 | 輸入特徵對預測結果的歸因 | ⚠️ 需額外實作 |
| Attention 權重視覺化 | GAT（未實作） | 注意力層 | 不同交易鄰居的重要性權重 | ❌ 需改用 GAT |

---

### 3.2 SHAP — 解釋帳戶行為特徵

**適用場景**：解釋「為什麼這個帳號的靜態行為特徵導致高風險評分」

**目前可用特徵**（來自 `shared_features.py`）：

| 特徵名稱 | 業務說明 | 典型詐欺模式 |
|---------|---------|------------|
| `crypto_night_tx_ratio` | 加密貨幣夜間交易比例 | 詐欺帳號常在深夜（23:00–06:00）操作 |
| `crypto_to_wallet_hhi` | 轉出錢包集中度（HHI 指數） | 高集中度 = 反覆轉給同一個外部錢包 |
| `crypto_net_flow` | 加密貨幣淨流量（流入 - 流出） | 大量流入後快速清空 = 洗錢中轉 |
| `twd_outflow_sum` | 法幣總流出金額 | 異常大額出金 |
| `crypto_external_ratio` | 外部轉帳比例（非平台內部） | 高比例 = 資金快速流出平台 |
| `graph_degree` | 圖連接度（關聯用戶數） | 高度數 = 與多個可疑帳號有往來 |
| `community_size` | 所屬社群大小 | 大型社群中的詐欺帳號風險更高 |
| `crypto_amount_cv` | 加密貨幣金額變異係數 | 金額極不穩定 = 刻意規避偵測 |

**儀表板呈現**：
- 橫向長條圖，正向貢獻（推向詐欺）顯示紅色，負向貢獻（推向正常）顯示藍色
- 每個特徵旁顯示業務說明文字，而非原始英文特徵名稱
- 支援展開查看特徵的實際數值與全體用戶的分布位置（百分位數）

---

### 3.3 SHAP on Aggregated Embedding — 解釋 GraphSAGE 宏觀特徵

**適用場景**：解釋 GraphSAGE 聚合鄰居資訊後，哪些宏觀行為特徵影響最大

**做法**：
1. 將 GraphSAGE 最後一層的 embedding（128 維）或輸入特徵矩陣（含圖特徵）丟入 SHAP TreeExplainer / KernelExplainer
2. 解釋「鄰居平均風險分數」、「鄰居平均交易量」等聚合特徵的貢獻
3. 與 LightGBM SHAP 並排顯示，讓分析師比較兩個模型的判斷依據是否一致

---

### 3.4 GNNExplainer — 解釋關鍵子圖與資金路徑

**適用場景**：針對特定黑名單預測，找出「哪些交易路徑最可疑」

**解釋邏輯**：
- GNNExplainer 對每個目標節點（高風險用戶）學習一個邊遮罩（edge mask）與特徵遮罩（feature mask）
- 邊遮罩高的邊 = 對黑名單判定貢獻最大的交易關係
- 特徵遮罩高的特徵 = 在圖傳播中最關鍵的節點屬性

**儀表板呈現**：
- 在圖網絡頁面，GNNExplainer 識別出的「關鍵邊」以紅色加粗顯示
- 關鍵邊旁標示貢獻分數（0–1）
- 提供文字說明：「模型認為 U-00234 → U-00891 這條轉帳路徑對黑名單判定貢獻最大（貢獻分數：0.87）」

**實作需求**：需在 `train_graphsage.py` 訓練完成後，額外執行 `torch_geometric.explain.GNNExplainer`

---

### 3.5 Integrated Gradients (IG) — 量化特徵歸因

**適用場景**：量化每個輸入特徵對 GraphSAGE 預測結果的梯度歸因

**做法**：
1. 以全零向量（或特徵均值）作為 baseline
2. 沿 baseline → 實際輸入的路徑積分梯度
3. 輸出每個特徵的歸因分數，正值 = 推向詐欺，負值 = 推向正常

**與 SHAP 的差異**：
- SHAP 解釋的是「相對於平均預測的偏差」，適合 LightGBM 等樹模型
- IG 解釋的是「梯度路徑上的累積貢獻」，更適合神經網路（GraphSAGE）
- 兩者結果可互相驗證，若一致性高則可信度更強

**實作需求**：使用 `captum` 套件（`pip install captum`）

---

### 3.6 可解釋性整合流程

```
用戶被預測為黑名單（風險分數 > 閾值）
         │
         ├─→ LightGBM SHAP
         │     └─→ 解釋：哪些帳戶行為特徵（金額、頻率、時間）導致高分
         │
         ├─→ GraphSAGE SHAP on Embedding
         │     └─→ 解釋：鄰居的哪些宏觀特徵影響了聚合結果
         │
         ├─→ GNNExplainer（關鍵子圖）
         │     └─→ 解釋：哪條交易路徑最可疑（視覺化在圖網絡上）
         │
         └─→ LLM 整合以上三層解釋
               └─→ 生成白話文診斷書，供客服與合規人員使用
```


---

## 四、LLM 串接設計

### 4.1 架構概覽

LLM 使用 **Amazon Bedrock**，透過 AWS SDK（`boto3`）呼叫，無需管理模型基礎設施。

```
前端問答框
    │  用戶輸入問題
    ▼
後端 API（/api/llm/ask）
    │
    ├─→ 組裝 Context（見 4.2）
    │
    ├─→ 呼叫 Amazon Bedrock（串流輸出）
    │     模型選項：
    │     - anthropic.claude-3-5-sonnet-20241022-v2:0（推薦，推理能力強）
    │     - amazon.nova-pro-v1:0（AWS 原生，成本較低）
    │     API：bedrock-runtime.invoke_model_with_response_stream()
    │
    └─→ 串流回傳前端（SSE）
```

---

### 4.2 後端 Context 組裝規格

後端在呼叫 LLM 前，依當前頁面情境自動組裝以下資訊：

#### 單一用戶查詢頁（Context Type: `single_user`）

```json
{
  "context_type": "single_user",
  "user_profile": {
    "user_id": "U-00234",
    "age": 32,
    "career": "自由業",
    "income_source": "投資",
    "has_kyc_level1": true,
    "has_kyc_level2": false
  },
  "prediction": {
    "risk_score": 0.91,
    "is_blacklist": true,
    "model": "Ensemble (LightGBM + GraphSAGE)",
    "threshold": 0.35
  },
  "shap_top10": [
    {"feature": "crypto_night_tx_ratio", "value": 0.87, "shap": 0.42, "description": "加密貨幣夜間交易比例"},
    {"feature": "crypto_to_wallet_hhi", "value": 0.93, "shap": 0.31, "description": "轉出錢包集中度"},
    {"feature": "crypto_net_flow",       "value": -145000, "shap": 0.28, "description": "加密貨幣淨流量（流入-流出）"},
    {"feature": "graph_degree",          "value": 8,      "shap": 0.19, "description": "關聯用戶數（圖連接度）"},
    {"feature": "crypto_external_ratio", "value": 0.95,   "shap": 0.17, "description": "外部轉帳比例"}
  ],
  "graph_features": {
    "graph_degree": 8,
    "graph_clustering": 0.12,
    "graph_component_size": 15,
    "community_size": 12,
    "community_density": 0.34
  },
  "neighbor_summary": {
    "total_neighbors": 8,
    "high_risk_neighbors": 3,
    "confirmed_fraud_neighbors": 1
  },
  "transaction_summary": {
    "total_tx_count": 47,
    "crypto_tx_count": 31,
    "twd_tx_count": 12,
    "crypto_night_tx_ratio": 0.87,
    "crypto_total_amount_twd": 142000,
    "twd_total_amount": 38000
  },
  "feature_glossary": {
    "crypto_night_tx_ratio": "加密貨幣夜間交易比例（23:00–06:00）",
    "crypto_to_wallet_hhi": "轉出錢包集中度，HHI 越高代表反覆轉給同一個外部錢包",
    "crypto_net_flow": "加密貨幣淨流量，負值代表大量流出",
    "graph_degree": "在交易網絡中的連接數，越高代表與越多用戶有往來",
    "crypto_external_ratio": "轉往平台外部錢包的比例"
  },
  "conversation_history": []
}
```

#### 圖網絡群聚頁（Context Type: `graph_cluster`）

在 `single_user` context 基礎上，額外加入：

```json
{
  "context_type": "graph_cluster",
  "cluster_summary": {
    "root_user_id": "U-00234",
    "hop": 1,
    "total_nodes": 9,
    "high_risk_nodes": 4,
    "confirmed_fraud_nodes": 1,
    "nodes": [
      {"user_id": "U-00234", "risk_score": 0.91, "is_blacklist": true,  "degree": 8},
      {"user_id": "U-00891", "risk_score": 0.78, "is_blacklist": true,  "degree": 5},
      {"user_id": "U-01203", "risk_score": 0.23, "is_blacklist": false, "degree": 2}
    ],
    "gnnexplainer_top_edges": [
      {"from": "U-00234", "to": "U-00891", "importance": 0.87, "amount_twd": 85000},
      {"from": "U-00891", "to": "U-02341", "importance": 0.72, "amount_twd": 62000}
    ]
  }
}
```

---

### 4.3 System Prompt 設計

```
你是 A4T 詐欺偵測系統的智能分析助理，專門協助風控分析師、合規人員與客服人員理解模型的預測結果。

你的職責：
1. 用繁體中文、清晰易懂的語言解釋模型判斷依據
2. 將技術特徵名稱翻譯為業務語言（參考 feature_glossary）
3. 針對圖網絡問題，說明資金流向模式與可疑路徑
4. 生成可直接對外使用的白話文診斷書（客服模式）
5. 不捏造資料，若資訊不足請明確說明

回答風格：
- 風控分析師：提供技術細節，引用 SHAP 數值與特徵名稱
- 客服人員：使用白話文，避免技術術語，語氣中立不帶指控
- 合規人員：強調法規相關性，提供可引用的結構化說明

限制：
- 不做最終凍結/放行決策，只提供分析依據
- 不透露模型內部參數或訓練資料細節
```

---

### 4.4 典型問答範例

**問**：「為什麼 U-00234 被判定為黑名單？」

**LLM 回答**（風控分析師模式）：
> U-00234 的集成模型風險分數為 0.91，超過閾值 0.35，主要由以下三個因素驅動：
> 1. 夜間交易比例極高（87%）：該用戶 87% 的加密貨幣交易發生在深夜 23:00–06:00，SHAP 貢獻值 +0.42，是最主要的風險訊號。相較之下，全體正常用戶的夜間交易比例中位數僅 8%，U-00234 位於第 99.2 百分位。
> 2. 轉出錢包高度集中（HHI=0.93）：幾乎所有加密貨幣都轉往同一個外部錢包，符合洗錢中轉的典型模式，SHAP 貢獻值 +0.31。全體用戶的 HHI 中位數為 0.41，U-00234 屬極端值。
> 3. 圖網絡關聯風險（graph_degree=8）：該用戶在交易網絡中連接 8 個用戶，其中 3 個被預測為高風險、1 個已確認詐欺，社群大小 12 人。全體用戶的平均 degree 為 1.7，U-00234 的連接度為平均值的 4.7 倍。

**綜合結論**：U-00234 同時具備「深夜高頻操作」、「資金單向集中流出」、「與已知詐欺帳號直接關聯」三項高風險特徵，三者在模型中相互強化，導致集成風險分數達 0.91。建議優先列入人工審查。

---

**問**：「這個群聚的資金流向模式是什麼？」

**LLM 回答**（圖網絡群聚模式）：
> 根據 1-hop 圖網絡分析，U-00234 所在群聚呈現典型的「星型中轉」結構：
>
> **資金流向**：U-00234（hub，degree=8）在 3 天內接收來自 2 個已知詐欺帳號（U-00891、U-02341）共約 14.7 萬元的加密貨幣，隨後在 6 小時內將 95% 的資金轉出至 1 個外部錢包（`crypto_external_ratio=0.95`）。GNNExplainer 識別 U-00234 → 外部錢包 這條路徑的貢獻分數為 0.87，為群聚中最可疑的交易路徑。
>
> **參考特徵數值**（與全體用戶比較）：
>
> | 特徵 | U-00234 | 全體中位數 | 百分位 |
> |------|---------|-----------|--------|
> | `crypto_net_flow`（淨流量） | -145,000 TWD | -1,200 TWD | 第 98 百分位 |
> | `crypto_external_ratio`（外部轉帳比例） | 0.95 | 0.31 | 第 97 百分位 |
> | `graph_degree`（連接度） | 8 | 2 | 第 96 百分位 |
> | `community_density`（社群密度） | 0.34 | 0.08 | 第 94 百分位 |
>
> **結論**：此群聚高度符合「收款 → 快速轉出 → 外部提領」的洗錢中轉模式。群聚內 44% 節點為高風險（全體詐欺率約 3–5%），建議對群聚內所有節點啟動關聯審查。

> 【模擬資料】：上述特徵數值與百分位數為示意用途，實際數值需從訓練資料計算。

---

**問**：「幫我生成一段可以向用戶說明的文字」

**LLM 回答**（客服模式）：
> 您好，您的帳號因系統偵測到以下異常行為而被暫時限制：(1) 您的多筆加密貨幣交易集中在深夜時段，與一般用戶的使用習慣有明顯差異；(2) 您的資金轉出模式較為集中，系統判斷存在異常風險。如您認為此判斷有誤，歡迎提供相關交易說明，我們將進行人工審查。


---

## 五、新用戶 / 新資料預測設計

### 5.1 單筆即時預測流程

```
用戶輸入（表單 or JSON）
    │
    ▼
後端 /api/predict/single
    │
    ├─→ 特徵工程（呼叫 shared_features._compute_user_features()）
    │     輸入：user_info + 各類交易記錄
    │     輸出：特徵向量（與訓練時相同的 feature_names 順序）
    │
    ├─→ 圖特徵計算（可選）
    │     若提供歷史交易對象，計算 graph_degree / community_size
    │     若無歷史資料，圖特徵填入全體均值（【模擬資料】）
    │
    ├─→ 載入已訓練模型（outputs/*.pkl）
    │     LightGBM model + GraphSAGE model + Ensemble meta-learner
    │
    ├─→ 推論 → 風險分數（0–1）
    │
    └─→ 計算 SHAP 解釋 → 回傳前端
```

**前端表單欄位**（對應 `_compute_user_features()` 的輸入）：

| 欄位群組 | 欄位 | 說明 |
|---------|------|------|
| 基本資料 | age, sex, career, income_source | 來自 user_info |
| KYC 狀態 | has_kyc_level1, has_kyc_level2 | 是否完成實名驗證 |
| 加密貨幣交易 | crypto_tx_count, crypto_total_amount, crypto_night_tx_ratio | 交易摘要 |
| 法幣交易 | twd_tx_count, twd_total_amount, twd_outflow_sum | 法幣轉帳摘要 |
| 流向特徵 | crypto_inflow_sum, crypto_outflow_sum, crypto_external_ratio | 資金流向 |
| 錢包集中度 | crypto_to_wallet_hhi, crypto_to_wallet_diversity | 對手方分散程度 |

---

### 5.2 批次預測流程

資料來源：BitoPro 官方 API（https://aws-event-api.bitopro.com/）
- 預測目標清單：`predict_label.jsonl`（僅含 user_id，無標籤）
- Swagger 文件：https://aws-event-docs.bitopro.com/

```
API 拉取新用戶交易資料
    │  GET /users/{user_id}/transactions（各交易類型）
    ▼
後端 /api/predict/batch（非同步任務）
    │
    ├─→ 驗證資料完整性（欄位對應 feature_names）
    ├─→ 批次特徵工程（逐行呼叫 _compute_user_features()）
    ├─→ 批次推論（LightGBM + GraphSAGE + Ensemble）
    ├─→ 高風險用戶（分數 > 閾值）自動加入案件管理列表
    └─→ 推送通知 + 可下載結果 CSV
```

亦支援手動上傳 CSV 作為 API 不可用時的 fallback，欄位需對應 `feature_names`。

**結果 CSV 欄位**：
`user_id, risk_score, is_blacklist, top_feature_1, top_feature_1_shap, top_feature_2, top_feature_2_shap, top_feature_3, top_feature_3_shap`

---

### 5.3 新用戶的圖特徵處理策略

新用戶尚未出現在訓練圖中，圖特徵無法直接計算，有以下三種策略：

| 策略 | 做法 | 適用情境 |
|------|------|---------|
| 填入全體均值 | `graph_degree=0, community_size=1` | 完全新用戶，無任何交易記錄 |
| 動態插入圖 | 將新用戶的交易對象加入現有圖，重新計算局部圖特徵 | 新用戶已有少量交易記錄 |
| 僅用 LightGBM | 跳過 GraphSAGE，只用 LightGBM 預測 | 圖特徵不可用時的 fallback |

目前系統採用「填入全體均值」作為預設策略，動態插入圖需額外實作。


---

## 六、Live Demo 流程腳本

> 建議 Demo 時長：15–20 分鐘
> 目標受眾：產品主管、風控主管、技術評審

### 第一幕：風險總覽（2 分鐘）

**操作**：開啟首頁

**說明重點**：
- 「這是今日的風險概況，目前有 47 位用戶被模型預測為高風險，其中 12 位待審查」
- 指向詐欺率趨勢圖：「過去 30 天詐欺率穩定在 3–5%，昨日有小幅上升，值得關注」
- 指向風險分數分布圖：「大多數用戶集中在低風險區間，高分尾端的用戶是我們的重點審查對象」

> 【模擬資料】：趨勢圖與分布圖數據為模擬，實際數值需從 `outputs/ensemble_results.pkl` 的 `oof_predictions` 計算。

---

### 第二幕：單一用戶深度分析（5 分鐘）

**操作**：在搜尋列輸入 `U-00234`，進入用戶查詢頁

**說明重點**：
1. 指向風險分數儀表板：「這位用戶的集成模型風險分數是 0.91，屬於極高風險」
2. 指向 SHAP 圖：「最主要的風險訊號是夜間交易比例 87%，以及轉出錢包高度集中——這兩個特徵合計貢獻了超過 70% 的風險分數」
3. 指向交易時間熱力圖：「你可以清楚看到，這位用戶幾乎所有交易都集中在凌晨 2–4 點，這在正常用戶中極為罕見」
4. 指向白話文診斷書：「這段說明是由 LLM 自動生成的，客服人員可以直接複製給用戶」

---

### 第三幕：圖網絡分析（5 分鐘）

**操作**：點擊「查看圖網絡」，進入圖網絡分析頁

**說明重點**：
1. 「以 U-00234 為中心，我們可以看到他的直接關聯用戶。紅色節點是已確認詐欺，橘色是模型預測高風險」
2. 指向圖下方的 LLM 自動摘要區塊：「這段文字是頁面載入時 LLM 自動生成的，不需要提問——它直接告訴你這個節點在群聚中扮演什麼角色，以及整個群聚的風險特性」
3. 指向最粗的邊：「這條邊代表 U-00234 和 U-00891 之間有一筆 8.5 萬元的加密貨幣轉帳，是整個群聚中金額最大的交易」
4. 指向 GNNExplainer 高亮邊：「紅色加粗的邊是 GNNExplainer 認為對黑名單判定貢獻最大的交易路徑」
5. 在 LLM 問答框輸入：「這個群聚的資金流向模式是什麼？」
6. 等待 LLM 串流回答，說明：「LLM 不只給出結論，還附上了關鍵特徵的實際數值，以及跟全體用戶比較的百分位數——這讓分析師可以快速判斷這個案子的嚴重程度」

---

### 第四幕：新用戶即時預測（3 分鐘）

**操作**：進入「即時預測 > 批次預測」

**說明重點**：
1. 「系統直接串接 BitoPro 官方 API（https://aws-event-api.bitopro.com/），自動拉取 `predict_label.jsonl` 中的待預測用戶交易資料，不需要手動匯出 CSV」
2. 點擊「立即執行批次預測」，系統開始拉取資料並推論
3. 「約 30 秒後，右上角出現通知，高風險用戶自動進入案件管理列表」
4. 切換到「單筆預測」分頁，貼上一筆預先準備好的新用戶 JSON 資料
5. 點擊「立即評估」，2 秒內回傳風險分數與 SHAP 解釋
6. 「這讓我們在用戶完成 KYC 的當下就能評估風險，不需要等到下一次批次更新」

> 【模擬資料】：Demo 用的單筆預測 JSON 為模擬資料，特徵數值刻意設計為高風險模式。`predict_label.jsonl` 無真實標籤，批次預測結果準確率無法即時驗證。

---

### 第五幕：模型監控與閾值調整（2 分鐘）

**操作**：進入「模型監控」頁

**說明重點**：
1. 「目前集成模型的 CV ROC-AUC 是 0.XX，PR-AUC 是 0.XX」
2. 拖動閾值滑桿從 0.35 調整到 0.50：「調高閾值可以減少誤判，但會漏掉更多真實詐欺——這個 trade-off 可以根據業務需求動態調整」


---

## 七、可實現 vs. 目前缺少的功能

### ✅ 目前可直接實現

以下功能所需的資料與模型輸出均已存在：

| 功能 | 資料來源 | 說明 |
|------|---------|------|
| 風險分數查詢 | `outputs/ensemble_results.pkl` → `oof_predictions` | 直接讀取 OOF 預測值 |
| SHAP 特徵貢獻圖（LightGBM） | 重新訓練時呼叫 `shap.TreeExplainer` | LightGBM 原生支援 |
| 交易時間熱力圖 | `data/crypto_transfer.jsonl` → `created_at` 欄位 | 解析小時與星期幾 |
| 圖網絡視覺化（1-hop） | `shared_features.compute_graph_features()` 建立的 NetworkX 圖 | 邊來自內部轉帳 + 共享 IP |
| 節點顏色（黑名單狀態） | `data/train_label.jsonl` → `status` 欄位 | 直接對應 |
| 社群偵測結果 | `shared_features.compute_community_features()` → `partition` | Louvain 社群 ID |
| 關鍵數字卡片 | `shared_features._compute_user_features()` 的各項統計特徵 | 直接從特徵向量讀取 |
| 案件狀態標記 | 需要一個輕量資料庫（SQLite / JSON 檔） | 儲存人工標記結果 |
| 閾值調整滑桿 | `oof_predictions` + `y_true` | 重新計算 Precision/Recall |
| 全局 SHAP 重要性圖 | LightGBM `feature_importances_` 或 SHAP summary | 已有 `feature_names` |
| LLM 問答（基礎） | 組裝 context + 呼叫 LLM API | 需要 API key |
| 單筆即時預測 | 呼叫 `_compute_user_features()` + 載入 `.pkl` 模型 | 需序列化訓練好的模型物件 |

---

### ⚠️ 需要額外實作（技術可行，但目前缺少）

| 功能 | 缺少什麼 | 實作難度 |
|------|---------|---------|
| GNNExplainer 關鍵子圖 | 需在訓練後執行 `torch_geometric.explain.GNNExplainer`，目前訓練腳本未包含此步驟 | 中 |
| Integrated Gradients | 需安裝 `captum`，並在 GraphSAGE 推論時保留計算圖（`retain_graph=True`） | 中 |
| SHAP on GraphSAGE Embedding | 需將 GraphSAGE 最後一層 embedding 輸出並套用 `shap.KernelExplainer` | 中 |
| 邊的方向（資金流向） | 目前圖為無向圖（`nx.Graph`），需改為有向圖（`nx.DiGraph`）並保留轉帳方向 | 低 |
| 邊的粗細（交易金額） | 需在建圖時儲存邊的金額屬性（`G.add_edge(u, v, weight=amount)`） | 低 |
| Hub / Authority Score | 需執行 `nx.hits(G)` 計算 HITS 演算法 | 低 |
| 批次預測排程 | 需要排程系統（APScheduler / Celery / cron） | 中 |
| 稽核軌跡記錄 | 需要資料庫儲存操作記錄 | 低 |
| PDF 報告匯出 | 需要 `reportlab` 或 `weasyprint` 套件 | 低 |

---

### ❌ 目前系統缺少的資料（無法實現）

| 功能 | 缺少的資料 | 說明 |
|------|-----------|------|
| 詐欺率趨勢折線圖（30 天） | 每日預測結果的時間序列 | 目前只有一次性的 OOF 預測，沒有每日快照 |
| 交易時間熱力圖（星期維度） | `created_at` 的星期幾資訊 | 需確認 JSONL 資料中 `created_at` 格式是否包含完整日期 |
| Attention 權重視覺化 | GAT（Graph Attention Network）模型 | 目前使用 GraphSAGE，不含 attention 機制；需改用 `GATConv` |
| 用戶歷史案件記錄 | 案件管理資料庫 | 目前無任何人工審查記錄的儲存機制 |
| 新用戶的動態圖插入 | 即時更新的圖結構 | 目前圖在訓練時靜態建立，新用戶無法動態加入 |
| 模型版本歷史比較 | 多次訓練的歷史 `.pkl` 結果 | 目前只有最新一次的訓練結果 |
| `predict_label.jsonl` 的真實標籤 | 競賽/業務方提供的測試集標籤 | 目前 `predict_label.jsonl` 只有 user_id，無 status 欄位，無法計算測試集指標 |

---

### 模擬資料說明彙整

以下功能在 Demo 或開發階段需使用模擬資料，正式上線前需替換為真實資料：

| 功能 | 模擬內容 | 替換條件 |
|------|---------|---------|
| 詐欺率趨勢圖 | 【模擬資料】30 天的每日詐欺率數值 | 建立每日預測快照機制後替換 |
| 新用戶 Demo JSON | 【模擬資料】刻意設計為高風險的特徵數值 | 使用真實新用戶資料後替換 |
| 案件管理列表 | 【模擬資料】預先填入的審查狀態 | 建立案件資料庫後替換 |
| GNNExplainer 邊重要性 | 【模擬資料】隨機邊權重 | 實作 GNNExplainer 後替換 |
| Hub / Authority Score | 【模擬資料】以 `graph_degree` 近似替代 | 實作 `nx.hits()` 後替換 |

---

## 八、技術選型與 AWS 部署架構

所有元件均可部署在 AWS 上，以下為選型與對應的 AWS 服務：

| 層次 | 技術選型 | AWS 服務 | 說明 |
|------|---------|---------|------|
| 前端框架 | React + TypeScript | Amazon S3 + CloudFront | 靜態資源托管，CDN 加速 |
| 圖視覺化 | `react-force-graph` 或 `Cytoscape.js` | （前端套件，隨前端部署） | 支援大型圖、互動操作、自訂節點樣式 |
| 圖表庫 | `Recharts` 或 `Plotly.js` | （前端套件，隨前端部署） | SHAP 長條圖、熱力圖、趨勢折線圖 |
| 後端框架 | FastAPI（Python） | ECS Fargate | 模型需常駐記憶體，Fargate 比 Lambda 更適合 |
| 模型服務 | 載入 `.pkl` 直接推論 | ECS Fargate（推薦）或 SageMaker Endpoint | 避免重複載入，使用 singleton 模式 |
| LLM 串接 | Amazon Bedrock | Amazon Bedrock（Claude 3.5 Sonnet / Nova Pro） | 無需管理模型，支援串流輸出，IAM 權限控管 |
| 資料儲存 | JSON 檔案模擬 | Amazon S3（`case_status.json`、`audit_log.json`） | 目前以 JSON 模擬，正式上線後可遷移至 Amazon DynamoDB |
| API 路由 | — | Amazon API Gateway | 統一管理前後端 API 路由與 CORS |
| 任務排程 | 批次預測排程 | Amazon EventBridge + ECS Task | 每日定時觸發批次預測 |
| 即時通知 | 批次完成推送 | Amazon SNS | 通知前端有新高風險用戶 |
| 日誌與監控 | 應用程式日誌 | Amazon CloudWatch | 記錄 API 呼叫、模型推論延遲、錯誤追蹤 |

### AWS 部署架構圖（文字版）

```
用戶瀏覽器
    │
    ▼
CloudFront（CDN）
    │
    ├─→ S3（React 前端靜態資源）
    │
    └─→ API Gateway
            │
            └─→ ECS Fargate（FastAPI 後端 + 模型服務）
                    ├─→ S3（讀取 .pkl 模型、case_status.json、audit_log.json）
                    └─→ Amazon Bedrock（LLM 問答 / 診斷書生成 / 圖網絡摘要）

EventBridge（每日排程）
    └─→ ECS Task（批次預測）
            ├─→ BitoPro API（拉取新用戶交易資料）
            ├─→ S3（讀取模型、寫入預測結果）
            └─→ SNS（推送完成通知 → 前端）

CloudWatch（全程監控 ECS / API Gateway / Bedrock 呼叫）
```

### Bedrock 呼叫範例（Python）

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model_with_response_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_question}]
    })
)

# SSE 串流輸出
for event in response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if chunk["type"] == "content_block_delta":
        yield chunk["delta"]["text"]
```

---

*文件版本：v2.0 | 更新日期：2026-03-26*
