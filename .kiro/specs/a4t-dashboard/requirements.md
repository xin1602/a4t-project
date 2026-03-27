# 需求文件

## 簡介

A4T 詐欺偵測儀表板是一套供加密貨幣交易所內部人員使用的 MVP Demo 系統，整合 LightGBM + GraphSAGE 集成模型的推論結果，提供風控分析師、客服人員與資安工程師三種核心使用情境。系統以 React + TypeScript 前端搭配 FastAPI 後端，透過 Amazon Bedrock（Claude 3.5 Sonnet）生成內部調查建議與圖網絡群聚摘要，並以 JSON 檔案作為 MVP 階段的資料儲存方案。

---

## 詞彙表

- **Dashboard**：A4T 詐欺偵測儀表板前端應用程式
- **API_Server**：FastAPI 後端服務，負責模型推論、LLM 串接與資料存取
- **Inference_Engine**：負責載入 `outputs/*.pkl` 並執行批次或單筆推論的後端模組
- **Graph_Renderer**：前端圖網絡視覺化元件（基於 react-force-graph）
- **LLM_Service**：透過 Amazon Bedrock 呼叫 Claude 3.5 Sonnet 的後端服務模組
- **Case_Store**：以 `case_status.json` 模擬的案件分類資料儲存
- **Explainer_Service**：負責執行 GNNExplainer 與 Integrated Gradients 歸因計算的後端模組
- **Risk_Score**：集成模型輸出的詐欺風險分數，範圍 0–1，閾值為 0.35
- **Risk_Level**：依 Risk_Score 劃分的等級：低（< 0.35）、中（0.35–0.6）、高（0.6–0.8）、極高（> 0.8）
- **Case_Status**：案件分類狀態，共三種：「加入黑名單」、「持續觀察」、「正常用戶」，以及初始狀態「待分類」
- **Audit_Trail**：案件分類操作的稽核軌跡，記錄操作人員、時間戳記與分類結果
- **GNNExplainer**：基於 torch_geometric.explain 的圖神經網路解釋器，識別關鍵交易邊
- **IG**：Integrated Gradients（梯度積分歸因），使用 captum 量化 GraphSAGE 特徵貢獻
- **SHAP**：SHapley Additive exPlanations，用於解釋 LightGBM 特徵貢獻
- **Hop_Depth**：圖網絡展開深度，範圍 1–5，控制從根節點延伸的鄰居層數
- **Blacklist_Chain**：沿黑名單節點邊延伸的連通子圖，展開至遇到第一個「正常用戶」節點為止

---

## 需求

### 需求 1：首頁風險總覽

**使用者故事：** 身為風控分析師，我希望在登入後立即看到高風險用戶的整體概況，以便快速決定今日審查優先順序。

#### 驗收標準

1. THE Dashboard SHALL 在首頁顯示三個 KPI 卡片：高風險用戶數（Risk_Score > 0.35）、待分類案件數、今日批次預測狀態。
2. THE Dashboard SHALL 在首頁顯示全體用戶 Risk_Score 的分布直方圖（使用 Recharts）。
3. THE Dashboard SHALL 在首頁顯示待分類高風險用戶列表，依 Risk_Score 降冪排列，最多顯示 10 筆。
4. WHEN 分析師點擊列表中的用戶，THE Dashboard SHALL 導航至該用戶的用戶查詢頁。
5. IF 尚無批次預測結果，THEN THE Dashboard SHALL 在 KPI 卡片顯示「尚未執行」狀態，而非顯示錯誤。

---

### 需求 2：用戶查詢頁

**使用者故事：** 身為風控分析師或客服人員，我希望透過 user_id 搜尋特定用戶，查看其風險評估結果與可解釋性資訊，以便進行人工審查。

#### 驗收標準

1. WHEN 使用者在搜尋列輸入有效的 user_id 並送出，THE Dashboard SHALL 顯示該用戶的 Risk_Score、Risk_Level 標籤，以及交易量、夜間交易比例、外部轉帳比例三個關鍵數字卡片。
2. THE Dashboard SHALL 在用戶查詢頁顯示 SHAP Top 10 特徵貢獻橫向長條圖（正值紅色、負值藍色）。
3. THE Dashboard SHALL 在用戶查詢頁顯示交易時間熱力圖（24 小時 × 7 天，使用 Recharts）。
4. THE Dashboard SHALL 在 SHAP 圖下方顯示「GraphSAGE 特徵歸因（IG）」橫向長條圖，正值紅色、負值藍色。
5. WHERE SHAP 與 IG 的 Top 3 特徵完全一致，THE Dashboard SHALL 顯示「✓ 兩個模型判斷依據一致」提示文字。
6. THE Dashboard SHALL 在用戶查詢頁嵌入圖網絡元件，預設展開深度為 1 hop。
7. IF 輸入的 user_id 不存在於系統資料中，THEN THE API_Server SHALL 回傳 404 錯誤，THE Dashboard SHALL 顯示「查無此用戶」提示。

---

### 需求 3：圖網絡分析頁

**使用者故事：** 身為風控分析師，我希望以互動式圖網絡探索高風險用戶的關聯結構，以便識別詐欺群聚與資金流向。

#### 驗收標準

1. THE Graph_Renderer SHALL 以指定 user_id 為根節點，依 Hop_Depth 展開圖網絡，預設深度為 1。
2. THE Dashboard SHALL 在圖網絡分析頁提供深度滑桿（Depth Slider），範圍 1–5 hop。
3. WHEN 使用者調整深度滑桿，THE Graph_Renderer SHALL 即時重新渲染圖網絡，無需手動重新整理。
4. WHILE Hop_Depth 大於 1，THE Graph_Renderer SHALL 沿黑名單節點的邊繼續延伸，直到遇到第一個 Case_Status 為「正常用戶」的節點為止，不再往下展開。
5. THE Graph_Renderer SHALL 依 Case_Status 以不同顏色渲染節點：紅色（加入黑名單）、橘色（待分類高風險）、黃色（持續觀察）、灰色（正常用戶）。
6. THE Graph_Renderer SHALL 依 `graph_degree` 決定節點大小，依交易金額決定邊的粗細，並以箭頭標示資金流向。
7. THE Graph_Renderer SHALL 在 GNNExplainer edge_mask 大於 0.5 的邊上，以紅色加粗顯示並標示貢獻分數。
8. WHEN 使用者將滑鼠懸停於節點，THE Graph_Renderer SHALL 顯示 tooltip，內容包含：user_id、Risk_Level、Risk_Score、加密轉帳總額、夜間交易比例、Case_Status，以及「點擊進入用戶查詢頁」提示。
9. WHEN 使用者點擊節點，THE Dashboard SHALL 導航至該節點對應的用戶查詢頁。
10. THE Graph_Renderer SHALL 支援縮放、拖曳、節點固定等互動操作。
11. THE Dashboard SHALL 在圖網絡頁面載入時自動觸發 LLM_Service 生成群聚摘要，摘要內容包含節點角色分析與群聚詐欺比例。
12. IF LLM_Service 呼叫逾時或失敗，THEN THE Dashboard SHALL 顯示「摘要生成失敗，請稍後重試」，不影響圖網絡的正常顯示。

---

### 需求 4：案件分類操作

**使用者故事：** 身為風控分析師，我希望對高風險用戶進行人工分類，並讓系統記錄稽核軌跡，以便後續追蹤與合規審查。

#### 驗收標準

1. THE Dashboard SHALL 在用戶查詢頁右側固定顯示案件分類操作區，提供三個按鈕：「加入黑名單」、「持續觀察」、「正常用戶」。
2. WHEN 分析師點擊分類按鈕，THE API_Server SHALL 將操作人員識別碼、UTC 時間戳記、user_id 與分類結果寫入 Audit_Trail，並更新 Case_Store 中的 Case_Status。
3. WHEN 案件分類完成，THE Graph_Renderer SHALL 即時更新該節點的顏色，反映新的 Case_Status。
4. THE Dashboard SHALL 在分類操作區顯示目前的 Case_Status。
5. IF Case_Store 寫入失敗，THEN THE API_Server SHALL 回傳錯誤，THE Dashboard SHALL 顯示「分類儲存失敗，請重試」，並保留原有 Case_Status 不變。

---

### 需求 5：內部調查建議

**使用者故事：** 身為客服人員，我希望查看系統生成的內部調查建議，以便了解應採取的調查步驟與後續處理方向，而不需要自行解讀模型輸出。

#### 驗收標準

1. THE Dashboard SHALL 在用戶查詢頁顯示「內部調查建議」區塊，內容由 LLM_Service 根據用戶的 Risk_Score、SHAP Top 5 特徵、鄰居詐欺摘要與 KYC 資料生成。
2. THE LLM_Service SHALL 在內部調查建議中包含以下四個調查方向：交易時間異常核查、資金流向追蹤、關聯帳號審查、KYC 資料核實。
3. THE LLM_Service SHALL 在內部調查建議中包含建議後續處理步驟，涵蓋短期（暫停功能）、中期（STR 報告）與長期（持續監控）三個層次。
4. THE Dashboard SHALL 在內部調查建議區塊標示「僅供內部人員使用」警示，不對外揭露任何資訊。
5. THE LLM_Service SHALL 使用 internal_investigation 模式的 System Prompt，明確限制不做最終凍結決策、不捏造資料。
6. IF LLM_Service 呼叫失敗，THEN THE Dashboard SHALL 顯示「建議生成失敗，請稍後重試」，不影響頁面其他區塊的正常顯示。
7. THE LLM_Service SHALL 不在任何對外介面或 API 回應中揭露內部調查建議的內容。

---

### 需求 6：批次預測

**使用者故事：** 身為資安工程師，我希望上傳新資料或觸發 API 拉取，執行批次預測並查看高風險用戶結果，以便及時發現新增詐欺帳號。

#### 驗收標準

1. THE Dashboard SHALL 在批次預測頁面提供兩種資料來源選項：API 自動拉取（從 `predict_label.jsonl` 取得 user_id）與手動上傳 CSV（欄位需對應 `feature_names`）。
2. WHEN 使用者點擊「執行批次預測」，THE API_Server SHALL 呼叫 `shared_features._compute_user_features()` 進行特徵工程，再載入 `outputs/*.pkl` 執行推論。
3. WHILE 批次預測執行中，THE Dashboard SHALL 顯示進度條，標示已處理用戶數與總用戶數。
4. WHEN 批次預測完成，THE Dashboard SHALL 顯示高風險用戶結果列表（Risk_Score > 0.35），欄位包含 user_id、Risk_Score、Risk_Level，並提供「查看」按鈕跳轉至用戶查詢頁。
5. THE Dashboard SHALL 在批次預測結果頁面提供「下載結果 CSV」功能，CSV 欄位為：`user_id, risk_score, is_blacklist, top_feature_1, top_feature_1_shap, top_feature_2, top_feature_2_shap, top_feature_3, top_feature_3_shap`。
6. IF 上傳的 CSV 欄位與 `feature_names` 不符，THEN THE API_Server SHALL 回傳欄位驗證錯誤，THE Dashboard SHALL 顯示具體缺少的欄位名稱。
7. IF 批次預測過程中發生推論錯誤，THEN THE API_Server SHALL 記錄錯誤日誌，THE Dashboard SHALL 顯示「批次預測失敗」並提供重試選項。

---

### 需求 7：GNNExplainer 關鍵邊識別

**使用者故事：** 身為風控分析師，我希望在圖網絡中看到哪些交易邊對黑名單判定貢獻最大，以便快速聚焦於最關鍵的資金往來關係。

#### 驗收標準

1. THE Explainer_Service SHALL 使用 `torch_geometric.explain.GNNExplainer`（epochs=200）針對目標節點計算 edge_mask 與 node_mask。
2. THE API_Server SHALL 提供 `GET /api/explain/gnn/{user_id}` 端點，回傳 Top K 邊的 edge_mask 值與 Top 10 節點特徵重要性。
3. THE Explainer_Service SHALL 將計算結果快取至 `outputs/gnn_explanations.pkl`，後續查詢優先從快取讀取。
4. IF 快取不存在，THEN THE Explainer_Service SHALL 即時計算 GNNExplainer 結果，計算時間應在 5 秒內完成。
5. THE Graph_Renderer SHALL 在 edge_mask 大於 0.5 的邊上以紅色加粗顯示，並在邊旁標示貢獻分數（保留兩位小數）。
6. THE Graph_Renderer SHALL 在節點 tooltip 中顯示來自 node_mask 的 Top 3 重要特徵名稱。
7. IF `train_graphsage.py` 的輸出 pkl 未包含 `model_state_dict`、`scaler`、`edge_index`，THEN THE Explainer_Service SHALL 回傳明確錯誤訊息，說明需先重新執行訓練腳本。

---

### 需求 8：Integrated Gradients 特徵歸因

**使用者故事：** 身為風控分析師，我希望查看 GraphSAGE 模型的特徵梯度歸因，並與 LightGBM 的 SHAP 值對照，以便驗證兩個模型的判斷依據是否一致。

#### 驗收標準

1. THE Explainer_Service SHALL 使用 `captum.attr.IntegratedGradients`（n_steps=50，baseline 為全零向量）計算目標節點的特徵歸因。
2. THE API_Server SHALL 提供 `GET /api/explain/ig/{user_id}` 端點，回傳所有特徵的 IG 歸因分數。
3. WHILE 執行 IG 計算，THE Explainer_Service SHALL 確保 GraphSAGE 模型處於 `model.eval()` 模式，以避免 BatchNorm1d 統計值影響梯度計算。
4. THE Dashboard SHALL 在用戶查詢頁的 SHAP 圖下方顯示「GraphSAGE 特徵歸因（IG）」橫向長條圖，正值紅色、負值藍色。
5. THE Dashboard SHALL 在 IG 圖表中標示正值代表「推向詐欺」、負值代表「推向正常」的說明文字。
6. WHERE SHAP 與 IG 的 Top 3 特徵完全一致，THE Dashboard SHALL 顯示「✓ 兩個模型判斷依據一致」提示。
7. THE Explainer_Service SHALL 將 IG 計算結果快取，若快取不存在則即時計算，計算時間應在 5 秒內完成。

---

### 需求 9：模型推論前置條件

**使用者故事：** 身為資安工程師，我希望系統在啟動時驗證所需的模型檔案是否完整，以便在 Demo 前確認環境就緒。

#### 驗收標準

1. WHEN API_Server 啟動，THE Inference_Engine SHALL 驗證 `outputs/` 目錄下存在 LightGBM、GraphSAGE、Ensemble 三個 `.pkl` 檔案。
2. IF 任一必要的 `.pkl` 檔案不存在，THEN THE API_Server SHALL 在啟動日誌中輸出明確的缺少檔案路徑，並拒絕啟動。
3. WHEN GNNExplainer 或 IG 功能被呼叫，THE Explainer_Service SHALL 驗證 GraphSAGE 的 pkl 包含 `model_state_dict`、`scaler`、`edge_index` 三個必要欄位。
4. IF GraphSAGE pkl 缺少上述任一欄位，THEN THE Explainer_Service SHALL 回傳錯誤碼 422，並說明需重新執行 `train_graphsage.py` 並加入模型儲存邏輯。
5. THE Inference_Engine SHALL 在推論時使用 `outputs/*.pkl` 中儲存的 `scaler` 進行特徵標準化，確保與訓練時一致。
