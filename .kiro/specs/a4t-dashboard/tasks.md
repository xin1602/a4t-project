# 實作計畫：A4T 詐欺偵測儀表板

## 概覽

以 React + TypeScript 前端搭配 FastAPI 後端，整合 LightGBM + GraphSAGE 集成模型推論結果，實作 MVP Demo 儀表板。後端以 Python 撰寫，前端以 TypeScript 撰寫。

---

## 任務清單

- [x] 1. 專案骨架與環境設定
  - 建立後端目錄結構：`backend/app/`（routers、services、models、core）
  - 建立前端目錄結構：`frontend/src/`（pages、components、api）
  - 建立後端 `requirements-dashboard.txt`，包含 fastapi、uvicorn、hypothesis、captum、torch-geometric
  - 建立前端 `package.json`，包含 react、typescript、recharts、react-force-graph、fast-check
  - 建立 `backend/app/core/config.py`，定義 `REQUIRED_PKL_FILES` 與 `REQUIRED_GRAPHSAGE_KEYS` 常數
  - _需求：9.1、9.2_

- [ ] 2. 後端資料模型與型別定義
  - [x] 2.1 建立 `backend/app/models/schemas.py`
    - 定義 `PredictionResult`、`CaseRecord`、`AuditEntry`、`GnnExplanation`、`IgExplanation` dataclass
    - 定義 `RiskLevel` enum（low/medium/high/critical）與閾值常數（0.35/0.6/0.8）
    - 定義 `CaseStatus` enum（pending/blacklisted/watching/normal）
    - 定義合法狀態轉換集合 `VALID_TRANSITIONS`
    - _需求：2.1、4.2、8.1_
  - [ ]* 2.2 撰寫屬性測試：Risk_Level 劃分正確性
    - **屬性 2：用戶查詢回傳資料完整性**
    - **驗證：需求 2.1**

- [ ] 3. ModelService — 模型載入與推論
  - [x] 3.1 建立 `backend/app/services/model_service.py`
    - 實作 `ModelService` Singleton，啟動時載入 `outputs/*.pkl`
    - 實作 `_load_models()`：驗證三個必要 pkl 存在，缺少則記錄路徑並拋出例外
    - 實作 `predict_single(user_id)` 與 `predict_batch(user_ids)`，使用 pkl 中的 `scaler` 標準化
    - 實作 `get_risk_level(score)` 依閾值回傳 RiskLevel
    - _需求：9.1、9.2、9.5、6.2_
  - [ ]* 3.2 撰寫屬性測試：pkl 欄位完整性驗證
    - **屬性 12：模型 pkl 欄位完整性驗證**
    - **驗證：需求 9.1、9.3**

- [x] 4. 修改 train_graphsage.py — 加入 ExplainerService 所需欄位
  - 在 `train_graphsage.py` 儲存區塊加入 `model_state_dict`、`scaler`、`edge_index`、`X_scaled` 四個欄位
  - _需求：9.3、9.4_

- [ ] 5. GraphService — 子圖展開與序列化
  - [x] 5.1 建立 `backend/app/services/graph_service.py`
    - 實作 `GraphService`，從 ModelService 取得 NetworkX 圖並快取
    - 實作 `get_subgraph(root_user_id, hop_depth, case_status_map)`：依 Hop_Depth 展開，沿黑名單節點延伸，遇到 `normal` 節點停止不再往下
    - 實作 `serialize_subgraph(subgraph)`：轉換為 `{nodes, edges}` JSON，節點含 riskScore/riskLevel/caseStatus/graphDegree，邊含 amount/edgeMask
    - _需求：3.1、3.4、3.6_
  - [ ]* 5.2 撰寫屬性測試：圖展開停止規則
    - **屬性 4：圖展開停止規則（Blacklist Chain）**
    - **驗證：需求 3.4**

- [ ] 6. CaseStore — 案件分類與稽核軌跡
  - [x] 6.1 建立 `backend/app/services/case_service.py`
    - 實作讀寫 `case_status.json` 的 CRUD，初始化時若不存在則建立空檔
    - 實作 `update_case(case_id, new_status, operator)`：驗證狀態轉換合法性，非法轉換回傳 HTTP 422
    - 實作 append-only 寫入 `audit_log.json`，記錄 operator、timestamp_utc（ISO 8601 UTC）、user_id、old_status、new_status
    - _需求：4.2、4.5_
  - [ ]* 6.2 撰寫屬性測試：案件狀態機轉換合法性
    - **屬性 8：案件狀態機轉換合法性**
    - **驗證：需求 4.2**
  - [ ]* 6.3 撰寫屬性測試：稽核軌跡寫入完整性
    - **屬性 7：稽核軌跡寫入完整性**
    - **驗證：需求 4.2**

- [ ] 7. ExplainerService — GNNExplainer 與 Integrated Gradients
  - [x] 7.1 建立 `backend/app/services/explainer_service.py`
    - 實作 `explain_gnn(user_id)`：使用 `torch_geometric.explain.GNNExplainer`（epochs=200），優先從 `outputs/gnn_explanations.pkl` 快取讀取，快取不存在則即時計算
    - 驗證 GraphSAGE pkl 包含 `model_state_dict`、`scaler`、`edge_index`、`X_scaled`，缺少則回傳 HTTP 422
    - 實作 `explain_ig(user_id)`：使用 `captum.attr.IntegratedGradients`（n_steps=50，baseline=全零向量），確保 `model.eval()` 模式，結果快取
    - _需求：7.1、7.3、7.4、8.1、8.3、9.3_
  - [ ]* 7.2 撰寫屬性測試：edge_mask 值域
    - **屬性 6：GNNExplainer edge_mask 值域**
    - **驗證：需求 7.1**
  - [ ]* 7.3 撰寫屬性測試：IG 歸因語意正確性
    - **屬性 11：IG 歸因值語意正確性**
    - **驗證：需求 8.1、8.5**

- [x] 8. LlmService — Amazon Bedrock SSE 串流
  - 建立 `backend/app/services/llm_service.py`
  - 實作 `stream_investigation(user_id, context)`：組裝 `InternalInvestigationContext`（risk_score、shap_top5、neighbor_summary、user_profile），使用 `internal_investigation` System Prompt，透過 Bedrock invoke_model_with_response_stream 串流
  - 實作 `stream_cluster_summary(graph_data)`：`graph_cluster` 模式，串流群聚摘要
  - System Prompt 明確限制：不做最終凍結決策、不捏造資料
  - _需求：5.1、5.2、5.3、5.5、3.11_

- [ ] 9. FastAPI 路由層
  - [x] 9.1 建立 `backend/app/routers/overview.py`
    - 實作 `GET /api/overview`：回傳高風險用戶數（score > 0.35）、待分類案件數、批次預測狀態，以及 Risk_Score 分布直方圖資料
    - 高風險待分類列表依 Risk_Score 降冪排列，最多 10 筆
    - 尚無批次結果時回傳 `batch_status: "尚未執行"`
    - _需求：1.1、1.2、1.3、1.5_
  - [ ]* 9.2 撰寫屬性測試：高風險用戶列表排序與截斷
    - **屬性 1：高風險用戶列表排序與截斷**
    - **驗證：需求 1.3**
  - [x] 9.3 建立 `backend/app/routers/users.py`
    - 實作 `GET /api/users/{user_id}`：回傳 risk_score、risk_level、shap_values、ig_attributions、交易量、夜間交易比例、外部轉帳比例
    - user_id 不存在回傳 HTTP 404
    - _需求：2.1、2.4、2.7_
  - [ ]* 9.4 撰寫屬性測試：用戶查詢回傳資料完整性
    - **屬性 2：用戶查詢回傳資料完整性**
    - **驗證：需求 2.1、2.4**
  - [x] 9.5 建立 `backend/app/routers/graph.py`
    - 實作 `GET /api/graph/{user_id}?hop=1`：呼叫 GraphService，回傳序列化子圖
    - 實作 `GET /api/explain/gnn/{user_id}` 與 `GET /api/explain/ig/{user_id}`
    - _需求：3.1、7.2、8.2_
  - [x] 9.6 建立 `backend/app/routers/cases.py`
    - 實作 `GET /api/cases` 與 `PATCH /api/cases/{case_id}`
    - 狀態轉換非法回傳 HTTP 422，Case_Store 寫入失敗回傳 HTTP 500
    - _需求：4.1、4.2、4.5_
  - [x] 9.7 建立 `backend/app/routers/predict.py`
    - 實作 `POST /api/predict/batch`：支援 API 自動拉取（`predict_label.jsonl`）與手動上傳 CSV
    - CSV 欄位驗證：不符合 `feature_names` 時回傳 HTTP 422 並列出缺少欄位
    - 批次結果過濾 risk_score > 0.35，提供 CSV 下載端點
    - _需求：6.1、6.2、6.4、6.5、6.6_
  - [ ]* 9.8 撰寫屬性測試：批次預測高風險過濾
    - **屬性 9：批次預測高風險過濾**
    - **驗證：需求 6.4**
  - [ ]* 9.9 撰寫屬性測試：CSV 欄位驗證
    - **屬性 10：CSV 欄位驗證**
    - **驗證：需求 6.6**
  - [x] 9.10 建立 `backend/app/routers/llm.py`
    - 實作 `POST /api/llm/investigate` 與 `POST /api/llm/cluster-summary`，以 SSE（Server-Sent Events）串流回應
    - Bedrock 逾時或失敗回傳 HTTP 503
    - _需求：5.1、3.11、3.12_

- [x] 10. 後端啟動驗證與 main.py
  - 建立 `backend/app/main.py`：FastAPI app 初始化，掛載所有 router
  - 在 `lifespan` 事件中呼叫 `ModelService.get_instance()`，驗證必要 pkl 存在，缺少則記錄路徑並拒絕啟動
  - _需求：9.1、9.2_

- [x] 11. 檢查點 — 後端單元測試
  - 確認所有後端路由可正常啟動，所有測試通過，請向使用者確認是否繼續。

- [x] 12. 前端 API 客戶端與型別定義
  - 建立 `frontend/src/api/client.ts`：封裝所有 REST 呼叫與 SSE 連線
  - 建立 `frontend/src/types/index.ts`：定義 `GraphNode`、`GraphEdge`、`GraphRendererProps`、`PredictionResult`、`CaseRecord` 等 TypeScript 介面
  - _需求：2.1、3.1_

- [ ] 13. 共用元件實作
  - [x] 13.1 建立 `frontend/src/components/KpiCard.tsx`
    - 顯示標題、數值、狀態（支援「尚未執行」空狀態）
    - _需求：1.1、1.5_
  - [x] 13.2 建立 `frontend/src/components/RiskHistogram.tsx`
    - 使用 Recharts BarChart 顯示 Risk_Score 分布直方圖
    - _需求：1.2_
  - [x] 13.3 建立 `frontend/src/components/ShapChart.tsx`
    - 使用 Recharts 橫向 BarChart，正值紅色（`#EF4444`）、負值藍色（`#3B82F6`），顯示 Top 10 特徵
    - _需求：2.2_
  - [x] 13.4 建立 `frontend/src/components/IgChart.tsx`
    - 同 ShapChart 樣式，標示「推向詐欺 / 推向正常」說明文字
    - _需求：2.4、8.4、8.5_
  - [x] 13.5 建立 `frontend/src/components/HeatmapChart.tsx`
    - 使用 Recharts 實作 24 小時 × 7 天交易時間熱力圖
    - _需求：2.3_
  - [x] 13.6 建立 `frontend/src/components/CaseClassifier.tsx`
    - 固定於右側，顯示目前 Case_Status 與三個分類按鈕
    - 分類成功後觸發 onStatusChange callback；失敗顯示「分類儲存失敗，請重試」
    - _需求：4.1、4.4、4.5_
  - [x] 13.7 建立 `frontend/src/components/LlmPanel.tsx`
    - 透過 SSE 串流顯示 LLM 輸出，標示「僅供內部人員使用」警示
    - 失敗時顯示對應錯誤訊息，不影響頁面其他區塊
    - _需求：5.4、5.6、3.12_
  - [x] 13.8 建立 `frontend/src/components/DepthSlider.tsx`
    - 範圍 1–5 hop 的滑桿，onChange 即時觸發重新渲染
    - _需求：3.2、3.3_

- [ ] 14. GraphRenderer 元件
  - [x] 14.1 建立 `frontend/src/components/GraphRenderer.tsx`
    - 使用 react-force-graph 渲染節點與邊，依 `caseStatus` 套用顏色映射（`#EF4444`/`#F97316`/`#EAB308`/`#9CA3AF`）
    - 依 `graphDegree` 決定節點大小，依交易金額決定邊粗細，箭頭標示資金流向
    - edge_mask > 0.5 的邊以紅色加粗並標示貢獻分數（兩位小數）
    - 支援縮放、拖曳、節點固定
    - _需求：3.5、3.6、3.7、3.10、7.5_
  - [x] 14.2 實作節點 tooltip 與點擊導航
    - Hover tooltip 顯示：user_id、Risk_Level、Risk_Score、加密轉帳總額、夜間交易比例、Case_Status、Top 3 重要特徵（來自 node_mask）、「點擊進入用戶查詢頁」提示
    - 點擊節點導航至用戶查詢頁
    - _需求：3.8、3.9、7.6_
  - [ ]* 14.3 撰寫屬性測試：節點顏色映射正確性（fast-check）
    - **屬性 5：節點顏色映射正確性**
    - **驗證：需求 3.5**

- [ ] 15. 頁面實作
  - [x] 15.1 建立 `frontend/src/pages/Overview.tsx`
    - 三個 KPI 卡片 + 風險分布直方圖 + 待分類高風險用戶列表（最多 10 筆，降冪排列）
    - 點擊列表項目導航至用戶查詢頁
    - _需求：1.1、1.2、1.3、1.4、1.5_
  - [x] 15.2 建立 `frontend/src/pages/UserDetail.tsx`
    - 搜尋列 + Risk_Score/Risk_Level 卡片 + 三個關鍵數字卡片
    - SHAP 圖 + IG 圖 + Top 3 一致性提示（「✓ 兩個模型判斷依據一致」）
    - 交易時間熱力圖 + 嵌入 GraphRenderer（預設 1 hop）+ CaseClassifier + LlmPanel
    - user_id 不存在顯示「查無此用戶」
    - _需求：2.1–2.7、4.1、5.1、5.4、5.6_
  - [ ]* 15.3 撰寫屬性測試：SHAP/IG Top 3 一致性判斷邏輯（fast-check）
    - **屬性 3：SHAP 與 IG Top 3 一致性判斷**
    - **驗證：需求 2.5、8.6**
  - [x] 15.4 建立 `frontend/src/pages/GraphAnalysis.tsx`
    - DepthSlider + GraphRenderer（全頁）+ LlmPanel（群聚摘要）
    - 頁面載入時自動觸發 LLM 群聚摘要
    - _需求：3.1–3.12_
  - [x] 15.5 建立 `frontend/src/pages/BatchPredict.tsx`
    - 兩種資料來源選項（API 拉取 / 手動上傳 CSV）
    - 執行中顯示進度條（已處理 / 總用戶數）
    - 結果列表（risk_score > 0.35）+ 「查看」按鈕 + 「下載結果 CSV」
    - CSV 欄位驗證失敗顯示具體缺少欄位；推論失敗顯示「批次預測失敗」並提供重試
    - _需求：6.1–6.7_

- [x] 16. 前端路由與 App 骨架
  - 建立 `frontend/src/App.tsx`：設定 React Router，定義四個頁面路由（`/`、`/users/:userId`、`/graph/:userId`、`/batch`）
  - 建立側邊導覽列元件
  - _需求：1.4、3.9_

- [x] 17. 最終檢查點 — 整合驗證
  - 確認所有屬性測試與單元測試通過，前後端整合正常，請向使用者確認是否有問題。

---

## 備註

- 標記 `*` 的子任務為選填，可跳過以加速 MVP 交付
- 每個任務均對應具體需求編號，確保可追溯性
- 屬性測試後端使用 Hypothesis（Python），前端使用 fast-check（TypeScript）
- 所有屬性測試最少執行 100 次迭代
- 稽核軌跡（audit_log.json）為 append-only，不允許修改或刪除
