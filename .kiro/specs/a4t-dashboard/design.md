# A4T 詐欺偵測儀表板 — 設計文件

> MVP Demo 版本，聚焦情境 A（風控分析師）、B（客服人員）、C（資安工程師）。

---

## 概覽（Overview）

A4T 儀表板是一套供加密貨幣交易所內部人員使用的詐欺偵測輔助系統。系統整合 LightGBM + GraphSAGE 集成模型的推論結果，提供三種核心使用情境：

- **情境 A**：風控分析師進行高風險用戶審查、圖網絡探索與案件分類
- **情境 B**：客服人員搜尋用戶並查看 LLM 生成的內部調查建議
- **情境 C**：資安工程師上傳資料、執行批次預測並查看結果

MVP 階段以 JSON 檔案模擬資料庫，不實作排程、PDF 報告、模型監控等功能。

---

## 架構（Architecture）

```
┌─────────────────────────────────────────────────────────┐
│                    前端（React + TypeScript）             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ 首頁總覽  │  │ 用戶查詢  │  │ 圖網絡   │  │ 批次   │  │
│  │ Overview │  │  User    │  │  Graph   │  │Predict │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│       └─────────────┴─────────────┴─────────────┘       │
│                         REST / SSE                       │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  後端（FastAPI / Python）                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ModelService │  │GraphService │  │ ExplainerService│  │
│  │ (Singleton) │  │(NetworkX 快取)│  │GNNExplainer+IG │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │           │
│  ┌──────┴──────────────────────────────────┐ │           │
│  │           LlmService (Bedrock SSE)      │ │           │
│  └─────────────────────────────────────────┘ │           │
│                                               │           │
│  ┌────────────────────────────────────────────┴────────┐  │
│  │  資料層：outputs/*.pkl │ case_status.json │ audit_log.json │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
              Amazon Bedrock（Claude 3.5 Sonnet）
```

### 資料流

```
批次預測請求
  → shared_features._compute_user_features()
  → ModelService.predict_batch()
  → 高風險用戶（score > 0.35）寫入結果列表

案件分類請求
  → PATCH /api/cases/{case_id}
  → 驗證狀態轉換合法性
  → 更新 case_status.json
  → 寫入 audit_log.json（不可變）

LLM 請求（SSE）
  → 組裝 context（risk_score, shap_top5, neighbor_summary, user_profile）
  → Bedrock invoke_model_with_response_stream
  → 逐 token 串流至前端
```


---

## 元件與介面（Components and Interfaces）

### 前端元件

#### 頁面結構

```
src/
├── pages/
│   ├── Overview.tsx          # 首頁風險總覽
│   ├── UserDetail.tsx        # 用戶查詢頁
│   ├── GraphAnalysis.tsx     # 圖網絡分析頁
│   └── BatchPredict.tsx      # 批次預測頁
├── components/
│   ├── KpiCard.tsx           # KPI 卡片
│   ├── RiskHistogram.tsx     # 風險分數分布直方圖（Recharts）
│   ├── ShapChart.tsx         # SHAP 橫向長條圖（Recharts）
│   ├── IgChart.tsx           # IG 歸因橫向長條圖（Recharts）
│   ├── HeatmapChart.tsx      # 交易時間熱力圖（Recharts）
│   ├── GraphRenderer.tsx     # 圖網絡元件（react-force-graph）
│   ├── CaseClassifier.tsx    # 案件分類操作區
│   ├── LlmPanel.tsx          # LLM 輸出面板（SSE 串流）
│   └── DepthSlider.tsx       # 深度調整滑桿（1–5 hop）
└── api/
    └── client.ts             # API 呼叫封裝
```

#### GraphRenderer 元件介面

```typescript
interface GraphRendererProps {
  rootUserId: string;
  hopDepth: number;           // 1–5
  onNodeClick: (userId: string) => void;
}

interface GraphNode {
  id: string;
  riskScore: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  caseStatus: 'pending' | 'blacklisted' | 'watching' | 'normal';
  graphDegree: number;
  cryptoTotalAmount: number;
  nightTxRatio: number;
}

interface GraphEdge {
  source: string;
  target: string;
  amount: number;
  edgeMask?: number;          // GNNExplainer edge_mask，若 > 0.5 則紅色加粗
}
```

節點顏色映射：

| Case_Status | 顏色 | Hex |
|-------------|------|-----|
| blacklisted | 紅色 | `#EF4444` |
| pending（高風險） | 橘色 | `#F97316` |
| watching | 黃色 | `#EAB308` |
| normal | 灰色 | `#9CA3AF` |

### 後端服務介面

#### ModelService（Singleton）

```python
class ModelService:
    """啟動時載入所有模型，全程單例"""
    _instance: Optional['ModelService'] = None

    def __init__(self):
        self.lgbm_model = None       # LightGBM 模型
        self.sage_model = None       # SAGE_Model（PyTorch）
        self.ensemble_model = None   # Stacking 集成模型
        self.scaler = None           # StandardScaler
        self.feature_names: List[str] = []

    @classmethod
    def get_instance(cls) -> 'ModelService':
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_models()
        return cls._instance

    def _load_models(self) -> None:
        """載入 outputs/*.pkl，驗證必要欄位"""
        ...

    def predict_single(self, user_id: str) -> PredictionResult:
        ...

    def predict_batch(self, user_ids: List[str]) -> List[PredictionResult]:
        ...
```

#### GraphService

```python
class GraphService:
    """NetworkX 圖快取，提供子圖序列化"""

    def get_subgraph(
        self,
        root_user_id: str,
        hop_depth: int,                    # 1–5
        case_status_map: Dict[str, str],   # user_id -> Case_Status
    ) -> SubgraphData:
        """
        展開規則：
        - 沿黑名單節點的邊繼續延伸
        - 遇到 Case_Status == 'normal' 的節點停止，不再往下展開
        """
        ...

    def serialize_subgraph(self, subgraph: nx.Graph) -> SubgraphData:
        """轉換為前端可用的 nodes + edges JSON"""
        ...
```

#### ExplainerService

```python
class ExplainerService:
    """GNNExplainer 與 Integrated Gradients 歸因計算"""

    CACHE_PATH = 'outputs/gnn_explanations.pkl'

    def explain_gnn(self, user_id: str) -> GnnExplanation:
        """
        使用 torch_geometric.explain.GNNExplainer（epochs=200）
        回傳 edge_mask（Top K 邊）與 node_mask（Top 10 特徵）
        優先從快取讀取，快取不存在則即時計算（< 5 秒）
        """
        ...

    def explain_ig(self, user_id: str) -> IgExplanation:
        """
        使用 captum.attr.IntegratedGradients（n_steps=50，baseline=全零向量）
        確保 model.eval() 模式
        正值 = 推向詐欺，負值 = 推向正常
        """
        ...
```

#### LlmService

```python
class LlmService:
    """Amazon Bedrock Claude 3.5 Sonnet，SSE 串流輸出"""

    CONTEXT_MODES = ['internal_investigation', 'graph_cluster']

    async def stream_investigation(
        self,
        user_id: str,
        context: InternalInvestigationContext,
    ) -> AsyncGenerator[str, None]:
        """internal_investigation 模式，SSE 串流"""
        ...

    async def stream_cluster_summary(
        self,
        graph_data: SubgraphData,
    ) -> AsyncGenerator[str, None]:
        """graph_cluster 模式，SSE 串流"""
        ...
```

### API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/api/overview` | 首頁 KPI + 高風險用戶列表 |
| GET | `/api/users/{user_id}` | 用戶詳細資料（風險分數、SHAP、IG） |
| GET | `/api/graph/{user_id}?hop=1` | 子圖資料（支援 1–5 hop） |
| GET | `/api/explain/gnn/{user_id}` | GNNExplainer edge_mask + node_mask |
| GET | `/api/explain/ig/{user_id}` | IG 歸因分數 |
| POST | `/api/predict/batch` | 批次預測 |
| GET | `/api/cases` | 案件列表 |
| PATCH | `/api/cases/{case_id}` | 更新案件狀態（狀態機驗證） |
| POST | `/api/llm/investigate` | SSE：內部調查建議 |
| POST | `/api/llm/cluster-summary` | SSE：群聚摘要 |


---

## 資料模型（Data Models）

### PredictionResult

```python
@dataclass
class PredictionResult:
    user_id: str
    risk_score: float           # 0.0–1.0
    is_blacklist: bool          # risk_score > 0.35
    risk_level: str             # 'low' | 'medium' | 'high' | 'critical'
    shap_values: Dict[str, float]   # feature_name -> shap_value
    top_features: List[str]         # Top 10 特徵名稱（依 |shap| 降冪）
```

Risk_Level 劃分：

| 等級 | 範圍 |
|------|------|
| low（低） | < 0.35 |
| medium（中） | 0.35–0.6 |
| high（高） | 0.6–0.8 |
| critical（極高） | > 0.8 |

### CaseRecord（case_status.json）

```json
{
  "case_id": "CASE-00234",
  "user_id": "U-00234",
  "status": "pending",
  "risk_score": 0.91,
  "created_at": "2026-03-26T08:00:00Z",
  "updated_at": "2026-03-26T10:30:00Z",
  "updated_by": "analyst_01"
}
```

有效狀態值：`pending` | `blacklisted` | `watching` | `normal`

### AuditEntry（audit_log.json）

```json
{
  "entry_id": "AUD-001",
  "case_id": "CASE-00234",
  "user_id": "U-00234",
  "operator": "analyst_01",
  "timestamp_utc": "2026-03-26T10:30:00Z",
  "old_status": "pending",
  "new_status": "blacklisted"
}
```

稽核軌跡為**只寫入（append-only）**，不允許修改或刪除。

### 案件狀態機

```
                  ┌─────────────┐
                  │   pending   │  ← 初始狀態（批次預測高風險用戶自動建立）
                  └──────┬──────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌──────────┐  ┌────────────┐
   │blacklisted │  │ watching │  │   normal   │
   └────────────┘  └──────────┘  └────────────┘
```

合法轉換：`pending → blacklisted`、`pending → watching`、`pending → normal`、`watching → blacklisted`、`watching → normal`、`blacklisted → watching`（降級）

### InternalInvestigationContext（LLM context）

```json
{
  "context_type": "internal_investigation",
  "user_id": "U-00234",
  "prediction": {
    "risk_score": 0.91,
    "is_blacklist": true,
    "threshold": 0.35
  },
  "shap_top5": [
    {"feature": "crypto_night_tx_ratio", "value": 0.87, "shap": 0.42}
  ],
  "neighbor_summary": {
    "confirmed_fraud_neighbors": 1,
    "high_risk_neighbors": 3
  },
  "user_profile": {
    "career": "自由業",
    "income_source": "投資",
    "has_kyc_level1": true,
    "has_kyc_level2": false
  }
}
```

### GnnExplanation

```python
@dataclass
class GnnExplanation:
    user_id: str
    edge_mask: Dict[str, float]     # edge_id -> mask_value（0.0–1.0）
    node_mask_top10: List[Tuple[str, float]]  # (feature_name, importance)
    computed_at: str                # ISO 8601 UTC
```

### IgExplanation

```python
@dataclass
class IgExplanation:
    user_id: str
    attributions: Dict[str, float]  # feature_name -> attribution_value
    # 正值 = 推向詐欺，負值 = 推向正常
    convergence_delta: float        # captum 收斂誤差
    computed_at: str
```

### train_graphsage.py 修改需求

目前訓練腳本只儲存 OOF 預測結果，ExplainerService 需要以下額外欄位：

```python
# 在 train_graphsage.py 的儲存區塊加入
results['model_state_dict'] = model.state_dict()
results['scaler'] = scaler
results['edge_index'] = edge_index.cpu()
results['X_scaled'] = X_scaled      # 推論時需要完整特徵矩陣
```

若 pkl 缺少上述欄位，ExplainerService 回傳 HTTP 422 並說明需重新執行訓練腳本。


---

## 正確性屬性（Correctness Properties）

*屬性（property）是在系統所有合法執行中都應成立的特性或行為，本質上是對系統應做什麼的形式化陳述。屬性作為人類可讀規格與機器可驗證正確性保證之間的橋樑。*

### 屬性反思（Property Reflection）

在撰寫屬性前，先整理 prework 中可測試的項目並消除冗餘：

- 1.3（列表降冪排列 ≤ 10 筆）與 6.4（高風險過濾 > 0.35）都涉及列表過濾與排序，但測試對象不同，保留兩者
- 3.4（圖展開停止規則）與 3.5（節點顏色映射）是獨立的圖渲染邏輯，保留兩者
- 4.2（稽核軌跡寫入完整性）與 4.3（節點顏色即時更新）測試不同層次，保留兩者
- 7.1（edge_mask 範圍）與 8.1（IG 歸因語意）測試不同模型的不同屬性，保留兩者
- 9.1（啟動驗證）與 9.3（欄位驗證）可合併為一個「模型 pkl 欄位完整性驗證」屬性

---

### 屬性 1：高風險用戶列表排序與截斷

*對任意*高風險用戶集合（Risk_Score > 0.35），首頁顯示的列表應依 Risk_Score 降冪排列，且長度不超過 10 筆。

**驗證：需求 1.3**

---

### 屬性 2：用戶查詢回傳資料完整性

*對任意*存在於系統中的 user_id，`GET /api/users/{user_id}` 的回應應包含 risk_score（0.0–1.0）、risk_level（四個合法值之一）、shap_values（非空字典）、ig_attributions（非空字典）。

**驗證：需求 2.1、2.4**

---

### 屬性 3：SHAP 與 IG Top 3 一致性判斷

*對任意* SHAP Top 10 特徵列表與 IG 歸因列表，「兩個模型判斷依據一致」提示的顯示條件，當且僅當兩者 Top 3 特徵名稱集合完全相同時成立。

**驗證：需求 2.5、8.6**

---

### 屬性 4：圖展開停止規則（Blacklist Chain）

*對任意*圖結構、任意根節點與任意展開深度（1–5），展開後的子圖中，不應存在任何「正常用戶（normal）」節點的子節點——即正常用戶節點只能作為葉節點出現，不得繼續往下展開。

**驗證：需求 3.4**

---

### 屬性 5：節點顏色映射正確性

*對任意*節點的 Case_Status 值，顏色映射函數應回傳對應的唯一顏色：`blacklisted → #EF4444`、`pending → #F97316`、`watching → #EAB308`、`normal → #9CA3AF`，且不存在未定義的映射。

**驗證：需求 3.5**

---

### 屬性 6：GNNExplainer edge_mask 值域

*對任意*用戶的 GNNExplainer 計算結果，所有 edge_mask 值應落在 [0.0, 1.0] 閉區間內，不存在負值或大於 1 的值。

**驗證：需求 7.1**

---

### 屬性 7：稽核軌跡寫入完整性

*對任意*案件分類操作（PATCH /api/cases/{case_id}），寫入 audit_log.json 的記錄應包含 operator（非空）、timestamp_utc（合法 ISO 8601 UTC 格式）、user_id、old_status、new_status 五個欄位，且 old_status ≠ new_status。

**驗證：需求 4.2**

---

### 屬性 8：案件狀態機轉換合法性

*對任意*狀態轉換請求，只有合法轉換（pending→blacklisted、pending→watching、pending→normal、watching→blacklisted、watching→normal、blacklisted→watching）應被接受；其他轉換應回傳 HTTP 422。

**驗證：需求 4.2**

---

### 屬性 9：批次預測高風險過濾

*對任意*批次預測結果集合，回傳給前端的高風險用戶列表中，每一筆記錄的 risk_score 都應嚴格大於 0.35，不存在低於或等於閾值的記錄。

**驗證：需求 6.4**

---

### 屬性 10：CSV 欄位驗證

*對任意*上傳的 CSV 檔案，若其欄位集合不是 feature_names 的超集，API 應回傳包含所有缺少欄位名稱的錯誤訊息，且缺少欄位列表應與實際缺少的欄位完全一致。

**驗證：需求 6.6**

---

### 屬性 11：IG 歸因值語意正確性

*對任意*用戶的 IG 歸因計算，當模型對該用戶的預測分數高於基準（全零向量的預測分數）時，歸因值為正的特徵數量應多於歸因值為負的特徵數量（整體推向詐欺方向）。

**驗證：需求 8.1、8.5**

---

### 屬性 12：模型 pkl 欄位完整性驗證

*對任意*啟動或 Explainer 呼叫場景，若 outputs/ 目錄下缺少任一必要 pkl 檔案，或 GraphSAGE pkl 缺少 `model_state_dict`、`scaler`、`edge_index`、`X_scaled` 任一欄位，系統應回傳包含具體缺少路徑或欄位名稱的錯誤，而非靜默失敗。

**驗證：需求 9.1、9.3**


---

## 錯誤處理（Error Handling）

### 後端錯誤碼規範

| HTTP 狀態碼 | 情境 |
|-------------|------|
| 404 | user_id 不存在於系統資料中 |
| 422 | 案件狀態轉換不合法；CSV 欄位驗證失敗；GraphSAGE pkl 缺少必要欄位 |
| 500 | 模型推論內部錯誤；Case_Store 寫入失敗 |
| 503 | Bedrock LLM 呼叫逾時或失敗 |

### 前端降級策略

| 失敗情境 | 前端行為 |
|----------|----------|
| LLM 群聚摘要失敗 | 顯示「摘要生成失敗，請稍後重試」，圖網絡正常顯示 |
| LLM 內部調查建議失敗 | 顯示「建議生成失敗，請稍後重試」，頁面其他區塊正常 |
| Case_Store 寫入失敗 | 顯示「分類儲存失敗，請重試」，保留原有 Case_Status |
| 批次預測失敗 | 顯示「批次預測失敗」並提供重試按鈕 |
| user_id 不存在 | 顯示「查無此用戶」提示 |
| 尚無批次預測結果 | KPI 卡片顯示「尚未執行」，不顯示錯誤 |

### API_Server 啟動驗證

```python
# 啟動時執行，任一失敗則拒絕啟動
REQUIRED_PKL_FILES = [
    'outputs/lgbm_v12_results.pkl',
    'outputs/graphsage_v13_results.pkl',
    'outputs/ensemble_results.pkl',
]

REQUIRED_GRAPHSAGE_KEYS = [
    'model_state_dict', 'scaler', 'edge_index', 'X_scaled'
]
```

---

## 測試策略（Testing Strategy）

### 雙軌測試方法

本系統採用單元測試（unit tests）與屬性測試（property-based tests）並行的策略，兩者互補：

- **單元測試**：驗證具體範例、邊界條件與整合點
- **屬性測試**：以隨機輸入驗證普遍性屬性，每個屬性測試最少執行 **100 次迭代**

### 屬性測試工具

後端（Python）使用 **Hypothesis** 作為屬性測試框架：

```bash
pip install hypothesis
```

前端（TypeScript）使用 **fast-check** 作為屬性測試框架：

```bash
npm install --save-dev fast-check
```

### 屬性測試對應表

每個屬性測試必須以標籤（tag）標示對應的設計屬性，格式：

```
Feature: a4t-dashboard, Property {N}: {property_text}
```

| 設計屬性 | 測試檔案 | 測試函數 |
|----------|----------|----------|
| 屬性 1：高風險列表排序截斷 | `tests/test_overview.py` | `test_prop_high_risk_list_sorted_and_capped` |
| 屬性 2：用戶查詢回傳完整性 | `tests/test_users_api.py` | `test_prop_user_response_schema` |
| 屬性 3：SHAP/IG Top 3 一致性 | `tests/test_explainer.py` | `test_prop_top3_consistency_logic` |
| 屬性 4：圖展開停止規則 | `tests/test_graph_service.py` | `test_prop_graph_expansion_stops_at_normal` |
| 屬性 5：節點顏色映射 | `tests/test_graph_renderer.ts` | `test_prop_node_color_mapping` |
| 屬性 6：edge_mask 值域 | `tests/test_explainer.py` | `test_prop_edge_mask_range` |
| 屬性 7：稽核軌跡完整性 | `tests/test_cases_api.py` | `test_prop_audit_entry_completeness` |
| 屬性 8：狀態機轉換合法性 | `tests/test_cases_api.py` | `test_prop_state_machine_transitions` |
| 屬性 9：批次預測高風險過濾 | `tests/test_predict_api.py` | `test_prop_batch_predict_threshold` |
| 屬性 10：CSV 欄位驗證 | `tests/test_predict_api.py` | `test_prop_csv_field_validation` |
| 屬性 11：IG 歸因語意 | `tests/test_explainer.py` | `test_prop_ig_attribution_semantics` |
| 屬性 12：pkl 欄位完整性 | `tests/test_model_service.py` | `test_prop_pkl_field_validation` |

### 屬性測試範例

```python
# Feature: a4t-dashboard, Property 4: 圖展開停止規則
from hypothesis import given, settings
import hypothesis.strategies as st

@given(
    graph=st.builds(random_graph_with_statuses),
    root=st.sampled_from(user_ids),
    depth=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=100)
def test_prop_graph_expansion_stops_at_normal(graph, root, depth):
    """Normal nodes should only appear as leaves, never expanded further."""
    result = graph_service.get_subgraph(root, depth, graph.case_status_map)
    for node in result.nodes:
        if node.case_status == 'normal':
            # Normal nodes must have no children in the expanded subgraph
            assert len(result.get_children(node.id)) == 0
```

```python
# Feature: a4t-dashboard, Property 8: 案件狀態機轉換合法性
VALID_TRANSITIONS = {
    ('pending', 'blacklisted'), ('pending', 'watching'), ('pending', 'normal'),
    ('watching', 'blacklisted'), ('watching', 'normal'), ('blacklisted', 'watching'),
}

@given(
    old_status=st.sampled_from(['pending', 'blacklisted', 'watching', 'normal']),
    new_status=st.sampled_from(['pending', 'blacklisted', 'watching', 'normal']),
)
@settings(max_examples=100)
def test_prop_state_machine_transitions(client, old_status, new_status):
    """Only valid transitions should be accepted."""
    response = client.patch(f'/api/cases/CASE-001', json={'status': new_status})
    if (old_status, new_status) in VALID_TRANSITIONS:
        assert response.status_code == 200
    else:
        assert response.status_code == 422
```

### 單元測試重點

單元測試聚焦於以下具體情境（不重複屬性測試已覆蓋的範圍）：

- **啟動驗證**：pkl 檔案存在/缺少的各種組合
- **LLM context 組裝**：驗證 `internal_investigation` context 包含四個調查方向欄位
- **SSE 串流**：驗證 Bedrock 串流回應正確轉發至前端
- **快取讀寫**：GNNExplainer 快取命中與未命中的行為
- **CSV 下載**：批次預測結果 CSV 欄位格式正確性
- **空狀態**：尚無批次預測結果時首頁 KPI 顯示「尚未執行」

---

*文件版本：MVP v1.0 | 更新日期：2026-03-26*
