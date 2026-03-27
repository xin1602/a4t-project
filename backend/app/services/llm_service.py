"""
LlmService — Amazon Bedrock Claude 3.5 Sonnet, SSE streaming output.

Provides two streaming modes:
  - internal_investigation: Internal fraud investigation advisory for analysts
  - graph_cluster: Cluster summary for graph network analysis

IMPORTANT: This service is for INTERNAL USE ONLY.
  - Does NOT make final account freeze decisions
  - Does NOT fabricate data
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

import boto3
from fastapi import HTTPException

from backend.app.services.graph_service import SubgraphData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bedrock configuration
# ---------------------------------------------------------------------------

_BEDROCK_REGION = "us-west-2"
_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_MAX_TOKENS = 2048
_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_INTERNAL_INVESTIGATION = """你是詐欺偵測系統的分析助手。請用繁體中文，針對提供的用戶數據做簡短分析。

【輸出格式】嚴格按以下格式輸出：

## 風險評分
說明此分數在全體用戶中的位置（正常用戶平均 0.008，詐欺用戶平均 0.58），判斷是否異常。不超過 2 句。

## 關鍵特徵分析
針對 SHAP 貢獻最大的前 5 個特徵，每個用一行，格式：
- **特徵名**（SHAP +X.XX）：此用戶值 → 與全體比較結論

## GNNExplainer 圖網絡分析
若有 GNNExplainer 資料，說明：
- 模型認為哪些交易路徑對黑名單判定貢獻最大
- 哪些節點特徵在圖傳播中最關鍵
若無 GNNExplainer 資料，跳過此段。

## 調查建議
一句話建議下一步調查方向。

## 建議提問
列出 3 個分析師可能想追問的問題，每個一行，用 `- ` 開頭。問題要具體、跟此用戶數據相關。

【規則】
- 不要廢話、不要重複數據、不要加聲明
- 直接給結論和數據解讀
- 建議提問要具體到此用戶的異常點"""

_SYSTEM_PROMPT_GRAPH_CLUSTER = """你是一位專業的圖網絡分析輔助系統，專門協助資安工程師分析詐欺帳號的關聯網絡結構。

【重要聲明】
- 本系統僅供內部人員使用，嚴禁對外揭露
- 本系統不做最終凍結決策，所有決策須由授權人員審核後執行
- 本系統不捏造任何資料，所有分析均基於提供的實際圖結構數據

【分析重點】
請依據提供的子圖資料，分析以下面向：

1. **群聚結構特徵**：識別高風險節點的聚集模式
2. **關鍵節點識別**：找出圖中影響力最大的節點（高度數、高風險）
3. **傳播路徑分析**：分析詐欺風險在網絡中的傳播路徑
4. **GNNExplainer 關鍵邊分析**：若有 GNNExplainer 資料，重點分析 edge_mask 高的邊代表的交易路徑，說明模型認為哪些連結對黑名單判定貢獻最大
5. **GNNExplainer 關鍵特徵**：若有 node_mask_top10，說明哪些節點特徵在圖傳播中最關鍵

請使用繁體中文提供結構化的群聚摘要分析。"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ShapFeature:
    """A single SHAP feature entry."""
    feature: str
    value: float
    shap: float


@dataclass
class PredictionInfo:
    """Prediction summary for LLM context."""
    risk_score: float
    is_blacklist: bool
    threshold: float = 0.35


@dataclass
class NeighborSummary:
    """Graph neighbor summary for LLM context."""
    confirmed_fraud_neighbors: int
    high_risk_neighbors: int


@dataclass
class UserProfile:
    """User profile information for LLM context."""
    career: str
    income_source: str
    has_kyc_level1: bool
    has_kyc_level2: bool


@dataclass
class InternalInvestigationContext:
    """
    Context assembled for the internal_investigation LLM mode.

    Matches the InternalInvestigationContext JSON schema defined in design.md.
    """
    user_id: str
    prediction: PredictionInfo
    shap_top5: List[ShapFeature] = field(default_factory=list)
    neighbor_summary: NeighborSummary = field(
        default_factory=lambda: NeighborSummary(0, 0)
    )
    user_profile: UserProfile = field(
        default_factory=lambda: UserProfile("", "", False, False)
    )
    gnn_node_mask_top10: List[dict] = field(default_factory=list)
    context_type: str = "internal_investigation"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the JSON format expected by the LLM."""
        d = {
            "context_type": self.context_type,
            "user_id": self.user_id,
            "prediction": {
                "risk_score": self.prediction.risk_score,
                "is_blacklist": self.prediction.is_blacklist,
                "threshold": self.prediction.threshold,
            },
            "shap_top5": [
                {
                    "feature": f.feature,
                    "value": f.value,
                    "shap": f.shap,
                }
                for f in self.shap_top5
            ],
            "neighbor_summary": {
                "confirmed_fraud_neighbors": self.neighbor_summary.confirmed_fraud_neighbors,
                "high_risk_neighbors": self.neighbor_summary.high_risk_neighbors,
            },
            "user_profile": {
                "career": self.user_profile.career,
                "income_source": self.user_profile.income_source,
                "has_kyc_level1": self.user_profile.has_kyc_level1,
                "has_kyc_level2": self.user_profile.has_kyc_level2,
            },
        }
        if self.gnn_node_mask_top10:
            d["gnn_explainer_node_mask_top10"] = self.gnn_node_mask_top10
        return d


# ---------------------------------------------------------------------------
# LlmService
# ---------------------------------------------------------------------------


class LlmService:
    """
    Amazon Bedrock Claude 3.5 Sonnet, SSE streaming output.

    Supports two context modes:
      - internal_investigation: Fraud investigation advisory
      - graph_cluster: Graph cluster summary
    """

    CONTEXT_MODES = ["internal_investigation", "graph_cluster"]

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._last_request_time: float = 0.0  # Bedrock rate limit: 1 RPS

    def _get_client(self) -> Any:
        """Lazily initialize the Bedrock Runtime client."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=_BEDROCK_REGION,
            )
        return self._client

    def _build_request_body(self, system_prompt: str, user_content: str) -> str:
        """Build the Anthropic Messages API request body as a JSON string."""
        payload = {
            "anthropic_version": _ANTHROPIC_VERSION,
            "max_tokens": _MAX_TOKENS,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content},
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    async def _stream_from_bedrock(
        self,
        system_prompt: str,
        user_content: str,
    ) -> AsyncGenerator[str, None]:
        """
        Core streaming logic: invoke Bedrock and yield text delta chunks.

        Enforces 1 RPS rate limit per hackathon Bedrock constraints.

        Raises:
            HTTPException(503): on Bedrock timeout or failure.
        """
        # Enforce Bedrock 1 RPS rate limit (hackathon requirement)
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        self._last_request_time = time.monotonic()

        client = self._get_client()
        body = self._build_request_body(system_prompt, user_content)

        try:
            response = client.invoke_model_with_response_stream(
                modelId=_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
        except Exception as exc:
            logger.error("Bedrock invoke failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"LLM service unavailable: {exc}",
            ) from exc

        try:
            event_stream = response["body"]
            for event in event_stream:
                chunk = event.get("chunk")
                if chunk is None:
                    continue
                raw_bytes = chunk.get("bytes")
                if not raw_bytes:
                    continue
                try:
                    data = json.loads(raw_bytes)
                except json.JSONDecodeError:
                    continue

                # Extract text delta from Anthropic streaming format
                delta = data.get("delta", {})
                text = delta.get("text")
                if text:
                    yield text

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Bedrock stream processing failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"LLM stream error: {exc}",
            ) from exc

    async def stream_investigation(
        self,
        user_id: str,
        context: InternalInvestigationContext,
    ) -> AsyncGenerator[str, None]:
        """
        internal_investigation mode — stream fraud investigation advisory.

        Assembles InternalInvestigationContext into JSON and streams
        Claude's analysis via Bedrock invoke_model_with_response_stream.

        Args:
            user_id: The user being investigated.
            context: Pre-assembled investigation context.

        Yields:
            Text delta chunks from the LLM.

        Raises:
            HTTPException(503): on Bedrock failure.
        """
        context_json = json.dumps(context.to_dict(), ensure_ascii=False, indent=2)
        user_content = (
            f"請針對以下用戶風險資料進行內部調查分析：\n\n```json\n{context_json}\n```"
        )

        logger.info(
            "stream_investigation: user_id=%s, risk_score=%.3f",
            user_id,
            context.prediction.risk_score,
        )

        async for chunk in self._stream_from_bedrock(
            _SYSTEM_PROMPT_INTERNAL_INVESTIGATION, user_content
        ):
            yield chunk

    async def stream_cluster_summary(
        self,
        graph_data: SubgraphData,
    ) -> AsyncGenerator[str, None]:
        """
        graph_cluster mode — stream graph cluster summary analysis.

        Serializes SubgraphData and streams Claude's cluster analysis
        via Bedrock invoke_model_with_response_stream.

        Args:
            graph_data: Subgraph data containing nodes and edges.

        Yields:
            Text delta chunks from the LLM.

        Raises:
            HTTPException(503): on Bedrock failure.
        """
        # Build a concise summary of the graph for the LLM
        node_count = len(graph_data.nodes)
        edge_count = len(graph_data.edges)

        # Summarize node risk distribution
        risk_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        high_risk_nodes = []

        for node in graph_data.nodes:
            risk_level = node.get("riskLevel", "low")
            case_status = node.get("caseStatus", "pending")
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            status_counts[case_status] = status_counts.get(case_status, 0) + 1

            if node.get("riskScore", 0) >= 0.6:
                high_risk_nodes.append({
                    "id": node.get("id"),
                    "riskScore": node.get("riskScore"),
                    "riskLevel": risk_level,
                    "caseStatus": case_status,
                    "graphDegree": node.get("graphDegree", 0),
                })

        # Identify high-importance edges (edge_mask > 0.3)
        important_edges = [
            e for e in graph_data.edges
            if e.get("edgeMask", 0) > 0.3
        ]

        graph_summary = {
            "context_type": "graph_cluster",
            "statistics": {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "risk_distribution": risk_counts,
                "status_distribution": status_counts,
            },
            "high_risk_nodes": high_risk_nodes[:10],
            "gnn_explainer": {
                "important_edges_count": len(important_edges),
                "important_edges": [
                    {
                        "from": e.get("source"),
                        "to": e.get("target"),
                        "edge_mask": e.get("edgeMask"),
                        "amount_twd": e.get("amount", 0),
                    }
                    for e in sorted(important_edges, key=lambda x: x.get("edgeMask", 0), reverse=True)[:10]
                ],
                "note": "edge_mask 越高代表該交易路徑對黑名單判定貢獻越大（由 GNNExplainer 計算）",
            },
        }

        graph_json = json.dumps(graph_summary, ensure_ascii=False, indent=2)
        user_content = (
            f"請針對以下圖網絡子圖資料進行群聚結構分析：\n\n```json\n{graph_json}\n```"
        )

        logger.info(
            "stream_cluster_summary: nodes=%d, edges=%d, high_risk=%d",
            node_count,
            edge_count,
            len(high_risk_nodes),
        )

        async for chunk in self._stream_from_bedrock(
            _SYSTEM_PROMPT_GRAPH_CLUSTER, user_content
        ):
            yield chunk
