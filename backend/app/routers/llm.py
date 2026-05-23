"""
LLM router — POST /api/llm/investigate, POST /api/llm/cluster-summary

Both endpoints stream responses via SSE (Server-Sent Events).
On Bedrock timeout or failure, returns HTTP 503.
"""
from __future__ import annotations

import json as _json
import logging
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.services.graph_service import SubgraphData
from backend.app.services.llm_service import (
    InternalInvestigationContext,
    LlmService,
    NeighborSummary,
    PredictionInfo,
    ShapFeature,
    UserProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Singleton LlmService instance
_llm_service = LlmService()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ShapFeatureItem(BaseModel):
    feature: str
    value: float
    shap: float


class NeighborSummaryItem(BaseModel):
    confirmed_fraud_neighbors: int
    high_risk_neighbors: int


class UserProfileItem(BaseModel):
    career: str
    income_source: str
    has_kyc_level1: bool
    has_kyc_level2: bool


class InvestigateRequest(BaseModel):
    user_id: str
    risk_score: float
    shap_top5: List[ShapFeatureItem] = []
    neighbor_summary: NeighborSummaryItem = NeighborSummaryItem(
        confirmed_fraud_neighbors=0, high_risk_neighbors=0
    )
    user_profile: UserProfileItem = UserProfileItem(
        career="", income_source="", has_kyc_level1=False, has_kyc_level2=False
    )


class GraphNodeItem(BaseModel):
    model_config = {"extra": "allow"}


class GraphEdgeItem(BaseModel):
    model_config = {"extra": "allow"}


class ClusterSummaryRequest(BaseModel):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


class ChatRequest(BaseModel):
    user_id: str
    question: str
    risk_score: float = 0.0
    shap_top5: List[ShapFeatureItem] = []


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


async def _sse_generator(
    stream: AsyncGenerator[str, None],
) -> AsyncGenerator[bytes, None]:
    """
    Wrap an async text generator into SSE-formatted byte chunks.
    Multi-line chunks use multiple 'data:' lines per SSE spec.
    """
    try:
        async for chunk in stream:
            # SSE spec: multi-line data uses "data: line1\ndata: line2\n\n"
            lines = chunk.split('\n')
            sse_data = '\n'.join(f"data: {line}" for line in lines)
            yield f"{sse_data}\n\n".encode("utf-8")
    except HTTPException as exc:
        if exc.status_code == 503:
            logger.error("LLM service unavailable: %s", exc.detail)
            yield b"data: [ERROR] LLM service unavailable\n\n"
        else:
            raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/llm/investigate")
async def llm_investigate(body: InvestigateRequest) -> StreamingResponse:
    """
    POST /api/llm/investigate

    Assembles InternalInvestigationContext from the request body and streams
    Claude's internal investigation advisory via SSE.

    Returns:
        StreamingResponse with media_type="text/event-stream"

    Raises:
        HTTP 503 (via SSE error event) on Bedrock failure.
    """
    # Build SHAP top 10 from ModelService for richer context
    from backend.app.services.model_service import ModelService as _MS
    shap_features: list = []
    try:
        ms = _MS.get_instance()
        pred = ms.predict_single(body.user_id)
        top10 = sorted(
            pred.shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        shap_features = [
            ShapFeature(feature=name, value=pred.shap_values.get(name, 0), shap=val)
            for name, val in top10
        ]
    except Exception:
        shap_features = [
            ShapFeature(feature=f.feature, value=f.value, shap=f.shap)
            for f in body.shap_top5
        ]

    context = InternalInvestigationContext(
        user_id=body.user_id,
        prediction=PredictionInfo(
            risk_score=body.risk_score,
            is_blacklist=body.risk_score > 0.35,
        ),
        shap_top5=shap_features,
        neighbor_summary=NeighborSummary(
            confirmed_fraud_neighbors=body.neighbor_summary.confirmed_fraud_neighbors,
            high_risk_neighbors=body.neighbor_summary.high_risk_neighbors,
        ),
        user_profile=UserProfile(
            career=body.user_profile.career,
            income_source=body.user_profile.income_source,
            has_kyc_level1=body.user_profile.has_kyc_level1,
            has_kyc_level2=body.user_profile.has_kyc_level2,
        ),
    )

    # Inject GNNExplainer node_mask if available
    try:
        from backend.app.services.explainer_service import ExplainerService
        explainer = ExplainerService.get_instance()
        if explainer.has_explanation(body.user_id):
            node_mask = explainer.get_node_mask_top10(body.user_id)
            context.gnn_node_mask_top10 = [
                {"feature": name, "importance": round(val, 6)}
                for name, val in node_mask
            ]
    except Exception as exc:
        logger.debug("GNNExplainer data not available for %s: %s", body.user_id, exc)

    stream = _llm_service.stream_investigation(body.user_id, context)

    return StreamingResponse(
        _sse_generator(stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/llm/cluster-summary")
async def llm_cluster_summary(body: ClusterSummaryRequest) -> StreamingResponse:
    """
    POST /api/llm/cluster-summary

    Assembles SubgraphData from the request body and streams
    Claude's cluster summary analysis via SSE.

    Returns:
        StreamingResponse with media_type="text/event-stream"

    Raises:
        HTTP 503 (via SSE error event) on Bedrock failure.
    """
    graph_data = SubgraphData(nodes=body.nodes, edges=body.edges)

    stream = _llm_service.stream_cluster_summary(graph_data)

    return StreamingResponse(
        _sse_generator(stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/llm/chat")
async def llm_chat(body: ChatRequest) -> StreamingResponse:
    """
    POST /api/llm/chat — free-form Q&A about a specific user.
    Streams Claude's response via SSE.

    The system prompt includes the user's full risk context so the LLM
    can answer questions with data-backed reasoning.
    """
    from backend.app.services.model_service import ModelService

    # Gather rich user context from ModelService
    model_service = ModelService.get_instance()
    user_info = model_service._user_info.get(body.user_id, {})
    fraud_label = model_service._fraud_labels.get(body.user_id)

    # Transaction counts per type
    twd_count = len(model_service._twd_by_user.get(body.user_id, []))
    crypto_count = len(model_service._crypto_by_user.get(body.user_id, []))
    trading_count = len(model_service._trading_by_user.get(body.user_id, []))
    swap_count = len(model_service._swap_by_user.get(body.user_id, []))

    # Gather SHAP top 10 from ModelService (more complete than frontend-sent data)
    try:
        prediction = model_service.predict_single(body.user_id)
        shap_top10 = sorted(
            prediction.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
    except KeyError:
        shap_top10 = [(f.feature, f.shap) for f in body.shap_top5]

    shap_info = "\n".join(
        f"  - {name}: shap={val:.4f}" for name, val in shap_top10
    ) or "  (no SHAP data)"

    # GNNExplainer node_mask info
    gnn_info = ""
    try:
        from backend.app.services.explainer_service import ExplainerService
        explainer = ExplainerService.get_instance()
        if explainer.has_explanation(body.user_id):
            node_mask = explainer.get_node_mask_top10(body.user_id)
            if node_mask:
                gnn_lines = "\n".join(
                    f"  - {name}: importance={val:.6f}" for name, val in node_mask
                )
                gnn_info = f"\n【GNNExplainer 關鍵特徵 Top 10】\n{gnn_lines}\n"
    except Exception:
        pass

    label_str = "詐欺" if fraud_label == 1 else "正常" if fraud_label == 0 else "未知（不在訓練集）"

    user_context = (
        f"【用戶資料】\n"
        f"user_id: {body.user_id}\n"
        f"風險分數: {body.risk_score:.4f}\n"
        f"真實標籤: {label_str}\n"
        f"用戶資訊: {_json.dumps(user_info, ensure_ascii=False, default=str)}\n"
        f"\n【交易統計】\n"
        f"TWD 轉帳: {twd_count} 筆\n"
        f"加密貨幣轉帳: {crypto_count} 筆\n"
        f"USDT/TWD 交易: {trading_count} 筆\n"
        f"USDT 兌換: {swap_count} 筆\n"
        f"\n【SHAP Top 10 特徵】\n{shap_info}\n"
        f"{gnn_info}"
    )

    user_content = f"{user_context}\n分析師提問：{body.question}"

    system_prompt = (
        "你是 A4T 詐欺偵測系統的分析助手。你已經被提供了該用戶的完整風險數據。\n"
        "請用繁體中文回答分析師的提問。\n\n"
        "【規則】\n"
        "- 回答要精準、有數據支撐\n"
        "- 引用具體數值時標明來源（如 SHAP 值、交易筆數）\n"
        "- 不要重複列出已提供的原始數據，直接給分析結論\n"
        "- 如果問題超出提供的數據範圍，明確說明\n"
        "- 保持簡潔，不要廢話"
    )

    stream = _llm_service._stream_from_openrouter(system_prompt, user_content)

    return StreamingResponse(
        _sse_generator(stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
