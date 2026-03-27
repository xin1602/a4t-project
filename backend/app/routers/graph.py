"""
Graph router — subgraph expansion and GNNExplainer endpoints.

Endpoints:
  GET /api/graph/{user_id}?hop=1      — BFS subgraph with GNNExplainer overlay
  GET /api/explain/gnn/{user_id}      — GNNExplainer edge_mask + node_mask_top10
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.app.services.case_service import CaseService
from backend.app.services.explainer_service import ExplainerService
from backend.app.services.graph_service import GraphService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class GraphNodeResponse(BaseModel):
    id: str
    riskScore: float
    riskLevel: str
    caseStatus: str
    graphDegree: int
    cryptoTotalAmount: float
    nightTxRatio: float
    isBlacklist: bool = False
    fraudLabel: Optional[int] = None


class GraphEdgeResponse(BaseModel):
    source: str
    target: str
    amount: float
    edgeMask: Optional[float] = None


class GnnNodeImportance(BaseModel):
    feature: str
    importance: float


class SubgraphResponse(BaseModel):
    nodes: List[GraphNodeResponse]
    edges: List[GraphEdgeResponse]
    hasGnnExplanation: bool = False
    gnnNodeMaskTop10: List[GnnNodeImportance] = []


class GnnExplanationResponse(BaseModel):
    user_id: str
    edge_mask: Dict[str, float]
    node_mask_top10: List[Tuple[str, float]]
    computed_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_case_status_map() -> Dict[str, str]:
    """Return a dict mapping user_id -> case status from CaseService."""
    try:
        case_service = CaseService.get_instance()
        all_cases = case_service.get_all_cases()
        return {c.user_id: c.status for c in all_cases}
    except Exception as exc:
        logger.warning("Failed to fetch case status map: %s", exc)
        return {}


def _clamp_hop(hop: int) -> int:
    return max(1, min(5, hop))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/graph/{user_id}", response_model=SubgraphResponse)
def get_subgraph(
    user_id: str,
    hop: int = Query(default=1, description="BFS hop depth (1-5)"),
) -> SubgraphResponse:
    """
    BFS subgraph with automatic GNNExplainer overlay.

    If the root user has a pre-computed GNNExplainer explanation,
    the edge_mask values are automatically overlaid on the subgraph edges,
    and node_mask_top10 is included in the response.
    """
    hop_depth = _clamp_hop(hop)
    graph_service = GraphService.get_instance()

    graph_service._ensure_graph()
    if user_id not in graph_service._G:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in graph.")

    case_status_map = _get_case_status_map()

    # Try to get GNNExplainer edge_mask for overlay
    edge_mask_map: Dict[str, float] = {}
    has_gnn = False
    gnn_node_mask: List[GnnNodeImportance] = []

    try:
        explainer = ExplainerService.get_instance()
        if explainer.has_explanation(user_id):
            gnn_result = explainer.explain_gnn(user_id)
            edge_mask_map = gnn_result.edge_mask
            has_gnn = True
            gnn_node_mask = [
                GnnNodeImportance(feature=name, importance=round(val, 6))
                for name, val in gnn_result.node_mask_top10
            ]
            logger.info("GNNExplainer overlay applied for user %s", user_id)
    except Exception as exc:
        logger.debug("No GNN explanation for %s: %s", user_id, exc)

    subgraph_data = graph_service.get_subgraph(
        root_user_id=user_id,
        hop_depth=hop_depth,
        case_status_map=case_status_map,
    )

    # Overlay edge_mask if available
    if edge_mask_map:
        for edge in subgraph_data.edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            for key in [f"{src}|{tgt}", f"{tgt}|{src}"]:
                if key in edge_mask_map:
                    edge["edgeMask"] = edge_mask_map[key]
                    break

    nodes = [GraphNodeResponse(**n) for n in subgraph_data.nodes]
    edges = [GraphEdgeResponse(**e) for e in subgraph_data.edges]

    return SubgraphResponse(
        nodes=nodes,
        edges=edges,
        hasGnnExplanation=has_gnn,
        gnnNodeMaskTop10=gnn_node_mask,
    )


@router.get("/explain/gnn/{user_id}", response_model=GnnExplanationResponse)
def get_gnn_explanation(user_id: str) -> GnnExplanationResponse:
    """
    Return pre-computed GNNExplainer results for a user.

    Only available for top-K highest-risk users (pre-computed during training).
    """
    explainer_service = ExplainerService.get_instance()
    result = explainer_service.explain_gnn(user_id)

    return GnnExplanationResponse(
        user_id=result.user_id,
        edge_mask=result.edge_mask,
        node_mask_top10=result.node_mask_top10,
        computed_at=result.computed_at,
    )
