"""
GraphService — NetworkX subgraph expansion and serialization.

Builds and caches a NetworkX graph from ModelService's raw transaction data,
then exposes BFS-based subgraph expansion with blacklist-chain traversal rules.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Ensure project src/ is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = str(_PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from backend.app.core.config import (  # noqa: E402
    CASE_STATUS_NORMAL,
    CASE_STATUS_BLACKLISTED,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MEDIUM,
)
from backend.app.models.schemas import RiskLevel  # noqa: E402

logger = logging.getLogger(__name__)

INVALID_IP_HASHES: Set[str] = {
    "cfcd208495d565ef66e7dff9f98764da",
    "",
    "null",
    "00000000000000000000000000000000",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SubgraphData:
    """Serialized subgraph ready for the frontend GraphRenderer."""

    nodes: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GraphService
# ---------------------------------------------------------------------------


class GraphService:
    """
    Singleton-style service that builds a NetworkX graph from ModelService's
    cached transaction data and provides BFS subgraph expansion.
    """

    _instance: Optional["GraphService"] = None

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "GraphService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._G: Optional[nx.Graph] = None
        # edge_key -> total amount (TWD equivalent)
        self._edge_amounts: Dict[Tuple[str, str], float] = {}
        # prediction cache: user_id -> (risk_score, risk_level)
        self._pred_cache: Dict[str, Tuple[float, str]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """
        Build the NetworkX graph from ModelService's cached transaction data.
        Mirrors the logic in shared_features.compute_graph_features().
        """
        from backend.app.services.model_service import ModelService  # lazy import

        ms = ModelService.get_instance()

        G = nx.Graph()

        # Add all known users as nodes
        for uid in ms._user_info:
            G.add_node(uid)

        user_set: Set[str] = set(ms._user_info.keys())

        # --- Internal crypto transfers (sub_kind == 1) ---
        edge_amounts: Dict[Tuple[str, str], float] = defaultdict(float)

        for tx in ms._crypto_by_user.values():
            for t in tx:
                if t.get("sub_kind") == 1:
                    u = str(t.get("user_id", ""))
                    v = t.get("relation_user_id")
                    v = str(v) if v is not None else ""
                    if u in user_set and v and v in user_set and u != v:
                        # Accumulate TWD-equivalent amount
                        amount = (
                            t.get("ori_samount", 0)
                            * t.get("twd_srate", 1)
                            * 1e-16
                        )
                        key = (min(u, v), max(u, v))
                        edge_amounts[key] += amount
                        G.add_edge(u, v)

        # --- Shared IP edges ---
        ip_to_users: Dict[str, Set[str]] = defaultdict(set)
        for uid, txs in {**ms._crypto_by_user, **ms._twd_by_user}.items():
            if uid not in user_set:
                continue
            for t in txs:
                ip = t.get("source_ip_hash")
                if ip and ip not in INVALID_IP_HASHES:
                    ip_to_users[ip].add(uid)
        for ip, users in ip_to_users.items():
            users_list = list(users)
            if 2 <= len(users_list) <= 20:
                for i in range(len(users_list)):
                    for j in range(i + 1, len(users_list)):
                        u, v = users_list[i], users_list[j]
                        G.add_edge(u, v)
                        # Shared-IP edges have no direct amount; keep 0 default

        self._G = G
        self._edge_amounts = dict(edge_amounts)

        logger.info(
            "GraphService: built graph with %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

    def _ensure_graph(self) -> nx.Graph:
        """Return cached graph, building it on first call."""
        if self._G is None:
            self._build_graph()
        return self._G  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _get_prediction(self, user_id: str) -> Tuple[float, str]:
        """
        Return (risk_score, risk_level) for a user, using a local cache
        to avoid repeated ModelService calls.
        """
        if user_id in self._pred_cache:
            return self._pred_cache[user_id]

        try:
            from backend.app.services.model_service import ModelService

            result = ModelService.get_instance().predict_single(user_id)
            pair = (result.risk_score, result.risk_level)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not predict for %s: %s", user_id, exc)
            pair = (0.0, RiskLevel.low.value)

        self._pred_cache[user_id] = pair
        return pair

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_subgraph(
        self,
        root_user_id: str,
        hop_depth: int,
        case_status_map: Dict[str, str],
    ) -> SubgraphData:
        """
        BFS expansion from root_user_id up to hop_depth hops.

        Expansion rules:
        - 'normal' nodes: added to subgraph but NOT expanded further (leaf only)
        - 'blacklisted' nodes: expanded normally (blacklist-chain traversal)
        - 'pending' / 'watching' nodes: expanded normally

        Args:
            root_user_id: Starting node for BFS.
            hop_depth: Maximum number of hops (1–5).
            case_status_map: Mapping of user_id -> Case_Status string.

        Returns:
            SubgraphData with serialized nodes and edges.
        """
        G = self._ensure_graph()

        if root_user_id not in G:
            logger.warning("root_user_id '%s' not in graph", root_user_id)
            return SubgraphData()

        hop_depth = max(1, min(5, hop_depth))

        # BFS state
        visited: Set[str] = set()
        # queue entries: (node_id, current_depth)
        queue: deque[Tuple[str, int]] = deque()
        queue.append((root_user_id, 0))
        visited.add(root_user_id)

        subgraph_nodes: Set[str] = {root_user_id}

        while queue:
            node, depth = queue.popleft()

            # Do not expand beyond hop_depth
            if depth >= hop_depth:
                continue

            # Do not expand 'normal' nodes (they are leaves)
            node_status = case_status_map.get(node, "pending")
            if node_status == CASE_STATUS_NORMAL and node != root_user_id:
                continue

            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Build the induced subgraph
        subgraph = G.subgraph(subgraph_nodes)
        return self.serialize_subgraph(subgraph, case_status_map)

    def serialize_subgraph(
        self,
        subgraph: nx.Graph,
        case_status_map: Optional[Dict[str, str]] = None,
        edge_mask_map: Optional[Dict[str, float]] = None,
    ) -> SubgraphData:
        """
        Serialize a NetworkX subgraph into SubgraphData (nodes + edges dicts).

        Node fields: id, riskScore, riskLevel, caseStatus, graphDegree,
                     cryptoTotalAmount, nightTxRatio
        Edge fields: source, target, amount, edgeMask (optional)

        Args:
            subgraph: The NetworkX subgraph to serialize.
            case_status_map: Optional mapping of user_id -> Case_Status.
            edge_mask_map: Optional mapping of "u|v" -> edge_mask float.

        Returns:
            SubgraphData with nodes and edges lists.
        """
        if case_status_map is None:
            case_status_map = {}
        if edge_mask_map is None:
            edge_mask_map = {}

        # Degree within the *full* graph (not just the subgraph)
        full_G = self._ensure_graph()

        nodes: List[dict] = []
        for uid in subgraph.nodes():
            risk_score, risk_level = self._get_prediction(uid)
            case_status = case_status_map.get(uid, "pending")
            graph_degree = full_G.degree(uid) if uid in full_G else 0

            # Fetch user-level stats from ModelService cache
            crypto_total, night_ratio = self._get_user_stats(uid)

            # Determine blacklist status:
            # - train_label users: use ground truth (fraud_label)
            # - predict_label users (fraud_label=None): use model prediction
            fraud_label = self._get_fraud_label(uid)
            if fraud_label is not None:
                is_blacklist = (fraud_label == 1)
            else:
                is_blacklist = (risk_score > 0.35)

            nodes.append(
                {
                    "id": uid,
                    "riskScore": round(risk_score, 4),
                    "riskLevel": risk_level,
                    "caseStatus": case_status,
                    "graphDegree": graph_degree,
                    "cryptoTotalAmount": round(crypto_total, 2),
                    "nightTxRatio": round(night_ratio, 4),
                    "isBlacklist": is_blacklist,
                    "fraudLabel": fraud_label,
                }
            )

        edges: List[dict] = []
        for u, v in subgraph.edges():
            key = (min(u, v), max(u, v))
            amount = self._edge_amounts.get(key, 0.0)

            # Look up edge_mask by both orderings
            mask_key1 = f"{u}|{v}"
            mask_key2 = f"{v}|{u}"
            edge_mask = edge_mask_map.get(mask_key1) or edge_mask_map.get(mask_key2)

            edge_dict: dict = {
                "source": u,
                "target": v,
                "amount": round(amount, 2),
            }
            if edge_mask is not None:
                edge_dict["edgeMask"] = round(float(edge_mask), 4)

            edges.append(edge_dict)

        return SubgraphData(nodes=nodes, edges=edges)

    # ------------------------------------------------------------------
    # User stats helper
    # ------------------------------------------------------------------

    def _get_fraud_label(self, user_id: str) -> Optional[int]:
        """Return fraud label (1=fraud, 0=normal, None=unknown)."""
        try:
            from backend.app.services.model_service import ModelService
            ms = ModelService.get_instance()
            return ms._fraud_labels.get(user_id)
        except Exception:
            return None

    def _get_user_stats(self, user_id: str) -> Tuple[float, float]:
        """
        Return (crypto_total_amount_twd, night_tx_ratio) for a user
        from ModelService's cached transaction data.
        """
        try:
            from backend.app.services.model_service import ModelService

            ms = ModelService.get_instance()
            crypto_txs = ms._crypto_by_user.get(user_id, [])

            # Crypto total amount in TWD
            crypto_total = sum(
                tx.get("ori_samount", 0) * tx.get("twd_srate", 1) * 1e-16
                for tx in crypto_txs
            )

            # Night tx ratio (23:00–06:00)
            from shared_features import parse_hour, is_night_hour  # noqa: E402

            all_txs = (
                ms._crypto_by_user.get(user_id, [])
                + ms._twd_by_user.get(user_id, [])
            )
            total_count = len(all_txs)
            if total_count == 0:
                return crypto_total, 0.0

            night_count = sum(
                is_night_hour(parse_hour(tx.get("created_at", "")))
                for tx in all_txs
            )
            return crypto_total, night_count / total_count

        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not get user stats for %s: %s", user_id, exc)
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Force graph rebuild on next access (e.g., after data reload)."""
        self._G = None
        self._edge_amounts = {}
        self._pred_cache = {}
        logger.info("GraphService cache invalidated")
