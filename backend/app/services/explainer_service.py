"""
ExplainerService — Pre-computed GNNExplainer lookup from GraphSAGE V13 GPU Complete pkl.

The new training script (train_graphsage_v13_gpu_complete.py) pre-computes
GNNExplainer results for the TOP-K highest-risk users and stores them in the pkl
under the key 'gnn_explanations'.

This service loads those pre-computed results and serves them via a simple lookup,
eliminating the need for real-time GNNExplainer computation (which is slow).

Structure of gnn_explanations in pkl:
    {
        user_id: {
            'edge_mask': np.ndarray (shape: [n_edges]),
            'node_mask': np.ndarray (shape: [n_nodes, n_features]) or None,
            'error': str (only if computation failed)
        },
        ...
    }
"""
from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project src/ is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = str(_PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from backend.app.core.config import (  # noqa: E402
    GRAPHSAGE_PKL_FILE,
    REQUIRED_GRAPHSAGE_KEYS,
)
from backend.app.models.schemas import GnnExplanation  # noqa: E402

logger = logging.getLogger(__name__)


class ExplainerService:
    """
    Singleton service providing pre-computed GNNExplainer explanations
    from the GraphSAGE V13 GPU Complete training results.
    """

    _instance: Optional["ExplainerService"] = None

    @classmethod
    def get_instance(cls) -> "ExplainerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._sage_data: Optional[dict] = None
        # Pre-processed GNN explanations: user_id -> GnnExplanation
        self._gnn_cache: Dict[str, GnnExplanation] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> dict:
        """Load and validate the GraphSAGE pkl on first access."""
        if self._sage_data is not None:
            return self._sage_data

        from fastapi import HTTPException

        sage_path = _PROJECT_ROOT / GRAPHSAGE_PKL_FILE
        if not sage_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"GraphSAGE pkl not found at '{sage_path}'.",
            )

        with open(sage_path, "rb") as f:
            self._sage_data = pickle.load(f)

        missing = [k for k in REQUIRED_GRAPHSAGE_KEYS if k not in self._sage_data]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"GraphSAGE pkl missing required fields: {missing}",
            )

        # Pre-process gnn_explanations into GnnExplanation objects
        self._preprocess_gnn_explanations()
        self._loaded = True
        logger.info(
            "ExplainerService: loaded pkl with %d pre-computed GNN explanations",
            len(self._gnn_cache),
        )
        return self._sage_data

    def _preprocess_gnn_explanations(self) -> None:
        """
        Convert raw gnn_explanations from pkl into structured GnnExplanation objects.

        For each user, extracts:
        - edge_mask: raw numpy array -> Dict[str, float] keyed by "src_idx|tgt_idx"
        - node_mask: numpy array -> top-10 feature importances
        """
        if self._sage_data is None:
            return

        raw_explanations = self._sage_data.get("gnn_explanations", {})
        feature_names = self._sage_data.get("feature_names", [])
        user_ids = self._sage_data.get("user_ids", [])
        computed_at = datetime.now(timezone.utc).isoformat()

        for uid, exp_data in raw_explanations.items():
            if exp_data.get("error"):
                logger.warning("GNN explanation error for %s: %s", uid, exp_data["error"])
                continue

            # Normalize uid to string for consistent lookup
            uid_str = str(uid)

            # Process edge_mask
            edge_mask_dict: Dict[str, float] = {}
            raw_edge_mask = exp_data.get("edge_mask")
            if raw_edge_mask is not None and isinstance(raw_edge_mask, np.ndarray):
                # Store top edges by importance (threshold > 0.1 to keep it manageable)
                for i, val in enumerate(raw_edge_mask):
                    if float(val) > 0.1:
                        edge_mask_dict[str(i)] = round(float(val), 4)

            # Process node_mask -> top-10 feature importances
            node_mask_top10: List[Tuple[str, float]] = []
            raw_node_mask = exp_data.get("node_mask")
            if raw_node_mask is not None and isinstance(raw_node_mask, np.ndarray):
                # node_mask shape: (n_nodes, n_features) — get the row for this user
                user_idx = None
                uid_str = str(uid)
                for i, u in enumerate(user_ids):
                    if str(u) == uid_str:
                        user_idx = i
                        break

                if user_idx is not None and raw_node_mask.ndim == 2:
                    node_importances = raw_node_mask[user_idx]
                elif raw_node_mask.ndim == 1:
                    node_importances = raw_node_mask
                else:
                    node_importances = None

                if node_importances is not None:
                    sorted_indices = sorted(
                        range(len(node_importances)),
                        key=lambda i: abs(node_importances[i]),
                        reverse=True,
                    )[:10]
                    node_mask_top10 = [
                        (
                            feature_names[i] if i < len(feature_names) else f"feature_{i}",
                            round(float(node_importances[i]), 6),
                        )
                        for i in sorted_indices
                    ]

            self._gnn_cache[uid_str] = GnnExplanation(
                user_id=uid_str,
                edge_mask=edge_mask_dict,
                node_mask_top10=node_mask_top10,
                computed_at=computed_at,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_users(self) -> List[str]:
        """Return list of user_ids that have pre-computed GNN explanations."""
        self._ensure_loaded()
        return list(self._gnn_cache.keys())

    def has_explanation(self, user_id: str) -> bool:
        """Check if a pre-computed GNN explanation exists for user_id."""
        self._ensure_loaded()
        return str(user_id) in self._gnn_cache

    def explain_gnn(self, user_id: str) -> GnnExplanation:
        """
        Look up pre-computed GNNExplainer results for a user.

        Returns cached GnnExplanation with edge_mask and node_mask_top10.

        Raises:
            HTTPException(404): if user_id has no pre-computed explanation.
        """
        from fastapi import HTTPException

        self._ensure_loaded()
        uid_str = str(user_id)

        if uid_str in self._gnn_cache:
            return self._gnn_cache[uid_str]

        # Try numeric conversion
        try:
            numeric_id = int(user_id)
            if str(numeric_id) in self._gnn_cache:
                return self._gnn_cache[str(numeric_id)]
        except (ValueError, TypeError):
            pass

        raise HTTPException(
            status_code=404,
            detail=(
                f"No pre-computed GNN explanation for user '{user_id}'. "
                f"Available: {len(self._gnn_cache)} users. "
                "GNNExplainer is only pre-computed for top-K highest-risk users."
            ),
        )

    def get_node_mask_top10(self, user_id: str) -> List[Tuple[str, float]]:
        """
        Get top-10 most important features from GNNExplainer node_mask.

        Returns empty list if no explanation exists (graceful fallback).
        """
        self._ensure_loaded()
        uid_str = str(user_id)
        if uid_str in self._gnn_cache:
            return self._gnn_cache[uid_str].node_mask_top10
        return []
