"""
Users router ??GET /api/users/{user_id}

Returns per-user fraud risk details including risk score, SHAP values,
Integrated Gradients attributions, transaction statistics, and case status.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.case_service import CaseService
from backend.app.services.explainer_service import ExplainerService
from backend.app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class UserDetailResponse(BaseModel):
    user_id: str
    risk_score: float                       # 0.0-1.0
    risk_level: str                         # low | medium | high | critical
    shap_values: Dict[str, float]           # feature_name -> shap_value
    top_features: List[str]                 # Top 10 features sorted by |shap| desc
    feature_values: Dict[str, float]        # feature_name -> raw feature value
    feature_stats: Dict[str, Any]           # feature_name -> stats (continuous or categorical)
    ig_attributions: Dict[str, float]       # feature_name -> ig_attribution (graceful fallback)
    tx_volume: int                          # total transaction count across all types
    night_tx_ratio: float                   # ratio of night-time transactions
    counterparty_count: int                 # number of unique counterparties
    case_status: str                        # pending | blacklisted | watching | normal
    fraud_label: Optional[int]               # 1=fraud, 0=normal, null=unknown (not in train_label)
    heatmap_data: List[Dict]                # [{hour, day, count}, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_tx_stats(model_service: ModelService, user_id: str) -> tuple[int, float]:
    """
    Compute tx_volume and night_tx_ratio from ModelService cached data.

    tx_volume = total count of twd + crypto + trading + swap transactions.
    night_tx_ratio = fraction of transactions occurring between 22:00??6:00.

    Returns:
        (tx_volume, night_tx_ratio)
    """
    twd_txs = model_service._twd_by_user.get(user_id, [])
    crypto_txs = model_service._crypto_by_user.get(user_id, [])
    trading_txs = model_service._trading_by_user.get(user_id, [])
    swap_txs = model_service._swap_by_user.get(user_id, [])

    all_txs = twd_txs + crypto_txs + trading_txs + swap_txs
    tx_volume = len(all_txs)

    if tx_volume == 0:
        return 0, 0.0

    # Count night-time transactions (22:00??6:00 UTC)
    night_count = 0
    for tx in all_txs:
        ts = tx.get("created_at") or tx.get("updated_at") or tx.get("timestamp") or tx.get("time") or ""
        if ts:
            try:
                # Parse hour from ISO 8601 string: "2026-01-15T23:30:00Z"
                hour = int(ts[11:13])
                if hour >= 22 or hour < 6:
                    night_count += 1
            except (ValueError, IndexError):
                pass

    night_tx_ratio = night_count / tx_volume
    return tx_volume, night_tx_ratio


def _count_counterparties(model_service: ModelService, user_id: str) -> int:
    """
    Count unique counterparties across all transaction types.
    Uses relation_user_id from crypto_transfer and user_id pairs from twd_transfer.
    """
    counterparties: set = set()

    # crypto_transfer: relation_user_id
    for tx in model_service._crypto_by_user.get(user_id, []):
        rel = tx.get("relation_user_id")
        if rel is not None:
            counterparties.add(str(rel))

    # twd_transfer: relation_user_id
    for tx in model_service._twd_by_user.get(user_id, []):
        rel = tx.get("relation_user_id")
        if rel is not None:
            counterparties.add(str(rel))

    return len(counterparties)


def _compute_heatmap_data(model_service: ModelService, user_id: str) -> List[Dict]:
    """
    Build 24h x 7day heatmap data from transaction timestamps.
    Returns list of {hour: int, day: int (0=Mon), count: int}.
    """
    from datetime import datetime as dt

    twd_txs = model_service._twd_by_user.get(user_id, [])
    crypto_txs = model_service._crypto_by_user.get(user_id, [])
    trading_txs = model_service._trading_by_user.get(user_id, [])
    swap_txs = model_service._swap_by_user.get(user_id, [])
    all_txs = twd_txs + crypto_txs + trading_txs + swap_txs

    counts: Dict[tuple, int] = {}
    for tx in all_txs:
        ts = tx.get("created_at") or tx.get("updated_at") or tx.get("timestamp") or tx.get("time") or ""
        if not ts or len(ts) < 13:
            continue
        try:
            hour = int(ts[11:13])
            parsed = dt.fromisoformat(ts.replace("Z", "+00:00"))
            day = parsed.weekday()  # 0=Monday
            key = (hour, day)
            counts[key] = counts.get(key, 0) + 1
        except (ValueError, IndexError):
            pass

    return [{"hour": h, "day": d, "count": c} for (h, d), c in counts.items()]


def _get_case_status(user_id: str) -> str:
    """
    Retrieve case status for user_id from CaseService.

    Returns "pending" if no case exists or on any error.
    """
    try:
        case_service = CaseService.get_instance()
        case_id = f"CASE-{user_id}"
        case = case_service.get_case(case_id)
        return case.status
    except HTTPException as exc:
        if exc.status_code == 404:
            return "pending"
        logger.warning("CaseService error for user %s: %s", user_id, exc.detail)
        return "pending"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get case status for user %s: %s", user_id, exc)
        return "pending"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}", response_model=UserDetailResponse)
def get_user_detail(user_id: str) -> UserDetailResponse:
    """
    Return detailed fraud risk information for a single user.

    - risk_score, risk_level, shap_values, top_features: from ModelService.predict_single()
    - ig_attributions: from ExplainerService.explain_ig() (graceful fallback to {})
    - tx_volume, night_tx_ratio: computed from ModelService cached transaction data
    - external_transfer_ratio: computed from cached crypto transactions
    - case_status: from CaseService (defaults to "pending" if no case exists)

    Raises:
        HTTP 404 if user_id is not found in the system.
    """
    # ------------------------------------------------------------------
    # 1. Prediction (raises KeyError -> 404 if user not found)
    # ------------------------------------------------------------------
    model_service = ModelService.get_instance()
    try:
        prediction = model_service.predict_single(user_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found in the system.",
        )

    # ------------------------------------------------------------------
    # 2. Raw feature values for waterfall chart display
    # ------------------------------------------------------------------
    feature_values: Dict[str, float] = {}
    try:
        from src.shared_features import _compute_user_features
        feat_dict = _compute_user_features(
            user_id=user_id,
            user_info=model_service._user_info[user_id],
            twd_txs=model_service._twd_by_user.get(user_id, []),
            crypto_txs=model_service._crypto_by_user.get(user_id, []),
            trading_txs=model_service._trading_by_user.get(user_id, []),
            swap_txs=model_service._swap_by_user.get(user_id, []),
        )
        feature_values = {k: float(v) for k, v in feat_dict.items() if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature value computation failed for %s: %s", user_id, exc)

    # ------------------------------------------------------------------
    # 2.5 Global feature stats for comparison
    # ------------------------------------------------------------------
    all_stats = model_service.get_feature_stats()
    # Only include stats for top features to keep response small
    feature_stats = {
        f: all_stats[f] for f in prediction.top_features if f in all_stats
    }

    # ------------------------------------------------------------------
    # 3. IG attributions — skip synchronous computation (too slow for
    #    51k-node forward pass × 50 steps).  Frontend should call
    #    GET /api/explain/ig/{user_id} separately if needed.
    # ------------------------------------------------------------------
    ig_attributions: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 3. Transaction statistics from cached data
    # ------------------------------------------------------------------
    tx_volume, night_tx_ratio = _compute_tx_stats(model_service, user_id)

    # ------------------------------------------------------------------
    # 4. Counterparty count
    # ------------------------------------------------------------------
    counterparty_count = _count_counterparties(model_service, user_id)

    # ------------------------------------------------------------------
    # 5. Fraud label from train_label.jsonl
    # ------------------------------------------------------------------
    fraud_label = model_service._fraud_labels.get(user_id)

    # ------------------------------------------------------------------
    # 6. Case status
    # ------------------------------------------------------------------
    case_status = _get_case_status(user_id)

    # ------------------------------------------------------------------
    # 7. Heatmap data (24h x 7day)
    # ------------------------------------------------------------------
    heatmap_data = _compute_heatmap_data(model_service, user_id)

    return UserDetailResponse(
        user_id=prediction.user_id,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level,
        shap_values=prediction.shap_values,
        top_features=prediction.top_features,
        feature_values=feature_values,
        feature_stats=feature_stats,
        ig_attributions=ig_attributions,
        tx_volume=tx_volume,
        night_tx_ratio=night_tx_ratio,
        counterparty_count=counterparty_count,
        case_status=case_status,
        fraud_label=fraud_label,
        heatmap_data=heatmap_data,
    )

