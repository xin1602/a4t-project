"""
Data models and schemas for A4T Dashboard backend.
Defines dataclasses, enums, and state machine constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Risk level thresholds
# ---------------------------------------------------------------------------

RISK_THRESHOLD_LOW: float = 0.35      # below this -> low
RISK_THRESHOLD_MEDIUM: float = 0.6    # 0.35–0.6   -> medium
RISK_THRESHOLD_HIGH: float = 0.8      # 0.6–0.8    -> high
                                       # above 0.8  -> critical


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk level classification based on risk score thresholds."""
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class CaseStatus(str, Enum):
    """Valid case status values for the case state machine."""
    pending = "pending"
    blacklisted = "blacklisted"
    watching = "watching"
    normal = "normal"


# ---------------------------------------------------------------------------
# State machine: valid transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Set[Tuple[str, str]] = {
    (CaseStatus.pending,     CaseStatus.blacklisted),
    (CaseStatus.pending,     CaseStatus.watching),
    (CaseStatus.pending,     CaseStatus.normal),
    (CaseStatus.watching,    CaseStatus.blacklisted),
    (CaseStatus.watching,    CaseStatus.normal),
    (CaseStatus.blacklisted, CaseStatus.watching),   # downgrade
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Result of a single-user fraud risk prediction."""
    user_id: str
    risk_score: float                   # 0.0–1.0
    is_blacklist: bool                  # risk_score > RISK_THRESHOLD_LOW (0.35)
    risk_level: str                     # 'low' | 'medium' | 'high' | 'critical'
    shap_values: Dict[str, float]       # feature_name -> shap_value
    top_features: List[str]             # Top 10 feature names sorted by |shap| desc


@dataclass
class CaseRecord:
    """A case record stored in case_status.json."""
    case_id: str
    user_id: str
    status: str                         # CaseStatus value
    risk_score: float
    created_at: str                     # ISO 8601 UTC
    updated_at: str                     # ISO 8601 UTC
    updated_by: str                     # operator identifier


@dataclass
class AuditEntry:
    """An immutable audit log entry stored in audit_log.json."""
    entry_id: str
    case_id: str
    user_id: str
    operator: str
    timestamp_utc: str                  # ISO 8601 UTC
    old_status: str                     # CaseStatus value
    new_status: str                     # CaseStatus value


@dataclass
class GnnExplanation:
    """GNNExplainer output for a single user."""
    user_id: str
    edge_mask: Dict[str, float]         # edge_id -> mask_value (0.0–1.0)
    node_mask_top10: List[Tuple[str, float]]  # (feature_name, importance)
    computed_at: str                    # ISO 8601 UTC


@dataclass
class IgExplanation:
    """Integrated Gradients attribution output for a single user."""
    user_id: str
    attributions: Dict[str, float]      # feature_name -> attribution_value
                                        # positive = pushes toward fraud
                                        # negative = pushes toward normal
    convergence_delta: float            # captum convergence error
    computed_at: str                    # ISO 8601 UTC
