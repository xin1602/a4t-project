"""
Core configuration constants for A4T Dashboard backend.
Defines required model files and GraphSAGE pkl field validation.
"""
from typing import List

# Required pkl files that must exist in outputs/ before API server starts
REQUIRED_PKL_FILES: List[str] = [
    "outputs/lgbm_v13_no_graph_results.pkl",
    "outputs/graphsage_v13_gpu_complete_results.pkl",
    "outputs/ensemble_results.pkl",
]

# Required keys that must be present in the GraphSAGE pkl for ExplainerService
REQUIRED_GRAPHSAGE_KEYS: List[str] = [
    "oof_predictions",
    "y_true",
    "user_ids",
    "feature_names",
]

# GraphSAGE pkl filename (v13 GPU Complete with GNNExplainer)
GRAPHSAGE_PKL_FILE: str = "outputs/graphsage_v13_gpu_complete_results.pkl"

# Risk score thresholds
RISK_THRESHOLD_LOW: float = 0.35
RISK_THRESHOLD_MEDIUM: float = 0.6
RISK_THRESHOLD_HIGH: float = 0.8

# Case status values
CASE_STATUS_PENDING = "pending"
CASE_STATUS_BLACKLISTED = "blacklisted"
CASE_STATUS_WATCHING = "watching"
CASE_STATUS_NORMAL = "normal"

# Data file paths
CASE_STATUS_FILE = "outputs/case_status.json"
AUDIT_LOG_FILE = "outputs/audit_log.json"
GNN_EXPLANATIONS_CACHE = "outputs/gnn_explanations.pkl"
