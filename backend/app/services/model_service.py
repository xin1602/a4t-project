"""
ModelService — Singleton for loading pkl models and running inference.

Loads outputs/*.pkl at startup, caches user data for fast inference,
and exposes predict_single / predict_batch / get_risk_level.
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project src/ is importable (shared_features lives there)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/app/services -> root
_SRC_DIR = str(_PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from shared_features import _compute_user_features  # noqa: E402

from backend.app.core.config import (  # noqa: E402
    REQUIRED_PKL_FILES,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MEDIUM,
)
from backend.app.models.schemas import PredictionResult, RiskLevel  # noqa: E402

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton that loads all model pkl files at startup and serves predictions."""

    _instance: Optional["ModelService"] = None

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_models()
        return cls._instance

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Model artefacts loaded from pkl files
        self.lgbm_model = None
        self.sage_model = None
        self.ensemble_model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self._lgbm_fold_models: List = []  # LightGBM fold models for real-time inference

        # Pre-computed OOF prediction scores: user_id (str) -> risk_score
        self._oof_scores: Dict[str, float] = {}
        # Pre-computed SHAP-like feature importances from lgbm pkl
        self._lgbm_oof_scores: Dict[str, float] = {}
        # Real SHAP values per user: user_id (str) -> Dict[feature_name, shap_value]
        self._shap_values: Dict[str, Dict[str, float]] = {}

        # Cached raw data for fast inference
        self._user_info: Dict[str, dict] = {}
        self._twd_by_user: Dict[str, List[dict]] = defaultdict(list)
        self._crypto_by_user: Dict[str, List[dict]] = defaultdict(list)
        self._trading_by_user: Dict[str, List[dict]] = defaultdict(list)
        self._swap_by_user: Dict[str, List[dict]] = defaultdict(list)
        self._data_loaded: bool = False

        # Ground truth labels from train_label.jsonl: user_id (str) -> 0 or 1
        self._fraud_labels: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load outputs/*.pkl and validate required files exist."""
        missing: List[str] = []
        for rel_path in REQUIRED_PKL_FILES:
            full_path = _PROJECT_ROOT / rel_path
            if not full_path.exists():
                logger.error("Required pkl file missing: %s", full_path)
                missing.append(str(full_path))

        if missing:
            raise FileNotFoundError(
                "Missing required pkl files:\n" + "\n".join(f"  - {p}" for p in missing)
            )

        # Load LightGBM results
        lgbm_path = _PROJECT_ROOT / "outputs/lgbm_v13_no_graph_results.pkl"
        with open(lgbm_path, "rb") as f:
            lgbm_results = pickle.load(f)
        self.feature_names = lgbm_results.get("feature_names", [])
        self._lgbm_fold_models = lgbm_results.get("fold_models", [])
        # Build lgbm OOF score lookup
        lgbm_uids = lgbm_results.get("user_ids", [])
        lgbm_oof = lgbm_results.get("oof_predictions", np.array([]))
        for uid, score in zip(lgbm_uids, lgbm_oof):
            self._lgbm_oof_scores[str(uid)] = float(score)

        # Load real SHAP values if available
        lgbm_shap = lgbm_results.get("oof_shap_values")
        lgbm_fnames = lgbm_results.get("feature_names", [])
        if lgbm_shap is not None and len(lgbm_uids) == len(lgbm_shap):
            for i, uid in enumerate(lgbm_uids):
                row = lgbm_shap[i]
                self._shap_values[str(uid)] = {
                    lgbm_fnames[j]: float(row[j])
                    for j in range(min(len(lgbm_fnames), len(row)))
                }
            logger.info("Loaded real SHAP values for %d users", len(self._shap_values))
        else:
            logger.warning("No oof_shap_values in LightGBM pkl; SHAP will use feature-value fallback")

        logger.info("Loaded LightGBM pkl: %s (%d OOF scores)", lgbm_path, len(self._lgbm_oof_scores))

        # Load GraphSAGE results (V13 GPU Complete with GNNExplainer)
        sage_path = _PROJECT_ROOT / "outputs/graphsage_v13_gpu_complete_results.pkl"
        with open(sage_path, "rb") as f:
            sage_results = pickle.load(f)
        self.scaler = sage_results.get("scaler")
        logger.info("Loaded GraphSAGE pkl: %s", sage_path)

        # Load ensemble results — use oof_predictions as the primary score lookup
        ensemble_path = _PROJECT_ROOT / "outputs/ensemble_results.pkl"
        with open(ensemble_path, "rb") as f:
            ensemble_results = pickle.load(f)
        # Build ensemble OOF score lookup: user_id -> risk_score
        ens_uids = ensemble_results.get("user_ids", [])
        ens_oof = ensemble_results.get("oof_predictions", np.array([]))
        for uid, score in zip(ens_uids, ens_oof):
            self._oof_scores[str(uid)] = float(score)
        # Use feature_names from ensemble if available, else keep lgbm's
        self.feature_names = ensemble_results.get("feature_names") or self.feature_names
        logger.info("Loaded ensemble pkl: %s (%d OOF scores)", ensemble_path, len(self._oof_scores))

        # Pre-load user data for inference
        self._load_user_data()

    # ------------------------------------------------------------------
    # User data caching
    # ------------------------------------------------------------------

    def _load_user_data(self) -> None:
        """Load JSONL data files into memory for fast per-user feature computation."""
        import json

        from config import get_data_dir  # type: ignore

        try:
            data_dir = get_data_dir()
        except FileNotFoundError as exc:
            logger.warning("Could not load user data: %s", exc)
            return

        def _read_jsonl(filename: str) -> List[dict]:
            path = os.path.join(data_dir, filename)
            rows: List[dict] = []
            if not os.path.exists(path):
                logger.warning("Data file not found: %s", path)
                return rows
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        # user_info
        for row in _read_jsonl("user_info.jsonl"):
            self._user_info[str(row["user_id"])] = row

        # transactions
        for tx in _read_jsonl("twd_transfer.jsonl"):
            self._twd_by_user[str(tx["user_id"])].append(tx)
        for tx in _read_jsonl("crypto_transfer.jsonl"):
            self._crypto_by_user[str(tx["user_id"])].append(tx)
        for tx in _read_jsonl("usdt_twd_trading.jsonl"):
            self._trading_by_user[str(tx["user_id"])].append(tx)
        for tx in _read_jsonl("usdt_swap.jsonl"):
            self._swap_by_user[str(tx["user_id"])].append(tx)

        self._data_loaded = True

        # train_label (ground truth)
        for row in _read_jsonl("train_label.jsonl"):
            self._fraud_labels[str(row["user_id"])] = int(row.get("status", 0))

        logger.info(
            "User data cached: %d users, %d twd, %d crypto, %d trading, %d swap, %d labels",
            len(self._user_info),
            sum(len(v) for v in self._twd_by_user.values()),
            sum(len(v) for v in self._crypto_by_user.values()),
            sum(len(v) for v in self._trading_by_user.values()),
            sum(len(v) for v in self._swap_by_user.values()),
            len(self._fraud_labels),
        )

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _build_feature_vector(self, user_id: str) -> np.ndarray:
        """Compute feature dict for user_id and return as ordered numpy array."""
        if user_id not in self._user_info:
            raise KeyError(f"user_id '{user_id}' not found in cached data")

        feat_dict = _compute_user_features(
            user_id=user_id,
            user_info=self._user_info[user_id],
            twd_txs=self._twd_by_user.get(user_id, []),
            crypto_txs=self._crypto_by_user.get(user_id, []),
            trading_txs=self._trading_by_user.get(user_id, []),
            swap_txs=self._swap_by_user.get(user_id, []),
        )

        if self.feature_names:
            vec = np.array([feat_dict.get(k, 0.0) for k in self.feature_names], dtype=float)
        else:
            # Fallback: use sorted keys
            keys = sorted(feat_dict.keys())
            vec = np.array([feat_dict[k] for k in keys], dtype=float)

        return vec

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler if available, otherwise return raw features."""
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def _run_inference(self, X_scaled: np.ndarray) -> np.ndarray:
        """Placeholder — not used in OOF lookup mode."""
        logger.warning("No trained model available; returning zero scores")
        return np.zeros(X_scaled.shape[0])

    def _compute_shap(self, X_scaled: np.ndarray) -> List[Dict[str, float]]:
        """Return empty SHAP dicts — no trained model available for SHAP computation."""
        return [{}] * X_scaled.shape[0]

    def _get_oof_score(self, user_id: str) -> float:
        """Look up the pre-computed ensemble OOF score for a user."""
        return self._oof_scores.get(user_id, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _realtime_lgbm_predict(self, user_id: str) -> tuple:
        """
        Real-time LightGBM inference + SHAP for users not in OOF scores
        (i.e. predict_label users).

        Returns:
            (risk_score: float, shap_dict: Dict[str, float])
        """
        if not self._lgbm_fold_models:
            return 0.0, {}

        import shap

        # Build feature vector
        X = self._build_feature_vector(user_id).reshape(1, -1)

        # Average predictions across fold models
        preds = []
        shap_vals_list = []
        for model in self._lgbm_fold_models:
            # Predict probability
            pred = model.predict(X, num_iteration=model.best_iteration)[0]
            preds.append(pred)

            # SHAP values
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            # For binary classification, shap_values returns [neg_class, pos_class]
            if isinstance(sv, list):
                sv = sv[1]  # positive class
            shap_vals_list.append(sv[0])

        avg_score = float(np.mean(preds))
        avg_shap = np.mean(shap_vals_list, axis=0)

        fnames = self.feature_names
        shap_dict = {
            fnames[i]: float(avg_shap[i])
            for i in range(min(len(fnames), len(avg_shap)))
        }

        return avg_score, shap_dict

    def predict_single(self, user_id: str) -> PredictionResult:
        """
        Look up fraud risk for a single user from pre-computed OOF scores.
        For users not in OOF (predict_label), runs real-time LightGBM inference.

        Raises:
            KeyError: if user_id is not found in cached data.
        """
        if user_id not in self._user_info:
            raise KeyError(f"user_id '{user_id}' not found in cached data")

        # Check if user has pre-computed OOF score (train_label users)
        if user_id in self._oof_scores:
            score = self._oof_scores[user_id]
            # Use pre-computed SHAP values
            if user_id in self._shap_values:
                shap_values = self._shap_values[user_id]
            else:
                shap_values = {}
        else:
            # predict_label user — real-time LightGBM inference + SHAP
            try:
                score, shap_values = self._realtime_lgbm_predict(user_id)
                logger.info("Real-time LightGBM prediction for user %s: score=%.4f", user_id, score)
            except Exception as exc:
                logger.warning("Real-time prediction failed for %s: %s", user_id, exc)
                score = 0.0
                shap_values = {}

        top_features: List[str] = sorted(
            shap_values.keys(), key=lambda k: abs(shap_values[k]), reverse=True
        )[:10]

        return PredictionResult(
            user_id=user_id,
            risk_score=score,
            is_blacklist=score > RISK_THRESHOLD_LOW,
            risk_level=self.get_risk_level(score).value,
            shap_values=shap_values,
            top_features=top_features,
        )

    def predict_batch(self, user_ids: List[str]) -> List[PredictionResult]:
        """
        Look up fraud risk for a list of users from pre-computed OOF scores.

        Users not found in either OOF scores or user_info get score 0.
        """
        results: List[PredictionResult] = []

        for uid in user_ids:
            uid_str = str(uid)
            score = self._get_oof_score(uid_str)
            results.append(
                PredictionResult(
                    user_id=uid_str,
                    risk_score=score,
                    is_blacklist=score > RISK_THRESHOLD_LOW,
                    risk_level=self.get_risk_level(score).value,
                    shap_values={},
                    top_features=[],
                )
            )

        return results

    # ------------------------------------------------------------------
    # Global feature statistics (loaded from pre-computed JSON)
    # ------------------------------------------------------------------

    _feature_stats: Optional[Dict[str, dict]] = None

    def get_feature_stats(self) -> Dict[str, dict]:
        """
        Return global feature statistics from pre-computed JSON file.
        File: outputs/feature_stats.json (generated by scripts/compute_feature_stats.py)
        """
        if self.__class__._feature_stats is not None:
            return self.__class__._feature_stats

        import json
        stats_path = _PROJECT_ROOT / "outputs" / "feature_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                self.__class__._feature_stats = json.load(f)
            logger.info("Loaded feature stats from %s", stats_path)
        else:
            logger.warning("feature_stats.json not found at %s", stats_path)
            self.__class__._feature_stats = {}

        return self.__class__._feature_stats

    @staticmethod
    def get_risk_level(score: float) -> RiskLevel:
        """
        Map a risk score to a RiskLevel enum value.

        Thresholds:
            < 0.35  -> low
            0.35–0.6 -> medium
            0.6–0.8  -> high
            > 0.8   -> critical
        """
        if score < RISK_THRESHOLD_LOW:
            return RiskLevel.low
        if score < RISK_THRESHOLD_MEDIUM:
            return RiskLevel.medium
        if score < RISK_THRESHOLD_HIGH:
            return RiskLevel.high
        return RiskLevel.critical
