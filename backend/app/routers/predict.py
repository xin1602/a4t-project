"""
Predict router — POST /api/predict/batch, GET /api/predict/download

Supports two batch prediction modes:
  - "api": auto-fetch user_ids from data/predict_label.jsonl
  - "csv": upload a CSV file with feature columns

Results are filtered to risk_score > 0.35.
A CSV download endpoint returns full results with top-3 SHAP features.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.routers import overview
from backend.app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# In-memory cache for last batch results (all results, not just high-risk)
# ---------------------------------------------------------------------------
_last_batch_results: list = []

# Risk score threshold for "high risk"
_RISK_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictResultItem(BaseModel):
    user_id: str
    risk_score: float
    risk_level: str
    is_blacklist: bool


class BatchPredictResponse(BaseModel):
    total_users: int
    high_risk_count: int
    results: List[PredictResultItem]
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_predict_label_user_ids() -> List[str]:
    """
    Read user_ids for batch prediction.
    
    Uses train_label.jsonl (which has pre-computed ensemble OOF scores)
    for the MVP demo. In production, this would use predict_label.jsonl
    with a real-time inference pipeline.
    """
    from config import get_data_dir  # type: ignore

    data_dir = get_data_dir()
    
    # Use train_label for MVP demo (these users have OOF scores)
    path = os.path.join(data_dir, "train_label.jsonl")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"train_label.jsonl not found at: {path}")

    user_ids: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                user_ids.append(str(row["user_id"]))
    return user_ids


def _generate_csv_bytes(results: list) -> bytes:
    """
    Generate CSV bytes from a list of PredictionResult objects.
    Columns: user_id, risk_score, is_blacklist,
             top_feature_1, top_feature_1_shap,
             top_feature_2, top_feature_2_shap,
             top_feature_3, top_feature_3_shap
    """
    output = io.StringIO()
    fieldnames = [
        "user_id", "risk_score", "is_blacklist",
        "top_feature_1", "top_feature_1_shap",
        "top_feature_2", "top_feature_2_shap",
        "top_feature_3", "top_feature_3_shap",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for r in results:
        top3 = r.top_features[:3]
        row: dict = {
            "user_id": r.user_id,
            "risk_score": round(r.risk_score, 6),
            "is_blacklist": r.is_blacklist,
        }
        for i in range(3):
            feat_key = f"top_feature_{i + 1}"
            shap_key = f"top_feature_{i + 1}_shap"
            if i < len(top3):
                feat_name = top3[i]
                row[feat_key] = feat_name
                row[shap_key] = round(r.shap_values.get(feat_name, 0.0), 6)
            else:
                row[feat_key] = ""
                row[shap_key] = ""
        writer.writerow(row)

    return output.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def batch_predict(
    source: str = Form("api"),
    file: Optional[UploadFile] = File(None),
) -> BatchPredictResponse:
    """
    Run batch prediction.

    - source="api": read user_ids from predict_label.jsonl
    - source="csv": upload a CSV file; columns must be a superset of feature_names

    Returns high-risk users (risk_score > 0.35) in the results list.
    Updates overview batch state after completion.

    Raises:
        HTTP 422 if source="csv" and CSV is missing required feature columns.
        HTTP 500 on model inference errors.
    """
    global _last_batch_results

    model_service = ModelService.get_instance()

    if source == "api":
        try:
            user_ids = _read_predict_label_user_ids()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    elif source == "csv":
        if file is None:
            raise HTTPException(
                status_code=422,
                detail="source='csv' requires a file upload",
            )

        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        reader = csv.DictReader(io.StringIO(text))
        csv_fields = set(reader.fieldnames or [])

        # Validate that all required feature columns are present
        required_features = set(model_service.feature_names)
        missing = required_features - csv_fields
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "CSV is missing required feature columns",
                    "missing_fields": sorted(missing),
                },
            )

        # Extract user_ids from CSV
        rows = list(reader)
        if "user_id" not in csv_fields:
            raise HTTPException(
                status_code=422,
                detail="CSV must contain a 'user_id' column",
            )
        user_ids = [str(row["user_id"]) for row in rows if row.get("user_id")]

    else:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid source '{source}'. Must be 'api' or 'csv'.",
        )

    # Run batch prediction
    try:
        all_results = model_service.predict_batch(user_ids)
    except Exception as exc:  # noqa: BLE001
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    # Cache all results for CSV download
    _last_batch_results = all_results

    # Notify overview module
    overview.set_batch_results(all_results)

    # Filter to high-risk only for response
    high_risk = [r for r in all_results if r.risk_score > _RISK_THRESHOLD]

    return BatchPredictResponse(
        total_users=len(all_results),
        high_risk_count=len(high_risk),
        results=[
            PredictResultItem(
                user_id=r.user_id,
                risk_score=r.risk_score,
                risk_level=r.risk_level,
                is_blacklist=r.is_blacklist,
            )
            for r in high_risk
        ],
        status="completed",
    )


@router.get("/predict/download")
def download_batch_csv() -> StreamingResponse:
    """
    Download the last batch prediction results as a CSV file.

    Columns: user_id, risk_score, is_blacklist,
             top_feature_1, top_feature_1_shap,
             top_feature_2, top_feature_2_shap,
             top_feature_3, top_feature_3_shap

    Raises:
        HTTP 404 if no batch prediction has been run yet.
    """
    if not _last_batch_results:
        raise HTTPException(
            status_code=404,
            detail="No batch prediction results available. Run POST /api/predict/batch first.",
        )

    csv_bytes = _generate_csv_bytes(_last_batch_results)

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predict_results.csv"},
    )
