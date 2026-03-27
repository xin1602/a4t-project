"""
Cases router — GET /api/cases, PATCH /api/cases/{case_id}

Provides case listing and status update endpoints with state machine validation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.services.case_service import CaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CaseRecordResponse(BaseModel):
    case_id: str
    user_id: str
    status: str
    risk_score: float
    created_at: str
    updated_at: str
    updated_by: str


class CaseListResponse(BaseModel):
    cases: List[CaseRecordResponse]


class UpdateCaseRequest(BaseModel):
    status: str
    operator: Optional[str] = "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/cases", response_model=CaseListResponse)
def get_cases() -> CaseListResponse:
    """
    Return all case records.

    Raises:
        HTTP 500 if case_status.json cannot be read.
    """
    case_service = CaseService.get_instance()
    records = case_service.get_all_cases()
    return CaseListResponse(
        cases=[
            CaseRecordResponse(
                case_id=r.case_id,
                user_id=r.user_id,
                status=r.status,
                risk_score=r.risk_score,
                created_at=r.created_at,
                updated_at=r.updated_at,
                updated_by=r.updated_by,
            )
            for r in records
        ]
    )


@router.patch("/cases/{case_id}", response_model=CaseRecordResponse)
def update_case(case_id: str, body: UpdateCaseRequest) -> CaseRecordResponse:
    """
    Update a case's status.

    Raises:
        HTTP 404 if case_id not found.
        HTTP 422 if the status transition is invalid.
        HTTP 500 if case_status.json write fails.
    """
    operator = body.operator or "unknown"
    case_service = CaseService.get_instance()
    updated = case_service.update_case(case_id, body.status, operator)
    return CaseRecordResponse(
        case_id=updated.case_id,
        user_id=updated.user_id,
        status=updated.status,
        risk_score=updated.risk_score,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
        updated_by=updated.updated_by,
    )
