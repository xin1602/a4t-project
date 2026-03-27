"""
CaseService — CRUD for case_status.json and append-only audit_log.json.

Manages case lifecycle with state machine validation and immutable audit trail.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from backend.app.core.config import (
    AUDIT_LOG_FILE,
    CASE_STATUS_FILE,
)
from backend.app.models.schemas import (
    AuditEntry,
    CaseRecord,
    VALID_TRANSITIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CaseService
# ---------------------------------------------------------------------------


class CaseService:
    """
    Manages case_status.json (CRUD) and audit_log.json (append-only).

    File formats:
      case_status.json  -> {"cases": {case_id: CaseRecord_dict, ...}}
      audit_log.json    -> {"entries": [AuditEntry_dict, ...]}
    """

    _instance: Optional["CaseService"] = None

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "CaseService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._case_file = Path(CASE_STATUS_FILE)
        self._audit_file = Path(AUDIT_LOG_FILE)
        self._ensure_files()

    def _ensure_files(self) -> None:
        """Create empty JSON files if they do not exist."""
        self._case_file.parent.mkdir(parents=True, exist_ok=True)

        if not self._case_file.exists():
            self._write_json(self._case_file, {"cases": {}})
            logger.info("Created empty case_status.json at %s", self._case_file)

        if not self._audit_file.exists():
            self._write_json(self._audit_file, {"entries": []})
            logger.info("Created empty audit_log.json at %s", self._audit_file)

    # ------------------------------------------------------------------
    # Low-level JSON helpers
    # ------------------------------------------------------------------

    def _read_json(self, path: Path) -> dict:
        """Read and parse a JSON file."""
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read %s: %s", path, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read data file: {path.name}",
            ) from exc

    def _write_json(self, path: Path, data: dict) -> None:
        """Write data to a JSON file atomically."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error("Failed to write %s: %s", path, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write data file: {path.name}",
            ) from exc

    # ------------------------------------------------------------------
    # Case CRUD
    # ------------------------------------------------------------------

    def get_all_cases(self) -> List[CaseRecord]:
        """Return all case records as a list."""
        data = self._read_json(self._case_file)
        cases = data.get("cases", {})
        return [CaseRecord(**record) for record in cases.values()]

    def get_case(self, case_id: str) -> CaseRecord:
        """
        Return a single case record by case_id.

        Raises:
            HTTPException 404 if case_id not found.
        """
        data = self._read_json(self._case_file)
        cases = data.get("cases", {})
        if case_id not in cases:
            raise HTTPException(
                status_code=404,
                detail=f"Case '{case_id}' not found",
            )
        return CaseRecord(**cases[case_id])

    def create_case(self, user_id: str, risk_score: float) -> CaseRecord:
        """
        Create a new case for the given user_id.

        case_id format: CASE-{user_id}
        Initial status: pending

        Raises:
            HTTPException 500 if write fails.
        """
        case_id = f"CASE-{user_id}"
        now = datetime.utcnow().isoformat() + "Z"

        record = CaseRecord(
            case_id=case_id,
            user_id=user_id,
            status="pending",
            risk_score=risk_score,
            created_at=now,
            updated_at=now,
            updated_by="system",
        )

        data = self._read_json(self._case_file)
        cases = data.get("cases", {})
        cases[case_id] = self._record_to_dict(record)
        data["cases"] = cases
        self._write_json(self._case_file, data)

        logger.info("Created case %s for user %s", case_id, user_id)
        return record

    # ------------------------------------------------------------------
    # State machine update
    # ------------------------------------------------------------------

    def update_case(
        self,
        case_id: str,
        new_status: str,
        operator: str,
    ) -> CaseRecord:
        """
        Update a case's status with state machine validation.

        Validates the (old_status, new_status) transition against
        VALID_TRANSITIONS. Appends an AuditEntry to audit_log.json.

        Args:
            case_id:    The case to update.
            new_status: Target status string.
            operator:   Identifier of the person performing the update.

        Returns:
            Updated CaseRecord.

        Raises:
            HTTPException 404 if case not found.
            HTTPException 422 if transition is invalid.
            HTTPException 500 if write fails.
        """
        # Load current record (raises 404 if missing)
        case = self.get_case(case_id)
        old_status = case.status

        # Validate transition
        if (old_status, new_status) not in VALID_TRANSITIONS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid status transition: '{old_status}' -> '{new_status}'. "
                    f"Allowed transitions from '{old_status}': "
                    + str([t[1] for t in VALID_TRANSITIONS if t[0] == old_status])
                ),
            )

        now = datetime.utcnow().isoformat() + "Z"

        # Update case record
        case.status = new_status
        case.updated_at = now
        case.updated_by = operator

        data = self._read_json(self._case_file)
        cases = data.get("cases", {})
        cases[case_id] = self._record_to_dict(case)
        data["cases"] = cases
        self._write_json(self._case_file, data)

        # Append audit entry
        self._append_audit_entry(
            case_id=case_id,
            user_id=case.user_id,
            operator=operator,
            timestamp_utc=now,
            old_status=old_status,
            new_status=new_status,
        )

        logger.info(
            "Case %s updated: %s -> %s by %s",
            case_id,
            old_status,
            new_status,
            operator,
        )
        return case

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _append_audit_entry(
        self,
        case_id: str,
        user_id: str,
        operator: str,
        timestamp_utc: str,
        old_status: str,
        new_status: str,
    ) -> AuditEntry:
        """
        Append an immutable audit entry to audit_log.json.

        entry_id format: AUD-{first 8 chars of uuid4}
        """
        entry_id = "AUD-" + str(uuid.uuid4()).replace("-", "")[:8]

        entry = AuditEntry(
            entry_id=entry_id,
            case_id=case_id,
            user_id=user_id,
            operator=operator,
            timestamp_utc=timestamp_utc,
            old_status=old_status,
            new_status=new_status,
        )

        data = self._read_json(self._audit_file)
        entries = data.get("entries", [])
        entries.append(self._entry_to_dict(entry))
        data["entries"] = entries
        self._write_json(self._audit_file, data)

        return entry

    def get_audit_log(self, case_id: Optional[str] = None) -> List[AuditEntry]:
        """
        Return audit entries, optionally filtered by case_id.

        Args:
            case_id: If provided, return only entries for this case.
        """
        data = self._read_json(self._audit_file)
        entries = data.get("entries", [])
        result = [AuditEntry(**e) for e in entries]
        if case_id is not None:
            result = [e for e in result if e.case_id == case_id]
        return result

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_dict(record: CaseRecord) -> dict:
        return {
            "case_id": record.case_id,
            "user_id": record.user_id,
            "status": record.status,
            "risk_score": record.risk_score,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "updated_by": record.updated_by,
        }

    @staticmethod
    def _entry_to_dict(entry: AuditEntry) -> dict:
        return {
            "entry_id": entry.entry_id,
            "case_id": entry.case_id,
            "user_id": entry.user_id,
            "operator": entry.operator,
            "timestamp_utc": entry.timestamp_utc,
            "old_status": entry.old_status,
            "new_status": entry.new_status,
        }
