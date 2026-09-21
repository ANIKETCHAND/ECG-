"""
Clinician Review & Sign-Off Management Subsystem
================================================
Enforces physician review workflows, mandatory medical registration credentials,
and cryptographic digital signature sealing for medical-device traceability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid

from src.database.db_manager import DB_MANAGER, DatabaseManager


class ReviewAction(str, Enum):
    ACCEPTED = "ACCEPTED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class RegistrationValidationError(ValueError):
    """Raised when clinician medical registration number is missing or invalid."""
    pass


@dataclass
class ClinicianSignoff:
    review_id: str
    analysis_id: str
    record_id: str
    clinician_id: str
    clinician_name: str
    clinician_role: str
    registration_number: str
    action: ReviewAction
    final_diagnosis: str
    clinical_notes: str
    reviewed_at: str
    digital_signature_hash: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


def generate_digital_signature(
    analysis_id: str,
    clinician_id: str,
    registration_number: str,
    action: str,
    final_diagnosis: str,
    timestamp: str,
) -> str:
    """Generate tamper-evident SHA-256 digital signature of physician sign-off."""
    payload = f"{analysis_id}|{clinician_id}|{registration_number}|{action}|{final_diagnosis}|{timestamp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ClinicianReviewManager:
    """Manages the lifecycle of physician reviews and diagnostic endorsements."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or DB_MANAGER

    def submit_review(
        self,
        analysis_id: str,
        record_id: str,
        clinician_id: str,
        clinician_name: str,
        clinician_role: str,
        registration_number: str,
        action: ReviewAction,
        final_diagnosis: str,
        clinical_notes: str = "",
    ) -> ClinicianSignoff:
        """Submit a signed clinical review.

        Mandates a valid medical registration number (e.g. State Medical Council or National Medical Commission).

        Raises:
            RegistrationValidationError: If registration_number is blank or missing.
        """
        reg_clean = str(registration_number).strip()
        if not reg_clean or len(reg_clean) < 3:
            raise RegistrationValidationError(
                "Mandatory medical registration number missing or invalid. "
                "Physician sign-off requires verified medical registration credentials."
            )

        if not clinician_name.strip():
            raise ValueError("Clinician name cannot be blank.")

        if not final_diagnosis.strip():
            raise ValueError("Final diagnostic conclusion cannot be blank.")

        review_id = f"REV-{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        sig_hash = generate_digital_signature(
            analysis_id=analysis_id,
            clinician_id=clinician_id,
            registration_number=reg_clean,
            action=action.value,
            final_diagnosis=final_diagnosis,
            timestamp=timestamp,
        )

        signoff = ClinicianSignoff(
            review_id=review_id,
            analysis_id=analysis_id,
            record_id=record_id,
            clinician_id=clinician_id,
            clinician_name=clinician_name.strip(),
            clinician_role=clinician_role.strip(),
            registration_number=reg_clean,
            action=action,
            final_diagnosis=final_diagnosis.strip(),
            clinical_notes=clinical_notes.strip(),
            reviewed_at=timestamp,
            digital_signature_hash=sig_hash,
        )

        # Persist to database
        self.db.save_clinician_review(
            review_id=review_id,
            analysis_id=analysis_id,
            record_id=record_id,
            clinician_id=clinician_id,
            clinician_name=clinician_name.strip(),
            clinician_role=clinician_role.strip(),
            agreement_status=action.value,
            clinician_interpretation=final_diagnosis.strip(),
            clinical_notes=clinical_notes.strip(),
            registration_number=reg_clean,
        )

        return signoff

    @staticmethod
    def verify_signoff_integrity(signoff: ClinicianSignoff) -> bool:
        """Verify the cryptographic digital signature of a review."""
        expected = generate_digital_signature(
            analysis_id=signoff.analysis_id,
            clinician_id=signoff.clinician_id,
            registration_number=signoff.registration_number,
            action=signoff.action.value,
            final_diagnosis=signoff.final_diagnosis,
            timestamp=signoff.reviewed_at,
        )
        return expected == signoff.digital_signature_hash
