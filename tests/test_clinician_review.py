"""
Unit Tests for Phase 10: Clinician Review Subsystem
===================================================
Validates:
1. Physician sign-off requires non-empty medical council registration number.
2. Digital signature generation and cryptographic tamper-detection.
3. Successful database persistence of clinician reviews across actions (ACCEPTED, MODIFIED, REJECTED).
"""

import tempfile
from pathlib import Path
import pytest

from src.database.db_manager import DatabaseManager
from src.review.review_manager import (
    ClinicianReviewManager,
    RegistrationValidationError,
    ReviewAction,
)


@pytest.fixture
def temp_review_manager():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = str(Path(tmpdir) / "test_reviews.db")
        db = DatabaseManager(db_path=db_path)
        # Create prerequisite patient and record
        db.create_patient("P-1", "MRN-1", "A. Sharma")
        db.save_ecg_record("R-1", 360.0, ["II"], 10.0, "hash1", "CSV", patient_id="P-1")
        db.save_analysis_result(
            analysis_id="A-1",
            record_id="R-1",
            model_id="RF-1",
            model_version="1.0.0",
            prediction="Normal Sinus Rhythm",
            probabilities={"Normal": 0.95, "PVC": 0.05},
            signal_quality="GOOD",
            quality_score=0.9,
        )
        mgr = ClinicianReviewManager(db=db)
        yield mgr, db


def test_submit_review_success(temp_review_manager):
    mgr, db = temp_review_manager
    signoff = mgr.submit_review(
        analysis_id="A-1",
        record_id="R-1",
        clinician_id="DOC-42",
        clinician_name="Dr. Sunita Rao",
        clinician_role="Cardiologist",
        registration_number="MCI-12345-2015",
        action=ReviewAction.ACCEPTED,
        final_diagnosis="Normal Sinus Rhythm, narrow QRS, no ectopy",
        clinical_notes="Patient asymptomatic during recording.",
    )

    assert signoff.review_id.startswith("REV-")
    assert signoff.action == ReviewAction.ACCEPTED
    assert signoff.registration_number == "MCI-12345-2015"
    assert len(signoff.digital_signature_hash) == 64
    assert mgr.verify_signoff_integrity(signoff) is True

    # Check persistence in db
    row = db.get_clinician_review(signoff.review_id)
    assert row is not None
    assert row["clinician_name"] == "Dr. Sunita Rao"
    assert row["registration_number"] == "MCI-12345-2015"


def test_submit_review_rejects_missing_registration_number(temp_review_manager):
    mgr, _ = temp_review_manager
    with pytest.raises(RegistrationValidationError, match="registration number missing"):
        mgr.submit_review(
            analysis_id="A-1",
            record_id="R-1",
            clinician_id="DOC-42",
            clinician_name="Dr. Sunita Rao",
            clinician_role="Cardiologist",
            registration_number="  ",  # Blank!
            action=ReviewAction.ACCEPTED,
            final_diagnosis="Normal Sinus Rhythm",
        )


def test_digital_signature_tamper_detection(temp_review_manager):
    mgr, _ = temp_review_manager
    signoff = mgr.submit_review(
        analysis_id="A-1",
        record_id="R-1",
        clinician_id="DOC-42",
        clinician_name="Dr. Sunita Rao",
        clinician_role="Cardiologist",
        registration_number="MCI-9999",
        action=ReviewAction.MODIFIED,
        final_diagnosis="Sinus Bradycardia with Artifact",
    )
    assert mgr.verify_signoff_integrity(signoff) is True

    # Tamper with diagnosis
    tampered = signoff
    tampered.final_diagnosis = "Completely Different Diagnosis"
    assert mgr.verify_signoff_integrity(tampered) is False
