"""
Unit Tests for Database Manager and Audit Logger
"""

import tempfile
from pathlib import Path
import pytest
from src.database.db_manager import DatabaseManager
from src.audit.audit_logger import AuditLogger


def test_database_manager_crud():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = str(Path(tmpdir) / "test_hospital.db")
        db = DatabaseManager(db_path=db_path)

        # 1. Create Patient
        p = db.create_patient(
            patient_id="PAT-001",
            hospital_mrn="MRN-10023",
            name="Ramesh Sharma",
            age=58,
            sex="M",
            contact="+91-9876543210",
        )
        assert p.patient_id == "PAT-001"
        assert p.hospital_mrn == "MRN-10023"

        p_fetched = db.get_patient("PAT-001")
        assert p_fetched is not None
        assert p_fetched.name == "Ramesh Sharma"

        p_by_mrn = db.get_patient_by_mrn("MRN-10023")
        assert p_by_mrn is not None
        assert p_by_mrn.patient_id == "PAT-001"

        # 2. Save ECG Record
        ok = db.save_ecg_record(
            record_id="REC-9001",
            sampling_rate=360.0,
            lead_names=["II"],
            duration_sec=10.0,
            file_hash="abcd1234ef5678",
            source_format="CSV",
            patient_id="PAT-001",
            device="GE MAC 2000",
            signal_quality="GOOD",
            quality_score=0.92,
            uploaded_by="technician",
        )
        assert ok
        rec = db.get_ecg_record("REC-9001")
        assert rec is not None
        assert rec["device"] == "GE MAC 2000"
        assert rec["lead_names"] == ["II"]

        # 3. Save Analysis Result
        ok_anl = db.save_analysis_result(
            analysis_id="ANL-8001",
            record_id="REC-9001",
            model_id="ECG-RF",
            model_version="1.0.0",
            prediction="Normal",
            probabilities={"Normal": 0.96, "PVC": 0.04, "Other": 0.0},
            signal_quality="GOOD",
            quality_score=0.92,
            heart_rate_bpm=72.0,
            mean_rr_ms=833.3,
            detected_beats_count=12,
        )
        assert ok_anl
        anl = db.get_analysis_result("ANL-8001")
        assert anl is not None
        assert anl["prediction"] == "Normal"
        assert anl["probabilities"]["Normal"] == 0.96

        # 4. Save Clinician Review
        ok_rev = db.save_clinician_review(
            review_id="REV-7001",
            analysis_id="ANL-8001",
            record_id="REC-9001",
            clinician_id="USR-01",
            clinician_name="Dr. Ananya Roy",
            clinician_role="CARDIOLOGIST",
            agreement_status="CONFIRMED",
            clinician_interpretation="Normal Sinus Rhythm. No acute ischemia.",
            clinical_notes="Routine outpatient follow up.",
            registration_number="MCI-CARD-4421",
        )
        assert ok_rev
        rev = db.get_clinician_review("REV-7001")
        assert rev is not None
        assert rev["agreement_status"] == "CONFIRMED"
        assert rev["registration_number"] == "MCI-CARD-4421"

        # 5. Save Report
        ok_rep = db.save_report(
            report_id="REP-6001",
            record_id="REC-9001",
            analysis_id="ANL-8001",
            review_id="REV-7001",
            report_type="PDF",
            status="SEALED",
            file_path="/reports/ecg_report_pat001.pdf",
            report_sha256="1234567890abcdef",
        )
        assert ok_rep
        rep = db.get_report("REP-6001")
        assert rep is not None
        assert rep["status"] == "SEALED"


def test_audit_logger_cryptographic_chain():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = str(Path(tmpdir) / "audit_test.db")
        jsonl_path = str(Path(tmpdir) / "audit_test.jsonl")
        logger = AuditLogger(db_path=db_path, jsonl_path=jsonl_path)

        # Log sequential events
        e1 = logger.log_event(
            event_type="AUTH_LOGIN",
            user_id="U01",
            username="doctor",
            user_role="DOCTOR",
            action="Doctor logged in",
        )
        assert e1.sequence_id == 1
        assert e1.previous_hash == "0" * 64
        assert len(e1.entry_hash) == 64

        e2 = logger.log_event(
            event_type="ECG_UPLOAD",
            user_id="U01",
            username="doctor",
            user_role="DOCTOR",
            action="Uploaded ECG",
            record_id="REC-01",
        )
        assert e2.sequence_id == 2
        assert e2.previous_hash == e1.entry_hash

        e3 = logger.log_event(
            event_type="CLINICIAN_REVIEW",
            user_id="U01",
            username="doctor",
            user_role="DOCTOR",
            action="Confirmed diagnosis",
            record_id="REC-01",
        )
        assert e3.sequence_id == 3
        assert e3.previous_hash == e2.entry_hash

        # Verify chain integrity
        valid, issues = logger.verify_chain_integrity()
        assert valid
        assert len(issues) == 0

        # Tamper simulation: directly alter entry_hash in database
        with logger._db() as conn:
            conn.execute("UPDATE audit_trail SET action = 'Tampered Action' WHERE sequence_id = 2")
            conn.commit()

        # Integrity verification must catch the tamper
        valid_after_tamper, issues_after = logger.verify_chain_integrity()
        assert not valid_after_tamper
        assert len(issues_after) > 0
        assert any("tamper" in i.lower() or "broken" in i.lower() for i in issues_after)
