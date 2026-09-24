"""
Unit & Integration Tests for ECG Report Persistence & History Layer
===================================================================
Tests:
- Unique Report ID generation ('ECG-YYYY-XXXXXX')
- Complete snapshot JSONB persistence
- Idempotency & duplicate prevention
- History search, filtering, and listing
- Physician sign-and-seal attestation
- PDF retrieval and synthesis
- Zero ML rerun verification
"""

import json
import sys
from pathlib import Path
import pytest
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.report_persistence_service import REPORT_PERSISTENCE_SERVICE, PersistenceStatus
from database.db_manager import DB_MANAGER


@pytest.fixture
def sample_report_snapshot():
    return {
        "report_info": {
            "title": "Clinical ECG Analysis Report",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "patient_info": {
            "patient_id": "PT-TEST-HIST-01",
            "hospital_mrn": "MRN-HIST-999",
            "patient_name": "Devendra Mukherjee",
            "patient_age": 62,
            "patient_sex": "M",
            "blood_group": "O+",
        },
        "cardiac_parameters": {
            "heart_rate_bpm": 74.0,
            "heart_rate_category": "Normal Heart Rate",
            "detected_beats": 12,
            "mean_rr_ms": 810.0,
            "qrs_duration_ms": 92.0,
            "qt_interval_ms": 395.0,
            "qtc_interval_ms": 418.0,
        },
        "signal_quality": {
            "category": "GOOD",
            "quality_score": 0.98,
        },
        "ai_analysis": {
            "prediction": "Normal Sinus Rhythm",
            "primary_finding": "Normal Sinus Rhythm",
            "probabilities": {"Normal": 0.97, "PVC": 0.02, "Other": 0.01},
        },
        "clinical_decision_support": {
            "clinical_urgency": "ROUTINE",
            "clinical_considerations": "Routine outpatient monitoring.",
            "guideline_references": ["AHA/ACC/HRS Guidelines"],
        },
        "medication_safety": {
            "alert_level": "LOW",
            "active_warnings": [],
        },
    }


def test_generate_report_number():
    num = REPORT_PERSISTENCE_SERVICE.generate_report_number()
    year = str(datetime.now().year)
    assert num.startswith(f"ECG-{year}-")
    assert len(num) == 15  # ECG-2026-XXXXXX


def test_save_report_snapshot_and_retrieve(sample_report_snapshot):
    test_analysis_id = f"ANL-TEST-{datetime.now().timestamp()}"
    res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        patient_id="PT-TEST-HIST-01",
        ecg_id="ECG-TEST-HIST-01",
        analysis_id=test_analysis_id,
        status="DRAFT",
    )
    assert res["status"] == PersistenceStatus.SAVE_SUCCESS.value
    rep_id = res["report_id"]
    rep_num = res["report_number"]
    assert rep_num.startswith("ECG-")

    # Retrieve without re-running ML
    retrieved = REPORT_PERSISTENCE_SERVICE.get_report(rep_id)
    assert retrieved is not None
    r_data = retrieved.get("report_data") or {}
    assert r_data["patient_info"]["patient_name"] == "Devendra Mukherjee"
    assert r_data["cardiac_parameters"]["heart_rate_bpm"] == 74.0
    assert r_data["ai_analysis"]["prediction"] == "Normal Sinus Rhythm"


def test_idempotency_prevents_duplicate_reports(sample_report_snapshot):
    fixed_analysis_id = f"ANL-IDEMPOTENT-001"
    res1 = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        analysis_id=fixed_analysis_id,
        force_new_version=False,
    )
    res2 = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        analysis_id=fixed_analysis_id,
        force_new_version=False,
    )
    assert res1["report_id"] == res2["report_id"]
    assert res1["report_number"] == res2["report_number"]
    assert res2.get("is_idempotent_duplicate") is True


def test_list_reports_and_search(sample_report_snapshot):
    # Save identifiable report
    unique_name = f"Ananya_Test_{int(datetime.now().timestamp())}"
    sample_report_snapshot["patient_info"]["patient_name"] = unique_name
    sample_report_snapshot["patient_info"]["hospital_mrn"] = f"MRN-{unique_name[:8]}"

    res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        analysis_id=f"ANL-{unique_name}",
        force_new_version=True,
    )

    # Search by patient name
    found = REPORT_PERSISTENCE_SERVICE.list_reports(search_query=unique_name)
    assert len(found) >= 1
    assert any(unique_name in r["patient_name"] for r in found)

    # Search by report number
    found_by_num = REPORT_PERSISTENCE_SERVICE.list_reports(search_query=res["report_number"])
    assert len(found_by_num) >= 1
    assert found_by_num[0]["report_number"] == res["report_number"]


def test_sign_and_seal_report(sample_report_snapshot):
    unique_id = f"ANL-SEAL-{int(datetime.now().timestamp())}"
    save_res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        analysis_id=unique_id,
        status="DRAFT",
        force_new_version=True,
    )
    rep_id = save_res["report_id"]

    # Clinician signs and seals
    seal_res = REPORT_PERSISTENCE_SERVICE.sign_and_seal_report(
        report_id=rep_id,
        clinician_user_id="DOC-99",
        clinician_name="Dr. Anita Desai",
        clinician_role="CARDIOLOGIST",
        registration_number="MCI-77621",
        agreement_status="CONFIRMED",
        clinician_interpretation="Normal Sinus Rhythm confirmed by cardiologist.",
        clinical_notes="Discharge with standard advice.",
    )
    assert seal_res["status"] == PersistenceStatus.SAVE_SUCCESS.value

    # Verify report snapshot is updated
    updated_rep = REPORT_PERSISTENCE_SERVICE.get_report(rep_id)
    u_data = updated_rep.get("report_data") or {}
    review = u_data.get("clinician_review", {})
    assert review.get("agreement_status") == "CONFIRMED"
    assert review.get("clinician_name") == "Dr. Anita Desai"
    assert review.get("registration_number") == "MCI-77621"


def test_get_patient_timeline(sample_report_snapshot):
    mrn = f"MRN-TIMELINE-{int(datetime.now().timestamp())}"
    sample_report_snapshot["patient_info"]["hospital_mrn"] = mrn

    REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        analysis_id=f"ANL-T1-{mrn}",
        force_new_version=True,
    )
    timeline = REPORT_PERSISTENCE_SERVICE.get_patient_timeline(mrn)
    assert len(timeline) >= 1
    assert timeline[0]["hospital_mrn"] == mrn


def test_pdf_generation_from_snapshot(sample_report_snapshot):
    save_res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=sample_report_snapshot,
        analysis_id=f"ANL-PDF-{int(datetime.now().timestamp())}",
        force_new_version=True,
    )
    rep_id = save_res["report_id"]

    doc_pdf = REPORT_PERSISTENCE_SERVICE.get_report_pdf(rep_id, report_type="doctor")
    assert doc_pdf is not None
    assert doc_pdf.startswith(b"%PDF")
    assert len(doc_pdf) > 2000

    pat_pdf = REPORT_PERSISTENCE_SERVICE.get_report_pdf(rep_id, report_type="patient")
    assert pat_pdf is not None
    assert pat_pdf.startswith(b"%PDF")
    assert len(pat_pdf) > 1000
