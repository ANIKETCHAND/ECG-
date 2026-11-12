"""
Test Negative Authorization, Zero-Trust Access Control, and Clinical Safety Safeguards.

Addresses:
- SEC-01: No anonymous access to patient data or authenticated identity defaults.
- SEC-02: Strict bearer/session authorization enforcement on clinical endpoints.
- SEC-03: Role-based access control preventing patients from signing/sealing clinical reviews.
- CLIN-01: No silent normal defaults for serum potassium in medication safety evaluation.
- CLIN-02: No silent normal defaults for renal biomarkers (creatinine / eGFR).
- BIO-01: Optical ECG waveform extraction marks uncalibrated signals and disables ST evaluation.
- ML-01: Production model registry enforces non-overlapping patient-level split.
"""

import os
import pytest
from starlette.testclient import TestClient
from api.index import app
from src.clinical.medication_candidate_engine import (
    MultimodalMedicationCandidateEngine,
)
from src.ecg_input.waveform_extractor import extract_waveform_from_image
from pathlib import Path
import json
import numpy as np


@pytest.fixture
def client():
    return TestClient(app)


class TestNegativeAuthorization:
    """Zero-trust authentication and authorization tests."""

    def test_anonymous_auth_me_returns_401(self, client):
        """SEC-01: Unauthenticated request to /api/auth/me must return 401, not dr_aniket."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401
        data = response.json()
        assert "authentication" in data["detail"].lower()

    def test_default_auth_blocks_unauthenticated_patients_endpoint(self, client):
        """SEC-02: Zero trust by default - unauthenticated GET /api/patients is strictly blocked."""
        response = client.get("/api/patients")
        assert response.status_code == 401
        assert "authentication required" in response.json()["detail"].lower()

    def test_default_auth_blocks_unauthenticated_patient_creation(self, client):
        """SEC-02: Unauthenticated POST /api/patients is strictly rejected with 401."""
        response = client.post("/api/patients", json={"name": "Anonymous Intruder"})
        assert response.status_code == 401

    def test_patient_role_cannot_create_patient_dossier(self, client):
        """SEC-02: Patient accounts are forbidden from registering patient dossiers (403)."""
        from api.security import create_jwt_token
        from src.auth.auth_manager import User, UserRole

        patient_user = User(
            user_id="pat_99881",
            username="eve_patient",
            password_hash="",
            salt="",
            role=UserRole.PATIENT,
            full_name="Eve Patient",
            email="eve@example.com",
            hospital_id="HOSP-APEX",
        )
        token = create_jwt_token(patient_user)
        response = client.post(
            "/api/patients",
            json={"name": "Injected Patient Record"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "patient accounts cannot register" in response.json()["detail"].lower()

    def test_default_auth_blocks_unauthenticated_reports_endpoint(self, client):
        """SEC-02: Zero trust by default - unauthenticated GET /api/reports is blocked."""
        response = client.get("/api/reports")
        assert response.status_code == 401

    def test_default_auth_blocks_unauthenticated_conversations_endpoint(self, client):
        """SEC-02: Unauthenticated clinical messaging GET /api/conversations is blocked."""
        response = client.get("/api/conversations")
        assert response.status_code == 401

    def test_default_auth_blocks_unauthenticated_appointments_endpoint(self, client):
        """SEC-02: Unauthenticated clinical appointments GET /api/appointments is blocked."""
        response = client.get("/api/appointments")
        assert response.status_code == 401

    def test_patient_role_cannot_seal_clinical_review(self, client):
        """SEC-03: Patient accounts cannot seal clinical diagnostic reports (403 Forbidden)."""
        from api.security import create_jwt_token
        from src.auth.auth_manager import User, UserRole

        patient_user = User(
            user_id="pat_12345",
            username="john_patient",
            password_hash="",
            salt="",
            role=UserRole.PATIENT,
            full_name="John Patient",
            email="patient@example.com",
            hospital_id="HOSP-APEX",
        )
        patient_token = create_jwt_token(patient_user)

        review_payload = {
            "report_id": "rep_test_001",
            "clinician_name": "John Patient",
            "clinician_role": "PATIENT",
            "agreement_status": "CONFIRMED",
            "clinical_notes": "Patient attempting to approve own diagnosis.",
        }

        response = client.post(
            "/api/review",
            json=review_payload,
            headers={"Authorization": f"Bearer {patient_token}"},
        )
        assert response.status_code == 403
        assert "only licensed clinicians" in response.json()["detail"].lower()

    def test_unauthenticated_review_rejected_with_401(self, client):
        """SEC-03: Anonymous POST /api/review is rejected with 401."""
        response = client.post(
            "/api/review",
            json={"analysis_id": "ANL-999", "clinician_name": "Anonymous Attacker"}
        )
        assert response.status_code == 401

    def test_invalid_bearer_token_rejected(self, client):
        """SEC-04: Tampered or invalid JWT/bearer token must be rejected with 401."""
        response = client.get(
            "/api/patients",
            headers={"Authorization": "Bearer totally_forged_invalid_token_12345"}
        )
        assert response.status_code == 401

    def test_unauthenticated_dashboard_stats_returns_401(self, client):
        """SEC-02: Unauthenticated GET /api/dashboard/stats is strictly blocked with 401."""
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 401
        assert "authentication required" in response.json()["detail"].lower()

    def test_patient_role_cannot_access_hospital_dashboard_stats(self, client):
        """SEC-03: Patient accounts are forbidden from accessing hospital telemetry dashboard (403)."""
        from api.security import create_jwt_token
        from src.auth.auth_manager import User, UserRole

        patient_user = User(
            user_id="pat_dash_test",
            username="eve_pat",
            password_hash="",
            salt="",
            role=UserRole.PATIENT,
            full_name="Eve Patient",
            email="eve@hospital.org",
            hospital_id="HOSP-APEX",
        )
        token = create_jwt_token(patient_user)
        response = client.get(
            "/api/dashboard/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "access denied" in response.json()["detail"].lower()

    def test_unauthenticated_notifications_returns_401(self, client):
        """SEC-02: Unauthenticated GET /api/notifications is strictly blocked with 401."""
        response = client.get("/api/notifications")
        assert response.status_code == 401

    def test_unauthenticated_doctor_profile_update_returns_401(self, client):
        """SEC-02: Unauthenticated PUT /api/doctors/{id} is strictly blocked with 401."""
        response = client.put("/api/doctors/dr.smith", json={"full_name": "Tampered Name"})
        assert response.status_code == 401

    def test_patient_role_cannot_modify_doctor_credentials(self, client):
        """SEC-03: Patient accounts are forbidden from altering clinician credentials (403)."""
        from api.security import create_jwt_token
        from src.auth.auth_manager import User, UserRole

        patient_user = User(
            user_id="pat_rogue",
            username="rogue_pat",
            password_hash="",
            salt="",
            role=UserRole.PATIENT,
            full_name="Rogue Patient",
            email="rogue@example.com",
            hospital_id="HOSP-APEX",
        )
        token = create_jwt_token(patient_user)
        response = client.put(
            "/api/doctors/dr.smith",
            json={"registration_number": "MCI-FAKE-9999"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "access denied" in response.json()["detail"].lower()

    def test_unauthenticated_appointment_cancellation_returns_401(self, client):
        """SEC-02: Unauthenticated DELETE /api/appointments/{id} is blocked with 401."""
        response = client.delete("/api/appointments/APT-9999")
        assert response.status_code == 401

    def test_unauthenticated_ecg_compare_returns_401(self, client):
        """SEC-02: Unauthenticated POST /api/ecg/compare is blocked with 401."""
        response = client.post("/api/ecg/compare", json={"current_analysis": {}, "prior_analysis": {}})
        assert response.status_code == 401


class TestClinicalSafetySafeguards:
    """Clinical safety engine prevents silent substitutions of missing lab data."""

    def test_missing_potassium_triggers_mandatory_withholding(self):
        """CLIN-01: Engine must NOT substitute 4.2 mEq/L when potassium is None or missing."""
        engine = MultimodalMedicationCandidateEngine()
        patient_profile = {
            "age": 65,
            "sex": "M",
            "conditions": ["Atrial Fibrillation", "Hypertension"],
            "allergies": [],
            "current_medications": []
        }
        ecg_measurements = {
            "heart_rate": 110,
            "qt_interval": 410,
            "qtc_interval": 430,
            "has_long_qt": False,
        }
        # Labs with potassium explicitly None or missing
        laboratory_results = {
            "potassium": None,
            "creatinine": 1.0,
            "egfr": 75.0,
            "magnesium": 2.0
        }

        result = engine.generate_candidate_recommendations(
            ecg_finding="Ventricular Ectopy",
            ecg_measurements=ecg_measurements,
            patient_profile=patient_profile,
            vital_signs={"systolic_bp": 130, "diastolic_bp": 80},
            laboratory_results=laboratory_results
        )
        
        # Verify candidate contraindications or lab checks contain mandatory withholding note
        withholding_noted = False
        all_reasons = []
        for cand in result.candidates:
            all_reasons.extend(cand.contraindications_detected)
            all_reasons.extend(cand.relevant_laboratory_checks)
            for reason in cand.contraindications_detected + cand.relevant_laboratory_checks:
                if "MANDATORY LAB WITHHOLDING: Serum potassium" in reason:
                    withholding_noted = True
                    break
        assert withholding_noted, f"Expected potassium withholding warning. Reasons were: {all_reasons}"

    def test_missing_creatinine_triggers_mandatory_withholding(self):
        """CLIN-02: Engine must NOT substitute 0.95 mg/dL when creatinine is None or missing."""
        engine = MultimodalMedicationCandidateEngine()
        patient_profile = {
            "age": 72,
            "sex": "F",
            "conditions": ["Heart Failure", "Hypertension"],
            "allergies": [],
            "current_medications": []
        }
        ecg_measurements = {
            "heart_rate": 75,
            "qt_interval": 400,
            "qtc_interval": 420,
            "has_long_qt": False,
        }
        # Labs with creatinine explicitly None or missing
        laboratory_results = {
            "potassium": 4.2,
            "creatinine": None,
            "egfr": None,
            "magnesium": 2.0
        }

        result = engine.generate_candidate_recommendations(
            ecg_finding="Normal Sinus Rhythm",
            ecg_measurements=ecg_measurements,
            patient_profile=patient_profile,
            vital_signs={"systolic_bp": 125, "diastolic_bp": 75},
            laboratory_results=laboratory_results
        )

        renal_warning_found = False
        all_reasons = []
        for cand in result.candidates:
            all_reasons.extend(cand.contraindications_detected)
            all_reasons.extend(cand.relevant_laboratory_checks)
            for reason in cand.contraindications_detected + cand.relevant_laboratory_checks:
                if "Renal function" in reason or "Renal Panel: NOT RECORDED" in reason:
                    renal_warning_found = True
                    break
        assert renal_warning_found, f"Expected renal withholding warning. Reasons were: {all_reasons}"

    def test_hypokalemia_contraindication_blocks_torsadogenic_candidates(self):
        """CLIN-03: Hypokalemia (<3.5 mEq/L) blocks torsadogenic / antiarrhythmic candidate agents."""
        engine = MultimodalMedicationCandidateEngine()
        patient_profile = {
            "age": 60,
            "sex": "M",
            "conditions": ["Ventricular Ectopy", "Hypertension"],
            "allergies": [],
            "current_medications": []
        }
        ecg_measurements = {
            "heart_rate": 80,
            "qt_interval": 400,
            "qtc_interval": 420,
            "has_long_qt": False,
        }
        # Serum potassium below safety threshold (3.1 < 3.5 mEq/L)
        laboratory_results = {
            "potassium": 3.1,
            "creatinine": 1.0,
            "egfr": 80.0,
            "magnesium": 2.0
        }

        result = engine.generate_candidate_recommendations(
            ecg_finding="Ventricular Ectopy",
            ecg_measurements=ecg_measurements,
            patient_profile=patient_profile,
            vital_signs={"systolic_bp": 130, "diastolic_bp": 80},
            laboratory_results=laboratory_results
        )

        hypokalemia_blocked = False
        all_reasons = []
        for cand in result.candidates:
            all_reasons.extend(cand.contraindications_detected)
            for reason in cand.contraindications_detected:
                if "Hypokalemia" in reason and "Contraindicated" in reason:
                    hypokalemia_blocked = True
                    break
        assert hypokalemia_blocked, f"Expected hypokalemia contraindication block. Detected reasons: {all_reasons}"


class TestBiomedicalSignalSafety:
    """Biomedical waveform extraction safeguards."""

    def test_uncalibrated_optical_waveform_marks_safety_flags(self):
        """BIO-01: Optical extractor disables ST segment evaluation and does not rescale to 1.5 mV."""
        # Synthetic white canvas with a dark horizontal/sinusoidal trace
        h, w = 300, 1000
        synthetic_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Draw a synthetic line
        for x in range(w):
            y = int(150 + 40 * np.sin(x / 30.0))
            synthetic_img[max(0, min(h - 1, y)), x] = [0, 0, 0]

        result = extract_waveform_from_image(synthetic_img)
        assert result["is_calibrated"] is False
        assert result["can_evaluate_st"] is False
        assert result["calibration_status"] == "UNCALIBRATED_OPTICAL_EXTRACT"
        assert "withheld" in result["clinical_warning"].lower() or "uncalibrated" in result["clinical_warning"].lower()

    def test_dynamic_q_onset_computes_physiologic_qt(self):
        """BIO-02: Dynamic morphology-aware Q-onset detection accurately measures physiologic QT/QTc."""
        from src.measurements.measurement_engine import compute_ecg_measurements

        fs = 360.0
        # Generate 10 beats at 75 BPM (RR = 0.8s = 288 samples)
        # Normal physiologic QT is ~360-400 ms (~130-144 samples)
        peaks = np.array([288 * i for i in range(1, 10)])
        sig_len = peaks[-1] + 360
        sig = np.zeros(sig_len)
        for p in peaks:
            # Q-wave onset at p - 14 (about 39ms before R peak), trough at p - 7
            sig[p - 14 : p - 7] = np.linspace(0.0, -0.25, 7)
            sig[p - 7 : p] = np.linspace(-0.25, 0.0, 7)
            # R peak
            sig[p] = 1.2
            # S wave
            sig[p : p + 14] = np.linspace(1.2, -0.3, 14)
            sig[p + 14 : p + 28] = np.linspace(-0.3, 0.0, 14)
            # T wave: peak at p + 75 (~208ms), offset at p + 130 (~361ms)
            t_len = 55
            sig[p + 75 : p + 75 + t_len] = 0.35 * np.sin(np.linspace(0, np.pi, t_len))

        res = compute_ecg_measurements(sig, fs, peaks, leads_available=["II"])
        assert res.heart_rate_bpm == 75.0
        assert res.qt_interval_ms is not None
        # Physiologic QT for 75 BPM should be between 300 ms and 450 ms
        assert 300.0 <= res.qt_interval_ms <= 450.0
        assert res.qtc_bazett_ms is not None
        assert "dynamic Q-onset" in res.qt_status


class TestModelGovernanceSplits:
    """AI Model Governance and Dataset Leakage Validation."""

    def test_production_model_has_strict_record_split(self):
        """ML-01: Catalog must prove zero patient leakage between train and test sets."""
        catalog_path = Path("models/registry/catalog.json")
        assert catalog_path.exists()
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        
        prod_entry = catalog.get("ECG-RF-1.0.0")
        assert prod_entry is not None
        prov = prod_entry["provenance"]
        assert prov["data_status"] == "REAL_DATASET"
        assert prov["patient_level_split_enforced"] is True
        assert prov["may_be_promoted_to_production"] is True

        train_recs = set(prov["train_records"])
        test_recs = set(prov["test_records"])
        overlap = train_recs.intersection(test_recs)
        assert len(overlap) == 0, f"Found overlapping patients between train and test: {overlap}"
