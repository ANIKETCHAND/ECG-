"""
Unit Tests for Vercel Serverless Clinical API (api/index.py)
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "CDSCO" in data["regulatory_classification"]
    assert data["safety_principle"] == "NO RELIABLE INPUT = NO AI RESULT"


def test_api_sample_normal():
    response = client.get("/api/sample?sample_type=normal")
    assert response.status_code == 200
    data = response.json()
    assert data["sample_type"] == "normal"
    assert len(data["signal"]) > 500
    assert data["fs"] == 360.0


def test_api_analyze_clean_signal():
    sample_resp = client.get("/api/sample?sample_type=normal")
    signal = sample_resp.json()["signal"]

    payload = {"signal": signal, "fs": 360.0, "lead": "II"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "cardiac_parameters" in data
    assert "heart_rate_bpm" in data["cardiac_parameters"]
    assert "ai_classification" in data
    assert len(data["detected_r_peaks"]) > 0


def test_api_analyze_unusable_signal_safety_gate():
    # Flatline signal should trigger the safety gate and reject AI execution
    flatline = [0.0] * 500
    payload = {"signal": flatline, "fs": 360.0, "lead": "II"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UNUSABLE_SIGNAL"
    assert data["can_run_ai"] is False
    assert data["prediction"] == "NO_RESULT_SIGNAL_UNUSABLE"


def test_api_clinician_review():
    payload = {
        "analysis_id": "ANALYSIS-TEST-001",
        "clinician_name": "Dr. Sarah Rao",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MCI-48291",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": "Sinus rhythm confirmed on bedside telemetry.",
        "clinical_notes": "Patient stable.",
    }
    response = client.post("/api/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SEALED"
    assert data["clinician_name"] == "Dr. Sarah Rao"
    assert "review_id" in data


def test_api_analyze_with_multimodal_patient_context():
    sample_resp = client.get("/api/sample?sample_type=normal")
    signal = sample_resp.json()["signal"]

    payload = {
        "signal": signal,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Arjun Mehta",
        "patient_mrn": "MRN-10928",
        "age": 62,
        "sex": "M",
        "blood_group": "O+",
        "existing_conditions": "Hypertension, Atrial Fibrillation",
        "current_medications": "Metoprolol, Amiodarone",
        "vital_signs": {"systolic_bp": 118, "diastolic_bp": 76, "heart_rate": 42},
        "laboratory_results": {"potassium": 4.1, "creatinine": 1.1},
        "include_full_report": True,
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "medication_safety" in data
    assert "clinical_decision_support" in data
    assert "full_report" in data
    # Verify Blood Group and Vitals are included in full report
    assert data["full_report"]["patient_info"]["blood_group"] == "O+"
    assert data["full_report"]["vital_signs"]["status"] == "RECORDED"

    # Criteria metadata verification (Rule 4 & 5)
    assert "criteria_metadata" in data
    cm = data["criteria_metadata"]
    assert "ecg_model_inputs" in cm
    assert "patient_context_considered" in cm
    # Blood group must NEVER be claimed as an ML model input
    assert not any("blood" in item.lower() and "group" in item.lower() for item in cm["ecg_model_inputs"])
    # Dynamic patient context considered should capture available fields
    p_ctx = " ".join(cm["patient_context_considered"])
    assert "Age" in p_ctx
    assert "Sex" in p_ctx
    assert "Blood pressure" in p_ctx


def test_api_generate_pdf_endpoint():
    sample_resp = client.get("/api/sample?sample_type=normal")
    signal = sample_resp.json()["signal"]

    payload = {
        "signal": signal,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Sunita Patil",
        "age": 58,
        "sex": "F",
        "blood_group": "B+",
    }
    response = client.post("/api/report/pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 1000
    # PDF magic header bytes
    assert response.content[:4] == b"%PDF"

