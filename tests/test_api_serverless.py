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
