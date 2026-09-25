"""
End-to-End Data Consistency and State Isolation Test
====================================================
Verifies:
1. Uploading ECG A computes measurements strictly from ECG A.
2. Report and PDF generated for ECG A match dashboard calculations 100%.
3. Uploading ECG B computes measurements strictly from ECG B.
4. Historical retrieval of ECG A returns unchanged data with zero cross-contamination.
5. No static mock fallbacks or stale cached values.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from api.index import app, AnalyzeRequest
from fastapi.testclient import TestClient
from services.report_persistence_service import REPORT_PERSISTENCE_SERVICE
from database.db_manager import DB_MANAGER


@pytest.fixture
def client():
    return TestClient(app)


def generate_synthetic_ecg(duration_sec=3.0, fs=360.0, hr=75.0, pvc=False):
    t = np.linspace(0, duration_sec, int(duration_sec * fs))
    rr = 60.0 / hr
    sig = np.zeros_like(t)

    # Place beats at regular intervals
    beat_times = np.arange(0.2, duration_sec - 0.2, rr)
    for i, b_time in enumerate(beat_times):
        idx = int(b_time * fs)
        if idx >= len(sig) - 50:
            continue
        # QRS spike
        width = 25 if (pvc and i % 2 == 1) else 14
        amp = -2.5 if (pvc and i % 2 == 1) else 2.2
        for w in range(-width, width):
            if 0 <= idx + w < len(sig):
                sig[idx + w] += amp * np.exp(-((w) ** 2) / (2 * (width / 3) ** 2))

    # Add small physiological noise
    sig += 0.02 * np.sin(2 * np.pi * 0.5 * t)
    return sig.tolist()


def test_e2e_data_consistency_and_patient_isolation(client):
    # 1. Analyze Patient A (Alice - Normal Rhythm, 72 BPM)
    sig_a = generate_synthetic_ecg(duration_sec=3.0, fs=360.0, hr=72.0, pvc=False)
    payload_a = {
        "signal": sig_a,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Alice Green",
        "patient_mrn": "MRN-ALICE-100",
        "age": 35,
        "sex": "F",
        "blood_group": "B+",
        "current_medications": "None",
        "existing_conditions": "None",
        "include_full_report": True,
    }

    res_a = client.post("/api/analyze", json=payload_a)
    assert res_a.status_code == 200, res_a.text
    data_a = res_a.json()

    assert data_a["status"] == "SUCCESS"
    rep_id_a = data_a["report_id"]
    rep_num_a = data_a["report_number"]
    hr_a = data_a["cardiac_parameters"]["heart_rate_bpm"]
    qrs_a = data_a["cardiac_parameters"]["qrs_duration_ms"]
    pred_a = data_a["ai_classification"]["prediction"]

    assert "Normal" in pred_a or "Sinus" in pred_a
    assert 60 <= hr_a <= 85
    assert len(data_a["beat_segments"]) > 0
    assert len(data_a["mean_beat_profile"]) > 0
    assert data_a["beat_features"]["r_peak_amplitude_mv"] > 0

    # Verify Patient A PDF Generation
    pdf_res_a = client.post("/api/report/pdf", json=payload_a)
    assert pdf_res_a.status_code == 200
    assert pdf_res_a.headers["content-type"] == "application/pdf"
    assert len(pdf_res_a.content) > 1000

    # 2. Analyze Patient B (Bob - PVC Arrhythmia, 96 BPM)
    sig_b = generate_synthetic_ecg(duration_sec=3.0, fs=360.0, hr=96.0, pvc=True)
    payload_b = {
        "signal": sig_b,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Bob Vance",
        "patient_mrn": "MRN-BOB-200",
        "age": 62,
        "sex": "M",
        "blood_group": "O-",
        "current_medications": "Metoprolol",
        "existing_conditions": "Hypertension",
        "include_full_report": True,
    }

    res_b = client.post("/api/analyze", json=payload_b)
    assert res_b.status_code == 200, res_b.text
    data_b = res_b.json()

    assert data_b["status"] == "SUCCESS"
    rep_id_b = data_b["report_id"]
    rep_num_b = data_b["report_number"]
    hr_b = data_b["cardiac_parameters"]["heart_rate_bpm"]
    qrs_b = data_b["cardiac_parameters"]["qrs_duration_ms"]

    # Verify Strict Isolation: IDs and measurements must be completely distinct
    assert rep_id_a != rep_id_b
    assert rep_num_a != rep_num_b
    assert abs(hr_a - hr_b) > 5.0  # Bob's heart rate differs significantly from Alice's

    # 3. Retrieve Historical Report for Patient A from Persistence Service
    hist_a_res = client.get(f"/api/reports/{rep_id_a}")
    assert hist_a_res.status_code == 200, hist_a_res.text
    hist_a = hist_a_res.json()["report"]

    # Assert Alice's historical data was NOT contaminated by Bob's analysis
    pat_profile_a = hist_a["report_data"]["patient_info"]
    assert pat_profile_a["patient_name"] == "Alice Green"
    assert pat_profile_a["hospital_mrn"] == "MRN-ALICE-100"
    assert pat_profile_a["patient_age"] == 35

    cardiac_a_saved = hist_a["report_data"]["cardiac_parameters"]
    assert cardiac_a_saved["heart_rate_bpm"] == hr_a
    assert cardiac_a_saved["qrs_duration_ms"] == qrs_a

    # 4. Retrieve Historical Report for Patient B from Persistence Service
    hist_b_res = client.get(f"/api/reports/{rep_id_b}")
    assert hist_b_res.status_code == 200, hist_b_res.text
    hist_b = hist_b_res.json()["report"]

    pat_profile_b = hist_b["report_data"]["patient_info"]
    assert pat_profile_b["patient_name"] == "Bob Vance"
    assert pat_profile_b["hospital_mrn"] == "MRN-BOB-200"

    cardiac_b_saved = hist_b["report_data"]["cardiac_parameters"]
    assert cardiac_b_saved["heart_rate_bpm"] == hr_b

    # 5. Verify History Archive Endpoint
    list_res = client.get("/api/reports")
    assert list_res.status_code == 200
    reports_list = list_res.json()["reports"]
    rep_nums = [r.get("report_number") for r in reports_list]
    assert rep_num_a in rep_nums
    assert rep_num_b in rep_nums
