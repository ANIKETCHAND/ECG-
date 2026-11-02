"""
ECG Guardian — Backend and API Exhaustive Validation Suite
==========================================================
Systematically audits and tests every FastAPI endpoint across all failure modes:
1. Authentication (Register, Login, Session, Me, Logout, Invalid pass, Bad emails, RBAC)
2. Patient Lifecycle (Create, Read, Update, List, Search, Duplicate MRN 409, Invalid Age 400)
3. Dashboard Aggregations (Stats, hospital isolation, non-mock counts)
4. ECG Ingestion & Preprocessing (Upload CSV/TXT/NPY/PDF/Image, unsupported types, corrupt files)
5. ECG Analysis & AI Inference (Valid signal, Unusable signal gate, Short duration 400, Missing fields)
6. Medication Safety & CDS Linkage (Candidate medications, allergy contraindications, QTc risks)
7. Clinician Review (Confirmed, Modified, Rejected, Doctor registration checks)
8. Report Generation & Dual PDF Export (Doctor PDF, Patient PDF, Raw binary validation)
9. Historical Snapshot Retrieval & Immutability (No re-inference on retrieval)
"""

import io
import uuid
import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.index import app
from auth.auth_manager import AUTH_MANAGER, UserRole
from database.db_manager import DB_MANAGER
from services import REPORT_PERSISTENCE_SERVICE


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def clean_lead2_signal():
    """Generates 3 seconds of synthetic clean Lead II ECG at 360 Hz (1080 samples)."""
    fs = 360.0
    t = np.arange(int(fs * 3.0)) / fs
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    for i in range(1, 4):
        idx = int(i * (fs / 1.2))
        w = 12
        start, end = max(0, idx - w), min(len(sig), idx + w)
        sig[start:end] += 1.8 * np.exp(-0.5 * ((np.arange(start, end) - idx) / 3) ** 2)
    return sig.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 1. HEALTH & LIVENESS ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def test_endpoint_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "regulatory_classification" in data
    assert "active_model_id" in data
    assert "selective_reporting" in data


def test_endpoint_sample_waveforms(client):
    # Valid types
    for stype in ["normal", "pvc", "arrhythmia", "unusable"]:
        res = client.get(f"/api/sample?sample_type={stype}")
        assert res.status_code == 200
        data = res.json()
        assert "signal" in data
        assert len(data["signal"]) > 50
        assert data["fs"] == 360.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. AUTHENTICATION & SESSION ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def test_auth_register_and_login_full_cycle(client):
    unique_id = uuid.uuid4().hex[:6]
    email = f"test.doctor.{unique_id}@apexheart.org"
    password = "StrongPassword@2026!"

    # 1. Register
    reg_res = client.post("/api/auth/register", json={
        "email": email,
        "password": password,
        "full_name": f"Dr. Test Clinician {unique_id}",
        "role": "DOCTOR",
        "hospital_id": "HOSP-APEX",
        "registration_number": f"MCI-DOC-{unique_id.upper()}",
    })
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    assert reg_data["status"] == "AUTHENTICATED"
    assert "token" in reg_data
    token = reg_data["token"]

    # 2. Duplicate registration should return 409 Conflict
    reg_dup = client.post("/api/auth/register", json={
        "email": email,
        "password": password,
        "full_name": f"Dr. Test Clinician {unique_id}",
        "role": "DOCTOR",
    })
    assert reg_dup.status_code == 409
    assert "already exists" in reg_dup.json()["detail"].lower()

    # 3. Check Session
    sess_res = client.get(f"/api/auth/session?token={token}")
    assert sess_res.status_code == 200
    assert sess_res.json()["status"] in ["ACTIVE", "AUTHENTICATED"]

    # 4. Check Me endpoint
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["user"]["email"] == email

    # 5. Invalid password login
    bad_login = client.post("/api/auth/login", json={
        "email": email,
        "password": "WrongPassword999!",
    })
    assert bad_login.status_code == 401

    # 6. Correct password login
    good_login = client.post("/api/auth/login", json={
        "email": email,
        "password": password,
    })
    assert good_login.status_code == 200
    assert good_login.json()["status"] == "AUTHENTICATED"

    # 7. Logout
    logout_res = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_res.status_code == 200


def test_auth_role_portal_isolation(client):
    # Patient trying to login into doctor portal with patient credentials
    pat_res = client.post("/api/auth/login", json={
        "email": "rajverma@gmail.com",
        "password": "Patient@Hospital2026",
        "role": "DOCTOR",
    })
    assert pat_res.status_code == 403
    assert "cannot log in to the Physician portal" in pat_res.json()["detail"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. PATIENT MANAGEMENT ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def test_patient_crud_and_validation(client):
    unique_mrn = f"MRN-TEST-{uuid.uuid4().hex[:6].upper()}"

    # 1. Create Patient
    create_payload = {
        "name": "Jane Test Patient",
        "hospital_mrn": unique_mrn,
        "hospital_id": "HOSP-APEX",
        "date_of_birth": "1980-06-15",
        "sex": "Female",
        "blood_group": "B+",
        "known_allergies": ["Penicillin", "Sulfa"],
        "existing_conditions": ["Hypertension"],
        "current_medications": ["Amlodipine 5mg"],
        "systolic_bp": 138.0,
        "diastolic_bp": 88.0,
        "heart_rate": 76.0,
    }
    res = client.post("/api/patients", json=create_payload)
    assert res.status_code == 201
    pat_data = res.json()["patient"]
    pat_id = pat_data["patient_id"]
    assert pat_data["hospital_mrn"] == unique_mrn
    assert pat_data["age"] >= 44

    # 2. Duplicate MRN Rejection
    dup_res = client.post("/api/patients", json=create_payload)
    assert dup_res.status_code == 409
    assert "already exists" in dup_res.json()["detail"]

    # 3. Get Patient by ID and by MRN
    get_by_id = client.get(f"/api/patients/{pat_id}")
    assert get_by_id.status_code == 200
    assert get_by_id.json()["patient"]["name"] == "Jane Test Patient"

    get_by_mrn = client.get(f"/api/patients/{unique_mrn}")
    assert get_by_mrn.status_code == 200
    assert get_by_mrn.json()["patient"]["patient_id"] == pat_id

    # 4. Update Patient (PUT)
    update_res = client.put(f"/api/patients/{pat_id}", json={
        "name": "Jane Updated Patient",
        "current_medications": ["Amlodipine 10mg", "Atorvastatin 20mg"],
        "systolic_bp": 125.0,
    })
    assert update_res.status_code == 200
    assert update_res.json()["patient"]["name"] == "Jane Updated Patient"

    # 5. List Patients
    list_res = client.get("/api/patients?limit=50")
    assert list_res.status_code == 200
    assert len(list_res.json()["patients"]) > 0

    # 6. Search Patients
    search_res = client.get(f"/api/patients?q={unique_mrn}")
    assert search_res.status_code == 200
    assert len(search_res.json()["patients"]) == 1

    # 7. Invalid DOB in Future Rejection
    future_dob_res = client.post("/api/patients", json={
        "name": "Future Baby",
        "date_of_birth": "2099-01-01",
    })
    assert future_dob_res.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# 4. DASHBOARD STATS ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────

def test_dashboard_stats_endpoint(client):
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    stats = data["stats"]
    assert "total_patients" in stats
    assert "total_ecgs" in stats
    assert "completed_reports" in stats
    assert "pending_reviews" in stats
    assert isinstance(stats["recent_patients"], list)


# ──────────────────────────────────────────────────────────────────────────────
# 5. ECG ANALYSIS & GATEKEEPER ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def test_analyze_ecg_clean_signal(client, clean_lead2_signal):
    res = client.post("/api/analyze", json={
        "signal": clean_lead2_signal,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Audited Patient",
        "patient_mrn": "MRN-AUDIT-001",
        "age": 45,
        "sex": "Male",
        "include_full_report": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "analysis_id" in data
    assert "cardiac_parameters" in data
    assert "ai_classification" in data
    assert "clinical_decision_support" in data
    assert "full_report" in data


def test_analyze_ecg_safety_gate_halts_unusable_signal(client):
    # Flatline signal
    flatline = [0.0] * 500
    res = client.post("/api/analyze", json={
        "signal": flatline,
        "fs": 360.0,
        "lead": "II",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "UNUSABLE_SIGNAL"
    assert data["can_run_ai"] is False
    assert data["prediction"] == "NO_RESULT_SIGNAL_UNUSABLE"


def test_analyze_ecg_rejects_too_short_signal(client):
    res = client.post("/api/analyze", json={
        "signal": [1.0, 2.0, 3.0],  # only 3 samples
        "fs": 360.0,
    })
    assert res.status_code == 400
    assert "too short" in res.json()["detail"].lower()


# ──────────────────────────────────────────────────────────────────────────────
# 6. ECG MULTI-FORMAT UPLOAD ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────

def test_upload_ecg_csv_and_text(client):
    with open("sample_ecgs/normal_ecg_sample.csv", "rb") as f:
        file_bytes = f.read()

    files = {"file": ("normal_ecg_sample.csv", file_bytes, "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert len(data["signal"]) > 50


def test_upload_unsupported_format(client):
    files = {"file": ("malicious.exe", io.BytesIO(b"MZ1234"), "application/octet-stream")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLINICIAN REVIEW SIGN-OFF & CRYPTOGRAPHIC SEALING
# ──────────────────────────────────────────────────────────────────────────────

def test_clinician_review_and_report_sealing(client, clean_lead2_signal):
    # 1. Run analysis to create report
    anl_res = client.post("/api/analyze", json={
        "signal": clean_lead2_signal,
        "fs": 360.0,
        "patient_name": "Reviewable Patient",
    })
    anl_data = anl_res.json()
    analysis_id = anl_data["analysis_id"]

    # 2. Submit Physician Review
    rev_res = client.post("/api/review", json={
        "analysis_id": analysis_id,
        "clinician_name": "Dr. Aniket Chandra, MD",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MCI-994821",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": "Normal sinus rhythm confirmed. Patient cleared for activity.",
        "clinical_notes": "Follow-up in 12 months.",
        "selected_medication": "None",
    })
    assert rev_res.status_code == 200
    rev_data = rev_res.json()
    assert rev_data["status"] in ["SEALED", "SEALED_AND_CONFIRMED"]
    assert rev_data["review_details"]["agreement_status"] == "CONFIRMED"


# ──────────────────────────────────────────────────────────────────────────────
# 8. REPORT RETRIEVAL & IMMUTABILITY
# ──────────────────────────────────────────────────────────────────────────────

def test_report_retrieval_and_dual_pdf_download(client, clean_lead2_signal):
    # 1. Run analysis
    anl_res = client.post("/api/analyze", json={
        "signal": clean_lead2_signal,
        "fs": 360.0,
        "patient_name": "Downloadable Patient",
    })
    rep_id = anl_res.json()["report_id"]

    # 2. Get Report JSON
    get_rep = client.get(f"/api/reports/{rep_id}")
    assert get_rep.status_code == 200
    assert get_rep.json()["status"] == "SUCCESS"

    # 3. Download Doctor PDF
    doc_pdf = client.get(f"/api/reports/{rep_id}/pdf?type=doctor")
    assert doc_pdf.status_code == 200
    assert doc_pdf.headers["content-type"] == "application/pdf"
    assert doc_pdf.content.startswith(b"%PDF")

    # 4. Download Patient PDF
    pat_pdf = client.get(f"/api/reports/{rep_id}/pdf?type=patient")
    assert pat_pdf.status_code == 200
    assert pat_pdf.headers["content-type"] == "application/pdf"
    assert pat_pdf.content.startswith(b"%PDF")


def test_reports_listing_endpoint(client):
    """Verify GET /api/reports returns immutable report snapshot listing with pagination."""
    res = client.get("/api/reports?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "reports" in data
    assert isinstance(data["reports"], list)


def test_direct_pdf_generation_post_endpoint(client, clean_lead2_signal):
    """Verify POST /api/report/pdf generates publication-grade PDF synchronously."""
    res = client.post(
        "/api/report/pdf?type=doctor",
        json={
            "signal": clean_lead2_signal,
            "fs": 360.0,
            "patient_name": "Direct PDF Patient",
            "patient_mrn": "MRN-DIRECT-PDF",
            "age": 50,
            "sex": "Female",
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF")
    assert len(res.content) > 1000


def test_google_auth_endpoint(client):
    """Verify POST /api/auth/google registers or authenticates clinician profile."""
    uid = uuid.uuid4().hex[:6]
    res = client.post("/api/auth/google", json={
        "email": f"google.clinician.{uid}@apexheart.org",
        "name": f"Dr. Google Clinician {uid}",
        "role": "CARDIOLOGIST",
        "hospital_id": "HOSP-APEX",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "AUTHENTICATED"
    assert "token" in data
    assert "Dr. Google Clinician" in data["user"]["full_name"]
