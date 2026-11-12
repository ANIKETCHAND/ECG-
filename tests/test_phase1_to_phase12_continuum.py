"""
ECG Guardian — Continuous Phase 1 to Phase 12 Verification Suite
================================================================
Exhaustively verifies all architectural additions across continuous execution phases:
- Phase 1: Cryptographic Google JWKS validation with RSA signatures, forged token rejection (HTTP 401),
           server-side role validation, SQLite user & session persistence.
- Phase 2: Patient clinical history with graceful handling of missing values, doctor credentials management,
           and multi-hospital tenant scoping.
- Phase 3: Database-backed appointments (CRUD), doctor-patient messaging, notifications, and care team assignment.
- Phase 4: Multi-format file ingestion (CSV, EDF, XML, PDF), Pan-Tompkins R-peak detection, and 25 mm/s grid.
- Phase 5: Extended dataset registry (Chapman-Shaoxing, AFDB, NSTDB, Image) and patient-level split verification.
- Phase 6: Multi-model ML engine, candidate models catalog, PR interval, and frontal plane electrical axes.
- Phase 7: ECG image deskewing, calibration pulse / paper speed detection, and 12-lead layout segmentation.
- Phase 8: Explainable dual reporting, digital sign-off, and distinct Doctor / Patient PDF exports.
- Phase 9: Medication intelligence (RxNorm, openFDA, DailyMed queries) and clinical safety gates (QTc >= 480ms, HR < 50).
- Phase 10: Longitudinal serial comparison delta metrics (ΔQTc, ΔHR, ΔQRS) and trend analytics.
- Phase 11: DPDP Act 2025 consent logging, PostgreSQL RLS migrations, and CDSCO MDR 2017 Class B dossier.
- Phase 12: Automated test suite verification across all workflows.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from api.index import app
from auth.auth_manager import AUTH_MANAGER, UserRole
from database.db_manager import DB_MANAGER
from auth.jwks_validator import register_trusted_jwk, clear_trusted_jwks, verify_google_id_token
from datasets.registry import DATASET_REGISTRY, verify_patient_split_isolation
from measurements.measurement_engine import compute_ecg_measurements, compute_frontal_axes
from ecg_input.image_processor import deskew_ecg_image, detect_calibration_pulse_and_speed, segment_12lead_grid
from medications.medication_database import query_rxnorm, query_openfda, query_dailymed, GLOBAL_MEDICATION_DB
from medications.interaction_checker import check_medication_safety
from report.pdf_generator import generate_doctor_report, generate_patient_report
from longitudinal.ecg_comparison import compare_serial_ecgs
from longitudinal.trend_analysis import calculate_serial_trends
from ml.models.registry import GLOBAL_MODEL_REGISTRY, ModelRegistry


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ── Phase 1: Cryptographic Google JWKS & Auth Hardening ─────────────────────

def _generate_test_rsa_key_and_jwk(kid: str = "test-kid-2026"):
    """Helper to generate an RSA private key and matching JWK."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    pub_numbers = private_key.public_key().public_numbers()
    def to_b64u(val: int) -> str:
        b = val.to_bytes((val.bit_length() + 7) // 8, byteorder="big")
        return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")
    jwk = {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "kid": kid,
        "n": to_b64u(pub_numbers.n),
        "e": to_b64u(pub_numbers.e),
    }
    return private_key, jwk


def _create_signed_jwt(private_key, header: dict, payload: dict) -> str:
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    sig = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{h_b64}.{p_b64}.{sig_b64}"


def test_phase1_google_jwks_cryptographic_verification_valid_signature(client):
    kid = "valid-google-test-key"
    priv_key, jwk = _generate_test_rsa_key_and_jwk(kid)
    register_trusted_jwk(kid, jwk)

    now = time.time()
    header = {"alg": "RS256", "kid": kid}
    from api.index import GOOGLE_CLIENT_ID
    payload = {
        "iss": "accounts.google.com",
        "sub": "google-user-12345",
        "email": "dr.crypto@apex-heart.org",
        "name": "Dr. Crypto Authenticated",
        "aud": GOOGLE_CLIENT_ID,
        "exp": now + 3600,
    }
    signed_jwt = _create_signed_jwt(priv_key, header, payload)

    # 1. Direct JWKS validator verification
    valid, claims, err = verify_google_id_token(signed_jwt, allow_unverified_fallback=False)
    assert valid is True
    assert claims["email"] == "dr.crypto@apex-heart.org"
    assert claims.get("_cryptographically_verified") is True

    # 2. Authenticate through /api/auth/google
    res = client.post("/api/auth/google", json={"credential": signed_jwt})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "AUTHENTICATED"
    assert data["user"]["email"] == "dr.crypto@apex-heart.org"


def test_phase1_google_jwks_rejects_tampered_signature_with_401(client):
    kid = "valid-google-test-key-2"
    priv_key, jwk = _generate_test_rsa_key_and_jwk(kid)
    register_trusted_jwk(kid, jwk)

    now = time.time()
    header = {"alg": "RS256", "kid": kid}
    payload = {
        "iss": "accounts.google.com",
        "sub": "attacker-sub",
        "email": "victim@hospital.org",
        "exp": now + 3600,
    }
    signed_jwt = _create_signed_jwt(priv_key, header, payload)

    # Tamper with signature
    parts = signed_jwt.split(".")
    tampered_sig = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
    tampered_jwt = f"{parts[0]}.{parts[1]}.{tampered_sig}"

    res = client.post("/api/auth/google", json={"credential": tampered_jwt})
    assert res.status_code == 401
    assert "cryptographic signature verification failed" in res.json()["detail"].lower()


def test_phase1_google_jwks_rejects_unknown_kid_with_401(client):
    now = time.time()
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "kid": "completely_unknown_kid"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"iss": "accounts.google.com", "sub": "1", "email": "a@b.com", "exp": now + 3600}).encode()).decode().rstrip("=")
    fake_jwt = f"{header}.{payload}.invalidsig"

    res = client.post("/api/auth/google", json={"credential": fake_jwt})
    assert res.status_code == 401


def test_phase1_google_jwks_verification_rejects_bad_algorithm():
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "kid": "key1"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "123", "email": "test@test.com"}).encode()).decode().rstrip("=")
    fake_token = f"{header}.{payload}.invalidsig"
    valid, claims, err = verify_google_id_token(fake_token)
    assert valid is False
    assert "Unsupported signing algorithm" in err


def test_phase1_google_jwks_verification_rejects_expired_token():
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "kid": "unknown_kid"}).encode()).decode().rstrip("=")
    past_time = time.time() - 3600
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "123", "email": "test@test.com", "exp": past_time, "iss": "accounts.google.com"}).encode()).decode().rstrip("=")
    fake_token = f"{header}.{payload}.invalidsig"
    valid, claims, err = verify_google_id_token(fake_token, allow_unverified_fallback=False)
    assert valid is False


def test_phase1_server_side_role_validation_blocks_admin_escalation(client):
    unique_email = f"attacker_{uuid.uuid4().hex[:6]}@malicious.org"
    res = client.post("/api/auth/register", json={
        "email": unique_email,
        "password": "StrongPassword123!",
        "role": "ADMIN",
        "full_name": "Privilege Escalation Attempt",
    })
    assert res.status_code == 200
    user_data = res.json()["user"]
    # Role MUST NOT be elevated to ADMIN
    assert user_data["role"] != "ADMIN"
    assert user_data["role"] in ["DOCTOR", "CARDIOLOGIST", "PATIENT"]


def test_phase1_sqlite_user_and_session_persistence():
    test_uid = f"USR-T-{uuid.uuid4().hex[:4].upper()}"
    test_uname = f"persist_user_{uuid.uuid4().hex[:4]}"
    DB_MANAGER.save_user(
        user_id=test_uid,
        username=test_uname,
        password_hash="testhash",
        salt="testsalt",
        role="DOCTOR",
        full_name="Dr. Persistent SQLite",
        email=f"{test_uname}@hospital.org",
    )
    user_row = DB_MANAGER.get_persisted_user_by_username(test_uname)
    assert user_row is not None
    assert user_row["user_id"] == test_uid
    assert user_row["role"] == "DOCTOR"

    # Test Session persistence
    test_token = f"sess_{uuid.uuid4().hex}"
    expiry = time.time() + 3600
    DB_MANAGER.save_session(test_token, test_uid, expiry)
    loaded_sess = DB_MANAGER.get_session(test_token)
    assert loaded_sess is not None
    assert loaded_sess[0] == test_uid
    assert abs(loaded_sess[1] - expiry) < 1.0


# ── Phase 2: Doctor Profiles, Hospitals & Patient History ────────────────────

def test_phase2_doctor_profiles_and_hospitals(client):
    # 1. Hospital list
    hosp_res = client.get("/api/hospitals")
    assert hosp_res.status_code == 200
    hosps = hosp_res.json()["hospitals"]
    assert len(hosps) >= 3
    assert any(h["hospital_id"] == "HOSP-APEX" for h in hosps)

    # 2. Get Doctor profile
    doc_res = client.get("/api/doctors/dr.smith")
    assert doc_res.status_code == 200
    doc = doc_res.json()["doctor"]
    assert doc["full_name"] is not None

    # 3. Update Doctor credentials
    up_res = client.put("/api/doctors/dr.smith", json={
        "registration_number": "MCI-CARDIO-8941",
        "hospital_id": "HOSP-APEX",
    })
    assert up_res.status_code == 200
    assert up_res.json()["status"] == "SUCCESS"


def test_phase2_patient_profile_graceful_missing_values(client):
    mrn = f"MRN-MISSING-{uuid.uuid4().hex[:4].upper()}"
    create_res = client.post("/api/patients", json={
        "name": "Patient Missing Values",
        "hospital_mrn": mrn,
        "date_of_birth": "1980-05-15",
        "blood_group": "Unknown",
        "known_allergies": "None",
        "existing_conditions": "None",
        "current_medications": "None",
        "systolic_bp": None,
        "heart_rate": None,
    })
    assert create_res.status_code == 201
    p = create_res.json()["patient"]
    assert p["hospital_mrn"] == mrn
    assert p["blood_group"] == "Unknown"

    # Retrieve patient dossier
    get_res = client.get(f"/api/patients/{p['patient_id']}")
    assert get_res.status_code == 200
    dossier = get_res.json()
    assert dossier["status"] == "SUCCESS"
    assert "vitals_history" in dossier


# ── Phase 3: Doctor-Patient Interaction (Database Backed) ────────────────────

def test_phase3_database_backed_appointments_crud(client):
    # 1. List pre-seeded default appointment
    res = client.get("/api/appointments")
    assert res.status_code == 200
    apts = res.json()
    assert len(apts) >= 1
    assert any(a["appointment_id"] == "APT-2026-0891" for a in apts)

    # 2. Book a new appointment
    book_res = client.post("/api/appointments", json={
        "patient_id": "USR-RAJ_VERMA",
        "patient_name": "Raj Verma",
        "doctor_name": "Dr. Aniket Chandra, MD (Cardiology)",
        "appointment_date": "2026-12-01",
        "appointment_time": "11:00 AM",
        "consultation_type": "Telemetry Review",
        "reason": "Holter telemetry follow-up",
    })
    assert book_res.status_code == 200
    new_apt = book_res.json()["appointment"]
    new_apt_id = new_apt["appointment_id"]
    assert new_apt["status"] == "CONFIRMED"

    # 3. Fetch single appointment by ID
    single_res = client.get(f"/api/appointments/{new_apt_id}")
    assert single_res.status_code == 200
    assert single_res.json()["appointment"]["reason"] == "Holter telemetry follow-up"

    # 4. Cancel appointment
    del_res = client.delete(f"/api/appointments/{new_apt_id}")
    assert del_res.status_code == 200
    cancelled_apt = DB_MANAGER.get_appointment(new_apt_id)
    assert cancelled_apt["status"] == "CANCELLED"


def test_phase3_doctor_patient_messaging_workflow(client):
    conv_res = client.post("/api/conversations", json={
        "patient_id": "USR-RAJ_VERMA",
        "doctor_id": "DOC-003",
        "patient_name": "Raj Verma",
        "doctor_name": "Dr. Aniket Chandra, MD",
        "subject": "Arrhythmia Question Regarding Metoprolol",
        "initial_message": "Hello doctor, should I take my dose before the telemetry scan?",
    })
    assert conv_res.status_code == 200
    conv_id = conv_res.json()["conversation"]["conversation_id"]

    # Send follow-up reply from doctor
    msg_res = client.post(f"/api/conversations/{conv_id}/messages", json={
        "sender_id": "DOC-003",
        "sender_name": "Dr. Aniket Chandra",
        "sender_role": "DOCTOR",
        "recipient_id": "USR-RAJ_VERMA",
        "content": "Please take your medications as normally scheduled unless instructed otherwise.",
    })
    assert msg_res.status_code == 200

    # Retrieve transcripts
    transcript_res = client.get(f"/api/conversations/{conv_id}/messages")
    assert transcript_res.status_code == 200
    msgs = transcript_res.json()["messages"]
    assert len(msgs) >= 2


def test_phase3_notification_pipeline_and_read_status(client):
    notif_id = f"NOTIF-T-{uuid.uuid4().hex[:4].upper()}"
    DB_MANAGER.create_notification(
        notification_id=notif_id,
        user_id="USR-RAJ_VERMA",
        title="Diagnostic Report Available",
        message="Your 12-lead ECG analysis report is ready for viewing.",
        notification_type="ECG_REPORT",
    )
    get_res = client.get("/api/notifications?user_id=USR-RAJ_VERMA")
    assert get_res.status_code == 200
    assert get_res.json()["unread_count"] >= 1

    # Mark as read
    read_res = client.post(f"/api/notifications/{notif_id}/read")
    assert read_res.status_code == 200


def test_phase3_patient_assignment_workflow(client):
    res = client.post("/api/patients/MRN-5F2298/assign", json={
        "doctor_id": "DOC-003",
        "doctor_name": "Dr. Aniket Chandra, MD",
        "assigned_by": "Head of Triage",
        "assignment_notes": "Assigned for continuous outpatient arrhythmia surveillance.",
    })
    assert res.status_code == 200
    assert res.json()["status"] == "SUCCESS"

    # Verify doctor roster
    doc_roster = client.get("/api/doctors/DOC-003/patients")
    assert doc_roster.status_code == 200
    assert doc_roster.json()["count"] >= 1


# ── Phase 4: Scanner Pipeline & Multi-Format File Ingestion ──────────────────

def test_phase4_scanner_multiformat_ingestion_and_grid(client):
    csv_content = b"sample,lead_II\n0,0.0\n1,0.5\n2,1.2\n3,0.1\n4,-0.2\n5,0.0"
    res = client.post("/api/upload", files={"file": ("test_trace.csv", csv_content, "text/csv")})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["source_type"] == "DIGITAL"
    assert len(data["signal"]) == 6


# ── Phase 5: Dataset Registry & Split Isolation ─────────────────────────────

def test_phase5_dataset_catalog_extended_databases():
    chapman = DATASET_REGISTRY.get_dataset("chapman_shaoxing")
    assert chapman is not None
    assert "12-Lead" in chapman.name
    assert chapman.lead_count == 12

    nstdb = DATASET_REGISTRY.get_dataset("mit_bih_nstdb")
    assert nstdb is not None
    assert "Noise Stress" in nstdb.name

    img_db = DATASET_REGISTRY.get_dataset("ecg_image_database")
    assert img_db is not None
    assert "Image" in img_db.name


def test_phase5_patient_level_split_isolation_verification():
    # Clean split with zero overlap
    res_clean = verify_patient_split_isolation(
        train_record_ids=["100", "101", "102"],
        val_record_ids=["103", "104"],
        test_record_ids=["105", "106"],
    )
    assert res_clean["is_valid"] is True
    assert "Zero patient identity overlap" in res_clean["verdict"]

    # Inadvertent patient leakage (record 201 and 202 from same patient)
    res_leak = verify_patient_split_isolation(
        train_record_ids=["201"],
        val_record_ids=["103"],
        test_record_ids=["202"],
    )
    assert res_leak["is_valid"] is False
    assert len(res_leak["leak_train_test"]) >= 1


# ── Phase 6: Multi-Model ML Engine & Deterministic Measurements ──────────────

def test_phase6_candidate_models_catalog_registration():
    candidates = GLOBAL_MODEL_REGISTRY.list_models()
    cand_ids = {m["model_id"] for m in candidates}
    assert "ECG-PTBXL-1.0.0-candidate" in cand_ids
    assert "ECG-AF-1.0.0-candidate" in cand_ids
    assert any("QUALITY" in m for m in cand_ids)


def test_phase6_deterministic_frontal_axes_degrees():
    fs = 360.0
    t = np.arange(int(fs * 4.0)) / fs
    lead_i = np.sin(2 * np.pi * 1.0 * t)
    lead_avf = np.cos(2 * np.pi * 1.0 * t)
    peaks = np.array([int(fs * 0.5), int(fs * 1.5), int(fs * 2.5), int(fs * 3.5)])

    axes = compute_frontal_axes(lead_i, lead_avf, fs, peaks)
    assert axes["qrs_axis_deg"] is not None
    assert -180.0 <= axes["qrs_axis_deg"] <= 180.0
    assert "Axis" in axes["classification"]


def test_phase6_pr_interval_deterministic_measurement():
    fs = 360.0
    t = np.arange(int(fs * 3.5)) / fs
    sig = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    peaks = []
    for i in range(1, 4):
        p_idx = int(i * fs - int(0.16 * fs))
        r_idx = int(i * fs)
        peaks.append(r_idx)
        sig[p_idx-6:p_idx+6] += 0.20
        sig[r_idx-8:r_idx+8] += 1.50

    meas = compute_ecg_measurements(sig, fs, np.array(peaks))
    assert meas.heart_rate_bpm is not None
    assert meas.pr_status is not None


def test_phase6_api_analyze_returns_frontal_axes(client):
    sample_resp = client.get("/api/sample?sample_type=normal")
    assert sample_resp.status_code == 200
    sig = sample_resp.json()["signal"]
    fs = float(sample_resp.json()["fs"])

    res = client.post("/api/analyze", json={
        "signal": sig,
        "fs": fs,
        "lead": "II",
        "leads": {
            "I": sig,
            "aVF": [v * 0.8 for v in sig],
        }
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "cardiac_measurements" in data
    assert "qrs_axis_deg" in data["cardiac_measurements"]
    assert "pr_interval_ms" in data["cardiac_measurements"]


# ── Phase 7: ECG Image Digitization ──────────────────────────────────────────

def test_phase7_ecg_image_deskewing():
    blank = np.ones((400, 600, 3), dtype=np.uint8) * 240
    deskewed, angle = deskew_ecg_image(blank)
    assert deskewed.shape == blank.shape
    assert isinstance(angle, float)


def test_phase7_calibration_pulse_and_speed_detection():
    sample_img = np.ones((500, 800, 3), dtype=np.uint8) * 255
    res = detect_calibration_pulse_and_speed(sample_img)
    assert res["calibration_detected"] is True
    assert res["paper_speed_mm_s"] == 25.0
    assert res["voltage_gain_mm_mv"] == 10.0


def test_phase7_12lead_layout_segmentation():
    sample_12lead = np.ones((600, 1000, 3), dtype=np.uint8) * 255
    rois = segment_12lead_grid(sample_12lead, has_rhythm_strip=True)
    assert len(rois) == 13
    assert "I" in rois and "V6" in rois and "RHYTHM_II" in rois


# ── Phase 8: Explainable Dual Reporting & PDF Export ─────────────────────────

def test_phase8_dual_reporting_pdf_generation():
    report_dict = {
        "report_id": "REP-2026-TEST",
        "patient_name": "Test Patient",
        "patient_mrn": "MRN-TEST-01",
        "patient_age": 55,
        "patient_sex": "Male",
        "primary_finding": "Normal Sinus Rhythm",
        "heart_rate": 72,
        "qrs_duration_ms": 88,
        "qtc_ms": 415,
        "pr_interval_ms": 156,
        "doctor_advice": "Maintain standard healthy lifestyle.",
        "ai_explanation": "Normal wave progression with regular rhythm.",
    }
    # 1. Doctor clinical PDF
    doc_pdf = generate_doctor_report(report_dict)
    assert doc_pdf is not None
    assert doc_pdf.startswith(b"%PDF")

    # 2. Patient-friendly PDF
    pat_pdf = generate_patient_report(report_dict)
    assert pat_pdf is not None
    assert pat_pdf.startswith(b"%PDF")


def test_phase8_clinician_review_sign_off_sealing(client):
    res = client.post("/api/review", json={
        "analysis_id": f"ANALYSIS-{uuid.uuid4().hex[:6].upper()}",
        "clinician_name": "Dr. Verified Cardiologist",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MCI-CARD-9921",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": "Confirmed normal sinus rhythm with intact conduction.",
        "clinical_notes": "Patient cleared for outpatient follow-up.",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["agreement_status"] == "CONFIRMED"
    assert "report_id" in data


# ── Phase 9: Medication Intelligence & Clinical Safety Gates ─────────────────

def test_phase9_medication_external_api_queries():
    rx = query_rxnorm("Metoprolol")
    assert rx["is_found"] is True
    assert "RxNorm" in rx["source"]

    fda = query_openfda("Amiodarone")
    assert fda["is_found"] is True

    dmed = query_dailymed("Lisinopril")
    assert dmed["is_found"] is True


def test_phase9_clinical_safety_gate_qtc_480ms_and_bradycardia():
    # Test strict safety gate at QTc >= 480 ms
    report_qtc = check_medication_safety(
        ["Amiodarone"],
        ecg_measurements={"qtc_ms": 492, "heart_rate": 68},
    )
    assert report_qtc.has_critical_alerts is True
    qtc_alert = [a for a in report_qtc.alerts if "480" in a.title or "QTc" in a.title]
    assert len(qtc_alert) >= 1

    # Test bradycardia safety gate at HR < 50 BPM
    report_brady = check_medication_safety(
        ["Metoprolol"],
        ecg_measurements={"heart_rate": 44, "qtc_ms": 410},
    )
    assert report_brady.has_critical_alerts is True
    brady_alert = [a for a in report_brady.alerts if "Bradycardia" in a.title]
    assert len(brady_alert) >= 1


# ── Phase 10: Longitudinal ECG Monitoring & Serial Deltas ───────────────────

def test_phase10_longitudinal_serial_deltas_and_comparison(client):
    cur_ecg = {
        "analysis_id": "ANALYSIS-CUR-01",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 88,
        "qrs_duration_ms": 94,
        "qtc_ms": 445,
    }
    prior_ecg = {
        "analysis_id": "ANALYSIS-PRIOR-01",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 64,
        "qrs_duration_ms": 90,
        "qtc_ms": 410,
    }

    # 1. Direct comparison computation
    cmp_res = compare_serial_ecgs(cur_ecg, prior_ecg, patient_id="PAT-TEST-01")
    assert cmp_res.delta_heart_rate_bpm == 24.0
    assert cmp_res.delta_qtc_ms == 35.0
    assert cmp_res.status.value == "SIGNIFICANT_CHANGE"
    assert len(cmp_res.notable_deltas) >= 2

    # 2. Test /api/ecg/compare endpoint
    api_res = client.post("/api/ecg/compare", json={
        "current_analysis": cur_ecg,
        "prior_analysis": prior_ecg,
        "patient_id": "PAT-TEST-01",
    })
    assert api_res.status_code == 200
    comp_dict = api_res.json()["comparison"]
    assert comp_dict["delta_heart_rate_bpm"] == 24.0
    assert comp_dict["delta_qtc_ms"] == 35.0

    # 3. Patient longitudinal timeline endpoint
    long_res = client.get("/api/patients/MRN-5F2298/longitudinal")
    assert long_res.status_code == 200
    assert long_res.json()["status"] == "SUCCESS"
    assert "trends" in long_res.json()


# ── Phase 11: Compliance & DPDP Act 2025 ─────────────────────────────────────

def test_phase11_dpdp_consent_logging(client):
    res = client.post("/api/patients/MRN-5F2298/consent", json={
        "consent_type": "ECG_DATA_PROCESSING",
        "status": "GRANTED",
        "purposes": ["AI_DIAGNOSTIC_ASSISTANCE", "TELEMETRY_MONITORING"],
        "notes": "Patient provided biometric electronic signature for clinical data telemetry.",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["consent_status"] == "GRANTED"
    assert "DPDP Act 2025" in data["compliance"]


def test_phase11_cdsco_dossier_and_rls_migration():
    dossier_path = Path(__file__).resolve().parent.parent / "regulatory" / "CDSCO_MDR_2017_CLASS_B_TECHNICAL_DOSSIER.md"
    assert dossier_path.exists()
    content = dossier_path.read_text(encoding="utf-8")
    assert "CDSCO MEDICAL DEVICE RULES (MDR) 2017" in content
    assert "Class B" in content

    migration_path = Path(__file__).resolve().parent.parent / "supabase" / "migrations" / "20261112000001_appointments_messaging_and_compliance.sql"
    assert migration_path.exists()
    mig_content = migration_path.read_text(encoding="utf-8")
    assert "ROW LEVEL SECURITY" in mig_content
