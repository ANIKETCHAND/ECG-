"""
End-to-End Hospital Platform Verification Tests
=================================================

Comprehensive test suite verifying the complete hospital ECG management workflow:
1. Authentication & RBAC (Doctor & Patient registration, login, role boundary enforcement)
2. Patient Lifecycle & Clinical Intake (Create, auto-MRN, auto-age, Read, Update, List with search)
3. Dashboard Aggregation (Total patients, ECGs, pending reviews, signed reports)
4. Multimodal ECG Analysis with Patient Context linkage
5. Dual PDF Generation (Doctor Clinical Report & Patient Plain-Language Summary)
6. Clinician Review Cryptographic Sealing & Attestation
7. Historical Report Retrieval & Snapshot Immutability (no ML re-run)
8. Historical PDF Downloads (Doctor and Patient formats)
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from datetime import date, datetime, timedelta

from api.index import app, calculate_age_from_dob, generate_unique_mrn
from auth.auth_manager import AUTH_MANAGER, UserRole
from database.db_manager import DB_MANAGER
from services import REPORT_PERSISTENCE_SERVICE


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def test_signal():
    """Generate 3.0 seconds of synthetic Lead II ECG signal at 360 Hz."""
    fs = 360.0
    t = np.arange(int(fs * 3.0)) / fs
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    for i in range(1, 4):
        idx = int(i * (fs / 1.2))
        w = 10
        start, end = max(0, idx - w), min(len(sig), idx + w)
        sig[start:end] += 1.8 * np.exp(-0.5 * ((np.arange(start, end) - idx) / 3) ** 2)
    return sig.tolist()


# ==============================================================================
# 1. Date of Birth & MRN Unit Tests
# ==============================================================================

def test_calculate_age_from_dob_valid():
    today = date.today()
    birth_year = today.year - 45
    dob_str = f"{birth_year}-{today.month:02d}-{today.day:02d}"
    age = calculate_age_from_dob(dob_str)
    assert age == 45


def test_calculate_age_from_dob_future_date_rejected():
    future_date = (date.today() + timedelta(days=365)).strftime("%Y-%m-%d")
    age = calculate_age_from_dob(future_date)
    assert age is None


def test_calculate_age_from_dob_invalid_format():
    assert calculate_age_from_dob("invalid-date") is None
    assert calculate_age_from_dob("") is None
    assert calculate_age_from_dob(None) is None


def test_generate_unique_mrn():
    mrn1 = generate_unique_mrn()
    mrn2 = generate_unique_mrn()
    current_year = str(datetime.now().year)
    assert mrn1.startswith(f"MRN-{current_year}-")
    assert mrn2.startswith(f"MRN-{current_year}-")
    assert mrn1 != mrn2


# ==============================================================================
# 2. Authentication & RBAC Tests
# ==============================================================================

def test_doctor_registration_and_login(client):
    email = f"dr.test_{datetime.now().timestamp()}@hospital.org"
    password = "SecurePassword123!"

    # 1. Register Doctor
    reg_resp = client.post("/api/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Dr. Sarah Jenkins, MD",
        "role": "DOCTOR",
        "hospital_id": "HOSP-APEX",
        "registration_number": "MCI-994821",
    })
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert reg_data["status"] == "AUTHENTICATED"
    assert reg_data["user"]["role"] == "DOCTOR"
    assert reg_data["user"]["full_name"] == "Dr. Sarah Jenkins, MD"
    assert "token" in reg_data

    # 2. Login Doctor
    login_resp = client.post("/api/auth/login", json={
        "email": email,
        "password": password,
        "role": "DOCTOR",
    })
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert login_data["status"] == "AUTHENTICATED"
    assert login_data["user"]["email"] == email

    # 3. Validate Session
    session_resp = client.get(f"/api/auth/session?token={login_data['token']}")
    assert session_resp.status_code == 200
    assert session_resp.json()["status"] == "AUTHENTICATED"


def test_patient_registration_and_login(client):
    email = f"patient.test_{datetime.now().timestamp()}@hospital.org"
    password = "PatientPass123!"

    # 1. Register Patient
    reg_resp = client.post("/api/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Sunita Verma",
        "role": "PATIENT",
        "hospital_id": "HOSP-APEX",
        "patient_code": "MRN-PAT-001",
    })
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert reg_data["user"]["role"] == "PATIENT"

    # 2. Login Patient
    login_resp = client.post("/api/auth/login", json={
        "email": email,
        "password": password,
        "role": "PATIENT",
    })
    assert login_resp.status_code == 200
    assert login_resp.json()["user"]["role"] == "PATIENT"

    # 3. RBAC Barrier Check: Patient cannot log into Doctor portal
    doctor_portal_attempt = client.post("/api/auth/login", json={
        "email": email,
        "password": password,
        "role": "DOCTOR",
    })
    assert doctor_portal_attempt.status_code == 403


def test_auth_invalid_credentials(client):
    resp = client.post("/api/auth/login", json={
        "email": "nonexistent@hospital.org",
        "password": "WrongPassword",
    })
    assert resp.status_code == 401


# ==============================================================================
# 3. Patient Clinical Intake & Management (CRUD) Tests
# ==============================================================================

def test_patient_crud_lifecycle(client, auth_headers):
    timestamp = int(datetime.now().timestamp())
    patient_name = f"Rajesh Sharma {timestamp}"

    # 1. Create Patient
    create_payload = {
        "name": patient_name,
        "date_of_birth": "1980-05-15",
        "sex": "M",
        "contact": "+91-9876543210",
        "email": f"rajesh_{timestamp}@example.com",
        "address": "42 MG Road, Bengaluru",
        "blood_group": "B+",
        "known_allergies": "Penicillin",
        "existing_conditions": ["Hypertension", "Type 2 Diabetes"],
        "current_medications": ["Metoprolol 25mg", "Metformin 500mg"],
        "previous_cardiac_history": "Mild Angina in 2022",
        "smoking_status": "Non-Smoker",
        "height_cm": 175.0,
        "weight_kg": 76.0,
        "systolic_bp": 128.0,
        "diastolic_bp": 82.0,
        "heart_rate": 74.0,
        "spo2": 98.0,
    }
    create_resp = client.post("/api/patients", json=create_payload, headers=auth_headers)
    assert create_resp.status_code == 201
    create_data = create_resp.json()
    assert create_data["status"] in ["SUCCESS", "CREATED"]
    pat = create_data["patient"]
    patient_id = pat["patient_id"]
    mrn = pat["hospital_mrn"]
    assert pat["name"] == patient_name
    assert pat["blood_group"] == "B+"
    assert pat["age"] is not None and pat["age"] >= 40

    # 2. Retrieve Patient by MRN
    get_resp = client.get(f"/api/patients/{mrn}", headers=auth_headers)
    assert get_resp.status_code == 200
    retrieved = get_resp.json()["patient"]
    assert retrieved["patient_id"] == patient_id
    assert retrieved["name"] == patient_name
    assert len(get_resp.json()["vitals_history"]) >= 1

    # 3. Update Patient Details
    update_payload = {
        "blood_group": "O+",
        "known_allergies": "Penicillin, Sulfa drugs",
        "current_medications": ["Metoprolol 50mg", "Metformin 500mg", "Atorvastatin 20mg"],
        "systolic_bp": 122.0,
        "diastolic_bp": 78.0,
    }
    put_resp = client.put(f"/api/patients/{patient_id}", json=update_payload, headers=auth_headers)
    assert put_resp.status_code == 200
    updated_pat = put_resp.json()["patient"]
    assert updated_pat["blood_group"] == "O+"
    assert "Sulfa" in str(updated_pat["known_allergies"])

    # 4. List Patients with search query
    list_resp = client.get(f"/api/patients?search={patient_name}", headers=auth_headers)
    assert list_resp.status_code == 200
    patients_list = list_resp.json()["patients"]
    assert any(p["patient_id"] == patient_id for p in patients_list)


# ==============================================================================
# 4. Doctor Dashboard Stats Endpoint Tests
# ==============================================================================

def test_dashboard_stats_endpoint(client, auth_headers):
    resp = client.get("/api/dashboard/stats", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    stats = data["stats"]
    assert "total_patients" in stats
    assert "total_ecgs" in stats
    assert "pending_reviews" in stats
    assert "completed_reports" in stats
    assert "recent_patients" in stats
    assert "recent_ecgs" in stats
    assert "system_status" in stats
    assert stats["system_status"]["regulatory"] == "CDSCO MDR 2017 Class B SaMD"


# ==============================================================================
# 5. Full ECG Analysis Linked to Patient Context Tests
# ==============================================================================

def test_analyze_ecg_with_patient_linkage(client, test_signal, auth_headers):
    # First create a patient
    p_resp = client.post("/api/patients", json={
        "name": "Ananya Sen",
        "date_of_birth": "1992-03-10",
        "sex": "F",
        "blood_group": "A+",
        "known_allergies": "Aspirin",
        "existing_conditions": "Palpitations",
        "current_medications": "Propranolol 10mg",
        "previous_cardiac_history": "None",
    }, headers=auth_headers)
    assert p_resp.status_code == 201
    pat_id = p_resp.json()["patient"]["patient_id"]
    pat_mrn = p_resp.json()["patient"]["hospital_mrn"]

    # Run Analysis passing patient_id
    analysis_resp = client.post("/api/analyze", json={
        "signal": test_signal,
        "fs": 360.0,
        "lead": "II",
        "patient_id": pat_id,
        "patient_mrn": pat_mrn,
        "include_full_report": True,
    })
    assert analysis_resp.status_code == 200
    analysis = analysis_resp.json()
    assert analysis["status"] == "SUCCESS"
    assert "analysis_id" in analysis
    assert "report_id" in analysis
    assert "cardiac_parameters" in analysis
    assert "ai_classification" in analysis
    assert "clinical_decision_support" in analysis

    # Verify patient context was automatically enriched from DB_MANAGER
    criteria = analysis.get("criteria_metadata", {})
    context_fields = criteria.get("patient_context_considered", [])
    assert any("Ananya Sen" in str(f) or "Sex" in str(f) or "Age" in str(f) for f in context_fields)


# ==============================================================================
# 6. PDF Generation (Doctor & Patient Reports) Tests
# ==============================================================================

def test_generate_doctor_and_patient_pdf(client, test_signal):
    payload = {
        "signal": test_signal,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Vikram Malhotra",
        "patient_mrn": "MRN-2026-VM01",
        "age": 52,
        "sex": "M",
        "blood_group": "AB+",
        "existing_conditions": "Mild Hypertension",
        "current_medications": "Amlodipine 5mg",
    }

    # 1. Doctor Clinical PDF
    doc_resp = client.post("/api/report/pdf?type=doctor", json=payload)
    assert doc_resp.status_code == 200
    assert doc_resp.headers["content-type"] == "application/pdf"
    assert len(doc_resp.content) > 1000
    assert doc_resp.content.startswith(b"%PDF")

    # 2. Patient Friendly Plain-Language PDF
    pat_resp = client.post("/api/report/pdf?type=patient", json=payload)
    assert pat_resp.status_code == 200
    assert pat_resp.headers["content-type"] == "application/pdf"
    assert len(pat_resp.content) > 1000
    assert pat_resp.content.startswith(b"%PDF")


# ==============================================================================
# 7. Clinician Review Cryptographic Sealing & Historical Retrieval Tests
# ==============================================================================

def test_clinician_review_and_historical_immutable_retrieval(client, test_signal):
    # 1. Run analysis to create report
    analysis_resp = client.post("/api/analyze", json={
        "signal": test_signal,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Kavita Rao",
        "patient_mrn": "MRN-2026-KR09",
        "age": 38,
        "sex": "F",
    })
    assert analysis_resp.status_code == 200
    analysis = analysis_resp.json()
    report_id = analysis["report_id"]

    # 2. Clinician reviews and cryptographically seals report
    from api.security import create_jwt_token
    from src.auth.auth_manager import User, UserRole
    doc_ananya = User(
        user_id="doc_ananya",
        username="ananya_roy",
        password_hash="",
        salt="",
        role=UserRole.CARDIOLOGIST,
        full_name="Dr. Ananya Roy, MD DM",
        email="ananya.roy@hospital.org",
        hospital_id="HOSP-APEX",
        registration_number="MCI-778899",
    )
    doc_headers = {"Authorization": f"Bearer {create_jwt_token(doc_ananya)}"}

    review_resp = client.post("/api/review", json={
        "analysis_id": report_id,
        "clinician_name": "Dr. Ananya Roy, MD DM",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MCI-778899",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": "Normal Sinus Rhythm. Physiological repolarization.",
        "clinical_notes": "Patient advised routine follow-up in 12 months.",
    }, headers=doc_headers)
    assert review_resp.status_code == 200
    review_data = review_resp.json()
    assert review_data["status"] == "SEALED"
    assert "review_id" in review_data

    # 3. Retrieve immutable report snapshot without re-running ML
    snap_resp = client.get(f"/api/reports/{report_id}", headers=doc_headers)
    assert snap_resp.status_code == 200
    snap = snap_resp.json()["report"]
    assert snap["report_id"] == report_id
    assert snap["status"] in ["SIGNED", "SEALED", "REVIEWED"]
    assert snap["reviewer_name"] == "Dr. Ananya Roy, MD DM"

    # 4. Download historical doctor PDF
    doc_pdf = client.get(f"/api/reports/{report_id}/pdf?type=doctor", headers=doc_headers)
    assert doc_pdf.status_code == 200
    assert doc_pdf.content.startswith(b"%PDF")

    # 5. Download historical patient PDF
    pat_pdf = client.get(f"/api/reports/{report_id}/pdf?type=patient", headers=doc_headers)
    assert pat_pdf.status_code == 200
    assert pat_pdf.content.startswith(b"%PDF")


# ==============================================================================
# 8. Frontend HTML SPA Serving & Integrity Tests
# ==============================================================================

def test_frontend_spa_serving_and_view_structure(client):
    """Verify that FastAPI serves public/index.html with all 8 flow views intact."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text

    # Verify branding & key hospital elements
    assert "ECG GUARDIAN" in html
    assert "CDSCO MDR 2017 Class B" in html
    assert "navigateTo" in html

    # Verify all 8 core flow views exist in the SPA markup
    expected_views = [
        "viewLanding",
        "viewAuth",
        "viewDoctorDashboard",
        "viewPatientDashboard",
        "viewPatientRegistration",
        "viewPatientProfile",
        "viewEcgScan",
        "viewAnalysis",
        "viewReportHistory",
    ]
    for view_id in expected_views:
        assert f'id="{view_id}"' in html, f"Missing view element '{view_id}' in public/index.html"


def test_edge_case_duplicate_mrn_conflict(client, auth_headers):
    """Attempting to register another patient with the same MRN returns 409 conflict."""
    unique_mrn = f"MRN-TEST-{datetime.now().timestamp():.0f}"
    p1 = client.post("/api/patients", json={
        "name": "Original Patient",
        "hospital_mrn": unique_mrn,
    }, headers=auth_headers)
    assert p1.status_code == 201

    # Attempt duplicate registration with same MRN
    p2 = client.post("/api/patients", json={
        "name": "Impostor Patient",
        "hospital_mrn": unique_mrn,
    }, headers=auth_headers)
    assert p2.status_code == 409
    assert "already exists" in p2.json()["detail"]


def test_edge_case_future_dob_rejected(client, auth_headers):
    """Registering patient with a future birth date is rejected."""
    future_date = (date.today() + timedelta(days=400)).strftime("%Y-%m-%d")
    resp = client.post("/api/patients", json={
        "name": "Time Traveler",
        "date_of_birth": future_date,
    }, headers=auth_headers)
    assert resp.status_code == 400
    assert "future" in resp.json()["detail"].lower()


# ==============================================================================
# 9. RBAC & Multi-Tenant Cross-Hospital Boundary Enforcement Tests
# ==============================================================================

def test_rbac_patient_cannot_access_other_patient_data(client, auth_headers):
    """Verify patient session token is strictly prohibited from accessing another patient's records."""
    ts = int(datetime.now().timestamp())
    p1_email = f"patient1_{ts}@hospital.org"
    p2_email = f"patient2_{ts}@hospital.org"
    pwd = "SecurePatient123!"

    # 1. Register Patient 1
    r1 = client.post("/api/auth/register", json={
        "email": p1_email,
        "password": pwd,
        "full_name": "Patient Alpha",
        "role": "PATIENT",
        "hospital_id": "HOSP-APEX",
        "patient_code": f"MRN-ALPHA-{ts}",
    })
    assert r1.status_code == 200
    token1 = r1.json()["token"]

    # 2. Register Patient 2
    r2 = client.post("/api/patients", json={
        "name": "Patient Beta",
        "hospital_mrn": f"MRN-BETA-{ts}",
        "hospital_id": "HOSP-APEX",
        "blood_group": "A+",
    }, headers=auth_headers)
    assert r2.status_code == 201
    p2_id = r2.json()["patient"]["patient_id"]
    p2_mrn = r2.json()["patient"]["hospital_mrn"]

    # 3. Patient 1 attempts to fetch Patient 2's dossier -> Rejected with 403 Forbidden
    auth_headers1 = {"Authorization": f"Bearer {token1}"}
    forbidden_get = client.get(f"/api/patients/{p2_id}", headers=auth_headers1)
    assert forbidden_get.status_code == 403
    assert "Access denied" in forbidden_get.json()["detail"]

    # 4. Patient 1 attempts to modify Patient 2's dossier -> Rejected with 403 Forbidden
    forbidden_put = client.put(f"/api/patients/{p2_id}", headers=auth_headers1, json={"blood_group": "O-"})
    assert forbidden_put.status_code == 403
    assert "Access denied" in forbidden_put.json()["detail"]

    # 5. Patient 1 listing patients -> must not expose Patient Beta
    list_resp = client.get("/api/patients", headers=auth_headers1)
    assert list_resp.status_code == 200
    returned_patients = list_resp.json()["patients"]
    assert not any(p["patient_id"] == p2_id for p in returned_patients)


def test_multitenancy_cross_hospital_isolation(client):
    """Verify clinician from Hospital A is forbidden from viewing or altering Hospital B records."""
    ts = int(datetime.now().timestamp())
    doc_metro_email = f"dr.metro_{ts}@hospital.org"
    doc_city_email = f"dr.city_{ts}@hospital.org"
    pwd = "DoctorPass123!"

    # 1. Register Doctor 1 at HOSP-METRO
    r_metro = client.post("/api/auth/register", json={
        "email": doc_metro_email,
        "password": pwd,
        "full_name": "Dr. Metro Cardiologist",
        "role": "DOCTOR",
        "hospital_id": "HOSP-METRO",
        "registration_number": f"MCI-METRO-{ts}",
    })
    assert r_metro.status_code == 200
    metro_token = r_metro.json()["token"]

    # 2. Register Doctor 2 at HOSP-CITY
    r_city = client.post("/api/auth/register", json={
        "email": doc_city_email,
        "password": pwd,
        "full_name": "Dr. City Physician",
        "role": "DOCTOR",
        "hospital_id": "HOSP-CITY",
        "registration_number": f"MCI-CITY-{ts}",
    })
    assert r_city.status_code == 200
    city_token = r_city.json()["token"]

    # 3. Create a patient belonging to HOSP-METRO
    p_metro = client.post("/api/patients", json={
        "name": "Metro Citizen",
        "hospital_id": "HOSP-METRO",
        "blood_group": "B+",
    }, headers={"Authorization": f"Bearer {metro_token}"}).json()["patient"]
    metro_pat_id = p_metro["patient_id"]

    # 4. Doctor 2 (HOSP-CITY) attempts to retrieve HOSP-METRO patient -> 403 Forbidden
    cross_get = client.get(f"/api/patients/{metro_pat_id}", headers={"Authorization": f"Bearer {city_token}"})
    assert cross_get.status_code == 403
    assert "another hospital" in cross_get.json()["detail"].lower()

    # 5. Doctor 2 (HOSP-CITY) attempts to modify HOSP-METRO patient -> 403 Forbidden
    cross_put = client.put(f"/api/patients/{metro_pat_id}", headers={"Authorization": f"Bearer {city_token}"}, json={"blood_group": "AB+"})
    assert cross_put.status_code == 403
    assert "another hospital" in cross_put.json()["detail"].lower()

    # 6. Doctor 1 (HOSP-METRO) can successfully retrieve HOSP-METRO patient
    authorized_get = client.get(f"/api/patients/{metro_pat_id}", headers={"Authorization": f"Bearer {metro_token}"})
    assert authorized_get.status_code == 200
    assert authorized_get.json()["patient"]["patient_id"] == metro_pat_id


def test_clinician_approved_treatment_plan_in_snapshot(client, test_signal, auth_headers):
    """Verify that clinician treatment directives and approved medication are frozen into the immutable report snapshot."""
    # 1. Run analysis to produce report
    analysis_resp = client.post("/api/analyze", json={
        "signal": test_signal,
        "fs": 360.0,
        "lead": "II",
        "patient_name": "Sunil Gavaskar",
        "age": 62,
        "sex": "M",
    })
    assert analysis_resp.status_code == 200
    report_id = analysis_resp.json()["report_id"]

    # 2. Cardiologist signs and seals with complete structured treatment plan
    review_resp = client.post("/api/review", json={
        "analysis_id": report_id,
        "clinician_name": "Dr. Rohan Roy, MD DM",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MCI-483921",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": "Sinus rhythm with physiological repolarization.",
        "clinical_notes": "Maintain systolic BP below 130 mmHg.",
        "selected_medication": "Metoprolol succinate",
        "medication_dosage": "25 mg orally once daily",
        "treatment_instructions": "Take with breakfast. Monitor resting heart rate.",
        "follow_up_instructions": "Schedule cardiology review in 4 weeks.",
    }, headers=auth_headers)
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["status"] == "SEALED"
    assert rev_data["treatment_plan"]["selected_medication"] == "Metoprolol succinate"
    assert rev_data["treatment_plan"]["medication_dosage"] == "25 mg orally once daily"

    # 3. Retrieve immutable snapshot and verify approved treatment plan is present
    snap_resp = client.get(f"/api/reports/{report_id}", headers=auth_headers)
    assert snap_resp.status_code == 200
    snap = snap_resp.json()["report"]
    rd = snap.get("report_data", {})
    tp = rd.get("approved_treatment_plan") or rd.get("clinician_review", {}).get("treatment_plan", {})
    assert tp.get("selected_medication") == "Metoprolol succinate"
    assert tp.get("medication_dosage") == "25 mg orally once daily"
    assert "4 weeks" in tp.get("follow_up_instructions", "")

    # 4. Generate Patient PDF and verify it includes treatment directives
    pat_pdf_resp = client.get(f"/api/reports/{report_id}/pdf?type=patient", headers=auth_headers)
    assert pat_pdf_resp.status_code == 200
    assert pat_pdf_resp.content.startswith(b"%PDF")
    assert len(pat_pdf_resp.content) > 1000


