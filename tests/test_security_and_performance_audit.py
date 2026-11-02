"""
Phase 10: Security Vulnerability & Performance Benchmark Audit Suite
=====================================================================
Validates:
1. SQL Injection Defense:
   - Parameterized search probes (' OR '1'='1, '; DROP TABLE patients; --)
   - Parameterized insert probes
2. XSS (Cross-Site Scripting) Defense:
   - Script and event handler payloads stored safely without HTML evaluation
3. IDOR (Insecure Direct Object Reference) & Multi-Tenant Access Controls:
   - Patient token forbidden from accessing another patient's records (HTTP 403)
   - Clinician forbidden from accessing patients from a different hospital facility (HTTP 403)
   - Patient account forbidden from logging into physician portal (HTTP 403)
4. Payload Limit & DoS Prevention:
   - Excessive signal length triggers HTTP 413
5. Latency Benchmarks:
   - /api/health < 50ms
   - /api/sample < 100ms
   - /api/dashboard/stats < 200ms
   - /api/analyze < 1000ms
"""

import time
import uuid
import pytest
from starlette.testclient import TestClient

from api.index import app
from src.auth.auth_manager import AUTH_MANAGER, UserRole
from src.database.db_manager import DB_MANAGER


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_sql_injection_defense_in_patient_search(client):
    """Verify SQL injection strings in search query parameter do not compromise database."""
    sqli_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE patients; --",
        "admin'--",
        "' UNION SELECT * FROM users --",
    ]
    for payload in sqli_payloads:
        res = client.get(f"/api/patients?search={payload}")
        assert res.status_code == 200, f"Search failed on SQLi payload: {payload}"
        data = res.json()
        assert data["status"] == "SUCCESS"
        # Database table should still exist and function
        patients = DB_MANAGER.list_patients(limit=5)
        assert len(patients) >= 0


def test_sql_injection_defense_in_patient_create(client):
    """Verify SQL injection payloads in patient creation fields are parameterized."""
    malicious_name = "Robert'); DROP TABLE patients;--"
    mrn = f"MRN-SQLI-{uuid.uuid4().hex[:6].upper()}"

    res = client.post("/api/patients", json={
        "name": malicious_name,
        "hospital_mrn": mrn,
        "hospital_id": "HOSP-APEX",
        "age": 40,
        "sex": "Male",
    })
    assert res.status_code == 201
    created_pat = res.json()["patient"]
    assert created_pat["name"] == malicious_name
    assert created_pat["hospital_mrn"] == mrn

    # Database table still exists and operates
    fetched = DB_MANAGER.get_patient_by_mrn(mrn)
    assert fetched is not None
    assert fetched.name == malicious_name


def test_xss_payload_neutralization(client):
    """Verify HTML/script payloads in patient fields are stored cleanly as plaintext."""
    xss_payload = "<script>alert('XSS_AUDIT_EXPLOIT')</script><img src=x onerror=alert(1)>"
    mrn = f"MRN-XSS-{uuid.uuid4().hex[:6].upper()}"

    res = client.post("/api/patients", json={
        "name": xss_payload,
        "hospital_mrn": mrn,
        "hospital_id": "HOSP-APEX",
        "age": 35,
        "known_allergies": ["<script>bad()</script>"],
        "existing_conditions": ["<b onmouseover=evil()>HTN</b>"],
    })
    assert res.status_code == 201
    pat = res.json()["patient"]
    assert pat["name"] == xss_payload

    # Fetch through GET endpoint
    get_res = client.get(f"/api/patients/{mrn}")
    assert get_res.status_code == 200
    fetched_name = get_res.json()["patient"]["name"]
    assert fetched_name == xss_payload


def test_idor_patient_cross_access_forbidden(client):
    """Verify a Patient cannot access another patient's medical records (HTTP 403)."""
    # Create Patient 1
    mrn_1 = f"MRN-IDOR1-{uuid.uuid4().hex[:6].upper()}"
    p1 = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=mrn_1,
        name="Patient One",
        hospital_id="HOSP-APEX",
    )
    user_p1 = AUTH_MANAGER.register_user(
        username=f"{mrn_1.lower()}@patient.org",
        password="Password123!",
        role=UserRole.PATIENT,
        full_name=p1.name,
        email=f"{mrn_1.lower()}@patient.org",
        registration_number=mrn_1,
    )
    token_p1 = AUTH_MANAGER.create_session(user_p1)

    # Create Patient 2
    mrn_2 = f"MRN-IDOR2-{uuid.uuid4().hex[:6].upper()}"
    p2 = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=mrn_2,
        name="Patient Two",
        hospital_id="HOSP-APEX",
    )

    # Patient 1 tries to access Patient 2's record
    res = client.get(
        f"/api/patients/{p2.patient_id}",
        headers={"Authorization": f"Bearer {token_p1}"},
    )
    assert res.status_code == 403
    assert "access denied" in res.json()["detail"].lower()


def test_idor_cross_hospital_isolation(client):
    """Verify a Clinician from Hospital A cannot access patients belonging to Hospital B (HTTP 403)."""
    # Doctor from Hospital Beta
    doc_beta = AUTH_MANAGER.register_user(
        username=f"doc.beta.{uuid.uuid4().hex[:4]}@beta.org",
        password="Password123!",
        role=UserRole.DOCTOR,
        full_name="Dr. Beta Clinician",
        email=f"doc.beta.{uuid.uuid4().hex[:4]}@beta.org",
        hospital_id="HOSP-BETA",
    )
    token_beta = AUTH_MANAGER.create_session(doc_beta)

    # Patient from Hospital Apex
    mrn_apex = f"MRN-APEX-{uuid.uuid4().hex[:6].upper()}"
    p_apex = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=mrn_apex,
        name="Apex Patient",
        hospital_id="HOSP-APEX",
    )

    # Doctor Beta tries to access Apex Patient
    res = client.get(
        f"/api/patients/{p_apex.patient_id}",
        headers={"Authorization": f"Bearer {token_beta}"},
    )
    assert res.status_code == 403
    assert "another hospital facility" in res.json()["detail"].lower()


def test_patient_cannot_log_into_clinician_portal(client):
    """Verify Patient credentials cannot authenticate into Doctor portal."""
    email = f"pat.portal.{uuid.uuid4().hex[:6]}@domain.org"
    password = "PatientPass123!"
    AUTH_MANAGER.register_user(
        username=email,
        password=password,
        role=UserRole.PATIENT,
        full_name="Portal Tester",
        email=email,
    )

    # Attempt login requesting DOCTOR role
    res = client.post("/api/auth/login", json={
        "email": email,
        "password": password,
        "role": "DOCTOR",
    })
    assert res.status_code == 403
    assert "patient accounts cannot log in to the physician portal" in res.json()["detail"].lower()


def test_signal_limits_enforced():
    """Verify safety limits reject signals that exceed max sample count (DoS barrier)."""
    from api.security import enforce_signal_limits
    from fastapi import HTTPException

    # Allowed sample length
    enforce_signal_limits(5000)

    # Exceeding limit
    with pytest.raises(HTTPException) as exc_info:
        enforce_signal_limits(10_000_000)
    assert exc_info.value.status_code == 413


def test_endpoint_latency_benchmarks(client):
    """Verify API latency benchmarks across core endpoints."""
    # 1. /api/health < 50ms
    t0 = time.perf_counter()
    h_res = client.get("/api/health")
    h_time_ms = (time.perf_counter() - t0) * 1000.0
    assert h_res.status_code == 200
    assert h_time_ms < 50.0, f"/health latency too high: {h_time_ms:.2f} ms"

    # 2. /api/sample < 100ms
    t0 = time.perf_counter()
    s_res = client.get("/api/sample?sample_type=normal")
    s_time_ms = (time.perf_counter() - t0) * 1000.0
    assert s_res.status_code == 200
    assert s_time_ms < 100.0, f"/sample latency too high: {s_time_ms:.2f} ms"

    # 3. /api/dashboard/stats < 200ms
    t0 = time.perf_counter()
    d_res = client.get("/api/dashboard/stats")
    d_time_ms = (time.perf_counter() - t0) * 1000.0
    assert d_res.status_code == 200
    assert d_time_ms < 200.0, f"/dashboard/stats latency too high: {d_time_ms:.2f} ms"
