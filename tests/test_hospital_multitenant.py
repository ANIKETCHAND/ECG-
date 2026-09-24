"""
Unit Tests for Multi-Tenant Hospital Isolation & Management (Phases 2-5)
"""

import tempfile
from pathlib import Path
import pytest
from src.database.db_manager import DatabaseManager
from src.auth.auth_manager import AuthManager, UserRole


def test_hospital_registration_and_retrieval():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = DatabaseManager(db_path=str(Path(tmpdir) / "test.db"))

        hosp = db.create_hospital(
            hospital_id="HOSP-METRO",
            hospital_name="Metro Cardiac Institute",
            registration_number="REG-DEL-2026-001",
            city="New Delhi",
            state="Delhi",
            country="India",
            email="admin@metrocardiac.in",
            phone="+91-11-4000-5000",
        )
        assert hosp["hospital_id"] == "HOSP-METRO"
        assert hosp["hospital_name"] == "Metro Cardiac Institute"
        assert hosp["status"] == "ACTIVE"

        fetched = db.get_hospital("HOSP-METRO")
        assert fetched is not None
        assert fetched["city"] == "New Delhi"

        # Update hospital
        ok = db.update_hospital("HOSP-METRO", phone="+91-11-4000-5099")
        assert ok
        updated = db.get_hospital("HOSP-METRO")
        assert updated["phone"] == "+91-11-4000-5099"


def test_doctor_and_staff_management():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = DatabaseManager(db_path=str(Path(tmpdir) / "test.db"))

        db.create_hospital(
            hospital_id="HOSP-CITY",
            hospital_name="City Heart Hospital",
            registration_number="REG-BLR-2026-088",
        )

        doc = db.create_doctor(
            doctor_id="DOC-901",
            hospital_id="HOSP-CITY",
            name="Dr. Sunita Rao",
            email="sunita.rao@cityheart.in",
            specialization="Interventional Cardiology",
            license_number="KMC-DOC-44120",
            department="Cardiology",
            role="CARDIOLOGIST",
        )
        assert doc["doctor_id"] == "DOC-901"
        assert doc["license_number"] == "KMC-DOC-44120"

        docs = db.list_doctors(hospital_id="HOSP-CITY")
        assert len(docs) == 1
        assert docs[0]["name"] == "Dr. Sunita Rao"

        # Staff creation
        staff = db.create_staff(
            staff_id="STF-101",
            hospital_id="HOSP-CITY",
            name="Nurse Kavita",
            email="kavita@cityheart.in",
            role="NURSE",
            department="CCU",
        )
        assert staff["staff_id"] == "STF-101"

        all_staff = db.list_staff(hospital_id="HOSP-CITY")
        assert len(all_staff) == 1


def test_strict_multi_tenant_isolation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = DatabaseManager(db_path=str(Path(tmpdir) / "test.db"))

        # Create two distinct hospitals
        db.create_hospital(hospital_id="HOSP-A", hospital_name="Hospital Alpha", registration_number="REG-A")
        db.create_hospital(hospital_id="HOSP-B", hospital_name="Hospital Beta", registration_number="REG-B")

        # Register patient in Hospital A
        p_a = db.create_patient(
            patient_id="PAT-ALPHA-1",
            hospital_mrn="MRN-A1",
            hospital_id="HOSP-A",
            name="Patient In Alpha",
            known_allergies="Penicillin",
        )

        # Register patient in Hospital B
        p_b = db.create_patient(
            patient_id="PAT-BETA-1",
            hospital_mrn="MRN-B1",
            hospital_id="HOSP-B",
            name="Patient In Beta",
            known_allergies="None",
        )

        # 1. Hospital A cannot see Hospital B patients
        pats_a = db.list_patients(hospital_id="HOSP-A")
        assert len(pats_a) == 1
        assert pats_a[0].name == "Patient In Alpha"
        assert pats_a[0].patient_id == "PAT-ALPHA-1"

        pats_b = db.list_patients(hospital_id="HOSP-B")
        assert len(pats_b) == 1
        assert pats_b[0].name == "Patient In Beta"

        # 2. Hospital A cannot fetch Hospital B patient directly
        cross_fetch = db.get_patient("PAT-BETA-1", hospital_id="HOSP-A")
        assert cross_fetch is None, "Security violation: Hospital A accessed Hospital B patient!"

        # 3. Save ECG in Hospital A
        db.save_ecg_record(
            record_id="REC-A-01",
            sampling_rate=360.0,
            lead_names=["II"],
            duration_sec=10.0,
            file_hash="hash_a",
            source_format="CSV",
            hospital_id="HOSP-A",
            patient_id="PAT-ALPHA-1",
        )

        # Hospital B querying ECG records should see 0 records
        ecgs_b = db.list_ecg_records(hospital_id="HOSP-B")
        assert len(ecgs_b) == 0

        # Hospital B cannot fetch Hospital A ECG record
        cross_ecg = db.get_ecg_record("REC-A-01", hospital_id="HOSP-B")
        assert cross_ecg is None, "Security violation: Hospital B accessed Hospital A ECG record!"


def test_rbac_expanded_roles():
    auth = AuthManager()
    hosp_admin = auth.register_user(
        username="hadmin",
        password="AdminPass#2026",
        role=UserRole.HOSPITAL_ADMIN,
        full_name="Hospital Director",
        email="director@hospital.org",
        hospital_id="HOSP-A",
    )
    nurse = auth.register_user(
        username="ccu_nurse",
        password="NursePass#2026",
        role=UserRole.NURSE,
        full_name="Staff Nurse",
        email="nurse@hospital.org",
        hospital_id="HOSP-A",
    )
    patient = auth.register_user(
        username="pt_user",
        password="PatientPass#2026",
        role=UserRole.PATIENT,
        full_name="Patient User",
        email="patient@example.com",
        hospital_id="HOSP-A",
    )

    # Hospital Admin permissions
    assert auth.has_permission(hosp_admin, "hospital:manage")
    assert auth.has_permission(hosp_admin, "doctor:manage")
    assert auth.has_permission(hosp_admin, "audit:view_logs")

    # Nurse permissions
    assert auth.has_permission(nurse, "patient:create")
    assert auth.has_permission(nurse, "ecg:upload")
    assert not auth.has_permission(nurse, "hospital:manage")
    assert not auth.has_permission(nurse, "doctor:manage")

    # Patient permissions
    assert auth.has_permission(patient, "patient:view_self")
    assert not auth.has_permission(patient, "audit:view_logs")
    assert not auth.has_permission(patient, "doctor:manage")
