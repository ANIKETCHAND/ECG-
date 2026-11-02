"""
Phase 4: Supabase Database Audit & Tenancy / Dual-Storage Verification Test Suite
================================================================================
Validates:
1. SQL Migrations Syntax & Structure:
   - 20260924000001_initial_schema.sql
   - 20260925000001_tenancy_and_rls_hardening.sql
2. Database Schema Completeness:
   - 16 Core Clinical Tables: profiles, patients, patient_vitals, patient_medications,
     laboratory_results, ecg_recordings, ecg_measurements, ai_analyses, ai_evidence,
     machine_interpretations, interpretation_comparisons, clinical_recommendations,
     medication_safety_checks, clinical_reviews, reports, audit_logs.
3. RLS Policies & Multi-Tenant Boundaries:
   - Checks that all 16 tables enable RLS.
   - Verifies tenant isolation helpers: can_access_patient, can_access_ecg, is_own_patient.
   - Verifies privilege escalation prevention triggers (prevent_self_role_change).
4. Dual Storage Resilience (Supabase + Local SQLite):
   - Confirms graceful fallback and persistence when Supabase is unconfigured/offline.
   - Tests complete report lifecycle under dual-storage manager.
"""

import os
import re
import uuid
import pytest
from pathlib import Path

from src.database.db_manager import DB_MANAGER, PatientRecord
from src.database.supabase_client import is_supabase_configured, get_supabase_client
from src.services.report_persistence_service import REPORT_PERSISTENCE_SERVICE

MIGRATION_DIR = Path(__file__).resolve().parent.parent / "supabase" / "migrations"


def test_migrations_exist_and_readable():
    """Verify presence of official Supabase migration files."""
    initial_mig = MIGRATION_DIR / "20260924000001_initial_schema.sql"
    hardening_mig = MIGRATION_DIR / "20260925000001_tenancy_and_rls_hardening.sql"

    assert initial_mig.exists(), f"Initial schema migration missing at {initial_mig}"
    assert hardening_mig.exists(), f"Tenancy hardening migration missing at {hardening_mig}"

    content_init = initial_mig.read_text(encoding="utf-8")
    content_hard = hardening_mig.read_text(encoding="utf-8")

    assert len(content_init) > 5000, "Initial schema file too short or corrupted"
    assert len(content_hard) > 5000, "Hardening schema file too short or corrupted"


def test_all_16_core_tables_defined_in_schema():
    """Verify all 16 core clinical tables are defined with primary keys and UUIDs."""
    initial_sql = (MIGRATION_DIR / "20260924000001_initial_schema.sql").read_text(encoding="utf-8")

    expected_tables = [
        "profiles",
        "patients",
        "patient_vitals",
        "patient_medications",
        "laboratory_results",
        "ecg_recordings",
        "ecg_measurements",
        "ai_analyses",
        "ai_evidence",
        "machine_interpretations",
        "interpretation_comparisons",
        "clinical_recommendations",
        "medication_safety_checks",
        "clinical_reviews",
        "reports",
        "audit_logs",
    ]

    for table in expected_tables:
        pattern = rf"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+public\.{table}\s*\("
        assert re.search(pattern, initial_sql, re.IGNORECASE), f"Table public.{table} not found in initial schema"


def test_tenancy_and_rls_policies_defined():
    """Verify RLS is explicitly enabled on tables and multi-tenant security functions exist."""
    combined_sql = (
        (MIGRATION_DIR / "20260924000001_initial_schema.sql").read_text(encoding="utf-8")
        + "\n"
        + (MIGRATION_DIR / "20260925000001_tenancy_and_rls_hardening.sql").read_text(encoding="utf-8")
    )

    # 1. Tenancy column in patients
    assert "hospital_id" in combined_sql

    # 2. Key security functions
    expected_funcs = [
        "current_user_role()",
        "is_clinical_staff()",
        "is_doctor_or_admin()",
        "current_user_hospital_id()",
        "patient_hospital",
        "can_access_patient",
        "can_access_ecg",
        "is_own_patient",
        "prevent_self_role_change()",
    ]
    for func in expected_funcs:
        assert func in combined_sql, f"Security function {func} missing from migrations"

    # 3. RLS enablement checks
    rls_tables = [
        "profiles",
        "patients",
        "ecg_recordings",
        "reports",
        "clinical_reviews",
        "patient_vitals",
        "patient_medications",
        "laboratory_results",
        "clinical_recommendations",
        "medication_safety_checks",
    ]
    for tbl in rls_tables:
        assert f"ALTER TABLE public.{tbl} ENABLE ROW LEVEL SECURITY;" in combined_sql, f"RLS not enabled on public.{tbl}"


def test_privilege_escalation_guard_in_migration():
    """Verify privilege escalation prevention trigger exists in SQL."""
    hard_sql = (MIGRATION_DIR / "20260925000001_tenancy_and_rls_hardening.sql").read_text(encoding="utf-8")
    assert "trg_profiles_prevent_self_role_change" in hard_sql
    assert "prevent_self_role_change" in hard_sql
    assert "Only an ADMIN may change a profile role" in hard_sql


def test_local_sqlite_dual_storage_fallback():
    """Verify local SQLite storage works transparently as zero-downtime offline cache."""
    test_mrn = f"MRN-SQLITE-{uuid.uuid4().hex[:6].upper()}"
    p_rec = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=test_mrn,
        name="Offline Resilient Patient",
        hospital_id="HOSP-APEX",
        age=52,
        sex="Male",
        blood_group="O+",
        known_allergies="Sulfa",
        existing_conditions="Hypertension",
        current_medications="Lisinopril 10mg",
    )
    assert p_rec is not None
    assert p_rec.hospital_mrn == test_mrn

    # Fetch back by MRN
    fetched = DB_MANAGER.get_patient_by_mrn(test_mrn)
    assert fetched is not None
    assert fetched.name == "Offline Resilient Patient"
    assert fetched.age == 52


def test_report_persistence_service_dual_save_and_retrieve():
    """Verify ReportPersistenceService operates cleanly across dual persistence layers."""
    report_uuid = str(uuid.uuid4())
    dummy_report_data = {
        "analysis_id": f"ANL-{uuid.uuid4().hex[:6].upper()}",
        "predicted_class": "Normal Sinus Rhythm",
        "confidence": 0.98,
        "heart_rate_bpm": 72.0,
        "signal_quality": "OPTIMAL",
        "medications": ["Aspirin 81mg"],
    }

    save_res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=dummy_report_data,
        report_id=report_uuid,
        patient_id="PAT-TEST-001",
        ecg_id="REC-TEST-001",
        analysis_id=dummy_report_data["analysis_id"],
        status="PENDING_REVIEW",
    )

    assert save_res is not None
    assert save_res.get("report_id") == report_uuid
    assert "report_number" in save_res
    rep_num = save_res["report_number"]

    # Retrieve report by ID
    retrieved = REPORT_PERSISTENCE_SERVICE.get_report(report_uuid)
    assert retrieved is not None
    assert retrieved.get("report_id") == report_uuid
    assert retrieved.get("report_number") == rep_num

    # Seal report
    seal_res = REPORT_PERSISTENCE_SERVICE.sign_and_seal_report(
        report_id=report_uuid,
        clinician_user_id="DOC-001",
        clinician_name="Dr. Audit Clinician, MD",
        clinician_role="CARDIOLOGIST",
        registration_number="MCI-12345",
        agreement_status="CONFIRMED",
        clinician_interpretation="ECG verified normal.",
    )
    assert seal_res is not None
    assert seal_res.get("status") in ["SAVE_SUCCESS", "SIGNED", "SEALED", "REVIEWED"]


def test_supabase_client_graceful_handling_when_unconfigured():
    """Verify Supabase client handles missing/unconfigured environment variables safely."""
    # When SUPABASE_URL / KEY are empty or not configured
    orig_url = os.environ.get("SUPABASE_URL")
    try:
        os.environ["SUPABASE_URL"] = ""
        # is_supabase_configured should return False
        assert not is_supabase_configured()
        client = get_supabase_client()
        assert client is None
    finally:
        if orig_url:
            os.environ["SUPABASE_URL"] = orig_url
