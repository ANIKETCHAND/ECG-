"""
Hospital Database Management System — Multi-Tenant Architecture
==============================================================

Provides clinical persistence layer for:
- Hospitals (multi-tenant onboarding, licensing, settings)
- Doctors & Technicians & Staff (role-based verified registry)
- Patients (clinical demographics, hospital MRN, conditions, allergies, medications)
- ECG Records (ingestion metadata, file hash, signal quality, workflow state machine)
- AI Analysis Runs (predictions, probabilities, interval measurements)
- Clinician Reviews (diagnostic sign-off, overrides, registration number)
- Sealed Reports (PDF paths, SHA-256 validation)

Compliance References:
- CDSCO Medical Devices Rules, 2017
- IEC 62304 Section 5.3 (Software Architecture)
- ISO 27799 / Health Data Privacy
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class PatientRecord:
    patient_id: str
    hospital_mrn: str
    name: str
    hospital_id: str = "HOSP-APEX"
    age: Optional[int] = None
    sex: Optional[str] = None  # M, F, O
    contact: Optional[str] = None
    created_at: str = ""
    date_of_birth: Optional[str] = None
    emergency_contact: Optional[str] = None
    blood_group: Optional[str] = None
    rh_factor: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    known_allergies: Optional[str] = None
    food_allergies: Optional[str] = None
    other_allergies: Optional[str] = None
    existing_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    past_medications: Optional[str] = None
    discontinued_medications: Optional[str] = None
    previous_cardiac_history: Optional[str] = None
    family_history: Optional[str] = None
    previous_surgeries: Optional[str] = None
    previous_hospitalizations: Optional[str] = None
    smoking_status: Optional[str] = None
    other_relevant_history: Optional[str] = None
    past_medical_history: Optional[str] = None
    other_clinical_information: Optional[str] = None
    diabetes: Optional[str] = None
    hypertension: Optional[str] = None
    coronary_artery_disease: Optional[str] = None
    previous_heart_attack: Optional[str] = None
    heart_failure: Optional[str] = None
    arrhythmia_history: Optional[str] = None
    kidney_disease: Optional[str] = None
    liver_disease: Optional[str] = None
    thyroid_disorders: Optional[str] = None
    other_diagnosed_diseases: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatabaseManager:
    """Thread-safe SQLite clinical database manager with multi-tenant data isolation."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
                data_dir = Path("/tmp") / "data"
            else:
                base_dir = Path(__file__).resolve().parent.parent.parent
                data_dir = base_dir / "data"
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                data_dir = Path("/tmp") / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "hospital_clinical.db")
        else:
            self.db_path = db_path
            try:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                self.db_path = str(Path("/tmp") / "data" / Path(self.db_path).name)
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 30000")
        except Exception:
            pass
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        with self._lock:
            with self._db() as conn:
                cursor = conn.cursor()

                # 1. Hospitals Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS hospitals (
                    hospital_id TEXT PRIMARY KEY,
                    hospital_name TEXT NOT NULL,
                    registration_number TEXT NOT NULL,
                    address TEXT,
                    city TEXT,
                    state TEXT,
                    country TEXT,
                    contact TEXT,
                    email TEXT,
                    phone TEXT,
                    website TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE'
                )
                """)

                # 2. Doctors Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS doctors (
                    doctor_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL,
                    user_id TEXT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    specialization TEXT,
                    license_number TEXT NOT NULL,
                    department TEXT,
                    role TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE
                )
                """)

                # 3. Staff Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS staff (
                    staff_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL,
                    user_id TEXT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE
                )
                """)

                # 4. Patients Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    hospital_mrn TEXT UNIQUE NOT NULL,
                    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
                    name TEXT NOT NULL,
                    age INTEGER,
                    sex TEXT,
                    contact TEXT,
                    created_at TEXT NOT NULL,
                    date_of_birth TEXT,
                    emergency_contact TEXT,
                    blood_group TEXT,
                    known_allergies TEXT,
                    existing_conditions TEXT,
                    current_medications TEXT,
                    previous_cardiac_history TEXT,
                    family_history TEXT,
                    smoking_status TEXT,
                    other_relevant_history TEXT,
                    past_medical_history TEXT,
                    other_clinical_information TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE
                )
                """)

                # Handle incremental migrations for patients table if columns missing
                cursor.execute("PRAGMA table_info(patients)")
                p_cols = [row["name"] for row in cursor.fetchall()]
                if "hospital_id" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX'")
                if "past_medical_history" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN past_medical_history TEXT")
                if "other_clinical_information" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN other_clinical_information TEXT")
                if "rh_factor" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN rh_factor TEXT")
                if "height_cm" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN height_cm REAL")
                if "weight_kg" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN weight_kg REAL")
                if "bmi" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN bmi REAL")
                if "previous_surgeries" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN previous_surgeries TEXT")
                if "previous_hospitalizations" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN previous_hospitalizations TEXT")
                if "past_medications" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN past_medications TEXT")
                if "discontinued_medications" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN discontinued_medications TEXT")
                if "food_allergies" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN food_allergies TEXT")
                if "other_allergies" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN other_allergies TEXT")
                if "diabetes" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN diabetes TEXT")
                if "hypertension" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN hypertension TEXT")
                if "coronary_artery_disease" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN coronary_artery_disease TEXT")
                if "previous_heart_attack" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN previous_heart_attack TEXT")
                if "heart_failure" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN heart_failure TEXT")
                if "arrhythmia_history" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN arrhythmia_history TEXT")
                if "kidney_disease" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN kidney_disease TEXT")
                if "liver_disease" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN liver_disease TEXT")
                if "thyroid_disorders" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN thyroid_disorders TEXT")
                if "other_diagnosed_diseases" not in p_cols:
                    cursor.execute("ALTER TABLE patients ADD COLUMN other_diagnosed_diseases TEXT")

                # 5. ECG Records Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ecg_records (
                    record_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
                    patient_id TEXT,
                    device TEXT,
                    sampling_rate REAL NOT NULL,
                    lead_names TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    file_hash TEXT NOT NULL,
                    source_format TEXT NOT NULL,
                    signal_quality TEXT,
                    quality_score REAL,
                    uploaded_by TEXT,
                    uploaded_at TEXT NOT NULL,
                    raw_data_path TEXT,
                    workflow_status TEXT DEFAULT 'UPLOADED',
                    priority TEXT DEFAULT 'ROUTINE',
                    assigned_doctor TEXT,
                    FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE SET NULL
                )
                """)

                cursor.execute("PRAGMA table_info(ecg_records)")
                e_cols = [row["name"] for row in cursor.fetchall()]
                if "hospital_id" not in e_cols:
                    cursor.execute("ALTER TABLE ecg_records ADD COLUMN hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX'")

                # 6. Analysis Results Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    analysis_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    preprocessing_version TEXT,
                    lead_analyzed TEXT,
                    signal_quality TEXT,
                    quality_score REAL,
                    prediction TEXT NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    heart_rate_bpm REAL,
                    mean_rr_ms REAL,
                    detected_beats_count INTEGER,
                    warnings_json TEXT,
                    limitations_json TEXT,
                    processing_time_ms REAL,
                    analyzed_at TEXT NOT NULL,
                    FOREIGN KEY (record_id) REFERENCES ecg_records(record_id) ON DELETE CASCADE
                )
                """)

                # 7. Clinician Reviews Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS clinician_reviews (
                    review_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    clinician_id TEXT NOT NULL,
                    clinician_name TEXT NOT NULL,
                    clinician_role TEXT NOT NULL,
                    registration_number TEXT,
                    agreement_status TEXT NOT NULL,
                    clinician_interpretation TEXT NOT NULL,
                    clinical_notes TEXT,
                    reviewed_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_results(analysis_id) ON DELETE CASCADE,
                    FOREIGN KEY (record_id) REFERENCES ecg_records(record_id) ON DELETE CASCADE
                )
                """)

                cursor.execute("PRAGMA table_info(clinician_reviews)")
                crv_cols = [row["name"] for row in cursor.fetchall()]
                if "clinician_diagnosis" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN clinician_diagnosis TEXT")
                if "selected_medication" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN selected_medication TEXT")
                if "medication_dosage" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN medication_dosage TEXT")
                if "treatment_instructions" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN treatment_instructions TEXT")
                if "follow_up_instructions" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN follow_up_instructions TEXT")
                if "prescriptions_json" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN prescriptions_json TEXT")
                if "rejected_candidates_json" not in crv_cols:
                    cursor.execute("ALTER TABLE clinician_reviews ADD COLUMN rejected_candidates_json TEXT")

                # 8. Clinical Reports Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS clinical_reports (
                    report_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    review_id TEXT,
                    report_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    file_path TEXT,
                    report_sha256 TEXT,
                    generated_at TEXT NOT NULL,
                    FOREIGN KEY (record_id) REFERENCES ecg_records(record_id) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_results(analysis_id) ON DELETE CASCADE,
                    FOREIGN KEY (review_id) REFERENCES clinician_reviews(review_id) ON DELETE SET NULL
                )
                """)

                cursor.execute("PRAGMA table_info(clinical_reports)")
                cr_cols = [row["name"] for row in cursor.fetchall()]
                if "report_number" not in cr_cols:
                    cursor.execute("ALTER TABLE clinical_reports ADD COLUMN report_number TEXT")
                if "report_data" not in cr_cols:
                    cursor.execute("ALTER TABLE clinical_reports ADD COLUMN report_data TEXT")
                if "pdf_storage_path" not in cr_cols:
                    cursor.execute("ALTER TABLE clinical_reports ADD COLUMN pdf_storage_path TEXT")
                if "report_version" not in cr_cols:
                    cursor.execute("ALTER TABLE clinical_reports ADD COLUMN report_version INTEGER DEFAULT 1")


                # 9. Vital Signs Table (Phase 2 & 3: Time-Aware Observations)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS vital_signs (
                    vital_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    heart_rate_bpm REAL,
                    systolic_bp_mmhg REAL,
                    diastolic_bp_mmhg REAL,
                    spo2_percent REAL,
                    respiratory_rate_bpm REAL,
                    temperature_celsius REAL,
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'VITAL-SIGN',
                    entered_by TEXT DEFAULT 'Triage Staff',
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )
                """)

                # 10. Laboratory Results Table (Phase 2 & 3)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS laboratory_results (
                    lab_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    panel_name TEXT NOT NULL DEFAULT 'Cardiac & Metabolic Panel',
                    potassium_mmol_l REAL,
                    magnesium_mg_dl REAL,
                    serum_creatinine_mg_dl REAL,
                    egfr_ml_min REAL,
                    troponin_i_ng_ml REAL,
                    troponin_t_ng_ml REAL,
                    bnp_pg_ml REAL,
                    hemoglobin_g_dl REAL,
                    additional_json TEXT,
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'LABORATORY',
                    entered_by TEXT DEFAULT 'Hospital Laboratory',
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )
                """)

                # 11. Patient Symptoms Table (Phase 2 & 3)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_symptoms (
                    symptom_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    primary_symptom TEXT NOT NULL,
                    onset_timestamp TEXT,
                    duration_hours REAL,
                    severity TEXT NOT NULL DEFAULT 'MODERATE',
                    character TEXT,
                    associated_symptoms_json TEXT,
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'PATIENT-HISTORY',
                    entered_by TEXT DEFAULT 'Attending Physician',
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )
                """)

                # 12. Patient Allergies Table (Phase 2 & 3)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_allergies (
                    allergy_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    allergen TEXT NOT NULL,
                    reaction TEXT,
                    severity TEXT NOT NULL DEFAULT 'MODERATE',
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'PATIENT-HISTORY',
                    entered_by TEXT DEFAULT 'Clinical Staff',
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )
                """)

                # 13. Patient Medications Table (Phase 2 & 3)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_medications (
                    medication_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    drug_name TEXT NOT NULL,
                    generic_name TEXT,
                    dosage TEXT,
                    unit TEXT,
                    frequency TEXT,
                    route TEXT DEFAULT 'Oral',
                    start_date TEXT,
                    end_date TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    indication TEXT,
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'MEDICATION-RECORD',
                    entered_by TEXT DEFAULT 'Physician / Pharmacy',
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )
                """)

                # Incremental migrations for vitals, labs, allergies, and medications
                cursor.execute("PRAGMA table_info(vital_signs)")
                vs_cols = [row["name"] for row in cursor.fetchall()]
                if "height_cm" not in vs_cols:
                    cursor.execute("ALTER TABLE vital_signs ADD COLUMN height_cm REAL")
                if "weight_kg" not in vs_cols:
                    cursor.execute("ALTER TABLE vital_signs ADD COLUMN weight_kg REAL")
                if "bmi" not in vs_cols:
                    cursor.execute("ALTER TABLE vital_signs ADD COLUMN bmi REAL")

                cursor.execute("PRAGMA table_info(laboratory_results)")
                lr_cols = [row["name"] for row in cursor.fetchall()]
                if "sodium_mmol_l" not in lr_cols:
                    cursor.execute("ALTER TABLE laboratory_results ADD COLUMN sodium_mmol_l REAL")
                if "alt_u_l" not in lr_cols:
                    cursor.execute("ALTER TABLE laboratory_results ADD COLUMN alt_u_l REAL")
                if "ast_u_l" not in lr_cols:
                    cursor.execute("ALTER TABLE laboratory_results ADD COLUMN ast_u_l REAL")
                if "bilirubin_mg_dl" not in lr_cols:
                    cursor.execute("ALTER TABLE laboratory_results ADD COLUMN bilirubin_mg_dl REAL")
                if "alk_phos_u_l" not in lr_cols:
                    cursor.execute("ALTER TABLE laboratory_results ADD COLUMN alk_phos_u_l REAL")
                if "lft_summary" not in lr_cols:
                    cursor.execute("ALTER TABLE laboratory_results ADD COLUMN lft_summary TEXT")

                cursor.execute("PRAGMA table_info(patient_allergies)")
                pa_cols = [row["name"] for row in cursor.fetchall()]
                if "allergy_type" not in pa_cols:
                    cursor.execute("ALTER TABLE patient_allergies ADD COLUMN allergy_type TEXT DEFAULT 'DRUG'")
                if "previous_reactions" not in pa_cols:
                    cursor.execute("ALTER TABLE patient_allergies ADD COLUMN previous_reactions TEXT")

                cursor.execute("PRAGMA table_info(patient_medications)")
                pm_cols = [row["name"] for row in cursor.fetchall()]
                if "brand_name" not in pm_cols:
                    cursor.execute("ALTER TABLE patient_medications ADD COLUMN brand_name TEXT")
                if "status" not in pm_cols:
                    cursor.execute("ALTER TABLE patient_medications ADD COLUMN status TEXT DEFAULT 'CURRENT'")
                if "discontinued_date" not in pm_cols:
                    cursor.execute("ALTER TABLE patient_medications ADD COLUMN discontinued_date TEXT")
                if "discontinue_reason" not in pm_cols:
                    cursor.execute("ALTER TABLE patient_medications ADD COLUMN discontinue_reason TEXT")
                if "adverse_reactions" not in pm_cols:
                    cursor.execute("ALTER TABLE patient_medications ADD COLUMN adverse_reactions TEXT")

                # 14. Users Table (Phase 1 Database Hardening)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    hospital_id TEXT DEFAULT 'HOSP-APEX',
                    registration_number TEXT,
                    picture TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """)

                # 15. User Sessions Table (Phase 1 Database Hardening)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expiry REAL NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
                """)

                # 16. Appointments Table (Phase 3 Doctor-Patient Interaction)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
                    patient_id TEXT NOT NULL,
                    patient_mrn TEXT,
                    patient_name TEXT NOT NULL,
                    doctor_id TEXT,
                    doctor_name TEXT NOT NULL,
                    appointment_date TEXT NOT NULL,
                    appointment_time TEXT NOT NULL,
                    consultation_type TEXT DEFAULT 'In-Person Hospital Visit',
                    location TEXT,
                    reason TEXT,
                    notes TEXT,
                    status TEXT DEFAULT 'CONFIRMED',
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
                """)

                # 17. Conversations Table (Phase 3 Messaging)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    patient_name TEXT,
                    doctor_name TEXT,
                    subject TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

                # 18. Messages Table (Phase 3 Messaging)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    sender_role TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
                """)

                # 19. Notifications Table (Phase 3 Notifications)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    notification_type TEXT DEFAULT 'SYSTEM',
                    related_id TEXT,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """)

                # 20. Patient Assignments Table (Phase 3 Care Team Assignment)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    doctor_name TEXT NOT NULL,
                    assigned_by TEXT NOT NULL,
                    assignment_notes TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
                """)

                # Indices
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_hospital_patient ON patients(hospital_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ecg_patient ON ecg_records(patient_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ecg_hospital ON ecg_records(hospital_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_record ON analysis_results(record_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_record ON clinician_reviews(record_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_record ON clinical_reports(record_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_vitals_patient_time ON vital_signs(patient_id, recorded_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_labs_patient_time ON laboratory_results(patient_id, recorded_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_patient_time ON patient_symptoms(patient_id, recorded_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_meds_patient_active ON patient_medications(patient_id, is_active)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_appointments_hosp ON appointments(hospital_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_patient ON conversations(patient_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_doctor ON conversations(doctor_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_notif_user ON notifications(user_id, is_read)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions ON user_sessions(token)")

                # Seed Default Hospital
                cursor.execute("SELECT 1 FROM hospitals WHERE hospital_id = 'HOSP-APEX'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO hospitals (hospital_id, hospital_name, registration_number, address, city, state, country, contact, email, phone, website, created_at, status)
                    VALUES ('HOSP-APEX', 'Apex Heart & Vascular Hospital', 'REG-IND-MH-2026-0914', '108 Healthcare Blvd', 'Mumbai', 'Maharashtra', 'India', 'Dr. Anand Deshmukh', 'contact@apexheart.org', '+91-22-2456-7890', 'https://apexheart.org', ?, 'ACTIVE')
                    """, (datetime.now().isoformat(),))

                # Seed Default Doctors
                cursor.execute("SELECT 1 FROM doctors WHERE doctor_id = 'DOC-001'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO doctors (doctor_id, hospital_id, user_id, name, email, phone, specialization, license_number, department, role, status, created_at)
                    VALUES ('DOC-001', 'HOSP-APEX', 'USR-02', 'Dr. Rajesh Sen', 'doctor@apexheart.org', '+91-9820011223', 'Internal Medicine & Cardiology', 'MCI-DOC-19882', 'Cardiac Telemetry', 'DOCTOR', 'ACTIVE', ?)
                    """, (datetime.now().isoformat(),))

                cursor.execute("SELECT 1 FROM doctors WHERE doctor_id = 'DOC-002'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO doctors (doctor_id, hospital_id, user_id, name, email, phone, specialization, license_number, department, role, status, created_at)
                    VALUES ('DOC-002', 'HOSP-APEX', 'USR-01', 'Dr. Ananya Roy', 'cardiologist@apexheart.org', '+91-9820044556', 'Electrophysiology & Intervention', 'MCI-CARD-4421', 'Electrophysiology', 'CARDIOLOGIST', 'ACTIVE', ?)
                    """, (datetime.now().isoformat(),))

                cursor.execute("SELECT 1 FROM doctors WHERE doctor_id = 'DOC-003'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO doctors (doctor_id, hospital_id, user_id, name, email, phone, specialization, license_number, department, role, status, created_at)
                    VALUES ('DOC-003', 'HOSP-APEX', 'USR-ANIKET', 'Dr. Aniket Chandra', 'dr.aniket@hospital.org', '+91-9876500001', 'Cardiology & Electrophysiology', 'MD/34667', 'Cardiology Department', 'DOCTOR', 'ACTIVE', ?)
                    """, (datetime.now().isoformat(),))

                cursor.execute("SELECT 1 FROM patients WHERE hospital_mrn = 'MRN-5F2298'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO patients (patient_id, hospital_mrn, name, hospital_id, age, sex, date_of_birth, contact, blood_group, known_allergies, existing_conditions, current_medications, previous_cardiac_history, smoking_status, created_at)
                    VALUES ('PAT-RAJ', 'MRN-5F2298', 'Raj Verma', 'HOSP-APEX', 15, 'Male', '2011-03-12', '+91 98765 43210', 'A+', 'None', 'None', 'None', 'None', 'Non-Smoker', ?)
                    """, (datetime.now().isoformat(),))

                cursor.execute("SELECT 1 FROM patients WHERE hospital_mrn = 'MRN-5F2297'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO patients (patient_id, hospital_mrn, name, hospital_id, age, sex, date_of_birth, contact, blood_group, known_allergies, existing_conditions, current_medications, previous_cardiac_history, smoking_status, created_at)
                    VALUES ('PAT-PRIYA', 'MRN-5F2297', 'Priya Sharma', 'HOSP-APEX', 42, 'Female', '1984-07-22', '+91 98765 11223', 'B+', 'Penicillin', 'Mild Hypertension', 'Amlodipine 5mg', 'None', 'Non-Smoker', ?)
                    """, (datetime.now().isoformat(),))

                cursor.execute("SELECT 1 FROM patients WHERE hospital_mrn = 'MRN-5F2296'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO patients (patient_id, hospital_mrn, name, hospital_id, age, sex, date_of_birth, contact, blood_group, known_allergies, existing_conditions, current_medications, previous_cardiac_history, smoking_status, created_at)
                    VALUES ('PAT-AMAN', 'MRN-5F2296', 'Aman Singh', 'HOSP-APEX', 58, 'Male', '1968-11-05', '+91 98765 33445', 'O+', 'None', 'Type 2 Diabetes', 'Metformin 500mg', 'None', 'Former Smoker', ?)
                    """, (datetime.now().isoformat(),))

                cursor.execute("SELECT 1 FROM patients WHERE hospital_mrn = 'MRN-5F2295'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO patients (patient_id, hospital_mrn, name, hospital_id, age, sex, date_of_birth, contact, blood_group, known_allergies, existing_conditions, current_medications, previous_cardiac_history, smoking_status, created_at)
                    VALUES ('PAT-NEHA', 'MRN-5F2295', 'Neha Gupta', 'HOSP-APEX', 36, 'Female', '1990-04-18', '+91 98765 55667', 'AB+', 'Sulfa Drugs', 'Palpitations', 'None', 'None', 'Non-Smoker', ?)
                    """, (datetime.now().isoformat(),))

                # Seed Default Appointment (Phase 3)
                cursor.execute("SELECT 1 FROM appointments WHERE appointment_id = 'APT-2026-0891'")
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO appointments (
                        appointment_id, hospital_id, patient_id, patient_mrn, patient_name,
                        doctor_id, doctor_name, appointment_date, appointment_time, consultation_type,
                        location, reason, notes, status, created_at
                    ) VALUES (
                        'APT-2026-0891', 'HOSP-APEX', 'USR-RAJ_VERMA', 'MRN-5F2298', 'Raj Verma',
                        'DOC-003', 'Dr. Aniket Chandra, MD (Cardiology)', '2026-11-15', '10:30 AM',
                        'In-Person Hospital Visit', 'Apex Heart & Vascular Hospital, Consultation Room 402',
                        'Routine Post-ECG Follow-Up & Medication Review',
                        'Follow-up after baseline 12-lead ECG analysis. Evaluate blood pressure trends and lipid profile.',
                        'CONFIRMED', '2026-10-31T10:30:00Z'
                    )
                    """)

                conn.commit()

    # --- Hospital Multi-Tenant Operations ---
    def create_hospital(
        self,
        hospital_id: str,
        hospital_name: str,
        registration_number: str,
        address: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = "India",
        contact: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        website: Optional[str] = None,
        status: str = "ACTIVE",
    ) -> Dict[str, Any]:
        created_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT INTO hospitals (hospital_id, hospital_name, registration_number, address, city, state, country, contact, email, phone, website, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (hospital_id, hospital_name, registration_number, address, city, state, country, contact, email, phone, website, created_at, status)
                )
                conn.commit()
        return self.get_hospital(hospital_id) or {}

    def get_hospital(self, hospital_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM hospitals WHERE hospital_id = ?", (hospital_id,)).fetchone()
            return dict(row) if row else None

    def list_hospitals(self) -> List[Dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute("SELECT * FROM hospitals ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    def update_hospital(self, hospital_id: str, **kwargs) -> bool:
        valid_cols = {"hospital_name", "registration_number", "address", "city", "state", "country", "contact", "email", "phone", "website", "status"}
        updates = {k: v for k, v in kwargs.items() if k in valid_cols}
        if not updates:
            return False
        set_clauses = [f"{col} = ?" for col in updates.keys()]
        values = list(updates.values()) + [hospital_id]
        with self._lock:
            with self._db() as conn:
                conn.execute(f"UPDATE hospitals SET {', '.join(set_clauses)} WHERE hospital_id = ?", values)
                conn.commit()
        return True

    # --- Doctor Management Operations ---
    def create_doctor(
        self,
        doctor_id: str,
        hospital_id: str,
        name: str,
        email: str,
        phone: Optional[str] = None,
        specialization: Optional[str] = "Cardiology",
        license_number: str = "MCI-PENDING",
        department: Optional[str] = "Cardiology",
        role: str = "DOCTOR",
        status: str = "ACTIVE",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        created_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT INTO doctors (doctor_id, hospital_id, user_id, name, email, phone, specialization, license_number, department, role, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doctor_id, hospital_id, user_id, name, email, phone, specialization, license_number, department, role, status, created_at)
                )
                conn.commit()
        return self.get_doctor(doctor_id) or {}

    def get_doctor(self, doctor_id: str, hospital_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            if hospital_id:
                row = conn.execute("SELECT * FROM doctors WHERE doctor_id = ? AND hospital_id = ?", (doctor_id, hospital_id)).fetchone()
            else:
                row = conn.execute("SELECT * FROM doctors WHERE doctor_id = ?", (doctor_id,)).fetchone()
            return dict(row) if row else None

    def list_doctors(self, hospital_id: Optional[str] = None, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = "SELECT * FROM doctors WHERE 1=1"
            params: List[Any] = []
            if hospital_id:
                query += " AND hospital_id = ?"
                params.append(hospital_id)
            if status_filter:
                query += " AND status = ?"
                params.append(status_filter)
            query += " ORDER BY created_at ASC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def update_doctor_status(self, doctor_id: str, status: str) -> bool:
        with self._lock:
            with self._db() as conn:
                conn.execute("UPDATE doctors SET status = ? WHERE doctor_id = ?", (status, doctor_id))
                conn.commit()
        return True

    # --- Staff Management Operations ---
    def create_staff(
        self,
        staff_id: str,
        hospital_id: str,
        name: str,
        email: str,
        role: str,
        department: Optional[str] = "Clinical Staff",
        status: str = "ACTIVE",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        created_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT INTO staff (staff_id, hospital_id, user_id, name, email, role, department, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (staff_id, hospital_id, user_id, name, email, role, department, status, created_at)
                )
                conn.commit()
        with self._db() as conn:
            row = conn.execute("SELECT * FROM staff WHERE staff_id = ?", (staff_id,)).fetchone()
            return dict(row) if row else {}

    def list_staff(self, hospital_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._db() as conn:
            if hospital_id:
                rows = conn.execute("SELECT * FROM staff WHERE hospital_id = ? ORDER BY created_at ASC", (hospital_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM staff ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    # --- Patient Operations with Multi-Tenant Scoping ---
    def create_patient(
        self,
        patient_id: str,
        hospital_mrn: str,
        name: str,
        hospital_id: str = "HOSP-APEX",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        contact: Optional[str] = None,
        date_of_birth: Optional[str] = None,
        emergency_contact: Optional[str] = None,
        blood_group: Optional[str] = None,
        rh_factor: Optional[str] = None,
        height_cm: Optional[float] = None,
        weight_kg: Optional[float] = None,
        bmi: Optional[float] = None,
        known_allergies: Optional[str] = None,
        food_allergies: Optional[str] = None,
        other_allergies: Optional[str] = None,
        existing_conditions: Optional[str] = None,
        current_medications: Optional[str] = None,
        past_medications: Optional[str] = None,
        discontinued_medications: Optional[str] = None,
        previous_cardiac_history: Optional[str] = None,
        family_history: Optional[str] = None,
        previous_surgeries: Optional[str] = None,
        previous_hospitalizations: Optional[str] = None,
        smoking_status: Optional[str] = None,
        other_relevant_history: Optional[str] = None,
        past_medical_history: Optional[str] = None,
        other_clinical_information: Optional[str] = None,
        diabetes: Optional[str] = None,
        hypertension: Optional[str] = None,
        coronary_artery_disease: Optional[str] = None,
        previous_heart_attack: Optional[str] = None,
        heart_failure: Optional[str] = None,
        arrhythmia_history: Optional[str] = None,
        kidney_disease: Optional[str] = None,
        liver_disease: Optional[str] = None,
        thyroid_disorders: Optional[str] = None,
        other_diagnosed_diseases: Optional[str] = None,
    ) -> PatientRecord:
        created_at = datetime.now().isoformat()
        if bmi is None and height_cm and weight_kg and height_cm > 0:
            bmi = round(float(weight_kg) / ((float(height_cm) / 100.0) ** 2), 1)

        cols = [
            "patient_id", "hospital_mrn", "hospital_id", "name", "age", "sex", "contact", "created_at",
            "date_of_birth", "emergency_contact", "blood_group", "rh_factor", "height_cm", "weight_kg", "bmi",
            "known_allergies", "food_allergies", "other_allergies",
            "existing_conditions", "current_medications", "past_medications", "discontinued_medications",
            "previous_cardiac_history", "family_history", "previous_surgeries", "previous_hospitalizations",
            "smoking_status", "other_relevant_history", "past_medical_history", "other_clinical_information",
            "diabetes", "hypertension", "coronary_artery_disease", "previous_heart_attack",
            "heart_failure", "arrhythmia_history", "kidney_disease", "liver_disease",
            "thyroid_disorders", "other_diagnosed_diseases", "updated_at"
        ]
        vals = [
            patient_id, hospital_mrn, hospital_id, name, age, sex, contact, created_at,
            date_of_birth, emergency_contact, blood_group, rh_factor, height_cm, weight_kg, bmi,
            known_allergies, food_allergies, other_allergies,
            existing_conditions, current_medications, past_medications, discontinued_medications,
            previous_cardiac_history, family_history, previous_surgeries, previous_hospitalizations,
            smoking_status, other_relevant_history, past_medical_history, other_clinical_information,
            diabetes, hypertension, coronary_artery_disease, previous_heart_attack,
            heart_failure, arrhythmia_history, kidney_disease, liver_disease,
            thyroid_disorders, other_diagnosed_diseases, created_at
        ]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        with self._lock:
            with self._db() as conn:
                conn.execute(
                    f"INSERT INTO patients ({col_names}) VALUES ({placeholders})",
                    vals,
                )
                conn.commit()
        return PatientRecord(
            patient_id=patient_id,
            hospital_mrn=hospital_mrn,
            hospital_id=hospital_id,
            name=name,
            age=age,
            sex=sex,
            contact=contact,
            created_at=created_at,
            date_of_birth=date_of_birth,
            emergency_contact=emergency_contact,
            blood_group=blood_group,
            rh_factor=rh_factor,
            height_cm=height_cm,
            weight_kg=weight_kg,
            bmi=bmi,
            known_allergies=known_allergies,
            food_allergies=food_allergies,
            other_allergies=other_allergies,
            existing_conditions=existing_conditions,
            current_medications=current_medications,
            past_medications=past_medications,
            discontinued_medications=discontinued_medications,
            previous_cardiac_history=previous_cardiac_history,
            family_history=family_history,
            previous_surgeries=previous_surgeries,
            previous_hospitalizations=previous_hospitalizations,
            smoking_status=smoking_status,
            other_relevant_history=other_relevant_history,
            past_medical_history=past_medical_history,
            other_clinical_information=other_clinical_information,
            diabetes=diabetes,
            hypertension=hypertension,
            coronary_artery_disease=coronary_artery_disease,
            previous_heart_attack=previous_heart_attack,
            heart_failure=heart_failure,
            arrhythmia_history=arrhythmia_history,
            kidney_disease=kidney_disease,
            liver_disease=liver_disease,
            thyroid_disorders=thyroid_disorders,
            other_diagnosed_diseases=other_diagnosed_diseases,
            updated_at=created_at,
        )

    def update_patient(self, patient_id: str, **kwargs) -> bool:
        valid_cols = {
            "name", "age", "sex", "contact", "date_of_birth", "emergency_contact",
            "blood_group", "rh_factor", "height_cm", "weight_kg", "bmi",
            "known_allergies", "food_allergies", "other_allergies",
            "existing_conditions", "current_medications", "past_medications", "discontinued_medications",
            "previous_cardiac_history", "family_history", "previous_surgeries", "previous_hospitalizations",
            "smoking_status", "other_relevant_history", "past_medical_history", "other_clinical_information",
            "diabetes", "hypertension", "coronary_artery_disease", "previous_heart_attack",
            "heart_failure", "arrhythmia_history", "kidney_disease", "liver_disease",
            "thyroid_disorders", "other_diagnosed_diseases"
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_cols}
        if not updates:
            return False
        # Calculate BMI if height and weight updated or either one provided
        if "height_cm" in updates and "weight_kg" in updates:
            h = updates.get("height_cm")
            w = updates.get("weight_kg")
            if h and w and float(h) > 0:
                updates["bmi"] = round(float(w) / ((float(h) / 100.0) ** 2), 1)

        updates["updated_at"] = datetime.now().isoformat()
        set_clauses = [f"{col} = ?" for col in updates.keys()]
        values = list(updates.values()) + [patient_id]
        with self._lock:
            with self._db() as conn:
                conn.execute(f"UPDATE patients SET {', '.join(set_clauses)} WHERE patient_id = ?", values)
                conn.commit()
        return True

    def _row_to_patient(self, row: sqlite3.Row) -> PatientRecord:
        d = dict(row)
        def _to_bool(val: Any) -> Optional[bool]:
            if val is None:
                return None
            return str(val).strip().lower() in ("1", "true", "yes", "t")

        return PatientRecord(
            patient_id=d.get("patient_id", ""),
            hospital_mrn=d.get("hospital_mrn", ""),
            hospital_id=d.get("hospital_id", "HOSP-APEX"),
            name=d.get("name", ""),
            age=d.get("age"),
            sex=d.get("sex"),
            contact=d.get("contact"),
            created_at=d.get("created_at", ""),
            date_of_birth=d.get("date_of_birth"),
            emergency_contact=d.get("emergency_contact"),
            blood_group=d.get("blood_group"),
            rh_factor=d.get("rh_factor"),
            height_cm=d.get("height_cm"),
            weight_kg=d.get("weight_kg"),
            bmi=d.get("bmi"),
            known_allergies=d.get("known_allergies"),
            food_allergies=d.get("food_allergies"),
            other_allergies=d.get("other_allergies"),
            existing_conditions=d.get("existing_conditions"),
            current_medications=d.get("current_medications"),
            past_medications=d.get("past_medications"),
            discontinued_medications=d.get("discontinued_medications"),
            previous_cardiac_history=d.get("previous_cardiac_history"),
            family_history=d.get("family_history"),
            previous_surgeries=d.get("previous_surgeries"),
            previous_hospitalizations=d.get("previous_hospitalizations"),
            smoking_status=d.get("smoking_status"),
            other_relevant_history=d.get("other_relevant_history"),
            past_medical_history=d.get("past_medical_history"),
            other_clinical_information=d.get("other_clinical_information"),
            diabetes=_to_bool(d.get("diabetes")),
            hypertension=_to_bool(d.get("hypertension")),
            coronary_artery_disease=_to_bool(d.get("coronary_artery_disease")),
            previous_heart_attack=_to_bool(d.get("previous_heart_attack")),
            heart_failure=_to_bool(d.get("heart_failure")),
            arrhythmia_history=_to_bool(d.get("arrhythmia_history")),
            kidney_disease=_to_bool(d.get("kidney_disease")),
            liver_disease=_to_bool(d.get("liver_disease")),
            thyroid_disorders=_to_bool(d.get("thyroid_disorders")),
            other_diagnosed_diseases=d.get("other_diagnosed_diseases"),
            updated_at=d.get("updated_at"),
        )

    def get_patient(self, patient_id: str, hospital_id: Optional[str] = None) -> Optional[PatientRecord]:
        with self._db() as conn:
            if hospital_id:
                row = conn.execute(
                    "SELECT * FROM patients WHERE patient_id = ? AND hospital_id = ?", (patient_id, hospital_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
                ).fetchone()
            return self._row_to_patient(row) if row else None

    def get_patient_by_mrn(self, hospital_mrn: str, hospital_id: Optional[str] = None) -> Optional[PatientRecord]:
        with self._db() as conn:
            if hospital_id:
                row = conn.execute(
                    "SELECT * FROM patients WHERE hospital_mrn = ? AND hospital_id = ?", (hospital_mrn, hospital_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM patients WHERE hospital_mrn = ?", (hospital_mrn,)
                ).fetchone()
            return self._row_to_patient(row) if row else None

    def list_patients(self, hospital_id: Optional[str] = None, limit: int = 100) -> List[PatientRecord]:
        with self._db() as conn:
            if hospital_id:
                rows = conn.execute(
                    "SELECT * FROM patients WHERE hospital_id = ? ORDER BY created_at DESC LIMIT ?", (hospital_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [self._row_to_patient(r) for r in rows]

    # --- ECG Record & Worklist Operations ---
    def save_ecg_record(
        self,
        record_id: str,
        sampling_rate: float,
        lead_names: List[str],
        duration_sec: float,
        file_hash: str,
        source_format: str,
        hospital_id: str = "HOSP-APEX",
        patient_id: Optional[str] = None,
        device: Optional[str] = None,
        signal_quality: Optional[str] = None,
        quality_score: Optional[float] = None,
        uploaded_by: Optional[str] = None,
        raw_data_path: Optional[str] = None,
        workflow_status: str = "UPLOADED",
        priority: str = "ROUTINE",
        assigned_doctor: Optional[str] = None,
    ) -> bool:
        uploaded_at = datetime.now().isoformat()
        lead_names_json = json.dumps(lead_names)
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ecg_records
                    (record_id, hospital_id, patient_id, device, sampling_rate, lead_names, duration_sec,
                     file_hash, source_format, signal_quality, quality_score, uploaded_by,
                     uploaded_at, raw_data_path, workflow_status, priority, assigned_doctor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id, hospital_id, patient_id, device, sampling_rate, lead_names_json, duration_sec,
                        file_hash, source_format, signal_quality, quality_score, uploaded_by,
                        uploaded_at, raw_data_path, workflow_status, priority, assigned_doctor,
                    ),
                )
                conn.commit()
        return True

    def update_ecg_workflow_status(self, record_id: str, status: str, assigned_doctor: Optional[str] = None) -> bool:
        with self._lock:
            with self._db() as conn:
                if assigned_doctor:
                    conn.execute(
                        "UPDATE ecg_records SET workflow_status = ?, assigned_doctor = ? WHERE record_id = ?",
                        (status, assigned_doctor, record_id),
                    )
                else:
                    conn.execute(
                        "UPDATE ecg_records SET workflow_status = ? WHERE record_id = ?",
                        (status, record_id),
                    )
                conn.commit()
        return True

    def get_ecg_record(self, record_id: str, hospital_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            if hospital_id:
                row = conn.execute(
                    "SELECT * FROM ecg_records WHERE record_id = ? AND hospital_id = ?", (record_id, hospital_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM ecg_records WHERE record_id = ?", (record_id,)
                ).fetchone()
            if row:
                d = dict(row)
                d["lead_names"] = json.loads(d["lead_names"])
                return d
            return None

    def list_ecg_records(self, hospital_id: Optional[str] = None, status_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = """
                SELECT r.*, p.name as patient_name, p.hospital_mrn,
                       a.prediction, a.quality_score as ai_quality_score, a.heart_rate_bpm
                FROM ecg_records r
                LEFT JOIN patients p ON r.patient_id = p.patient_id
                LEFT JOIN analysis_results a ON r.record_id = a.record_id
                WHERE 1=1
            """
            params: List[Any] = []
            if hospital_id:
                query += " AND r.hospital_id = ?"
                params.append(hospital_id)
            if status_filter:
                query += " AND r.workflow_status = ?"
                params.append(status_filter)
            query += " ORDER BY r.uploaded_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["lead_names"] = json.loads(d["lead_names"])
                results.append(d)
            return results

    # --- Analysis Result Operations ---
    def save_analysis_result(
        self,
        analysis_id: str,
        record_id: str,
        model_id: str,
        model_version: str,
        prediction: str,
        probabilities: Dict[str, float],
        signal_quality: str,
        quality_score: float,
        heart_rate_bpm: Optional[float] = None,
        mean_rr_ms: Optional[float] = None,
        detected_beats_count: int = 0,
        warnings: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        lead_analyzed: str = "II",
        preprocessing_version: str = "1.0.0",
        processing_time_ms: float = 0.0,
    ) -> bool:
        analyzed_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO analysis_results
                    (analysis_id, record_id, model_id, model_version, preprocessing_version,
                     lead_analyzed, signal_quality, quality_score, prediction, probabilities_json,
                     heart_rate_bpm, mean_rr_ms, detected_beats_count, warnings_json, limitations_json,
                     processing_time_ms, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id, record_id, model_id, model_version, preprocessing_version,
                        lead_analyzed, signal_quality, quality_score, prediction,
                        json.dumps(probabilities), heart_rate_bpm, mean_rr_ms,
                        detected_beats_count, json.dumps(warnings or []),
                        json.dumps(limitations or []), processing_time_ms, analyzed_at,
                    ),
                )
                conn.commit()
        return True

    def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_results WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["probabilities"] = json.loads(d["probabilities_json"])
                d["warnings"] = json.loads(d["warnings_json"])
                d["limitations"] = json.loads(d["limitations_json"])
                return d
            return None

    def get_latest_analysis_for_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_results WHERE record_id = ? ORDER BY analyzed_at DESC LIMIT 1",
                (record_id,),
            ).fetchone()
            if row:
                d = dict(row)
                d["probabilities"] = json.loads(d["probabilities_json"])
                d["warnings"] = json.loads(d["warnings_json"])
                d["limitations"] = json.loads(d["limitations_json"])
                return d
            return None

    # --- Clinician Review Operations ---
    def save_clinician_review(
        self,
        review_id: str,
        analysis_id: str,
        record_id: str,
        clinician_id: str,
        clinician_name: str,
        clinician_role: str,
        agreement_status: str,
        clinician_interpretation: str,
        clinical_notes: str = "",
        registration_number: Optional[str] = None,
    ) -> bool:
        reviewed_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO clinician_reviews
                    (review_id, analysis_id, record_id, clinician_id, clinician_name,
                     clinician_role, registration_number, agreement_status,
                     clinician_interpretation, clinical_notes, reviewed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_id, analysis_id, record_id, clinician_id, clinician_name,
                        clinician_role, registration_number, agreement_status,
                        clinician_interpretation, clinical_notes, reviewed_at,
                    ),
                )
                conn.commit()
        return True

    def get_clinician_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM clinician_reviews WHERE review_id = ?", (review_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_review_for_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM clinician_reviews WHERE analysis_id = ? ORDER BY reviewed_at DESC LIMIT 1",
                (analysis_id,),
            ).fetchone()
            return dict(row) if row else None

    # --- Report Operations ---
    def save_report(
        self,
        report_id: str,
        record_id: str,
        analysis_id: str,
        report_type: str,
        status: str,
        review_id: Optional[str] = None,
        file_path: Optional[str] = None,
        report_sha256: Optional[str] = None,
        report_number: Optional[str] = None,
        report_data: Optional[Any] = None,
        pdf_storage_path: Optional[str] = None,
        report_version: int = 1,
    ) -> bool:
        generated_at = datetime.now().isoformat()
        rep_data_str = json.dumps(report_data) if isinstance(report_data, dict) else (report_data if isinstance(report_data, str) else None)
        rep_num = report_number or f"ECG-{datetime.now().year}-{report_id[-6:].upper()}"
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO clinical_reports
                    (report_id, record_id, analysis_id, review_id, report_type, status,
                     file_path, report_sha256, generated_at, report_number, report_data,
                     pdf_storage_path, report_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report_id, record_id, analysis_id, review_id, report_type, status,
                        file_path, report_sha256, generated_at, rep_num, rep_data_str,
                        pdf_storage_path, report_version,
                    ),
                )
                conn.commit()
        return True

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM clinical_reports WHERE report_id = ? OR report_number = ?", (report_id, report_id)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("report_data") and isinstance(data["report_data"], str):
                try:
                    data["report_data"] = json.loads(data["report_data"])
                except Exception:
                    pass
            return data

    def get_report_by_number(self, report_number: str) -> Optional[Dict[str, Any]]:
        return self.get_report(report_number)

    def list_reports(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = """
                SELECT rep.*, rec.patient_id, p.name as patient_name, p.hospital_mrn,
                       ar.prediction as primary_prediction, ar.signal_quality
                FROM clinical_reports rep
                LEFT JOIN ecg_records rec ON rep.record_id = rec.record_id
                LEFT JOIN patients p ON rec.patient_id = p.patient_id
                LEFT JOIN analysis_results ar ON rep.analysis_id = ar.analysis_id
                WHERE 1=1
            """
            params: List[Any] = []
            if status_filter and status_filter != "ALL":
                query += " AND rep.status = ?"
                params.append(status_filter)
            query += " ORDER BY rep.generated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("report_data") and isinstance(d["report_data"], str):
                    try:
                        d["report_data"] = json.loads(d["report_data"])
                    except Exception:
                        pass
                result.append(d)
            return result

    # --- Time-Aware Multimodal Clinical Observation Operations (Phase 2 & 3) ---

    def record_vital_signs(
        self,
        vital_id: str,
        patient_id: str,
        heart_rate_bpm: Optional[float] = None,
        systolic_bp_mmhg: Optional[float] = None,
        diastolic_bp_mmhg: Optional[float] = None,
        spo2_percent: Optional[float] = None,
        respiratory_rate_bpm: Optional[float] = None,
        temperature_celsius: Optional[float] = None,
        temperature_c: Optional[float] = None,
        height_cm: Optional[float] = None,
        weight_kg: Optional[float] = None,
        bmi: Optional[float] = None,
        recorded_at: Optional[str] = None,
        source: str = "VITAL-SIGN",
        entered_by: Optional[str] = "Triage Staff",
    ) -> bool:
        t_rec = recorded_at or datetime.now().isoformat()
        eff_temp = temperature_celsius if temperature_celsius is not None else temperature_c
        if bmi is None and height_cm and weight_kg and height_cm > 0:
            bmi = round(float(weight_kg) / ((float(height_cm) / 100.0) ** 2), 1)
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vital_signs
                    (vital_id, patient_id, heart_rate_bpm, systolic_bp_mmhg, diastolic_bp_mmhg,
                     spo2_percent, respiratory_rate_bpm, temperature_celsius, height_cm, weight_kg, bmi,
                     recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (vital_id, patient_id, heart_rate_bpm, systolic_bp_mmhg, diastolic_bp_mmhg,
                     spo2_percent, respiratory_rate_bpm, eff_temp, height_cm, weight_kg, bmi,
                     t_rec, source, entered_by),
                )
                conn.commit()
        return True

    def get_patient_vitals(
        self, patient_id: str, as_of_timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve patient vital signs strictly recorded at or before as_of_timestamp."""
        with self._db() as conn:
            if as_of_timestamp:
                rows = conn.execute(
                    "SELECT * FROM vital_signs WHERE patient_id = ? AND recorded_at <= ? ORDER BY recorded_at ASC",
                    (patient_id, as_of_timestamp),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM vital_signs WHERE patient_id = ? ORDER BY recorded_at ASC",
                    (patient_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def record_laboratory_results(
        self,
        lab_id: str,
        patient_id: str,
        panel_name: str = "Cardiac & Metabolic Panel",
        potassium_mmol_l: Optional[float] = None,
        magnesium_mg_dl: Optional[float] = None,
        serum_creatinine_mg_dl: Optional[float] = None,
        egfr_ml_min: Optional[float] = None,
        sodium_mmol_l: Optional[float] = None,
        troponin_i_ng_ml: Optional[float] = None,
        troponin_t_ng_ml: Optional[float] = None,
        bnp_pg_ml: Optional[float] = None,
        hemoglobin_g_dl: Optional[float] = None,
        alt_u_l: Optional[float] = None,
        ast_u_l: Optional[float] = None,
        bilirubin_mg_dl: Optional[float] = None,
        alk_phos_u_l: Optional[float] = None,
        lft_summary: Optional[str] = None,
        additional_dict: Optional[Dict[str, Any]] = None,
        recorded_at: Optional[str] = None,
        source: str = "LABORATORY",
        entered_by: Optional[str] = "Hospital Laboratory",
    ) -> bool:
        t_rec = recorded_at or datetime.now().isoformat()
        add_json = json.dumps(additional_dict) if additional_dict else None
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO laboratory_results
                    (lab_id, patient_id, panel_name, potassium_mmol_l, magnesium_mg_dl,
                     serum_creatinine_mg_dl, egfr_ml_min, sodium_mmol_l, troponin_i_ng_ml, troponin_t_ng_ml,
                     bnp_pg_ml, hemoglobin_g_dl, alt_u_l, ast_u_l, bilirubin_mg_dl, alk_phos_u_l,
                     lft_summary, additional_json, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (lab_id, patient_id, panel_name, potassium_mmol_l, magnesium_mg_dl,
                     serum_creatinine_mg_dl, egfr_ml_min, sodium_mmol_l, troponin_i_ng_ml, troponin_t_ng_ml,
                     bnp_pg_ml, hemoglobin_g_dl, alt_u_l, ast_u_l, bilirubin_mg_dl, alk_phos_u_l,
                     lft_summary, add_json, t_rec, source, entered_by),
                )
                conn.commit()
        return True

    def get_patient_labs(
        self, patient_id: str, as_of_timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve patient laboratory panels strictly recorded at or before as_of_timestamp."""
        with self._db() as conn:
            if as_of_timestamp:
                rows = conn.execute(
                    "SELECT * FROM laboratory_results WHERE patient_id = ? AND recorded_at <= ? ORDER BY recorded_at ASC",
                    (patient_id, as_of_timestamp),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM laboratory_results WHERE patient_id = ? ORDER BY recorded_at ASC",
                    (patient_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def record_symptom(
        self,
        symptom_id: str,
        patient_id: str,
        primary_symptom: str,
        onset_timestamp: Optional[str] = None,
        duration_hours: Optional[float] = None,
        severity: str = "MODERATE",
        character: Optional[str] = None,
        associated_symptoms: Optional[List[str]] = None,
        recorded_at: Optional[str] = None,
        source: str = "PATIENT-HISTORY",
        entered_by: Optional[str] = "Attending Physician",
    ) -> bool:
        t_rec = recorded_at or datetime.now().isoformat()
        assoc_json = json.dumps(associated_symptoms) if associated_symptoms else None
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO patient_symptoms
                    (symptom_id, patient_id, primary_symptom, onset_timestamp, duration_hours,
                     severity, character, associated_symptoms_json, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (symptom_id, patient_id, primary_symptom, onset_timestamp, duration_hours,
                     severity, character, assoc_json, t_rec, source, entered_by),
                )
                conn.commit()
        return True

    def get_patient_symptoms(
        self, patient_id: str, as_of_timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._db() as conn:
            if as_of_timestamp:
                rows = conn.execute(
                    "SELECT * FROM patient_symptoms WHERE patient_id = ? AND recorded_at <= ? ORDER BY recorded_at ASC",
                    (patient_id, as_of_timestamp),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patient_symptoms WHERE patient_id = ? ORDER BY recorded_at ASC",
                    (patient_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def record_allergy(
        self,
        allergy_id: str,
        patient_id: str,
        allergen: str,
        reaction: Optional[str] = None,
        severity: str = "MODERATE",
        allergy_type: str = "DRUG",
        previous_reactions: Optional[str] = None,
        recorded_at: Optional[str] = None,
        source: str = "PATIENT-HISTORY",
        entered_by: Optional[str] = "Clinical Staff",
    ) -> bool:
        t_rec = recorded_at or datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO patient_allergies
                    (allergy_id, patient_id, allergen, reaction, severity, allergy_type, previous_reactions, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (allergy_id, patient_id, allergen, reaction, severity, allergy_type, previous_reactions, t_rec, source, entered_by),
                )
                conn.commit()
        return True

    def get_patient_allergies(
        self, patient_id: str, as_of_timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._db() as conn:
            if as_of_timestamp:
                rows = conn.execute(
                    "SELECT * FROM patient_allergies WHERE patient_id = ? AND recorded_at <= ? ORDER BY recorded_at ASC",
                    (patient_id, as_of_timestamp),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patient_allergies WHERE patient_id = ? ORDER BY recorded_at ASC",
                    (patient_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def record_medication(
        self,
        medication_id: str,
        patient_id: str,
        drug_name: str,
        generic_name: Optional[str] = None,
        brand_name: Optional[str] = None,
        dosage: Optional[str] = None,
        unit: Optional[str] = None,
        frequency: Optional[str] = None,
        route: str = "Oral",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_active: bool = True,
        status: str = "CURRENT",
        discontinued_date: Optional[str] = None,
        discontinue_reason: Optional[str] = None,
        adverse_reactions: Optional[str] = None,
        indication: Optional[str] = None,
        recorded_at: Optional[str] = None,
        source: str = "MEDICATION-RECORD",
        entered_by: Optional[str] = "Physician / Pharmacy",
    ) -> bool:
        t_rec = recorded_at or datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO patient_medications
                    (medication_id, patient_id, drug_name, generic_name, brand_name, dosage, unit,
                     frequency, route, start_date, end_date, is_active, status, discontinued_date,
                     discontinue_reason, adverse_reactions, indication, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (medication_id, patient_id, drug_name, generic_name, brand_name, dosage, unit,
                     frequency, route, start_date, end_date, 1 if is_active else 0, status,
                     discontinued_date, discontinue_reason, adverse_reactions, indication, t_rec, source, entered_by),
                )
                conn.commit()
        return True

    def get_patient_medications(
        self, patient_id: str, as_of_timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve patient medications strictly active / recorded as of timestamp."""
        with self._db() as conn:
            if as_of_timestamp:
                rows = conn.execute(
                    """
                    SELECT * FROM patient_medications
                    WHERE patient_id = ? AND recorded_at <= ?
                    AND (start_date IS NULL OR start_date <= ?)
                    AND (end_date IS NULL OR end_date >= ?)
                    ORDER BY recorded_at ASC
                    """,
                    (patient_id, as_of_timestamp, as_of_timestamp, as_of_timestamp),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patient_medications WHERE patient_id = ? ORDER BY recorded_at ASC",
                    (patient_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_patient_multimodal_profile(
        self, patient_id: str, as_of_timestamp: Optional[str] = None
    ) -> Optional[Any]:
        """Assembles a strongly-typed Patient object with time-aware observations."""
        from src.clinical.models import (
            Allergy,
            LaboratoryResults,
            Medication,
            Patient,
            Symptoms,
            VitalSigns,
        )

        p_row = self.get_patient(patient_id)
        if not p_row:
            return None

        # 1. Parse structured or legacy conditions
        cond_list: List[str] = []
        if p_row.existing_conditions:
            cond_list = [c.strip() for c in p_row.existing_conditions.split(",") if c.strip()]
        for specific_cond, label in [
            (p_row.diabetes, "Diabetes"),
            (p_row.hypertension, "Hypertension"),
            (p_row.coronary_artery_disease, "Coronary Artery Disease"),
            (p_row.previous_heart_attack, "Previous Heart Attack"),
            (p_row.heart_failure, "Heart Failure"),
            (p_row.arrhythmia_history, "Arrhythmia History"),
            (p_row.kidney_disease, "Kidney Disease"),
            (p_row.liver_disease, "Liver Disease"),
            (p_row.thyroid_disorders, "Thyroid Disorder"),
            (p_row.other_diagnosed_diseases, "Other Diagnosed Diseases"),
        ]:
            if specific_cond and specific_cond not in ["None", "No", "0", "false", False] and label not in cond_list:
                is_affirmative = (specific_cond is True) or str(specific_cond).strip().lower() in ["yes", "1", "true", "t"]
                cond_list.append(label if is_affirmative else f"{label} ({specific_cond})")

        cardiac_list: List[str] = []
        if p_row.previous_cardiac_history:
            cardiac_list = [c.strip() for c in p_row.previous_cardiac_history.split(",") if c.strip()]
        fam_list: List[str] = []
        if p_row.family_history:
            fam_list = [c.strip() for c in p_row.family_history.split(",") if c.strip()]
        surg_list: List[str] = []
        if p_row.previous_surgeries:
            surg_list = [s.strip() for s in p_row.previous_surgeries.split(",") if s.strip()]
        hosp_list: List[str] = []
        if p_row.previous_hospitalizations:
            hosp_list = [h.strip() for h in p_row.previous_hospitalizations.split(",") if h.strip()]

        # 2. Vitals (time-filtered)
        vitals_raw = self.get_patient_vitals(patient_id, as_of_timestamp=as_of_timestamp)
        vitals_objs = [
            VitalSigns(
                heart_rate_bpm=v.get("heart_rate_bpm"),
                systolic_bp_mmhg=v.get("systolic_bp_mmhg"),
                diastolic_bp_mmhg=v.get("diastolic_bp_mmhg"),
                spo2_percent=v.get("spo2_percent"),
                respiratory_rate_bpm=v.get("respiratory_rate_bpm"),
                temperature_celsius=v.get("temperature_celsius"),
                height_cm=v.get("height_cm") or p_row.height_cm,
                weight_kg=v.get("weight_kg") or p_row.weight_kg,
                bmi=v.get("bmi") or p_row.bmi,
                recorded_at=v.get("recorded_at"),
                source=v.get("source", "VITAL-SIGN"),
                entered_by=v.get("entered_by"),
            )
            for v in vitals_raw
        ]

        # 3. Labs (time-filtered)
        labs_raw = self.get_patient_labs(patient_id, as_of_timestamp=as_of_timestamp)
        labs_objs = [
            LaboratoryResults(
                panel_name=l.get("panel_name", "Cardiac & Metabolic Panel"),
                potassium_mmol_l=l.get("potassium_mmol_l"),
                magnesium_mg_dl=l.get("magnesium_mg_dl"),
                serum_creatinine_mg_dl=l.get("serum_creatinine_mg_dl"),
                egfr_ml_min=l.get("egfr_ml_min"),
                sodium_mmol_l=l.get("sodium_mmol_l"),
                troponin_i_ng_ml=l.get("troponin_i_ng_ml"),
                troponin_t_ng_ml=l.get("troponin_t_ng_ml"),
                bnp_pg_ml=l.get("bnp_pg_ml"),
                hemoglobin_g_dl=l.get("hemoglobin_g_dl"),
                alt_u_l=l.get("alt_u_l"),
                ast_u_l=l.get("ast_u_l"),
                bilirubin_mg_dl=l.get("bilirubin_mg_dl"),
                alk_phos_u_l=l.get("alk_phos_u_l"),
                lft_summary=l.get("lft_summary"),
                recorded_at=l.get("recorded_at"),
                source=l.get("source", "LABORATORY"),
                entered_by=l.get("entered_by"),
            )
            for l in labs_raw
        ]

        # 4. Symptoms (time-filtered)
        symp_raw = self.get_patient_symptoms(patient_id, as_of_timestamp=as_of_timestamp)
        symp_objs = [
            Symptoms(
                primary_symptom=s.get("primary_symptom"),
                onset_timestamp=s.get("onset_timestamp"),
                duration_hours=s.get("duration_hours"),
                severity=s.get("severity", "MODERATE"),
                character=s.get("character"),
                associated_symptoms=json.loads(s["associated_symptoms_json"]) if s.get("associated_symptoms_json") else [],
                recorded_at=s.get("recorded_at"),
                source=s.get("source", "PATIENT-HISTORY"),
                entered_by=s.get("entered_by"),
            )
            for s in symp_raw
        ]

        # 5. Allergies
        allergy_raw = self.get_patient_allergies(patient_id, as_of_timestamp=as_of_timestamp)
        allergy_objs = [
            Allergy(
                allergen=a["allergen"],
                reaction=a.get("reaction"),
                severity=a.get("severity", "MODERATE"),
                allergy_type=a.get("allergy_type", "DRUG"),
                previous_reactions=a.get("previous_reactions"),
                recorded_at=a.get("recorded_at"),
                source=a.get("source", "PATIENT-HISTORY"),
                entered_by=a.get("entered_by"),
            )
            for a in allergy_raw
        ]
        # Include legacy text allergy if no structured entries exist
        if not allergy_objs and p_row.known_allergies and p_row.known_allergies != "None":
            for alg in p_row.known_allergies.split(","):
                if alg.strip():
                    allergy_objs.append(Allergy(allergen=alg.strip()))

        food_allergy_objs: List[Allergy] = [a for a in allergy_objs if a.allergy_type == "FOOD"]
        if not food_allergy_objs and p_row.food_allergies and p_row.food_allergies != "None":
            for fa in p_row.food_allergies.split(","):
                if fa.strip():
                    food_allergy_objs.append(Allergy(allergen=fa.strip(), allergy_type="FOOD"))

        other_allergy_objs: List[Allergy] = [a for a in allergy_objs if a.allergy_type in ["ENVIRONMENTAL", "OTHER"]]
        if not other_allergy_objs and p_row.other_allergies and p_row.other_allergies != "None":
            for oa in p_row.other_allergies.split(","):
                if oa.strip():
                    other_allergy_objs.append(Allergy(allergen=oa.strip(), allergy_type="OTHER"))

        # 6. Medications (time-filtered)
        meds_raw = self.get_patient_medications(patient_id, as_of_timestamp=as_of_timestamp)
        meds_objs = [
            Medication(
                drug_name=m["drug_name"],
                generic_name=m.get("generic_name"),
                brand_name=m.get("brand_name"),
                dosage=m.get("dosage"),
                unit=m.get("unit"),
                frequency=m.get("frequency"),
                route=m.get("route", "Oral"),
                start_date=m.get("start_date"),
                end_date=m.get("end_date"),
                is_active=bool(m.get("is_active", 1)),
                status=m.get("status", "CURRENT"),
                discontinued_date=m.get("discontinued_date"),
                discontinue_reason=m.get("discontinue_reason"),
                adverse_reactions=m.get("adverse_reactions"),
                indication=m.get("indication"),
                recorded_at=m.get("recorded_at"),
                source=m.get("source", "MEDICATION-RECORD"),
                entered_by=m.get("entered_by"),
            )
            for m in meds_raw if bool(m.get("is_active", 1))
        ]
        if not meds_objs and p_row.current_medications and p_row.current_medications != "None":
            for med_str in p_row.current_medications.split(","):
                if med_str.strip():
                    meds_objs.append(Medication(drug_name=med_str.strip()))

        past_meds_objs = [
            Medication(
                drug_name=m["drug_name"],
                generic_name=m.get("generic_name"),
                brand_name=m.get("brand_name"),
                dosage=m.get("dosage"),
                unit=m.get("unit"),
                frequency=m.get("frequency"),
                route=m.get("route", "Oral"),
                start_date=m.get("start_date"),
                end_date=m.get("end_date"),
                is_active=False,
                status=m.get("status", "PREVIOUS"),
                discontinued_date=m.get("discontinued_date"),
                discontinue_reason=m.get("discontinue_reason"),
                adverse_reactions=m.get("adverse_reactions"),
                indication=m.get("indication"),
                recorded_at=m.get("recorded_at"),
                source=m.get("source", "MEDICATION-RECORD"),
                entered_by=m.get("entered_by"),
            )
            for m in meds_raw if not bool(m.get("is_active", 1))
        ]
        if not past_meds_objs and p_row.past_medications and p_row.past_medications != "None":
            for pmed in p_row.past_medications.split(","):
                if pmed.strip():
                    past_meds_objs.append(Medication(drug_name=pmed.strip(), is_active=False, status="PREVIOUS"))
        if p_row.discontinued_medications and p_row.discontinued_medications != "None":
            for dmed in p_row.discontinued_medications.split(","):
                if dmed.strip():
                    past_meds_objs.append(Medication(drug_name=dmed.strip(), is_active=False, status="DISCONTINUED"))

        # 7. Previous ECG records
        with self._db() as conn:
            p_rows = conn.execute(
                "SELECT record_id FROM ecg_records WHERE patient_id = ? ORDER BY uploaded_at ASC",
                (patient_id,)
            ).fetchall()
            prev_ecg_ids = [r["record_id"] for r in p_rows]

        return Patient(
            patient_id=p_row.patient_id,
            hospital_mrn=p_row.hospital_mrn,
            name=p_row.name,
            hospital_id=p_row.hospital_id,
            age=p_row.age,
            date_of_birth=p_row.date_of_birth,
            sex=p_row.sex,
            blood_group=p_row.blood_group,
            rh_factor=p_row.rh_factor,
            height_cm=p_row.height_cm,
            weight_kg=p_row.weight_kg,
            contact=p_row.contact,
            emergency_contact=p_row.emergency_contact,
            allergies=allergy_objs,
            food_allergies=food_allergy_objs,
            other_allergies=other_allergy_objs,
            smoking_status=p_row.smoking_status,
            alcohol_status=None,
            existing_conditions=cond_list,
            cardiac_history=cardiac_list,
            family_history=fam_list,
            previous_surgeries=surg_list,
            previous_hospitalizations=hosp_list,
            current_medications=meds_objs,
            past_medications=past_meds_objs,
            symptoms=symp_objs,
            vital_signs=vitals_objs,
            laboratory_results=labs_objs,
            previous_ecg_ids=prev_ecg_ids,
            created_at=p_row.created_at,
            updated_at=p_row.updated_at,
        )

    # --- Phase 1: User & Session Persistence Operations ---
    def save_user(self, user_id: str, username: str, password_hash: str, salt: str, role: str, full_name: str, email: str, hospital_id: str = "HOSP-APEX", registration_number: Optional[str] = None, picture: Optional[str] = None, is_active: bool = True, created_at: Optional[float] = None, updated_at: Optional[float] = None) -> None:
        now_ts = time.time()
        created_at = created_at or now_ts
        updated_at = updated_at or now_ts
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                INSERT INTO users (user_id, username, password_hash, salt, role, full_name, email, hospital_id, registration_number, picture, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    password_hash = excluded.password_hash,
                    salt = excluded.salt,
                    role = excluded.role,
                    full_name = excluded.full_name,
                    email = excluded.email,
                    hospital_id = excluded.hospital_id,
                    registration_number = excluded.registration_number,
                    picture = excluded.picture,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at
                """, (user_id, username.lower(), password_hash, salt, role, full_name, email, hospital_id, registration_number, picture, 1 if is_active else 0, created_at, updated_at))
                conn.commit()

    def get_persisted_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),)).fetchone()
            if row:
                return dict(row)
        return None

    def get_persisted_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)).fetchone()
            if row:
                return dict(row)
        return None

    def list_persisted_users(self, hospital_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._db() as conn:
            if hospital_id:
                rows = conn.execute("SELECT * FROM users WHERE hospital_id = ?", (hospital_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM users").fetchall()
            return [dict(r) for r in rows]

    def save_session(self, token: str, user_id: str, expiry: float) -> None:
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                INSERT OR REPLACE INTO user_sessions (token, user_id, expiry, created_at)
                VALUES (?, ?, ?, ?)
                """, (token, user_id, expiry, time.time()))
                conn.commit()

    def get_session(self, token: str) -> Optional[Tuple[str, float]]:
        with self._db() as conn:
            row = conn.execute("SELECT user_id, expiry FROM user_sessions WHERE token = ?", (token,)).fetchone()
            if row:
                return row["user_id"], float(row["expiry"])
        return None

    def delete_session(self, token: str) -> None:
        with self._lock:
            with self._db() as conn:
                conn.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
                conn.commit()

    # --- Phase 3: Appointment Operations ---
    def create_appointment(
        self,
        appointment_id: str,
        patient_id: str,
        patient_name: str,
        doctor_name: str,
        appointment_date: str,
        appointment_time: str,
        hospital_id: str = "HOSP-APEX",
        patient_mrn: Optional[str] = None,
        doctor_id: Optional[str] = None,
        consultation_type: str = "In-Person Hospital Visit",
        location: Optional[str] = "Apex Heart & Vascular Hospital, Consultation Room 402",
        reason: Optional[str] = "Routine Cardiology Follow-Up",
        notes: Optional[str] = "",
        status: str = "CONFIRMED",
    ) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                INSERT INTO appointments (
                    appointment_id, hospital_id, patient_id, patient_mrn, patient_name,
                    doctor_id, doctor_name, appointment_date, appointment_time,
                    consultation_type, location, reason, notes, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    appointment_id, hospital_id, patient_id, patient_mrn, patient_name,
                    doctor_id, doctor_name, appointment_date, appointment_time,
                    consultation_type, location, reason, notes, status, now_iso, now_iso
                ))
                conn.commit()
        return self.get_appointment(appointment_id)

    def get_appointment(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_appointments(
        self,
        patient_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = "SELECT * FROM appointments WHERE 1=1"
            params: List[Any] = []
            if patient_id and patient_id not in ["ALL", ""]:
                query += " AND (patient_id = ? OR patient_mrn = ?)"
                params.extend([patient_id, patient_id])
            if hospital_id:
                query += " AND hospital_id = ?"
                params.append(hospital_id)
            query += " ORDER BY appointment_date DESC, appointment_time DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def cancel_appointment(self, appointment_id: str) -> bool:
        now_iso = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                cur = conn.execute(
                    "UPDATE appointments SET status = 'CANCELLED', updated_at = ? WHERE appointment_id = ?",
                    (now_iso, appointment_id)
                )
                conn.commit()
                return cur.rowcount > 0

    # --- Phase 3: Doctor-Patient Messaging Operations ---
    def create_conversation(
        self,
        conversation_id: str,
        patient_id: str,
        doctor_id: str,
        patient_name: str,
        doctor_name: str,
        subject: str,
        hospital_id: str = "HOSP-APEX",
    ) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                INSERT INTO conversations (conversation_id, hospital_id, patient_id, doctor_id, patient_name, doctor_name, subject, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
                """, (conversation_id, hospital_id, patient_id, doctor_id, patient_name, doctor_name, subject, now_iso, now_iso))
                conn.commit()
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_conversations(
        self,
        user_id: Optional[str] = None,
        hospital_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = "SELECT * FROM conversations WHERE 1=1"
            params: List[Any] = []
            if user_id:
                query += " AND (patient_id = ? OR doctor_id = ?)"
                params.extend([user_id, user_id])
            if hospital_id:
                query += " AND hospital_id = ?"
                params.append(hospital_id)
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def send_message(
        self,
        message_id: str,
        conversation_id: str,
        sender_id: str,
        sender_name: str,
        sender_role: str,
        recipient_id: str,
        content: str,
    ) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                INSERT INTO messages (message_id, conversation_id, sender_id, sender_name, sender_role, recipient_id, content, is_read, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (message_id, conversation_id, sender_id, sender_name, sender_role, recipient_id, content, now_iso))
                conn.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (now_iso, conversation_id))
                conn.commit()
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "sender_role": sender_role,
            "recipient_id": recipient_id,
            "content": content,
            "is_read": False,
            "created_at": now_iso,
        }

    def list_messages(self, conversation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
                (conversation_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Phase 3: Notifications Operations ---
    def create_notification(
        self,
        notification_id: str,
        user_id: str,
        title: str,
        message: str,
        notification_type: str = "SYSTEM",
        related_id: Optional[str] = None,
        hospital_id: str = "HOSP-APEX",
    ) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                INSERT INTO notifications (notification_id, user_id, hospital_id, title, message, notification_type, related_id, is_read, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (notification_id, user_id, hospital_id, title, message, notification_type, related_id, now_iso))
                conn.commit()
        return {
            "notification_id": notification_id,
            "user_id": user_id,
            "hospital_id": hospital_id,
            "title": title,
            "message": message,
            "notification_type": notification_type,
            "related_id": related_id,
            "is_read": False,
            "created_at": now_iso,
        }

    def list_notifications(self, user_id: str, unread_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = "SELECT * FROM notifications WHERE (user_id = ? OR user_id = 'ALL')"
            params: List[Any] = [user_id]
            if unread_only:
                query += " AND is_read = 0"
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def mark_notification_read(self, notification_id: str) -> bool:
        with self._lock:
            with self._db() as conn:
                cur = conn.execute("UPDATE notifications SET is_read = 1 WHERE notification_id = ?", (notification_id,))
                conn.commit()
                return cur.rowcount > 0

    def mark_all_notifications_read(self, user_id: str) -> int:
        with self._lock:
            with self._db() as conn:
                cur = conn.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ? OR user_id = 'ALL'", (user_id,))
                conn.commit()
                return cur.rowcount

    # --- Phase 3: Patient Assignment Operations ---
    def assign_patient_to_doctor(
        self,
        assignment_id: str,
        patient_id: str,
        doctor_id: str,
        doctor_name: str,
        assigned_by: str,
        assignment_notes: Optional[str] = None,
        hospital_id: str = "HOSP-APEX",
    ) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    "UPDATE patient_assignments SET status = 'REASSIGNED', updated_at = ? WHERE patient_id = ? AND status = 'ACTIVE'",
                    (now_iso, patient_id)
                )
                conn.execute("""
                INSERT INTO patient_assignments (assignment_id, hospital_id, patient_id, doctor_id, doctor_name, assigned_by, assignment_notes, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
                """, (assignment_id, hospital_id, patient_id, doctor_id, doctor_name, assigned_by, assignment_notes or "", now_iso, now_iso))
                conn.commit()
        return {
            "assignment_id": assignment_id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "doctor_name": doctor_name,
            "assigned_by": assigned_by,
            "assignment_notes": assignment_notes,
            "status": "ACTIVE",
            "created_at": now_iso,
        }

    def get_doctor_assigned_patients(self, doctor_id: str, hospital_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._db() as conn:
            query = """
            SELECT a.assignment_id, a.patient_id, a.doctor_id, a.doctor_name, a.created_at as assigned_at,
                   p.hospital_mrn, p.name, p.age, p.sex, p.blood_group, p.existing_conditions
            FROM patient_assignments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.doctor_id = ? AND a.status = 'ACTIVE'
            """
            params: List[Any] = [doctor_id]
            if hospital_id:
                query += " AND a.hospital_id = ?"
                params.append(hospital_id)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]


# Global singleton instance
DB_MANAGER = DatabaseManager()

