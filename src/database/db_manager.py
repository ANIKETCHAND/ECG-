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
    known_allergies: Optional[str] = None
    existing_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    previous_cardiac_history: Optional[str] = None
    family_history: Optional[str] = None
    smoking_status: Optional[str] = None
    other_relevant_history: Optional[str] = None
    past_medical_history: Optional[str] = None
    other_clinical_information: Optional[str] = None
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
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
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
        known_allergies: Optional[str] = None,
        existing_conditions: Optional[str] = None,
        current_medications: Optional[str] = None,
        previous_cardiac_history: Optional[str] = None,
        family_history: Optional[str] = None,
        smoking_status: Optional[str] = None,
        other_relevant_history: Optional[str] = None,
        past_medical_history: Optional[str] = None,
        other_clinical_information: Optional[str] = None,
    ) -> PatientRecord:
        created_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT INTO patients (
                        patient_id, hospital_mrn, hospital_id, name, age, sex, contact, created_at,
                        date_of_birth, emergency_contact, blood_group, known_allergies,
                        existing_conditions, current_medications, previous_cardiac_history,
                        family_history, smoking_status, other_relevant_history,
                        past_medical_history, other_clinical_information, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id, hospital_mrn, hospital_id, name, age, sex, contact, created_at,
                        date_of_birth, emergency_contact, blood_group, known_allergies,
                        existing_conditions, current_medications, previous_cardiac_history,
                        family_history, smoking_status, other_relevant_history,
                        past_medical_history, other_clinical_information, created_at,
                    ),
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
            known_allergies=known_allergies,
            existing_conditions=existing_conditions,
            current_medications=current_medications,
            previous_cardiac_history=previous_cardiac_history,
            family_history=family_history,
            smoking_status=smoking_status,
            other_relevant_history=other_relevant_history,
            past_medical_history=past_medical_history,
            other_clinical_information=other_clinical_information,
            updated_at=created_at,
        )

    def update_patient(self, patient_id: str, **kwargs) -> bool:
        valid_cols = {
            "name", "age", "sex", "contact", "date_of_birth", "emergency_contact",
            "blood_group", "known_allergies", "existing_conditions", "current_medications",
            "previous_cardiac_history", "family_history", "smoking_status", "other_relevant_history",
            "past_medical_history", "other_clinical_information"
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_cols}
        if not updates:
            return False
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
            known_allergies=d.get("known_allergies"),
            existing_conditions=d.get("existing_conditions"),
            current_medications=d.get("current_medications"),
            previous_cardiac_history=d.get("previous_cardiac_history"),
            family_history=d.get("family_history"),
            smoking_status=d.get("smoking_status"),
            other_relevant_history=d.get("other_relevant_history"),
            past_medical_history=d.get("past_medical_history"),
            other_clinical_information=d.get("other_clinical_information"),
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
        recorded_at: Optional[str] = None,
        source: str = "VITAL-SIGN",
        entered_by: Optional[str] = "Triage Staff",
    ) -> bool:
        t_rec = recorded_at or datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vital_signs
                    (vital_id, patient_id, heart_rate_bpm, systolic_bp_mmhg, diastolic_bp_mmhg,
                     spo2_percent, respiratory_rate_bpm, temperature_celsius, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (vital_id, patient_id, heart_rate_bpm, systolic_bp_mmhg, diastolic_bp_mmhg,
                     spo2_percent, respiratory_rate_bpm, temperature_celsius, t_rec, source, entered_by),
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
        troponin_i_ng_ml: Optional[float] = None,
        troponin_t_ng_ml: Optional[float] = None,
        bnp_pg_ml: Optional[float] = None,
        hemoglobin_g_dl: Optional[float] = None,
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
                     serum_creatinine_mg_dl, egfr_ml_min, troponin_i_ng_ml, troponin_t_ng_ml,
                     bnp_pg_ml, hemoglobin_g_dl, additional_json, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (lab_id, patient_id, panel_name, potassium_mmol_l, magnesium_mg_dl,
                     serum_creatinine_mg_dl, egfr_ml_min, troponin_i_ng_ml, troponin_t_ng_ml,
                     bnp_pg_ml, hemoglobin_g_dl, add_json, t_rec, source, entered_by),
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
                    (allergy_id, patient_id, allergen, reaction, severity, recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (allergy_id, patient_id, allergen, reaction, severity, t_rec, source, entered_by),
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
        dosage: Optional[str] = None,
        unit: Optional[str] = None,
        frequency: Optional[str] = None,
        route: str = "Oral",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_active: bool = True,
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
                    (medication_id, patient_id, drug_name, generic_name, dosage, unit,
                     frequency, route, start_date, end_date, is_active, indication,
                     recorded_at, source, entered_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (medication_id, patient_id, drug_name, generic_name, dosage, unit,
                     frequency, route, start_date, end_date, 1 if is_active else 0,
                     indication, t_rec, source, entered_by),
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
        cardiac_list: List[str] = []
        if p_row.previous_cardiac_history:
            cardiac_list = [c.strip() for c in p_row.previous_cardiac_history.split(",") if c.strip()]
        fam_list: List[str] = []
        if p_row.family_history:
            fam_list = [c.strip() for c in p_row.family_history.split(",") if c.strip()]

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
                troponin_i_ng_ml=l.get("troponin_i_ng_ml"),
                troponin_t_ng_ml=l.get("troponin_t_ng_ml"),
                bnp_pg_ml=l.get("bnp_pg_ml"),
                hemoglobin_g_dl=l.get("hemoglobin_g_dl"),
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

        # 6. Medications (time-filtered)
        meds_raw = self.get_patient_medications(patient_id, as_of_timestamp=as_of_timestamp)
        meds_objs = [
            Medication(
                drug_name=m["drug_name"],
                generic_name=m.get("generic_name"),
                dosage=m.get("dosage"),
                unit=m.get("unit"),
                frequency=m.get("frequency"),
                route=m.get("route", "Oral"),
                start_date=m.get("start_date"),
                end_date=m.get("end_date"),
                is_active=bool(m.get("is_active", 1)),
                indication=m.get("indication"),
                recorded_at=m.get("recorded_at"),
                source=m.get("source", "MEDICATION-RECORD"),
                entered_by=m.get("entered_by"),
            )
            for m in meds_raw
        ]
        if not meds_objs and p_row.current_medications and p_row.current_medications != "None":
            for med_str in p_row.current_medications.split(","):
                if med_str.strip():
                    meds_objs.append(Medication(drug_name=med_str.strip()))

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
            allergies=allergy_objs,
            smoking_status=p_row.smoking_status,
            alcohol_status=None,
            existing_conditions=cond_list,
            cardiac_history=cardiac_list,
            family_history=fam_list,
            current_medications=meds_objs,
            symptoms=symp_objs,
            vital_signs=vitals_objs,
            laboratory_results=labs_objs,
            previous_ecg_ids=prev_ecg_ids,
            created_at=p_row.created_at,
            updated_at=p_row.updated_at,
        )


# Global singleton instance
DB_MANAGER = DatabaseManager()

