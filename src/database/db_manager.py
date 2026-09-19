"""
Hospital Database Management System
===================================

Provides clinical persistence layer for:
- Patients (demographics, hospital MRN)
- ECG Records (ingestion metadata, file hash, signal quality)
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
    age: Optional[int] = None
    sex: Optional[str] = None  # M, F, O
    contact: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatabaseManager:
    """Thread-safe SQLite clinical database manager with automatic connection closing."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            data_dir = base_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "hospital_clinical.db")
        else:
            self.db_path = db_path
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
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    hospital_mrn TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    age INTEGER,
                    sex TEXT,
                    contact TEXT,
                    created_at TEXT NOT NULL
                )
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ecg_records (
                    record_id TEXT PRIMARY KEY,
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
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE SET NULL
                )
                """)

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

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ecg_patient ON ecg_records(patient_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_record ON analysis_results(record_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_record ON clinician_reviews(record_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_report_record ON clinical_reports(record_id)")
                conn.commit()

    # --- Patient Operations ---
    def create_patient(self, patient_id: str, hospital_mrn: str, name: str,
                       age: Optional[int] = None, sex: Optional[str] = None,
                       contact: Optional[str] = None) -> PatientRecord:
        created_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT INTO patients (patient_id, hospital_mrn, name, age, sex, contact, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (patient_id, hospital_mrn, name, age, sex, contact, created_at),
                )
                conn.commit()
        return PatientRecord(
            patient_id=patient_id,
            hospital_mrn=hospital_mrn,
            name=name,
            age=age,
            sex=sex,
            contact=contact,
            created_at=created_at,
        )

    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            if row:
                return PatientRecord(
                    patient_id=row["patient_id"],
                    hospital_mrn=row["hospital_mrn"],
                    name=row["name"],
                    age=row["age"],
                    sex=row["sex"],
                    contact=row["contact"],
                    created_at=row["created_at"],
                )
            return None

    def get_patient_by_mrn(self, hospital_mrn: str) -> Optional[PatientRecord]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE hospital_mrn = ?", (hospital_mrn,)
            ).fetchone()
            if row:
                return PatientRecord(
                    patient_id=row["patient_id"],
                    hospital_mrn=row["hospital_mrn"],
                    name=row["name"],
                    age=row["age"],
                    sex=row["sex"],
                    contact=row["contact"],
                    created_at=row["created_at"],
                )
            return None

    def list_patients(self, limit: int = 100) -> List[PatientRecord]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                PatientRecord(
                    patient_id=r["patient_id"],
                    hospital_mrn=r["hospital_mrn"],
                    name=r["name"],
                    age=r["age"],
                    sex=r["sex"],
                    contact=r["contact"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    # --- ECG Record Operations ---
    def save_ecg_record(self, record_id: str, sampling_rate: float, lead_names: List[str],
                        duration_sec: float, file_hash: str, source_format: str,
                        patient_id: Optional[str] = None, device: Optional[str] = None,
                        signal_quality: Optional[str] = None, quality_score: Optional[float] = None,
                        uploaded_by: Optional[str] = None, raw_data_path: Optional[str] = None) -> bool:
        uploaded_at = datetime.now().isoformat()
        lead_names_json = json.dumps(lead_names)
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ecg_records
                    (record_id, patient_id, device, sampling_rate, lead_names, duration_sec,
                     file_hash, source_format, signal_quality, quality_score, uploaded_by, uploaded_at, raw_data_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id, patient_id, device, sampling_rate, lead_names_json, duration_sec,
                        file_hash, source_format, signal_quality, quality_score, uploaded_by, uploaded_at, raw_data_path,
                    ),
                )
                conn.commit()
        return True

    def get_ecg_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM ecg_records WHERE record_id = ?", (record_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["lead_names"] = json.loads(d["lead_names"])
                return d
            return None

    def list_ecg_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                """
                SELECT r.*, p.name as patient_name, p.hospital_mrn
                FROM ecg_records r
                LEFT JOIN patients p ON r.patient_id = p.patient_id
                ORDER BY r.uploaded_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["lead_names"] = json.loads(d["lead_names"])
                results.append(d)
            return results

    # --- Analysis Result Operations ---
    def save_analysis_result(self, analysis_id: str, record_id: str, model_id: str,
                             model_version: str, prediction: str, probabilities: Dict[str, float],
                             signal_quality: str, quality_score: float,
                             heart_rate_bpm: Optional[float] = None,
                             mean_rr_ms: Optional[float] = None,
                             detected_beats_count: int = 0,
                             warnings: Optional[List[str]] = None,
                             limitations: Optional[List[str]] = None,
                             lead_analyzed: str = "II",
                             preprocessing_version: str = "1.0.0",
                             processing_time_ms: float = 0.0) -> bool:
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
    def save_clinician_review(self, review_id: str, analysis_id: str, record_id: str,
                              clinician_id: str, clinician_name: str, clinician_role: str,
                              agreement_status: str, clinician_interpretation: str,
                              clinical_notes: str = "", registration_number: Optional[str] = None) -> bool:
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
    def save_report(self, report_id: str, record_id: str, analysis_id: str,
                    report_type: str, status: str, review_id: Optional[str] = None,
                    file_path: Optional[str] = None, report_sha256: Optional[str] = None) -> bool:
        generated_at = datetime.now().isoformat()
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO clinical_reports
                    (report_id, record_id, analysis_id, review_id, report_type, status,
                     file_path, report_sha256, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report_id, record_id, analysis_id, review_id, report_type, status,
                        file_path, report_sha256, generated_at,
                    ),
                )
                conn.commit()
        return True

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM clinical_reports WHERE report_id = ?", (report_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                """
                SELECT rep.*, rec.patient_id, p.name as patient_name, p.hospital_mrn
                FROM clinical_reports rep
                JOIN ecg_records rec ON rep.record_id = rec.record_id
                LEFT JOIN patients p ON rec.patient_id = p.patient_id
                ORDER BY rep.generated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]


# Global singleton instance
DB_MANAGER = DatabaseManager()
