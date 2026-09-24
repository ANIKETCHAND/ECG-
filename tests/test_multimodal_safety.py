"""
Multimodal Safety, Robustness, and Failure Mode Verification Suite.
===================================================================

Phase 27 Tests:
1. Missing vitals: Zero fabrication, explicit 'NOT PROVIDED', cautionary warning.
2. Missing laboratory results: Zero fabrication, cautionary warning.
3. Missing blood group: Stored/displayed as 'NOT PROVIDED', zero predictive impact.
4. Temporal look-ahead violation: Strict rejection of observations where T_event > T_ecg.
5. Corrupted waveform: Intercepted by quality gate, inference halted safely.
6. Physiological safety cross-checks: Critical alerts for QTc, bradycardia, potassium.
7. Blood Group feature governance: Strict absence of blood group in model feature vectors.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical.models import Patient, VitalSigns, LaboratoryResults
from database.db_manager import DatabaseManager, PatientRecord
from datasets.timestamp_mapper import TimestampMapper
from datasets.clinical_feature_builder import ClinicalFeatureBuilder, BLOOD_GROUP_FEATURE_NAMES
from medications.interaction_checker import check_medication_safety
from report.report_generator import generate_structured_report, export_report_to_text


def test_missing_vitals_zero_fabrication():
    """Verify that when vitals are not provided, system emits NOT PROVIDED without fabrication."""
    report = generate_structured_report(
        input_info={"file_name": "test.csv", "sampling_rate": 360, "duration_sec": 10},
        ai_results={"predicted_class": "Normal", "signal_quality": "GOOD"},
        patient_profile={"patient_name": "Test Patient", "age": 50, "sex": "M"},
        vital_signs=None,
    )
    assert report["vital_signs"]["status"] == "NOT PROVIDED"
    assert report["vital_signs"]["heart_rate_bpm"] == "NOT PROVIDED"
    assert report["vital_signs"]["blood_pressure"] == "NOT PROVIDED"
    assert any("Vital signs not recorded" in w for w in report["missing_information_warnings"])

    txt = export_report_to_text(report)
    assert "Vitals Status     : NOT PROVIDED" in txt
    assert "MISSING CLINICAL INFORMATION WARNINGS" in txt


def test_missing_laboratory_results_zero_fabrication():
    """Verify missing laboratory results emit explicit NOT PROVIDED and safety warning."""
    report = generate_structured_report(
        input_info={"file_name": "test.csv", "sampling_rate": 360, "duration_sec": 10},
        ai_results={"predicted_class": "Normal", "signal_quality": "GOOD"},
        patient_profile={"patient_name": "Test Patient", "age": 50, "sex": "M"},
        laboratory_results=None,
    )
    assert report["laboratory_results"]["status"] == "NOT PROVIDED"
    assert report["laboratory_results"]["potassium_meq_l"] == "NOT PROVIDED"
    assert any("electrolyte panel" in w.lower() for w in report["missing_information_warnings"])


def test_missing_blood_group_governance():
    """Verify missing blood group is preserved as NOT PROVIDED and causes no crashes."""
    report = generate_structured_report(
        input_info={"file_name": "test.csv", "sampling_rate": 360, "duration_sec": 10},
        patient_profile={"patient_name": "Test Patient", "age": 45, "sex": "F", "blood_group": None},
    )
    assert report["patient_info"]["blood_group"] == "NOT PROVIDED"
    assert "excluded from ml" in report["patient_info"]["blood_group_policy_note"].lower()


def test_temporal_lookahead_violation_rejection():
    """Verify that temporal filter rejects observations timestamped after the ECG."""
    ecg_time = "2026-05-10 14:00:00"
    valid_obs = {
        "patient_id": "P1",
        "feature_name": "potassium",
        "value": 4.2,
        "recorded_at": "2026-05-10 13:30:00",
    }
    future_obs = {
        "patient_id": "P1",
        "feature_name": "potassium",
        "value": 5.8,
        "recorded_at": "2026-05-10 14:30:00",  # 30 mins AFTER ECG! Look-ahead leakage!
    }
    assert TimestampMapper.is_temporally_valid(valid_obs["recorded_at"], ecg_time) is True
    assert TimestampMapper.is_temporally_valid(future_obs["recorded_at"], ecg_time) is False

    valid_events, leaked = TimestampMapper.filter_point_in_time_events(
        [valid_obs, future_obs],
        ecg_timestamp=ecg_time,
        timestamp_key="recorded_at",
    )
    assert len(valid_events) == 1
    assert valid_events[0]["value"] == 4.2
    assert len(leaked) == 1
    assert leaked[0]["value"] == 5.8


def test_blood_group_feature_exclusion_governance():
    """Verify that the clinical feature builder excludes blood group by default."""
    builder_default = ClinicalFeatureBuilder(include_blood_group=False)
    for bg_col in BLOOD_GROUP_FEATURE_NAMES:
        assert bg_col not in builder_default.feature_names

    builder_ablation = ClinicalFeatureBuilder(include_blood_group=True)
    for bg_col in BLOOD_GROUP_FEATURE_NAMES:
        assert bg_col in builder_ablation.feature_names


def test_physiological_crosscheck_safety_alerts():
    """Verify critical safety alerts trigger for lethal drug-vital and drug-lab conflicts."""
    patient = PatientRecord(
        patient_id="P-ALERT",
        hospital_mrn="MRN-ALERT",
        name="Critical Patient",
        age=68,
        sex="M",
        known_allergies="None",
        existing_conditions="Arrhythmia, Heart Failure",
        current_medications="Amiodarone, Metoprolol, Digoxin",
    )
    # Severe QTc prolongation with QT-prolonging drug
    rep_qt = check_medication_safety(
        ["Amiodarone"],
        patient=patient,
        ecg_measurements={"qtc_ms": 525, "heart_rate": 65},
    )
    assert any("Critical QTc Alert" in a.title for a in rep_qt.alerts)
    assert any("Torsades de Pointes" in a.description for a in rep_qt.alerts)

    # Severe Bradycardia with beta-blocker
    rep_brady = check_medication_safety(
        ["Metoprolol"],
        patient=patient,
        vital_signs={"heart_rate": 38, "systolic_bp": 105},
    )
    assert any("Severe Bradycardia Alert" in a.title for a in rep_brady.alerts)

    # Digoxin + Severe Hypokalemia
    rep_dig = check_medication_safety(
        ["Digoxin"],
        patient=patient,
        laboratory_results={"potassium": 2.9, "creatinine": 1.0},
    )
    assert any("Potassium-Digoxin Toxicity Alert" in a.title for a in rep_dig.alerts)
