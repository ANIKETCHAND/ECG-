"""
Unit tests for Cardiovascular Medication Knowledge Base & Safety Checker.
Verifies:
- Authoritative formulary lookup (Metoprolol, Amiodarone, Apixaban).
- Drug-drug interaction detection.
- Known allergy conflict identification.
- Condition-based contraindication alerts.
- Missing context safety halts ('INSUFFICIENT CLINICAL CONTEXT').
- Absolute safety rule: no autonomous prescribing or automated dosage generation.
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.db_manager import PatientRecord
from medications.medication_database import (
    GLOBAL_MEDICATION_DB,
    MedicationDatabase,
    MedicationEntry,
)
from medications.interaction_checker import (
    check_medication_safety,
    MedicationSafetyReport,
)


def test_formulary_lookup():
    db = GLOBAL_MEDICATION_DB
    all_meds = db.list_all()
    assert len(all_meds) >= 3

    metoprolol = db.get_by_name("Metoprolol")
    assert metoprolol is not None
    assert metoprolol.generic_name == "Metoprolol"
    assert "Beta-1" in metoprolol.drug_class
    assert "FDA Prescribing Information" in metoprolol.authoritative_source

    amiodarone = db.get_by_name("Cordarone")  # Brand name lookup
    assert amiodarone is not None
    assert amiodarone.generic_name == "Amiodarone"
    assert amiodarone.qt_prolongation_risk == "HIGH"


def test_interaction_checker_detects_drug_drug_interaction():
    patient = PatientRecord(
        patient_id="PAT-TEST-01",
        hospital_mrn="MRN-TEST-01",
        name="Test Patient",
        age=60,
        sex="M",
        known_allergies="None",
        existing_conditions="Hypertension",
        current_medications="Metoprolol, Amiodarone",
    )

    report = check_medication_safety(["Metoprolol", "Amiodarone"], patient=patient)
    assert isinstance(report, MedicationSafetyReport)
    assert len(report.alerts) > 0
    # Must identify bradycardia / heart block interaction
    interaction_alerts = [a for a in report.alerts if a.category == "INTERACTION"]
    assert len(interaction_alerts) > 0
    assert any("bradycardia" in a.description.lower() for a in interaction_alerts)
    assert any(a.severity in ["CRITICAL", "MAJOR"] for a in interaction_alerts)


def test_interaction_checker_detects_allergy_conflict():
    patient = PatientRecord(
        patient_id="PAT-TEST-02",
        hospital_mrn="MRN-TEST-02",
        name="Allergic Patient",
        age=55,
        sex="F",
        known_allergies="Iodine allergy",
        existing_conditions="Atrial Fibrillation",
        current_medications="Amiodarone",
    )

    report = check_medication_safety(["Amiodarone"], patient=patient)
    allergy_alerts = [a for a in report.alerts if a.category == "ALLERGY"]
    assert len(allergy_alerts) > 0
    assert allergy_alerts[0].severity == "CRITICAL"
    assert "iodine" in allergy_alerts[0].description.lower()


def test_interaction_checker_detects_condition_contraindication():
    patient = PatientRecord(
        patient_id="PAT-TEST-03",
        hospital_mrn="MRN-TEST-03",
        name="Cardiac Patient",
        age=70,
        sex="M",
        known_allergies="None",
        existing_conditions="Severe bradycardia, second-degree heart block",
        current_medications="Metoprolol",
    )

    report = check_medication_safety(["Metoprolol"], patient=patient)
    contra_alerts = [a for a in report.alerts if a.category == "CONTRAINDICATION"]
    assert len(contra_alerts) > 0
    assert any("heart block" in a.description.lower() or "bradycardia" in a.description.lower() for a in contra_alerts)


def test_missing_patient_triggers_insufficient_clinical_context():
    # Calling safety checker without a linked patient profile
    report = check_medication_safety(["Metoprolol", "Apixaban"], patient=None)
    assert report.context_completeness == "INSUFFICIENT"
    assert "Patient Record" in report.missing_context_fields
    context_alerts = [a for a in report.alerts if a.category == "CONTEXT_WARNING"]
    assert len(context_alerts) > 0
    assert context_alerts[0].severity == "CRITICAL"
    assert "INSUFFICIENT CLINICAL CONTEXT" in context_alerts[0].title


def test_incomplete_patient_profile_triggers_warning():
    patient = PatientRecord(
        patient_id="PAT-TEST-04",
        hospital_mrn="MRN-TEST-04",
        name="Incomplete Patient",
        age=None,
        sex="M",
        known_allergies="",
        existing_conditions="",
    )
    report = check_medication_safety(["Metoprolol"], patient=patient)
    assert report.context_completeness == "INSUFFICIENT"
    assert len(report.missing_context_fields) >= 2


def test_no_autonomous_prescriptions_or_doses_in_alerts():
    patient = PatientRecord(
        patient_id="PAT-TEST-05",
        hospital_mrn="MRN-TEST-05",
        name="Safe Patient",
        age=65,
        sex="M",
        known_allergies="None",
        existing_conditions="Hypertension",
        current_medications="Metoprolol",
    )
    report = check_medication_safety(["Metoprolol", "Amiodarone"], patient=patient)
    for alert in report.alerts:
        assert "take " not in alert.clinical_recommendation.lower()
        assert "mg daily" not in alert.clinical_recommendation.lower()
        assert "prescribe " not in alert.clinical_recommendation.lower()


def test_multimodal_qt_prolongation_alert():
    patient = PatientRecord(
        patient_id="PAT-TEST-06",
        hospital_mrn="MRN-TEST-06",
        name="QT Patient",
        age=60,
        sex="M",
        known_allergies="None",
        existing_conditions="Arrhythmia",
        current_medications="Amiodarone",
    )
    report = check_medication_safety(
        ["Amiodarone"],
        patient=patient,
        ecg_measurements={"qtc_ms": 495, "heart_rate": 68},
    )
    qtc_alerts = [a for a in report.alerts if "QTc" in a.title or a.category == "PHYSIOLOGICAL_CONFLICT"]
    assert len(qtc_alerts) > 0
    assert any("Torsades" in a.description or "QTc" in a.title for a in qtc_alerts)
    assert any(a.severity == "CRITICAL" for a in qtc_alerts)


def test_multimodal_bradycardia_alert():
    patient = PatientRecord(
        patient_id="PAT-TEST-07",
        hospital_mrn="MRN-TEST-07",
        name="Brady Patient",
        age=62,
        sex="M",
        known_allergies="None",
        existing_conditions="CAD",
        current_medications="Metoprolol",
    )
    report = check_medication_safety(
        ["Metoprolol"],
        patient=patient,
        vital_signs={"heart_rate": 42, "systolic_bp": 115},
    )
    brady_alerts = [a for a in report.alerts if "Bradycardia" in a.title]
    assert len(brady_alerts) > 0
    assert brady_alerts[0].severity == "CRITICAL"


def test_multimodal_electrolyte_and_renal_alerts():
    patient = PatientRecord(
        patient_id="PAT-TEST-08",
        hospital_mrn="MRN-TEST-08",
        name="Renal Patient",
        age=72,
        sex="M",
        known_allergies="None",
        existing_conditions="Heart Failure, CKD",
        current_medications="Digoxin, Apixaban, Spironolactone",
    )
    # 1. Digoxin + Hypokalemia
    rep_hypok = check_medication_safety(
        ["Digoxin"],
        patient=patient,
        laboratory_results={"potassium": 3.1, "creatinine": 1.1},
    )
    assert any("Potassium-Digoxin" in a.title for a in rep_hypok.alerts)

    # 2. Spironolactone + Hyperkalemia
    rep_hyperk = check_medication_safety(
        ["Spironolactone"],
        patient=patient,
        laboratory_results={"potassium": 5.4, "creatinine": 1.2},
    )
    assert any("Hyperkalemia" in a.title for a in rep_hyperk.alerts)

    # 3. Apixaban + Elevated Creatinine
    rep_renal = check_medication_safety(
        ["Apixaban"],
        patient=patient,
        laboratory_results={"potassium": 4.2, "creatinine": 1.8},
    )
    assert any("Apixaban Renal" in a.title for a in rep_renal.alerts)

