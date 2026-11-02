"""
Phase 8 & 9: Medication Safety, Pharmacology & Clinical Decision Support Audit Suite
=====================================================================================
Validates 10 Core Pharmacology Rules & Safety Gates:
1. Allergy Direct Conflict (Patient allergy: Aspirin -> Order: Aspirin -> CRITICAL ALLERGY)
2. Drug-Drug Interaction: Amiodarone + Warfarin (CYP2C9 inhibition / hemorrhage risk)
3. Drug-Drug Interaction: Amiodarone + Digoxin (P-glycoprotein inhibition / lethal toxicity)
4. Drug-Drug Interaction: Beta-blocker + Calcium Channel Blocker (Metoprolol + Diltiazem)
5. Multimodal Cross-Check: QT-Prolonging Agent + Prolonged QTc (>480 ms -> Torsades risk)
6. Multimodal Cross-Check: Bradycardia (HR < 50) + AV Nodal Blocking Agent
7. Patient Condition Contraindication (Asthma / Heart Block + Metoprolol)
8. Incomplete Context Safety Gate (Missing allergies & history -> INSUFFICIENT CONTEXT)
9. Complete Context Validation (Verified patient record -> COMPLETE CONTEXT)
10. Rule 1 Medical Ethics Verification (Zero autonomous prescribing, mandatory clinician sign-off)
"""

import uuid
import pytest
from src.database.db_manager import PatientRecord
from src.medications.interaction_checker import check_medication_safety
from src.clinical.medication_candidate_engine import build_authoritative_clinical_analysis
from src.clinical.recommendation_engine import GLOBAL_CDS_ENGINE


def test_rule_1_allergy_direct_conflict():
    """Rule 1: Direct patient allergy matching prescribed medication."""
    patient = PatientRecord(
        patient_id=str(uuid.uuid4()),
        hospital_mrn="MRN-ALLERGY-01",
        name="Allergic Patient",
        hospital_id="HOSP-APEX",
        age=42,
        sex="Female",
        known_allergies="Metoprolol, Penicillin",
        existing_conditions="CAD",
    )
    report = check_medication_safety(["Metoprolol"], patient=patient)
    allergy_alerts = [a for a in report.alerts if a.category == "ALLERGY"]

    assert len(allergy_alerts) >= 1
    assert allergy_alerts[0].severity == "CRITICAL"
    assert "Metoprolol" in allergy_alerts[0].title


def test_rule_2_drug_drug_amiodarone_warfarin():
    """Rule 2: Amiodarone + Warfarin interaction (CYP2C9 inhibition)."""
    report = check_medication_safety(["Amiodarone", "Warfarin"])
    interactions = [a for a in report.alerts if a.category == "INTERACTION"]

    assert any("Amiodarone" in a.title and "Warfarin" in a.title for a in interactions)
    inter = next(a for a in interactions if "Amiodarone" in a.title and "Warfarin" in a.title)
    assert inter.severity in ["MAJOR", "CRITICAL"]
    assert "CYP2C9" in inter.description or "INR" in inter.description or "bleeding" in inter.description.lower()


def test_rule_3_drug_drug_amiodarone_digoxin():
    """Rule 3: Amiodarone + Digoxin interaction (P-gp inhibition)."""
    report = check_medication_safety(["Amiodarone", "Digoxin"])
    interactions = [a for a in report.alerts if a.category == "INTERACTION"]

    assert any("Amiodarone" in a.title and "Digoxin" in a.title for a in interactions)
    inter = next(a for a in interactions if "Amiodarone" in a.title and "Digoxin" in a.title)
    assert inter.severity in ["MAJOR", "CRITICAL"]
    assert "digoxin" in inter.description.lower()


def test_rule_4_drug_drug_metoprolol_diltiazem():
    """Rule 4: Beta-blocker + Non-dihydropyridine CCB interaction."""
    report = check_medication_safety(["Metoprolol", "Diltiazem"])
    interactions = [a for a in report.alerts if a.category == "INTERACTION"]

    assert any("Metoprolol" in a.title and "Diltiazem" in a.title for a in interactions)
    inter = next(a for a in interactions if "Metoprolol" in a.title and "Diltiazem" in a.title)
    assert "bradycardia" in inter.description.lower() or "inotropic" in inter.description.lower() or "heart failure" in inter.management_recommendation.lower()


def test_rule_5_qtc_prolongation_critical_alert():
    """Rule 5: High QT prolongation risk drug with prolonged QTc measurement (> 480 ms)."""
    patient = PatientRecord(
        patient_id=str(uuid.uuid4()),
        hospital_mrn="MRN-QT-01",
        name="Prolonged QT Patient",
        hospital_id="HOSP-APEX",
        age=55,
        sex="Male",
        known_allergies="None",
        existing_conditions="Ventricular Arrhythmia",
    )
    report = check_medication_safety(
        ["Amiodarone"],
        patient=patient,
        ecg_measurements={"qtc_ms": 492.0, "heart_rate": 70.0},
    )
    qt_alerts = [a for a in report.alerts if "QTc Alert" in a.title or "Prolonged QTc" in a.title]

    assert len(qt_alerts) >= 1
    assert qt_alerts[0].severity == "CRITICAL"
    assert "Torsades de Pointes" in qt_alerts[0].description


def test_rule_6_bradycardia_nodal_blocker_conflict():
    """Rule 6: Nodal blocker in patient with resting bradycardia (< 50 bpm)."""
    patient = PatientRecord(
        patient_id=str(uuid.uuid4()),
        hospital_mrn="MRN-BRADY-01",
        name="Bradycardic Patient",
        hospital_id="HOSP-APEX",
        age=68,
        sex="Male",
        known_allergies="None",
        existing_conditions="CAD",
    )
    report = check_medication_safety(
        ["Metoprolol"],
        patient=patient,
        ecg_measurements={"heart_rate": 42.0},
    )
    brady_alerts = [a for a in report.alerts if "Bradycardia" in a.title or "AV Nodal" in a.title]

    assert len(brady_alerts) >= 1
    assert brady_alerts[0].severity == "CRITICAL"


def test_rule_7_condition_contraindication():
    """Rule 7: Patient condition contraindication (Heart Block + Metoprolol)."""
    p_mrn = f"MRN-CONTRA-{uuid.uuid4().hex[:6].upper()}"
    patient = PatientRecord(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=p_mrn,
        name="Heart Block Patient",
        hospital_id="HOSP-APEX",
        age=75,
        sex="Male",
        known_allergies="None",
        existing_conditions="Second-degree heart block, Bradycardia",
    )
    report = check_medication_safety(["Metoprolol"], patient=patient)
    contra_alerts = [a for a in report.alerts if a.category == "CONTRAINDICATION"]

    assert len(contra_alerts) >= 1
    assert contra_alerts[0].severity == "CRITICAL"


def test_rule_8_missing_context_safety_gate():
    """Rule 8: Missing clinical context triggers INSUFFICIENT status and warning alert."""
    report = check_medication_safety(["Metoprolol"], patient=None)

    assert report.context_completeness == "INSUFFICIENT"
    assert report.has_critical_alerts
    assert any("INSUFFICIENT CLINICAL CONTEXT" in a.title for a in report.alerts)


def test_rule_9_complete_context_validation():
    """Rule 9: Verified patient record with all fields achieves COMPLETE status."""
    patient = PatientRecord(
        patient_id=str(uuid.uuid4()),
        hospital_mrn="MRN-COMPLETE-01",
        name="Complete Patient",
        hospital_id="HOSP-APEX",
        age=50,
        sex="Female",
        known_allergies="None documented",
        existing_conditions="Mild Hypertension",
    )
    report = check_medication_safety(["Amlodipine"], patient=patient)

    assert report.context_completeness == "COMPLETE"
    assert not any(a.category == "CONTEXT_WARNING" for a in report.alerts)


def test_rule_10_authoritative_clinical_analysis_rule_1_compliance():
    """Rule 10: build_authoritative_clinical_analysis enforces CDSCO Rule 1: No autonomous prescribing."""
    auth_analysis = build_authoritative_clinical_analysis(
        ecg_finding="Premature Ventricular Contractions",
        ecg_measurements={"qrs_duration_ms": 130.0, "qtc_bazett_ms": 440.0, "heart_rate_bpm": 80.0},
        patient_profile={"age": 62, "sex": "Male", "existing_conditions": "CAD"},
        signal_quality_info={"category": "GOOD", "score": 1.0},
    )
    assert auth_analysis is not None
    # Verify candidates are marked as considerations, not autonomous prescriptions
    med_cds = auth_analysis.medication_decision_support
    assert med_cds is not None
    for cand in med_cds.candidates:
        assert cand.action_type in ["THERAPY_CONSIDERATION", "CLINICIAN_REVIEW_REQUIRED", "CONSIDERATION", "RECOMMENDATION"]
