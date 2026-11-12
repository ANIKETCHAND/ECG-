"""
Validation Suite for Patient-Specific ECG Analysis and Medication Intelligence.
==============================================================================

Verifies all requirements of the Clinical Decision Support and Medication Safety Engine:
1. Independent tests for:
   - Patient Context Engine
   - Medication Safety & Intelligence Engine
   - ECG Interpretation & Signal Quality Engine
2. Synthetic test cases for all 10 clinical archetypes:
   - Patient with Diabetes
   - Patient with Hypertension
   - Patient with Existing Cardiac Disease (CAD / Prior MI)
   - Patient with Kidney Disease (CKD)
   - Patient with Liver Disease (Cirrhosis / Transaminitis)
   - Patient with Known Drug Allergies
   - Patient with Polypharmacy (Multiple Medications)
   - Patient with Missing Medical History
   - Patient with Previous ECG Abnormalities
   - Patient with Conflicting Clinical Information
3. Complete End-to-End Hospital Workflow Integration:
   - Registration -> Vitals/Labs -> ECG Analysis -> CDS Explainability -> Doctor Review & Prescribing -> Two Reports.
"""

import json
import pytest
import numpy as np

from src.database.db_manager import DatabaseManager
from src.clinical.models import InformationSource, Patient
from src.clinical_context.context_engine import ClinicalContextFusionEngine
from src.clinical.recommendation_engine import ClinicalDecisionSupportEngine, FindingReliability
from src.clinical.medication_candidate_engine import (
    MultimodalMedicationCandidateEngine,
    build_authoritative_clinical_analysis,
)
from src.medications.interaction_checker import (
    check_medication_safety,
    reconcile_medications,
)
from src.medications.medication_database import (
    GLOBAL_MEDICATION_DB,
    resolve_drug_name,
)
from src.report.report_generator import generate_structured_report
from src.services.report_persistence_service import ReportPersistenceService


@pytest.fixture
def db(tmp_path):
    """Provides an isolated test database."""
    db_file = str(tmp_path / "test_ecg_clinical.db")
    manager = DatabaseManager(db_path=db_file)
    return manager


@pytest.fixture
def persistence_service(tmp_path):
    """Provides report persistence service tied to test database."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return ReportPersistenceService(local_reports_dir=reports_dir)


def create_synthetic_ecg_signal(duration_sec=10.0, fs=360.0, hr_bpm=72.0, noise_std=0.02):
    """Generate a clean synthetic lead II ECG signal for testing."""
    total_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # Base baseline
    sig = np.zeros(total_samples)
    
    # Add heartbeat pulses
    rr_interval_sec = 60.0 / hr_bpm
    beat_times = np.arange(0.2, duration_sec - 0.2, rr_interval_sec)
    
    for bt in beat_times:
        idx = int(bt * fs)
        # QRS spike
        sig[idx] += 1.5
        if idx > 0:
            sig[idx - 1] += 0.8
            sig[idx - 2] -= 0.2  # Q wave
        if idx + 1 < total_samples:
            sig[idx + 1] += 0.8
            sig[idx + 2] -= 0.3  # S wave
            
        # P wave
        p_idx = idx - int(0.16 * fs)
        if 0 <= p_idx < total_samples:
            sig[p_idx] += 0.2
            
        # T wave
        t_idx = idx + int(0.25 * fs)
        if 0 <= t_idx < total_samples:
            sig[t_idx] += 0.35
            
    # Add gentle noise
    sig += np.random.normal(0, noise_std, total_samples)
    return sig


# ==============================================================================
# 1. INDEPENDENT MODULE VERIFICATION
# ==============================================================================

def test_patient_context_engine_independent(db):
    """Test patient context engine independently: persists & retrieves full dossier without cross-leakage."""
    # Patient A: Complete dossier
    pat_a = db.create_patient(
        patient_id="PAT-ALPHA-01",
        hospital_mrn="MRN-2026-ALPHA",
        name="Alpha Patient",
        age=58,
        sex="Male",
        blood_group="A",
        rh_factor="Rh+",
        height_cm=178.0,
        weight_kg=84.0,
        known_allergies="Penicillin",
        food_allergies="Shellfish",
        other_allergies="Pollen",
        existing_conditions="Type 2 Diabetes, Hypertension",
        current_medications="Metformin 1000 mg, Lisinopril 20 mg",
        past_medications="Hydrochlorothiazide 25 mg",
        discontinued_medications="Amlodipine (edema)",
        previous_surgeries="Appendectomy 2010",
        previous_hospitalizations="Chest pain observation 2022",
        diabetes=True,
        hypertension=True,
    )
    
    # Patient B: Separate record with different blood group and conditions
    pat_b = db.create_patient(
        patient_id="PAT-BETA-02",
        hospital_mrn="MRN-2026-BETA",
        name="Beta Patient",
        age=42,
        sex="Female",
        blood_group="O",
        rh_factor="Rh-",
        known_allergies="Sulfa",
        existing_conditions="Asthma",
    )
    
    # Verify isolation
    retrieved_a = db.get_patient("PAT-ALPHA-01")
    retrieved_b = db.get_patient("PAT-BETA-02")
    
    assert retrieved_a.name == "Alpha Patient"
    assert retrieved_a.rh_factor == "Rh+"
    assert retrieved_a.height_cm == 178.0
    assert retrieved_a.bmi == round(84.0 / ((178.0 / 100.0) ** 2), 1)
    assert "Appendectomy" in retrieved_a.previous_surgeries
    assert "Shellfish" in retrieved_a.food_allergies
    assert retrieved_a.diabetes is True
    
    assert retrieved_b.name == "Beta Patient"
    assert retrieved_b.rh_factor == "Rh-"
    assert retrieved_b.blood_group == "O"
    assert "Sulfa" in retrieved_b.known_allergies
    # Ensure zero cross-contamination
    assert "Penicillin" not in (retrieved_b.known_allergies or "")
    assert "Diabetes" not in (retrieved_b.existing_conditions or "")
    
    # Test multimodal context fusion statements
    fusion_engine = ClinicalContextFusionEngine()
    profile_a = db.get_patient_multimodal_profile("PAT-ALPHA-01")
    statements = fusion_engine.fuse_context(
        ecg_finding="Sinus Rhythm",
        patient=profile_a,
    )
    stmt_texts = [s.statement for s in statements]
    assert any("Rh+" in s for s in stmt_texts)
    assert any("Appendectomy" in s for s in stmt_texts)
    assert any("Shellfish" in s for s in stmt_texts)
    assert any("Hydrochlorothiazide" in s for s in stmt_texts)


def test_medication_safety_engine_independent():
    """Test medication safety and intelligence engine independently with real repositories."""
    # 1. Lookup DailyMed, RxNorm, openFDA, SIDER attributes
    entry = GLOBAL_MEDICATION_DB.lookup_medication("carvedilol")
    assert entry is not None
    assert entry.generic_name.lower() == "carvedilol"
    assert "beta" in entry.drug_class.lower()
    assert entry.bradycardia_risk is True
    assert entry.hypotension_risk is True
    assert entry.rxnorm_cui is not None
    assert entry.dailymed_setid is not None
    assert len(entry.sider_adverse_effects) > 0
    
    # 2. Brand name resolution
    assert resolve_drug_name("Cordarone") == "amiodarone"
    assert resolve_drug_name("Toprol-XL") == "metoprolol"
    assert resolve_drug_name("Glucophage") == "metformin"
    assert resolve_drug_name("Jardiance") == "empagliflozin"

    # 3. Drug-Drug interaction check
    safety = check_medication_safety(
        medication_names=["amiodarone", "warfarin"],
        ecg_finding="Atrial Fibrillation",
        ecg_measurements={"heart_rate": 88.0, "qtc_ms": 430.0},
    )
    assert len(safety.alerts) > 0
    assert any(a.category == "INTERACTION" for a in safety.alerts)
    assert any("warfarin" in (a.title + " " + a.description + " " + a.clinical_recommendation).lower() for a in safety.alerts)
    
    # 4. Medication reconciliation check
    reconciled = reconcile_medications(
        current_medications=["Metoprolol 50 mg", "Lisinopril 10 mg"],
        previous_medications=["Metoprolol 25 mg", "Amlodipine 5 mg"],
        discontinued_medications=["Amlodipine 5 mg"],
    )
    assert "Amlodipine" in str(reconciled.discontinued)
    assert len(reconciled.dosage_changes) > 0


def test_ecg_interpretation_independent():
    """Test ECG interpretation independently without clinical history bias."""
    engine = ClinicalDecisionSupportEngine()
    
    # Reliable PVC finding
    rel_pvc = FindingReliability(
        reliable=True,
        signal_quality="GOOD",
        selective_gate_applied=True,
        reported_coverage=0.95,
    )
    rec_pvc = engine.evaluate_finding("Premature Ventricular Contractions (PVC)", heart_rate=78.0, reliability=rel_pvc)
    assert "PVC" in rec_pvc.finding
    assert rec_pvc.medication_recommendations_withheld is False
    assert len(rec_pvc.medication_recommendations) > 0
    
    # Unreliable read -> medications withheld per safety gate
    rel_bad = FindingReliability(
        reliable=False,
        reasons=["Signal quality POOR: baseline wander and excessive muscle artifact."],
        signal_quality="POOR",
    )
    rec_bad = engine.evaluate_finding("Atrial Fibrillation", heart_rate=115.0, reliability=rel_bad)
    assert rec_bad.medication_recommendations_withheld is True
    assert len(rec_bad.medication_recommendations) == 0
    assert len(rec_bad.withheld_medication_recommendations) > 0
    assert "poor" in rec_bad.withheld_reason.lower() or "withheld" in rec_bad.withheld_reason.lower()


# ==============================================================================
# 2. TEN SYNTHETIC CLINICAL ARCHETYPES
# ==============================================================================

def test_archetype_1_patient_with_diabetes():
    """Archetype 1: Diabetes Mellitus — Guideline options (SGLT2i, Metformin), autonomic neuropathy explainability."""
    engine = MultimodalMedicationCandidateEngine()
    cds = ClinicalDecisionSupportEngine()
    
    p_profile = {
        "age": 62,
        "sex": "Female",
        "conditions": "Type 2 Diabetes Mellitus, Dyslipidemia",
        "current_medications": "Metformin 1000 mg BID, Atorvastatin 20 mg",
        "allergies": "None",
    }
    vitals = {"systolic_bp": 128.0, "diastolic_bp": 82.0, "heart_rate": 84.0}
    labs = {"potassium": 4.3, "creatinine": 0.9, "egfr": 78.0}
    
    rec = cds.evaluate_finding("Sinus Tachycardia", heart_rate=102.0, patient=p_profile, vital_signs=vitals, laboratory_results=labs)
    assert any(c["condition"] == "Diabetes Mellitus" for c in rec.relevant_conditions)
    assert any("autonomic" in exp["ecg_relationship"].lower() for exp in rec.relevance_explanations if exp["factor"] == "Diabetes Mellitus")
    
    # Medication candidate synthesis
    res = engine.generate_candidate_recommendations(
        ecg_finding="Normal Sinus Rhythm",
        patient_profile=p_profile,
        vital_signs=vitals,
        laboratory_results=labs,
    )
    cand_names = [c.medication for c in res.candidates]
    assert any("SGLT2" in n for n in cand_names)


def test_archetype_2_patient_with_hypertension():
    """Archetype 2: Hypertension — Evaluates LV afterload strain, ACEi/ARB and CCB candidates, BP correlation."""
    cds = ClinicalDecisionSupportEngine()
    engine = MultimodalMedicationCandidateEngine()
    
    p_profile = {
        "age": 55,
        "sex": "Male",
        "conditions": "Essential Hypertension",
        "current_medications": "Amlodipine 5 mg OD",
    }
    vitals = {"systolic_bp": 165.0, "diastolic_bp": 98.0, "heart_rate": 74.0}
    labs = {"potassium": 4.1, "creatinine": 1.0, "egfr": 85.0}
    
    rec = cds.evaluate_finding("Normal Sinus Rhythm", heart_rate=74.0, patient=p_profile, vital_signs=vitals, laboratory_results=labs)
    assert any(c["condition"] == "Hypertension" for c in rec.relevant_conditions)
    assert any("hypertrophy" in exp["ecg_relationship"].lower() for exp in rec.relevance_explanations if exp["factor"] == "Hypertension")
    
    cand = engine.generate_candidate_recommendations("Normal Sinus Rhythm", patient_profile=p_profile, vital_signs=vitals, laboratory_results=labs)
    assert any("ACE" in c.drug_class or "Calcium" in c.drug_class for c in cand.candidates)


def test_archetype_3_patient_with_existing_cardiac_disease():
    """Archetype 3: CAD / Prior MI — Identifies myocardial scar conduction deceleration and antiplatelet/statin options."""
    cds = ClinicalDecisionSupportEngine()
    p_profile = {
        "age": 67,
        "sex": "Male",
        "conditions": "Coronary Artery Disease, Prior Anteroseptal Myocardial Infarction",
        "current_medications": "Aspirin 81 mg, Atorvastatin 40 mg, Metoprolol succinate 50 mg",
    }
    labs = {"potassium": 4.4, "creatinine": 1.1, "troponin": 0.01}
    
    rec = cds.evaluate_finding("Premature Ventricular Contractions (PVC)", heart_rate=70.0, patient=p_profile, laboratory_results=labs)
    assert any("Coronary" in c["condition"] or "MI" in c["condition"] for c in rec.relevant_conditions)
    assert any("scar" in exp["ecg_relationship"].lower() or "re-entrant" in exp["ecg_relationship"].lower() for exp in rec.relevance_explanations)
    # CAST trial warning for prior MI
    assert any("CAST" in w or "flecainide" in w.lower() for w in rec.contraindication_warnings)


def test_archetype_4_patient_with_kidney_disease():
    """Archetype 4: Chronic Kidney Disease — Renal threshold checking (eGFR < 30 contraindicates Metformin)."""
    engine = MultimodalMedicationCandidateEngine()
    cds = ClinicalDecisionSupportEngine()
    
    p_profile = {
        "age": 71,
        "sex": "Female",
        "conditions": "Chronic Kidney Disease Stage 4, Type 2 Diabetes",
        "current_medications": "None",
        "allergies": "None",
    }
    vitals = {"systolic_bp": 130.0, "diastolic_bp": 80.0, "heart_rate": 68.0}
    labs = {"potassium": 4.8, "creatinine": 2.8, "egfr": 22.0}  # Severe renal impairment
    
    cand_data = engine.generate_candidate_recommendations(
        ecg_finding="Normal Sinus Rhythm",
        patient_profile=p_profile,
        vital_signs=vitals,
        laboratory_results=labs,
    )
    # Metformin must be blocked due to lactic acidosis risk per KDIGO & FDA label
    metformin_cand = next((c for c in cand_data.candidates if "Biguanide" in c.medication or "Metformin" in c.medication), None)
    assert metformin_cand is not None
    assert metformin_cand.safety_status == "BLOCKED"
    assert any("lactic acidosis" in ci.lower() for ci in metformin_cand.contraindications_detected)


def test_archetype_5_patient_with_liver_disease():
    """Archetype 5: Liver Disease — Transaminitis / cirrhosis checks (Statins blocked, Amiodarone cautioned)."""
    engine = MultimodalMedicationCandidateEngine()
    cds = ClinicalDecisionSupportEngine()
    
    p_profile = {
        "age": 59,
        "sex": "Male",
        "conditions": "Decompensated Hepatic Cirrhosis, Dyslipidemia",
        "current_medications": "Spironolactone 50 mg",
        "allergies": "None",
    }
    vitals = {"systolic_bp": 110.0, "diastolic_bp": 70.0, "heart_rate": 72.0}
    labs = {"alt_u_l": 160.0, "ast_u_l": 185.0, "bilirubin_mg_dl": 2.8, "potassium": 4.2}  # Active transaminitis
    
    cand_data = engine.generate_candidate_recommendations(
        ecg_finding="Normal Sinus Rhythm",
        patient_profile=p_profile,
        vital_signs=vitals,
        laboratory_results=labs,
    )
    statin_cand = next((c for c in cand_data.candidates if "Statin" in c.medication), None)
    assert statin_cand is not None
    assert statin_cand.safety_status == "BLOCKED"
    assert any("Hepatic" in ci or "transaminitis" in ci.lower() for ci in statin_cand.contraindications_detected)


def test_archetype_6_patient_with_known_drug_allergies():
    """Archetype 6: Known Drug Allergies — Blocks contraindicated candidate classes and presents allergy alerts."""
    engine = MultimodalMedicationCandidateEngine()
    
    p_profile = {
        "age": 50,
        "sex": "Female",
        "conditions": "Hypertension, Palpitations",
        "allergies": "Beta-Blockers (Anaphylaxis), Aspirin (Hives)",
        "current_medications": "None",
    }
    cand_data = engine.generate_candidate_recommendations(
        ecg_finding="Premature Ventricular Contractions (PVC)",
        patient_profile=p_profile,
    )
    bb_cand = next((c for c in cand_data.candidates if "Beta" in c.medication), None)
    assert bb_cand is not None
    assert bb_cand.safety_status == "BLOCKED"
    assert bb_cand.allergy_check == "ALLERGY_ALERT"
    assert any("allergy conflict" in a["alert"].lower() for a in cand_data.allergy_checks)


def test_archetype_7_patient_with_polypharmacy():
    """Archetype 7: Taking Multiple Medications — Detects DDIs (Beta-Blocker + Non-DHP CCB -> severe AV block)."""
    engine = MultimodalMedicationCandidateEngine()
    
    p_profile = {
        "age": 68,
        "sex": "Male",
        "conditions": "Atrial Fibrillation, Hypertension",
        "current_medications": "Diltiazem 180 mg, Digoxin 0.125 mg, Warfarin 5 mg",
    }
    cand_data = engine.generate_candidate_recommendations(
        ecg_finding="Atrial Fibrillation",
        patient_profile=p_profile,
    )
    bb_cand = next((c for c in cand_data.candidates if "Beta" in c.medication), None)
    assert bb_cand is not None
    assert bb_cand.safety_status == "REVIEW_REQUIRED"
    assert any("Non-DHP" in inter for inter in bb_cand.drug_interactions_detected)


def test_archetype_8_patient_with_missing_medical_history():
    """Archetype 8: Missing Medical History — Zero fabrication, missing info audit, withholding routine recommendations."""
    cds = ClinicalDecisionSupportEngine()
    engine = MultimodalMedicationCandidateEngine()
    
    # Completely empty patient context
    rec = cds.evaluate_finding("Premature Ventricular Contractions (PVC)")
    assert len(rec.missing_information) >= 4
    assert any("Vital signs" in m for m in rec.missing_information)
    assert any("Laboratory" in m for m in rec.missing_information)
    
    cand_data = engine.generate_candidate_recommendations(
        ecg_finding="Premature Ventricular Contractions (PVC)",
        patient_profile={},
        vital_signs={},
        laboratory_results={},
        finding_reliability=FindingReliability(reliable=False),
    )
    assert cand_data.status in ["WITHHELD", "INSUFFICIENT_INFORMATION"]
    assert len(cand_data.missing_information) >= 3


def test_archetype_9_patient_with_previous_ecg_abnormalities():
    """Archetype 9: Previous ECG Abnormalities — Compares longitudinal changes and tracks evolution."""
    prev_comp = {
        "has_previous_ecg": True,
        "previous_record_id": "REC-HIST-001",
        "previous_timestamp": "2025-10-14T09:30:00",
        "delta_heart_rate_bpm": 15.0,
        "delta_qtc_ms": 35.0,
        "rhythm_evolution": "PVC BURDEN INCREASED",
        "comparison_summary": "Comparison with baseline: QTc prolonged by +35 ms; PVC burden increased.",
    }
    
    auth_ca = build_authoritative_clinical_analysis(
        ecg_finding="Premature Ventricular Contractions (PVC)",
        ecg_measurements={"heart_rate_bpm": 85.0, "qtc_bazett_ms": 455.0},
        previous_comparison=prev_comp,
    )
    assert auth_ca.previous_ecg_comparison.has_previous_ecg is True
    assert auth_ca.previous_ecg_comparison.delta_qtc_ms == 35.0
    assert "PVC BURDEN" in auth_ca.previous_ecg_comparison.rhythm_evolution


def test_archetype_10_patient_with_conflicting_clinical_information():
    """Archetype 10: Conflicting Clinical Info — Normal ECG in a patient with severe active chest pain."""
    cds = ClinicalDecisionSupportEngine()
    p_profile = {
        "age": 58,
        "sex": "Male",
        "symptoms": "Severe crushing substernal chest pain radiating to left arm, diaphoresis",
        "conditions": "Hypertension",
        "current_medications": "Amlodipine 10 mg",
    }
    labs = {"troponin": 0.08}  # Mildly elevated troponin
    
    rec = cds.evaluate_finding("Normal Sinus Rhythm", heart_rate=80.0, patient=p_profile, laboratory_results=labs)
    # The CDS considerations must warn that a normal resting ECG does NOT exclude transient ischemia or NSTEMI
    assert any("does not exclude" in c.lower() or "ischemia" in c.lower() for c in rec.clinical_considerations)
    assert any("symptoms" in a.lower() for a in rec.suggested_assessments)


# ==============================================================================
# 3. END-TO-END HOSPITAL WORKFLOW & TWO REPORTS
# ==============================================================================

def test_complete_hospital_workflow_and_two_reports(db, persistence_service):
    """Test full integrated hospital workflow: Register -> Vitals/Labs -> ECG Analysis -> CDS -> Doctor Review & Prescribing -> Two Reports."""
    # 1. Register Patient
    patient = db.create_patient(
        patient_id="PAT-WORKFLOW-99",
        hospital_mrn="MRN-WF-2026",
        name="Eleanor Vance",
        age=64,
        sex="Female",
        blood_group="B",
        rh_factor="Rh+",
        height_cm=165.0,
        weight_kg=70.0,
        known_allergies="Sulfa",
        existing_conditions="Hypertension, Heart Failure with Preserved Ejection Fraction",
        current_medications="Lisinopril 20 mg OD, Carvedilol 12.5 mg BID",
        diabetes=False,
        hypertension=True,
        heart_failure=True,
    )
    assert patient.patient_id == "PAT-WORKFLOW-99"
    
    # 2. Record Vitals & Labs
    db.record_vital_signs(
        vital_id="VIT-WF-001",
        patient_id=patient.patient_id,
        heart_rate_bpm=76.0,
        systolic_bp_mmhg=138.0,
        diastolic_bp_mmhg=86.0,
        spo2_percent=97.0,
        temperature_c=36.8,
    )
    db.record_laboratory_results(
        lab_id="LAB-WF-001",
        patient_id=patient.patient_id,
        potassium_mmol_l=4.2,
        sodium_mmol_l=140.0,
        serum_creatinine_mg_dl=1.0,
        egfr_ml_min=68.0,
        troponin_i_ng_ml=0.01,
        alt_u_l=25.0,
        ast_u_l=22.0,
        bilirubin_mg_dl=0.7,
    )
    
    # 3. Retrieve Context & Pre-populate Analysis
    prof = db.get_patient_multimodal_profile(patient.patient_id)
    vitals_dict = db.get_patient_vitals(patient.patient_id)[0]
    labs_dict = db.get_patient_labs(patient.patient_id)[0]
    
    meas_dict = {
        "heart_rate_bpm": 76.0,
        "mean_rr_ms": 789.0,
        "pr_interval_ms": 164.0,
        "qrs_duration_ms": 92.0,
        "qt_interval_ms": 380.0,
        "qtc_bazett_ms": 418.0,
    }
    
    # 4. CDS & Medication Decision Support
    cds = ClinicalDecisionSupportEngine()
    cds_rec = cds.evaluate_finding(
        ecg_finding="Normal Sinus Rhythm",
        heart_rate=76.0,
        patient=prof,
        vital_signs=vitals_dict,
        laboratory_results=labs_dict,
        ecg_measurements=meas_dict,
        reliability=FindingReliability(reliable=True, signal_quality="GOOD"),
    )
    assert len(cds_rec.relevant_conditions) > 0
    assert any("Hypertension" in c["condition"] for c in cds_rec.relevant_conditions)
    
    # Candidate generation
    med_engine = MultimodalMedicationCandidateEngine()
    candidates_data = med_engine.generate_candidate_recommendations(
        ecg_finding="Normal Sinus Rhythm",
        patient_profile=prof.to_dict(),
        vital_signs=vitals_dict,
        laboratory_results=labs_dict,
        finding_reliability=FindingReliability(reliable=True),
    )
    assert len(candidates_data.candidates) > 0
    
    # 5. Doctor Review & Authorized Prescription
    review_record = {
        "clinician_name": "Dr. Sarah Lin, MD",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MED-REG-44910",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": "Normal Sinus Rhythm with preserved PR/QRS. Well-controlled baseline hypertension.",
        "clinician_diagnosis": "Controlled Essential Hypertension (ICD-10 I10)",
        "clinical_notes": "Patient reports good exercise tolerance. Continue home BP logs.",
        "selected_medications": [{"name": "Lisinopril", "dosage": "20 mg OD"}],
        "rejected_medications": [{"name": "Class III Antiarrhythmic", "reason": "Not indicated for normal rhythm"}],
        "prescriptions": [
            {
                "medication": "Lisinopril",
                "generic_name": "Lisinopril",
                "dosage": "20 mg",
                "frequency": "Once daily with water in morning",
                "duration": "90 days",
                "refills": 3,
            }
        ],
        "treatment_instructions": "Take Lisinopril 20 mg OD every morning. Maintain low-sodium DASH diet.",
        "follow_up_instructions": "Review blood pressure log and CMP in 6 months.",
    }
    
    # Persist and seal review
    analysis_id = "ANL-TEST-WF-001"
    db.save_ecg_record(
        record_id="REC-WF-001",
        sampling_rate=360.0,
        lead_names=["II"],
        duration_sec=10.0,
        file_hash="AABBCC112233",
        source_format="DIGITAL",
        patient_id=patient.patient_id,
        signal_quality="GOOD",
        quality_score=0.96,
    )
    db.save_analysis_result(
        analysis_id=analysis_id,
        record_id="REC-WF-001",
        model_id="RandomForestClassifier",
        model_version="1.0.0",
        prediction="Normal Sinus Rhythm",
        probabilities={"Normal Sinus Rhythm": 0.98, "PVC": 0.02},
        signal_quality="GOOD",
        quality_score=0.96,
        heart_rate_bpm=76.0,
    )
    
    # Save preliminary report snapshot
    persistence_service.save_report_snapshot(
        report_data={
            "analysis_id": analysis_id,
            "patient_id": patient.patient_id,
            "record_id": "REC-WF-001",
            "findings": ["Normal Sinus Rhythm"],
            "model_prediction": "Normal Sinus Rhythm",
        },
        report_id=analysis_id,
        patient_id=patient.patient_id,
        ecg_id="REC-WF-001",
        analysis_id=analysis_id,
        status="PENDING_REVIEW",
    )
    
    seal_res = persistence_service.sign_and_seal_report(
        report_id=analysis_id,
        clinician_user_id="DOC-SARAH-01",
        clinician_name=review_record["clinician_name"],
        clinician_role=review_record["clinician_role"],
        registration_number=review_record["registration_number"],
        agreement_status=review_record["agreement_status"],
        clinician_interpretation=review_record["clinician_interpretation"],
        clinical_notes=review_record["clinical_notes"],
        clinician_diagnosis=review_record["clinician_diagnosis"],
        prescriptions=review_record["prescriptions"],
        rejected_medications=review_record["rejected_medications"],
        treatment_instructions=review_record["treatment_instructions"],
        follow_up_instructions=review_record["follow_up_instructions"],
    )
    assert seal_res["status"] in ["SAVE_SUCCESS", "SIGNED"]
    saved_rep = persistence_service.get_report(analysis_id)
    assert saved_rep["status"] == "SIGNED"
    
    # 6. Generate Two Separate Reports
    # A. Structured Technical Doctor Report
    doc_report = generate_structured_report(
        input_info={"format": "DIGITAL_LEAD_II", "sampling_rate": 360.0},
        ai_results={"signal_quality": "GOOD", "heart_rate_bpm": 76.0, "status": "COMPLETED", "primary_pattern": "Normal Sinus Rhythm", "beat_distribution": {"PVC": {"count": 0, "percentage": 0.0}}},
        patient_profile=prof.to_dict(),
        vital_signs=vitals_dict,
        laboratory_results=labs_dict,
        clinician_review=review_record,
        cds_report=cds_rec.to_dict(),
    )
    assert doc_report["patient_info"]["patient_name"] == "Eleanor Vance"
    assert doc_report["patient_info"]["rh_factor"] == "Rh+"
    assert doc_report["clinician_review"]["clinician_diagnosis"] == "Controlled Essential Hypertension (ICD-10 I10)"
    assert len(doc_report["clinician_review"]["prescriptions"]) == 1
    assert "Eleanor Vance" in doc_report["patient_info"]["patient_name"]
    
    # B. Generate PDF Reports (Patient vs Doctor)
    from src.report.pdf_generator import generate_doctor_report, generate_patient_report
    sig = create_synthetic_ecg_signal(duration_sec=6.0, hr_bpm=76.0)
    
    # Doctor PDF
    doc_pdf = generate_doctor_report(doc_report, waveform=sig, fs=360.0)
    assert len(doc_pdf) > 1000
    assert doc_pdf.startswith(b"%PDF")
    
    # Patient PDF
    pat_pdf = generate_patient_report(doc_report, waveform=sig, fs=360.0)
    assert len(pat_pdf) > 1000
    assert pat_pdf.startswith(b"%PDF")


def test_emergency_escalation_severe_hyperkalemia_and_shock():
    """Verify Section 5 Safety Gate: Critical clinical emergencies trigger EMERGENCY_ESCALATION and withhold routine candidates."""
    engine = MultimodalMedicationCandidateEngine()

    # Case 1: Critical Hyperkalemia (K+ = 6.5 mmol/L)
    res_k = engine.generate_candidate_recommendations(
        ecg_finding="Normal Sinus Rhythm",
        patient_profile={"patient_id": "PAT-EM-01", "name": "Crit K Patient", "existing_conditions": "Hypertension"},
        vital_signs={"systolic_bp": 130, "heart_rate": 72},
        laboratory_results={"potassium_mmol_l": 6.5, "creatinine_mg_dl": 2.1},
        finding_reliability=FindingReliability(reliable=True)
    )
    assert res_k.status == "EMERGENCY_ESCALATION"
    assert len(res_k.candidates) == 0
    assert "EMERGENCY ESCALATION" in res_k.status_reason
    assert "hyperkalemia" in res_k.status_reason.lower()

    # Case 2: Profound Bradycardia (HR = 36 bpm)
    res_brady = engine.generate_candidate_recommendations(
        ecg_finding="Sinus Bradycardia",
        patient_profile={"patient_id": "PAT-EM-02", "name": "Crit Brady Patient", "existing_conditions": "Hypertension"},
        vital_signs={"systolic_bp_mmhg": 120, "heart_rate_bpm": 36},
        laboratory_results={"potassium_mmol_l": 4.1},
        finding_reliability=FindingReliability(reliable=True)
    )
    assert res_brady.status == "EMERGENCY_ESCALATION"
    assert len(res_brady.candidates) == 0
    assert "bradycardia" in res_brady.status_reason.lower()

    # Case 3: Cardiogenic Shock / Profound Hypotension (SBP = 78 mmHg)
    res_shock = engine.generate_candidate_recommendations(
        ecg_finding="Normal Sinus Rhythm",
        patient_profile={"patient_id": "PAT-EM-03", "name": "Shock Patient", "existing_conditions": "Heart Failure"},
        vital_signs={"systolic_bp": 78, "heart_rate": 105},
        laboratory_results={"potassium_mmol_l": 4.0},
        finding_reliability=FindingReliability(reliable=True)
    )
    assert res_shock.status == "EMERGENCY_ESCALATION"
    assert len(res_shock.candidates) == 0
    assert "hypotension" in res_shock.status_reason.lower()


def test_multi_ingredient_combination_reconciliation():
    """Verify combination drug decomposition and multi-ingredient interaction / duplication detection."""
    from src.medications.interaction_checker import reconcile_medications

    # Patient is on Sacubitril/Valsartan (Entresto) + Valsartan (Diovan)
    med_list = [
        {"medication_name": "Sacubitril/Valsartan 24/26 mg", "dosage": "50 mg", "frequency": "BID"},
        {"medication_name": "Valsartan", "dosage": "80 mg", "frequency": "OD"}
    ]
    reconciliation = reconcile_medications(med_list)
    rec_dict = reconciliation.to_dict() if hasattr(reconciliation, "to_dict") else reconciliation
    dups = rec_dict.get("therapeutic_duplications", [])
    assert len(dups) > 0
    dup_details = " ".join([d.get("details", "") if isinstance(d, dict) else str(d) for d in dups])
    assert "ARB" in dup_details or "Valsartan" in dup_details or "therapeutic duplication" in dup_details.lower() or "Angiotensin" in dup_details


def test_standard_keys_potassium_and_vital_alerts():
    """Verify that check_medication_safety correctly reads standard multimodal dictionary keys."""
    from src.medications.interaction_checker import check_medication_safety

    # Digoxin with hypokalemia (potassium_mmol_l = 2.8)
    pat_dict = {
        "name": "Digoxin Patient",
        "known_allergies": "None",
        "current_medications": "Digoxin 0.25 mg OD",
        "potassium_mmol_l": 2.8,
        "heart_rate_bpm": 68
    }
    safety_res = check_medication_safety(
        medication_names=["Digoxin 0.25 mg OD"],
        patient=pat_dict
    )
    res_dict = safety_res.to_dict() if hasattr(safety_res, "to_dict") else safety_res
    alerts = res_dict.get("alerts", [])
    assert len(alerts) > 0
    all_issues = " ".join([f"{a.get('title','')} {a.get('description','')}" for a in alerts])
    assert "hypokalemia" in all_issues.lower() or "potassium" in all_issues.lower() or "digoxin" in all_issues.lower()


def test_clinician_review_endpoint_seals_report_with_distinct_analysis_id():
    """Verify that POST /api/review correctly seals the report in DB when report_id != analysis_id."""
    import uuid
    from fastapi.testclient import TestClient
    from api.index import app, DB_MANAGER, REPORT_PERSISTENCE_SERVICE

    client = TestClient(app)
    suffix = uuid.uuid4().hex[:6].upper()

    # 1. Setup Patient & ECG in DB
    p_id = f"PAT-REV-{suffix}"
    mrn = f"MRN-REV-{suffix}"
    DB_MANAGER.create_patient(
        patient_id=p_id,
        hospital_mrn=mrn,
        name="Review Test Patient",
        age=58,
        sex="Male"
    )
    rec_id = f"REC-REV-{suffix}"
    DB_MANAGER.save_ecg_record(
        record_id=rec_id,
        sampling_rate=360.0,
        lead_names=["II"],
        duration_sec=10.0,
        file_hash=f"REVHASH{suffix}",
        source_format="DIGITAL",
        patient_id=p_id,
        signal_quality="GOOD",
        quality_score=0.95
    )
    anl_id = f"ANL-REV-{suffix}"
    DB_MANAGER.save_analysis_result(
        analysis_id=anl_id,
        record_id=rec_id,
        model_id="RandomForestClassifier",
        model_version="1.0.0",
        prediction="Normal Sinus Rhythm",
        probabilities={"Normal Sinus Rhythm": 0.95},
        signal_quality="GOOD",
        quality_score=0.95,
        heart_rate_bpm=72.0
    )
    rep_id = f"REP-DISTINCT-{suffix}"  # Distinct from anl_id!
    REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data={
            "analysis_id": anl_id,
            "patient_id": p_id,
            "record_id": rec_id,
            "findings": ["Normal Sinus Rhythm"]
        },
        report_id=rep_id,
        patient_id=p_id,
        ecg_id=rec_id,
        analysis_id=anl_id,
        status="PENDING_REVIEW"
    )

    # Verify initial status is PENDING_REVIEW
    pre_rep = REPORT_PERSISTENCE_SERVICE.get_report(rep_id)
    assert pre_rep is not None
    assert pre_rep["status"] == "PENDING_REVIEW"

    # 2. Call POST /api/review sending analysis_id (simulating the UI payload)
    review_payload = {
        "analysis_id": anl_id,
        "clinician_id": "DOC-REV-42",
        "clinician_name": "Dr. Aniket Chandra, MD",
        "action": "CONFIRMED",
        "clinician_diagnosis": "Normal Sinus Rhythm with preserved intervals",
        "notes": "Clinician confirms benign rhythm.",
        "prescribed_medication": "None",
        "signature": "Dr. Aniket Chandra, MD (Attested)"
    }
    response = client.post("/api/review", json=review_payload)
    assert response.status_code == 200, f"Expected 200, got: {response.text}"
    res_data = response.json()
    assert res_data["status"] in ["SEALED", "success"]
    assert res_data["report_status"] == "SIGNED"

    # 3. Verify in database that report status is SIGNED and sealed
    post_rep = REPORT_PERSISTENCE_SERVICE.get_report(rep_id)
    assert post_rep["status"] == "SIGNED"
    clin_rev = post_rep.get("report_data", {}).get("clinician_review", {})
    assert clin_rev.get("clinician_diagnosis") == "Normal Sinus Rhythm with preserved intervals"
    assert clin_rev.get("status") == "SIGNED"


def test_patient_registration_endpoint_with_section1_fields(db):
    """Verify that POST /api/patients accepts and persists all Section 1 multimodal clinical fields."""
    import uuid
    from fastapi.testclient import TestClient
    from api.index import app

    client = TestClient(app)
    mrn = f"MRN-S1-{uuid.uuid4().hex[:6].upper()}"

    reg_payload = {
        "name": "Devika Raman",
        "hospital_mrn": mrn,
        "hospital_id": "HOSP-APEX",
        "date_of_birth": "1978-05-20",
        "age": 48,
        "sex": "Female",
        "contact": "+91 9876543210",
        "email": "devika.raman@example.com",
        "address": "Bangalore, India",
        "emergency_contact": "+91 9876500000",
        "blood_group": "B+",
        "rh_factor": "Rh+",
        "smoking_status": "Non-Smoker",
        "known_allergies": "Aspirin (urticaria)",
        "food_allergies": "Peanuts",
        "other_allergies": "Latex",
        "existing_conditions": "Type 2 Diabetes, Hypertension, CKD Stage 2",
        "diabetes": True,
        "hypertension": True,
        "coronary_artery_disease": False,
        "previous_heart_attack": False,
        "heart_failure": False,
        "arrhythmia_history": False,
        "kidney_disease": True,
        "liver_disease": False,
        "thyroid_disorders": False,
        "current_medications": "Metformin 500 mg BID, Telmisartan 40 mg OD",
        "past_medications": "Amlodipine (peripheral edema)",
        "discontinued_medications": "Amlodipine",
        "previous_surgeries": "Cholecystectomy (2018)",
        "previous_hospitalizations": "None in past 5 years",
        "family_history": "Father had MI at age 62",
        "height_cm": 162.0,
        "weight_kg": 64.0,
        "bmi": 24.4,
        "systolic_bp": 128.0,
        "diastolic_bp": 82.0,
        "heart_rate": 74.0,
        "spo2": 99.0,
        "temperature_c": 36.9,
        "potassium_mmol_l": 4.4,
        "sodium_mmol_l": 139.0,
        "creatinine_mg_dl": 1.1,
        "egfr_ml_min": 64.0,
        "troponin_i_ng_ml": 0.008,
        "alt_u_l": 22.0
    }

    res = client.post("/api/patients", json=reg_payload)
    assert res.status_code in [200, 201], f"Expected 200/201, got: {res.text}"
    p_data = res.json()["patient"]
    p_id = p_data["patient_id"]

    # Verify through GET /api/patients/{id}
    get_res = client.get(f"/api/patients/{p_id}")
    assert get_res.status_code == 200
    retrieved = get_res.json()["patient"]
    assert retrieved["name"] == "Devika Raman"
    assert retrieved["rh_factor"] == "Rh+"
    assert retrieved["food_allergies"] == "Peanuts"
    assert retrieved["other_allergies"] == "Latex"
    assert retrieved["diabetes"] in [True, "1", 1]
    assert retrieved["kidney_disease"] in [True, "1", 1]
    assert retrieved["height_cm"] == 162.0
    assert retrieved["weight_kg"] == 64.0


def test_vital_signs_temperature_c_and_aliases():
    """Verify VitalSigns model has temperature_c and aliases, avoiding AttributeError in context engine."""
    from src.clinical.models import VitalSigns, Patient
    from src.clinical_context.context_engine import ClinicalContextFusionEngine

    vs = VitalSigns(
        heart_rate_bpm=72.0,
        systolic_bp_mmhg=120.0,
        diastolic_bp_mmhg=80.0,
        spo2_percent=98.0,
        temperature_celsius=37.1,
        recorded_at="2026-11-12T10:00:00",
    )
    assert vs.temperature_c == 37.1
    assert vs.blood_pressure == "120/80 mmHg"
    assert vs.systolic_bp == 120.0

    pat = Patient(
        patient_id="PAT-TEMP-01",
        hospital_mrn="MRN-TEMP-01",
        name="Temp Test",
        vital_signs=[vs],
    )
    fusion = ClinicalContextFusionEngine()
    stmts = fusion.fuse_context("Normal Sinus Rhythm", patient=pat)
    v_stmt = next((s.statement for s in stmts if s.category == "VITAL_SIGN"), None)
    assert v_stmt is not None
    assert "37.1°C" in v_stmt
    assert "120/80 mmHg" in v_stmt


def test_hepatic_safety_and_beers_criteria_with_dictionary_patient():
    """Verify check_medication_safety works with standard dictionary patient profiles."""
    # 1. Hepatic transaminitis check
    safety_rep = check_medication_safety(
        medication_names=["Atorvastatin 40 mg"],
        patient={
            "patient_id": "PAT-HEP-01",
            "name": "Hepatic Patient",
            "existing_conditions": "Decompensated Cirrhosis",
            "alt_u_l": 160.0,
            "age": 62,
            "known_allergies": "None",
        },
    )
    alert_titles = [a.title for a in safety_rep.alerts]
    assert any("Hepatic Safety Advisory: Atorvastatin" in t for t in alert_titles)

    # 2. Beers criteria in elderly patient
    safety_beers = check_medication_safety(
        medication_names=["Digoxin 0.25 mg"],
        patient={
            "patient_id": "PAT-GER-01",
            "name": "Geriatric Patient",
            "age": 78,
            "known_allergies": "None",
            "existing_conditions": "Hypertension",
        },
    )
    alert_beers_titles = [a.title for a in safety_beers.alerts]
    assert any("AGS Beers Criteria Advisory: Digoxin" in t for t in alert_beers_titles)


def test_generate_structured_report_with_standard_multimodal_keys():
    """Verify generate_structured_report correctly maps standard multimodal lab and vital keys."""
    vitals = {
        "systolic_bp_mmhg": 136.0,
        "diastolic_bp_mmhg": 84.0,
        "heart_rate_bpm": 78.0,
        "spo2_percent": 99.0,
        "temperature_celsius": 36.8,
    }
    labs = {
        "potassium_mmol_l": 4.5,
        "sodium_mmol_l": 141.0,
        "serum_creatinine_mg_dl": 0.95,
        "egfr_ml_min": 82.0,
        "troponin_i_ng_ml": 0.01,
        "alt_u_l": 24.0,
    }
    rep = generate_structured_report(
        input_info={},
        vital_signs=vitals,
        laboratory_results=labs,
    )
    v_data = rep["vital_signs"]
    assert v_data["status"] == "RECORDED"
    assert v_data["blood_pressure"] == "136.0/84.0 mmHg"
    assert v_data["heart_rate_bpm"] == 78.0
    assert v_data["temperature_c"] == "36.8 °C"

    l_data = rep["laboratory_results"]
    assert l_data["status"] == "RECORDED"
    assert "4.5" in l_data["potassium_meq_l"]
    assert "0.95" in l_data["creatinine_mg_dl"]
    assert "24" in l_data["alt_u_l"]


def test_cds_missing_information_with_direct_vital_signs_and_labs():
    """Verify evaluate_finding does not flag vitals/labs as missing when passed as arguments."""
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        "Sinus Tachycardia",
        heart_rate=105.0,
        patient={"patient_id": "PAT-01", "name": "Vitals Patient", "existing_conditions": "None", "known_allergies": "None"},
        vital_signs={"systolic_bp_mmhg": 120.0, "heart_rate_bpm": 105.0},
        laboratory_results={"potassium_mmol_l": 4.1, "creatinine_mg_dl": 0.9},
    )
    assert not any("Vital signs: Not provided" in m for m in rec.missing_information)
    assert not any("Laboratory data: Not provided" in m for m in rec.missing_information)


def test_signed_report_status_not_downgraded_by_subsequent_snapshot_save(persistence_service):
    """Verify that saving a snapshot with PENDING_REVIEW does not demote an already SIGNED report."""
    rep_id = "REP-SEAL-TEST-99"
    # 1. Save preliminary report
    persistence_service.save_report_snapshot(
        report_data={"analysis_id": "ANL-99", "patient_id": "PAT-99"},
        report_id=rep_id,
        status="PENDING_REVIEW",
    )
    # 2. Seal report as SIGNED
    persistence_service.sign_and_seal_report(
        report_id=rep_id,
        clinician_user_id="DOC-99",
        clinician_name="Dr. Tester, MD",
        clinician_role="CARDIOLOGIST",
        registration_number="REG-99",
        agreement_status="CONFIRMED",
        clinician_interpretation="Normal study confirmed.",
    )
    sealed = persistence_service.get_report(rep_id)
    assert sealed["status"] == "SIGNED"

    # 3. Simulate another service saving snapshot with status='PENDING_REVIEW'
    persistence_service.save_report_snapshot(
        report_data={"analysis_id": "ANL-99", "patient_id": "PAT-99"},
        report_id=rep_id,
        status="PENDING_REVIEW",
    )
    rechecked = persistence_service.get_report(rep_id)
    assert rechecked["status"] == "SIGNED"
    assert "clinician_review" in rechecked.get("report_data", {})


def test_boolean_conditions_in_multimodal_profile(db):
    """Verify that boolean condition flags in DB produce clean labels without '(True)'."""
    p = db.create_patient(
        patient_id="PAT-BOOL-01",
        hospital_mrn="MRN-BOOL-01",
        name="Bool Cond Patient",
        diabetes=True,
        hypertension=True,
        kidney_disease=True,
    )
    profile = db.get_patient_multimodal_profile(p.patient_id)
    assert "Diabetes" in profile.existing_conditions
    assert "Hypertension" in profile.existing_conditions
    assert "Kidney Disease" in profile.existing_conditions
    assert not any("(True)" in c for c in profile.existing_conditions)


def test_single_combination_drug_does_not_flag_self_duplication():
    """Verify that a single combination medication (e.g. Entresto / Sacubitril/Valsartan alone) does not flag duplication."""
    from src.medications.interaction_checker import reconcile_medications
    r_entresto = reconcile_medications(["Entresto 49/51 mg BID"])
    assert r_entresto.therapeutic_duplications == []

    r_hctz = reconcile_medications(["Lisinopril/Hydrochlorothiazide 20/12.5 mg OD"])
    assert r_hctz.therapeutic_duplications == []


def test_duplicate_active_ingredients_detected():
    """Verify that multiple brand/generic names for the same chemical active ingredient are flagged as duplicates."""
    from src.medications.interaction_checker import reconcile_medications

    # 1. Prinivil and Zestril (both Lisinopril)
    r_lis = reconcile_medications(["Prinivil 10 mg"], proposed_medications=["Zestril 20 mg"])
    assert len(r_lis.therapeutic_duplications) > 0
    all_text = " ".join(r_lis.therapeutic_duplications)
    assert "Lisinopril" in all_text

    # 2. Lipitor and Atorvastatin
    r_atorv = reconcile_medications(["Lipitor 20 mg"], proposed_medications=["Atorvastatin 40 mg"])
    assert len(r_atorv.therapeutic_duplications) > 0
    all_text2 = " ".join(r_atorv.therapeutic_duplications)
    assert "Atorvastatin" in all_text2

    # 3. Lisinopril/HCTZ combination + Lisinopril
    r_combo = reconcile_medications(["Lisinopril/Hydrochlorothiazide 20/12.5 mg"], proposed_medications=["Lisinopril 10 mg"])
    assert len(r_combo.therapeutic_duplications) > 0
    all_text3 = " ".join(r_combo.therapeutic_duplications)
    assert "Lisinopril" in all_text3


def test_whitespace_does_not_trigger_false_dosage_change():
    """Verify whitespace differences do not trigger false positive dosage changes, but true differences do."""
    from src.medications.interaction_checker import reconcile_medications

    # Same dose, whitespace difference
    r_ws = reconcile_medications(["Metoprolol 25 mg"], previous_medications=["Metoprolol 25mg"])
    assert r_ws.dosage_changes == []

    # Actual dose increase: 25 mg -> 50 mg
    r_diff = reconcile_medications(["Metoprolol 50 mg"], previous_medications=["Metoprolol 25 mg"])
    assert len(r_diff.dosage_changes) == 1
    assert "Metoprolol 25 mg -> Metoprolol 50 mg" in r_diff.dosage_changes[0]


def test_newly_added_cardiovascular_medications_and_interactions():
    """Verify repository completeness and critical safety checks for HCTZ, Aggrenox, Dipyridamole, Farxiga, and Enalapril."""
    from src.medications.medication_database import GLOBAL_MEDICATION_DB
    from src.medications.interaction_checker import check_medication_safety

    # 1. Database lookups
    hctz = GLOBAL_MEDICATION_DB.get_by_name("Hydrochlorothiazide")
    assert hctz is not None
    assert "Microzide" in hctz.brand_names
    assert hctz.drug_class == "Thiazide Diuretic"

    aggrenox = GLOBAL_MEDICATION_DB.get_by_name("Aggrenox")
    assert aggrenox is not None
    assert "Aspirin/Dipyridamole" == aggrenox.generic_name

    dapa = GLOBAL_MEDICATION_DB.get_by_name("Farxiga")
    assert dapa is not None
    assert dapa.generic_name == "Dapagliflozin"

    enalapril = GLOBAL_MEDICATION_DB.get_by_name("Vasotec")
    assert enalapril is not None
    assert enalapril.generic_name == "Enalapril"

    # 2. Critical HCTZ + Digoxin Interaction
    hctz_dig_report = check_medication_safety(
        ["Hydrochlorothiazide 25 mg", "Digoxin 0.125 mg"],
        patient={"patient_id": "P-TEST", "age": 68, "allergies": [], "existing_conditions": ["Hypertension"]},
    )
    dig_alerts = [a for a in hctz_dig_report.alerts if "Digoxin" in a.title or "digitalis" in a.description.lower()]
    assert len(dig_alerts) > 0
    assert any(a.severity == "CRITICAL" for a in dig_alerts)

    # 3. Renal Threshold Advisory (eGFR < 30 mL/min for HCTZ)
    hctz_renal_rep = check_medication_safety(
        ["Hydrochlorothiazide 25 mg"],
        patient={"patient_id": "P-TEST", "age": 70, "allergies": [], "existing_conditions": ["Hypertension"], "egfr": 22},
    )
    renal_alerts = [a for a in hctz_renal_rep.alerts if "Renal" in a.title]
    assert len(renal_alerts) > 0
    assert any(a.severity == "CRITICAL" for a in renal_alerts)


def test_concurrent_report_signing_thread_safety(persistence_service):
    """Verify concurrent report creation and signing under multithreaded contention."""
    import concurrent.futures

    def worker(idx):
        p_id = f"PAT-THREAD-SAFE-{idx}"
        rep_data = {
            "patient_info": {"patient_id": p_id, "patient_name": f"Concurrent Patient {idx}"},
            "input_info": {"file_name": f"ecg_concur_{idx}.dat"},
            "ai_analysis": {"analysis_id": f"ANL-CONCUR-{idx}"},
            "interpretation": "Sinus rhythm",
        }
        res = persistence_service.save_report_snapshot(rep_data, pdf_bytes=b"%PDF-1.4 mock")
        return res.get("status")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(worker, range(12)))

    assert len(results) == 12
    assert all(st == "SAVE_SUCCESS" for st in results)



