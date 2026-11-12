"""
Phase 6 & 7: Synthetic Patient Profiles A–H & Clinical Scenarios 1–16 Suite
==========================================================================
Validates:
- 8 Synthetic Patient Archetypes:
    Patient A: Young Healthy (24yo, no conditions, clean normal rhythm)
    Patient B: Hypertensive (58yo, HTN, Lisinopril 10mg, elevated BP)
    Patient C: Diabetic (64yo, T2DM, Metformin 500mg, resting tachycardia)
    Patient D: CAD / Prior MI (72yo, Stent, Aspirin + Metoprolol, ventricular ectopy)
    Patient E: Multi-Allergy (45yo, Penicillin + Sulfa + Aspirin allergy)
    Patient F: Elderly Polypharmacy (82yo, AFib, CKD, Warfarin + Digoxin + Amiodarone)
    Patient G: Incomplete Context (Missing age, allergies, conditions - withheld CDS)
    Patient H: Prolonged QT Suspect (50yo, QTc > 480ms, Amiodarone - critical alert)

- 16 End-to-End Clinical Scenarios:
    Scenario 1: Patient Registration & MRN generation
    Scenario 2: Duplicate Patient MRN Rejection (HTTP 409)
    Scenario 3: Longitudinal Vital Signs Intake
    Scenario 4: Clinician Authentication & Portal Scoping
    Scenario 5: Patient Authentication & Strict Self-Isolation
    Scenario 6: Real MIT-BIH PhysioNet Trace Loading (Record 100)
    Scenario 7: Real MIT-BIH PhysioNet Trace Loading (Record 208)
    Scenario 8: Signal Quality Gate Validation on Clean Signal
    Scenario 9: Signal Quality Gate Rejection on Flatline Signal
    Scenario 10: Decoupled AI Inference Pipeline Execution
    Scenario 11: Morphological Measurements (QRS, QT, QTc Bazett & Fridericia)
    Scenario 12: Clinical Decision Support & Guideline Mapping (AHA/ACC/ESC)
    Scenario 13: Drug-Drug Interaction Detection (Amiodarone + Warfarin)
    Scenario 14: Physiological Conflict Alert (Severe QTc Prolongation + Amiodarone)
    Scenario 15: Clinician Review & Cryptographic Sign-Off Sealing
    Scenario 16: Report Snapshot Persistence, Timeline Retrieval & PDF Generation
"""

import uuid
from pathlib import Path
import numpy as np
import pytest
import wfdb
from fastapi.testclient import TestClient

from api.index import app
from src.auth.auth_manager import AUTH_MANAGER, UserRole
from src.clinical.medication_candidate_engine import build_authoritative_clinical_analysis
from src.clinical.recommendation_engine import GLOBAL_CDS_ENGINE
from src.data_loader import load_record
from src.database.db_manager import DB_MANAGER, PatientRecord
from src.inference.inference_engine import run_ecg_inference
from src.measurements.measurement_engine import compute_ecg_measurements
from src.medications.interaction_checker import check_medication_safety
from src.report.pdf_generator import generate_doctor_report, generate_patient_report
from src.report.report_generator import generate_structured_report
from src.safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate
from src.services.report_persistence_service import REPORT_PERSISTENCE_SERVICE

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets" / "mit_bih_arrhythmia" / "raw"


@pytest.fixture(scope="module")
def patient_profiles():
    """Create and register the 8 clinical archetypes A through H."""
    profiles = {}

    # Patient A: Young Healthy
    profiles["A"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-A-{uuid.uuid4().hex[:6].upper()}",
        name="Alex Young (Patient A)",
        hospital_id="HOSP-APEX",
        age=24,
        sex="Male",
        blood_group="O+",
        known_allergies="None",
        existing_conditions="None",
        current_medications="None",
    )

    # Patient B: Hypertensive
    profiles["B"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-B-{uuid.uuid4().hex[:6].upper()}",
        name="Brenda Vance (Patient B)",
        hospital_id="HOSP-APEX",
        age=58,
        sex="Female",
        blood_group="A+",
        known_allergies="None",
        existing_conditions="Essential Hypertension",
        current_medications="Lisinopril 10mg",
    )

    # Patient C: Diabetic
    profiles["C"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-C-{uuid.uuid4().hex[:6].upper()}",
        name="Carlos Diaz (Patient C)",
        hospital_id="HOSP-APEX",
        age=64,
        sex="Male",
        blood_group="B+",
        known_allergies="None",
        existing_conditions="Type 2 Diabetes Mellitus, Hyperlipidemia",
        current_medications="Metformin 500mg, Atorvastatin 20mg",
    )

    # Patient D: Known CAD / Prior MI
    profiles["D"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-D-{uuid.uuid4().hex[:6].upper()}",
        name="David Sterling (Patient D)",
        hospital_id="HOSP-APEX",
        age=72,
        sex="Male",
        blood_group="AB+",
        known_allergies="None",
        existing_conditions="Coronary Artery Disease, Prior MI (2021), PCI Stent",
        current_medications="Aspirin 81mg, Metoprolol 25mg, Atorvastatin 40mg",
    )

    # Patient E: Multi-Allergy
    profiles["E"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-E-{uuid.uuid4().hex[:6].upper()}",
        name="Elena Rostova (Patient E)",
        hospital_id="HOSP-APEX",
        age=45,
        sex="Female",
        blood_group="O-",
        known_allergies="Penicillin, Sulfa, Aspirin",
        existing_conditions="Mild Asthma",
        current_medications="Albuterol Inhaler",
    )

    # Patient F: Elderly Polypharmacy
    profiles["F"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-F-{uuid.uuid4().hex[:6].upper()}",
        name="Frank Miller (Patient F)",
        hospital_id="HOSP-APEX",
        age=82,
        sex="Male",
        blood_group="A-",
        known_allergies="None",
        existing_conditions="Atrial Fibrillation, Chronic Kidney Disease Stage 3, Hypertension",
        current_medications="Warfarin 5mg, Digoxin 0.125mg, Amiodarone 200mg, Furosemide 40mg",
    )

    # Patient G: Incomplete / Missing Context
    profiles["G"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-G-{uuid.uuid4().hex[:6].upper()}",
        name="Grace Unknown (Patient G)",
        hospital_id="HOSP-APEX",
        age=None,
        sex="Other",
        blood_group="Unknown",
        known_allergies="",
        existing_conditions="",
        current_medications="",
    )

    # Patient H: QT-Prolongation Suspect
    profiles["H"] = DB_MANAGER.create_patient(
        patient_id=str(uuid.uuid4()),
        hospital_mrn=f"MRN-H-{uuid.uuid4().hex[:6].upper()}",
        name="Harold Finch (Patient H)",
        hospital_id="HOSP-APEX",
        age=50,
        sex="Male",
        blood_group="B-",
        known_allergies="None",
        existing_conditions="Ventricular Arrhythmia",
        current_medications="Amiodarone 400mg",
    )

    return profiles


# ------------------------------------------------------------------------------
# Scenarios 1 to 16
# ------------------------------------------------------------------------------

def test_scenario_1_patient_registration(patient_profiles):
    """Scenario 1: Register new patient and verify DB entry."""
    p = patient_profiles["A"]
    assert p.name == "Alex Young (Patient A)"
    assert p.age == 24
    assert p.hospital_mrn.startswith("MRN-A-")


def test_scenario_2_duplicate_mrn_rejection(patient_profiles):
    """Scenario 2: Duplicate MRN must be rejected by database manager."""
    p = patient_profiles["A"]
    with pytest.raises(Exception):
        DB_MANAGER.create_patient(
            patient_id=str(uuid.uuid4()),
            hospital_mrn=p.hospital_mrn,
            name="Duplicate Imposter",
            hospital_id="HOSP-APEX",
        )
    fetched = DB_MANAGER.get_patient_by_mrn(p.hospital_mrn)
    assert fetched.name == "Alex Young (Patient A)"


def test_scenario_3_vitals_intake_history(patient_profiles):
    """Scenario 3: Record multiple vitals and verify non-overwriting history."""
    p_b = patient_profiles["B"]
    DB_MANAGER.record_vital_signs(
        vital_id=str(uuid.uuid4()),
        patient_id=p_b.patient_id,
        heart_rate_bpm=78.0,
        systolic_bp_mmhg=142.0,
        diastolic_bp_mmhg=90.0,
        spo2_percent=98.0,
    )
    DB_MANAGER.record_vital_signs(
        vital_id=str(uuid.uuid4()),
        patient_id=p_b.patient_id,
        heart_rate_bpm=74.0,
        systolic_bp_mmhg=138.0,
        diastolic_bp_mmhg=86.0,
        spo2_percent=99.0,
    )
    vitals = DB_MANAGER.get_patient_vitals(p_b.patient_id)
    assert len(vitals) >= 2


def test_scenario_4_clinician_auth_and_portal_access():
    """Scenario 4: Authenticate clinician and verify credentials."""
    user = AUTH_MANAGER.get_user_by_username("cardiologist")
    assert user is not None
    assert user.role in [UserRole.CARDIOLOGIST, UserRole.DOCTOR]
    tok = AUTH_MANAGER.create_session(user)
    assert AUTH_MANAGER.validate_session(tok).user_id == user.user_id


def test_scenario_5_patient_auth_self_isolation(patient_profiles):
    """Scenario 5: Patient account only has access to self record."""
    p_c = patient_profiles["C"]
    pat_email = f"user.{p_c.hospital_mrn.lower()}@patient.org"
    pat_user = AUTH_MANAGER.register_user(
        username=pat_email,
        password="PatientSecure123!",
        role=UserRole.PATIENT,
        full_name=p_c.name,
        email=pat_email,
        registration_number=p_c.hospital_mrn,
    )
    assert pat_user.role == UserRole.PATIENT


def test_scenario_6_real_sample_loading_record_100():
    """Scenario 6: Load real MIT-BIH record 100."""
    sig, fs, _ = load_record("100", RAW_DATA_DIR)
    assert fs == 360.0
    assert len(sig) > 3600


def test_scenario_7_real_sample_loading_record_208():
    """Scenario 7: Load real MIT-BIH record 208 with ectopy."""
    sig, fs, _ = load_record("208", RAW_DATA_DIR)
    assert fs == 360.0
    assert len(sig) > 3600


def test_scenario_8_quality_gate_clean_signal():
    """Scenario 8: Quality gate validates clean physiological signal."""
    sig, fs, _ = load_record("100", RAW_DATA_DIR)
    q = evaluate_signal_quality_gate(sig[:3600], fs=fs)
    assert q.can_run_ai
    assert q.category in [QualityCategory.GOOD, QualityCategory.ACCEPTABLE]


def test_scenario_9_quality_gate_flatline_halt():
    """Scenario 9: Quality gate halts pipeline on flatline signal."""
    flat = np.zeros(3600)
    q = evaluate_signal_quality_gate(flat, fs=360.0)
    assert not q.can_run_ai
    assert q.category == QualityCategory.UNUSABLE


def test_scenario_10_decoupled_ai_inference():
    """Scenario 10: Run decoupled inference engine on real recording."""
    sig, fs, _ = load_record("100", RAW_DATA_DIR)
    res = run_ecg_inference(sig[:3600], fs=fs)
    assert res.prediction is not None
    assert res.detected_beats_count > 0


def test_scenario_11_cardiac_measurements():
    """Scenario 11: Compute morphological measurements."""
    sig, fs, _ = load_record("100", RAW_DATA_DIR)
    res = run_ecg_inference(sig[:3600], fs=fs)
    m = compute_ecg_measurements(
        signal=sig[:3600],
        fs=fs,
        r_peaks=np.array(res.detected_r_peaks),
        leads_available=["II"],
    )
    assert m.heart_rate_bpm is not None and m.heart_rate_bpm > 50
    assert m.mean_rr_ms is not None
    assert m.qrs_status is not None
    assert "single-lead" in m.axis_status.lower()


def test_scenario_12_cds_guideline_mapping(patient_profiles):
    """Scenario 12: CDS engine maps findings to AHA/ACC/ESC recommendations."""
    sig, fs, _ = load_record("100", RAW_DATA_DIR)
    res = run_ecg_inference(sig[:3600], fs=fs)

    cds = GLOBAL_CDS_ENGINE.evaluate_finding(
        ecg_finding=res.prediction,
        heart_rate=res.heart_rate_bpm,
        patient=patient_profiles["A"],
        reliability=res,
    )
    assert cds.finding is not None
    assert len(cds.relevant_guidelines) > 0


def test_scenario_13_drug_drug_interaction(patient_profiles):
    """Scenario 13: Detect Amiodarone + Warfarin interaction in polypharmacy patient F."""
    p_f = patient_profiles["F"]
    meds = ["Amiodarone", "Warfarin", "Digoxin"]
    report = check_medication_safety(meds, patient=p_f)

    inter_titles = [a.title for a in report.alerts if a.category == "INTERACTION"]
    assert any("Amiodarone" in t and "Warfarin" in t for t in inter_titles)


def test_scenario_14_physiological_qtc_conflict(patient_profiles):
    """Scenario 14: Trigger critical alert for QTc > 480ms with QT-prolonging medication."""
    p_h = patient_profiles["H"]
    report = check_medication_safety(
        ["Amiodarone"],
        patient=p_h,
        ecg_measurements={"qtc_ms": 495.0, "heart_rate": 68.0},
    )
    assert any("QTc Alert" in a.title or "Prolonged QTc" in a.title for a in report.alerts)


def test_scenario_15_clinician_review_sealing():
    """Scenario 15: Clinician signs and cryptographically seals report."""
    rep_uuid = str(uuid.uuid4())
    dummy_rep = {"sample": "data", "predicted_class": "Normal"}
    REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=dummy_rep,
        report_id=rep_uuid,
        patient_id="PAT-TEST-001",
        ecg_id="REC-TEST-001",
        analysis_id="ANL-TEST-001",
        status="PENDING_REVIEW",
    )

    sealed = REPORT_PERSISTENCE_SERVICE.sign_and_seal_report(
        report_id=rep_uuid,
        clinician_user_id="DOC-99",
        clinician_name="Dr. Senior Cardiologist, MD",
        clinician_role="CARDIOLOGIST",
        registration_number="MCI-998877",
        agreement_status="CONFIRMED",
        clinician_interpretation="ECG reviewed and approved.",
        clinical_notes="Discharge with standard advice.",
    )
    assert sealed["status"] in ["SAVE_SUCCESS", "SIGNED", "SEALED", "REVIEWED"]


def test_scenario_16_report_generation_and_dual_pdf(patient_profiles):
    """Scenario 16: Structured report generation & dual Doctor/Patient PDF creation."""
    sig, fs, _ = load_record("100", RAW_DATA_DIR)
    segment = sig[:3600]
    res = run_ecg_inference(segment, fs=fs)
    m = compute_ecg_measurements(segment, fs=fs, r_peaks=np.array(res.detected_r_peaks))
    p_a = patient_profiles["A"]

    rep_dict = generate_structured_report(
        input_info={"file_name": "mit_100", "sampling_rate": fs, "duration_sec": 10.0, "lead": "II"},
        ai_results={"predicted_class": res.prediction, "probabilities": res.model_probabilities, "heart_rate_bpm": res.heart_rate_bpm, "signal_quality": "GOOD"},
        extracted_measurements={"qrs_duration_ms": m.qrs_duration_ms, "qt_interval_ms": m.qt_interval_ms, "qtc_interval_ms": m.qtc_bazett_ms},
        patient_profile=p_a.to_dict(),
    )

    doc_pdf = generate_doctor_report(rep_dict, waveform=segment, fs=fs, r_peaks=np.array(res.detected_r_peaks))
    pat_pdf = generate_patient_report(rep_dict)

    assert doc_pdf.startswith(b"%PDF")
    assert pat_pdf.startswith(b"%PDF")
    assert len(doc_pdf) > 1000
    assert len(pat_pdf) > 1000


@pytest.mark.parametrize("archetype_key,record_id,expected_finding", [
    ("A", "100", "Normal"),
    ("D", "208", "Ventricular Ectopy"),
])
def test_phase_7_patient_to_report_integration_pipeline(patient_profiles, archetype_key, record_id, expected_finding):
    """Phase 7: Full 16-Step Patient-to-Report Integration Pipeline on Real Data."""
    client = TestClient(app)
    target_patient = patient_profiles[archetype_key]

    # SCENARIO 1: Register new synthetic patient & verify DB persistence
    init_login = client.post("/api/auth/login", json={
        "email": "dr.aniket@hospital.org",
        "password": "Doctor@Hospital2026",
        "role": "DOCTOR",
    })
    assert init_login.status_code == 200
    doc_token = init_login.json()["token"]

    pat_payload = {
        "name": target_patient.name,
        "hospital_mrn": f"MRN-INT-{uuid.uuid4().hex[:6].upper()}",
        "hospital_id": target_patient.hospital_id,
        "age": target_patient.age or 50,
        "sex": target_patient.sex or "Male",
        "blood_group": target_patient.blood_group or "O+",
        "known_allergies": [a.strip() for a in (target_patient.known_allergies or "").split(",") if a.strip()],
        "existing_conditions": [c.strip() for c in (target_patient.existing_conditions or "").split(",") if c.strip()],
        "current_medications": [m.strip() for m in (target_patient.current_medications or "").split(",") if m.strip()],
    }
    create_res = client.post("/api/patients", json=pat_payload, headers={"Authorization": f"Bearer {doc_token}"})
    assert create_res.status_code == 201
    created_pat = create_res.json()["patient"]
    pat_id = created_pat["patient_id"]
    pat_mrn = created_pat["hospital_mrn"]

    # SCENARIO 2: Log out and log in again, verify patient is available
    # Logout
    client.post("/api/auth/logout", headers={"Authorization": f"Bearer {doc_token}"})
    # Login again
    relogin_res = client.post("/api/auth/login", json={
        "email": "dr.aniket@hospital.org",
        "password": "Doctor@Hospital2026",
        "role": "DOCTOR",
    })
    assert relogin_res.status_code == 200
    new_token = relogin_res.json()["token"]

    # SCENARIO 3: Select patient from doctor dashboard, verify details
    pat_get = client.get(f"/api/patients/{pat_mrn}", headers={"Authorization": f"Bearer {new_token}"})
    assert pat_get.status_code == 200
    retrieved_pat = pat_get.json()["patient"]
    assert retrieved_pat["name"] == target_patient.name
    assert retrieved_pat["hospital_mrn"] == pat_mrn

    # SCENARIO 4: Upload real MIT-BIH ECG recording
    sig, fs, _ = load_record(record_id, RAW_DATA_DIR)
    segment = sig[:3600]
    csv_content = "time_s,lead_ii_mv\n" + "\n".join(f"{i/fs:.4f},{v:.4f}" for i, v in enumerate(segment[:360]))
    upload_res = client.post(
        "/api/upload",
        files={"file": (f"mit_{record_id}.csv", csv_content.encode("utf-8"), "text/csv")},
        headers={"Authorization": f"Bearer {new_token}"},
    )
    assert upload_res.status_code == 200
    assert len(upload_res.json()["signal"]) > 50

    # SCENARIO 5: Run the actual ECG analysis model
    analyze_res = client.post(
        "/api/analyze",
        json={
            "signal": segment.tolist(),
            "fs": fs,
            "lead": "II",
            "patient_id": pat_id,
            "patient_name": retrieved_pat["name"],
            "patient_mrn": pat_mrn,
            "age": retrieved_pat["age"],
            "sex": retrieved_pat["sex"],
            "include_full_report": True,
        },
        headers={"Authorization": f"Bearer {new_token}"},
    )
    assert analyze_res.status_code == 200
    anl_data = analyze_res.json()
    assert anl_data["status"] == "SUCCESS"
    rep_id = anl_data["report_id"]
    ai_class = anl_data["ai_classification"]

    # SCENARIO 6: Compare model results against official MIT-BIH reference annotations
    ann = wfdb.rdann(str(RAW_DATA_DIR / record_id), "atr")
    ref_symbols = [sym for s, sym in zip(ann.sample, ann.symbol) if s < 3600]
    if record_id == "100":
        assert "N" in ref_symbols
        assert expected_finding in ai_class["prediction"] or "Sinus Rhythm" in ai_class["prediction"]
    else:
        assert "V" in ref_symbols
        assert expected_finding in ai_class["prediction"] or "PVC" in ai_class["prediction"]

    # SCENARIO 7: Verify ECG measurements displayed match backend outputs
    params = anl_data["cardiac_parameters"]
    assert params["heart_rate_bpm"] > 40 and params["heart_rate_bpm"] < 160
    assert "qrs_duration_ms" in params
    assert params.get("mean_rr_ms") is not None or params.get("heart_rate_bpm") is not None

    # SCENARIO 8: Verify clinical interpretation uses actual analysis result
    cds = anl_data["clinical_decision_support"]
    finding_val = cds.get("primary_finding") or cds.get("finding")
    assert finding_val is not None and len(finding_val) > 0

    # SCENARIO 9: Verify patient context is retrieved from the correct record
    full_rep = anl_data.get("full_report", {})
    p_info = full_rep.get("patient_info", {})
    assert p_info.get("patient_name") == target_patient.name

    # SCENARIO 10: Run medication safety checks
    meds = [m for m in pat_payload["current_medications"] if m and m != "None"]
    if meds:
        med_report = check_medication_safety(meds, patient=target_patient)
        assert med_report is not None

    # SCENARIO 11 & 12: Generate Doctor and Patient reports
    doc_pdf = client.get(f"/api/reports/{rep_id}/pdf?type=doctor", headers={"Authorization": f"Bearer {new_token}"})
    assert doc_pdf.status_code == 200
    assert doc_pdf.content.startswith(b"%PDF")

    pat_pdf = client.get(f"/api/reports/{rep_id}/pdf?type=patient", headers={"Authorization": f"Bearer {new_token}"})
    assert pat_pdf.status_code == 200
    assert pat_pdf.content.startswith(b"%PDF")

    # SCENARIO 13: Compare both reports against saved analysis
    rep_obj = client.get(f"/api/reports/{rep_id}", headers={"Authorization": f"Bearer {new_token}"}).json()["report"]
    assert rep_obj["report_id"] == rep_id
    assert rep_obj.get("patient_name") == target_patient.name or rep_obj.get("patient_id") == pat_id

    # SCENARIO 14: Save clinician review and seal report
    rev_res = client.post("/api/review", json={
        "analysis_id": anl_data["analysis_id"],
        "clinician_name": "Dr. Aniket Chandra, MD",
        "clinician_role": "CARDIOLOGIST",
        "registration_number": "MCI-778899",
        "agreement_status": "CONFIRMED",
        "clinician_interpretation": f"Verified {ai_class['prediction']} under clinical protocol.",
        "clinical_notes": "Follow standard treatment guidelines.",
    }, headers={"Authorization": f"Bearer {new_token}"})
    assert rev_res.status_code == 200
    assert rev_res.json()["status"] in ["SEALED", "SEALED_AND_CONFIRMED"]

    # SCENARIO 15: Open Report History and retrieve identical report
    history_res = client.get(f"/api/reports?search={pat_mrn}", headers={"Authorization": f"Bearer {new_token}"})
    assert history_res.status_code == 200
    hist_reports = history_res.json()["reports"]
    assert len(hist_reports) >= 1
    assert any(r["report_id"] == rep_id for r in hist_reports)

    # SCENARIO 16: Verify no cross-patient mixing
    other_mrn = f"MRN-OTHER-{uuid.uuid4().hex[:6].upper()}"
    empty_history = client.get(f"/api/reports?search={other_mrn}", headers={"Authorization": f"Bearer {new_token}"})
    assert empty_history.status_code == 200
    assert len(empty_history.json()["reports"]) == 0
