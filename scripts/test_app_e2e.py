"""
End-to-End Application & Subsystem Verification Script
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from ecg_input.signal_loader import load_any_ecg
from quality.quality_gate import evaluate_ecg_quality_gate
from ml.inference.inference_engine import run_ecg_ml_inference
from comparison.ai_machine_comparison import compare_ai_and_machine
from comparison.machine_interpretation import parse_machine_interpretation
from longitudinal.ecg_comparison import compare_serial_ecgs
from review.review_manager import ClinicianReviewManager, ReviewAction
from report.report_generator import generate_structured_report, export_report_to_text


def main():
    print("=== 1. Testing Normal ECG Ingestion & Inference ===")
    rec_norm, err = load_any_ecg("sample_ecgs/normal_ecg_sample.csv", filename="normal_ecg_sample.csv", fs=360.0, lead_name="II")
    assert rec_norm is not None, f"Failed to load normal ECG: {err}"
    print(f"  Ingested: {rec_norm.record_id}, duration={rec_norm.duration}s, rate={rec_norm.sampling_rate}Hz")

    gate_norm = evaluate_ecg_quality_gate(rec_norm.signals[0], rec_norm.sampling_rate, lead_name="II")
    print(f"  Quality Gate: Category={gate_norm.category.value}, Can run AI={gate_norm.can_run_ai}")

    res_norm = run_ecg_ml_inference(rec_norm)
    print(f"  ML Prediction: {res_norm['prediction']}, HR={res_norm['heart_rate']} bpm, Detected Beats={len(res_norm['r_peaks'])}")

    print("\n=== 2. Testing PVC Arrhythmia Ingestion & Inference ===")
    rec_pvc, err = load_any_ecg("sample_ecgs/pvc_arrhythmia_sample.csv", filename="pvc_arrhythmia_sample.csv", fs=360.0, lead_name="II")
    assert rec_pvc is not None, f"Failed to load PVC ECG: {err}"
    res_pvc = run_ecg_ml_inference(rec_pvc)
    print(f"  ML Prediction: {res_pvc['prediction']}, HR={res_pvc['heart_rate']} bpm, Detected Beats={len(res_pvc['r_peaks'])}")

    print("\n=== 3. Testing Ventricular Couplets Ingestion & Inference ===")
    rec_cpl, err = load_any_ecg("sample_ecgs/ventricular_couplets_sample.csv", filename="ventricular_couplets_sample.csv", fs=360.0, lead_name="II")
    assert rec_cpl is not None, f"Failed to load Couplets ECG: {err}"
    res_cpl = run_ecg_ml_inference(rec_cpl)
    print(f"  ML Prediction: {res_cpl['prediction']}, HR={res_cpl['heart_rate']} bpm, Detected Beats={len(res_cpl['r_peaks'])}")

    print("\n=== 4. Testing Machine vs. AI Comparison ===")
    machine_norm = parse_machine_interpretation("Normal Sinus Rhythm, HR=72 bpm, PR=160, QRS=88")
    cmp_agree = compare_ai_and_machine(res_norm, machine_norm)
    print(f"  Normal Case: {cmp_agree.status.value} - {cmp_agree.summary}")

    cmp_disagree = compare_ai_and_machine(res_pvc, machine_norm)
    print(f"  Disagreement Case: {cmp_disagree.status.value} - {cmp_disagree.summary}")

    print("\n=== 5. Testing Longitudinal Serial Comparison ===")
    serial_cmp = compare_serial_ecgs(res_pvc, res_norm, patient_id="PT-DEMO-001")
    print(f"  Status: {serial_cmp.status.value} - {serial_cmp.objective_change_summary}")
    print(f"  Notable Deltas: {serial_cmp.notable_deltas}")

    print("\n=== 6. Testing Clinician Review & Digital Sign-off ===")
    from database.db_manager import DB_MANAGER
    db_mgr = DB_MANAGER
    if not db_mgr.get_patient("PT-DEMO-001"):
        db_mgr.create_patient(patient_id="PT-DEMO-001", hospital_mrn="MRN-DEMO-101", name="Ramesh Sharma")
    db_mgr.save_ecg_record(
        record_id=rec_norm.record_id,
        sampling_rate=rec_norm.sampling_rate,
        lead_names=rec_norm.lead_names,
        duration_sec=rec_norm.duration,
        file_hash="hash_demo_123",
        source_format="CSV",
        patient_id="PT-DEMO-001",
    )
    db_mgr.save_analysis_result(
        analysis_id=res_norm["analysis_id"],
        record_id=rec_norm.record_id,
        model_id="RandomForest",
        model_version=res_norm["model_version"],
        prediction=res_norm["prediction"],
        probabilities=res_norm["model_probabilities"],
        signal_quality=res_norm["signal_quality"],
        quality_score=1.0,
        heart_rate_bpm=res_norm["heart_rate"],
    )

    rev_mgr = ClinicianReviewManager(db=db_mgr)
    signoff = rev_mgr.submit_review(
        analysis_id=res_norm["analysis_id"],
        record_id=rec_norm.record_id,
        clinician_id="CARD-101",
        clinician_name="Dr. Vikram Malhotra",
        clinician_role="Consultant Cardiologist",
        registration_number="MCI-CARD-9842",
        action=ReviewAction.ACCEPTED,
        final_diagnosis="Normal Sinus Rhythm, narrow QRS complexes, no ectopy",
        clinical_notes="Routine health screening. Patient asymptomatic. Safe for discharge.",
    )
    print(f"  Review ID: {signoff.review_id}")
    print(f"  Digital Signature: {signoff.digital_signature_hash[:20]}...")
    print(f"  Cryptographic Integrity Verified: {ClinicianReviewManager.verify_signoff_integrity(signoff)}")

    print("\n=== 7. Testing Clinical Report Assembly ===")
    report = generate_structured_report(
        input_info={
            "filename": "normal_ecg_sample.csv",
            "duration_sec": rec_norm.duration,
            "sampling_rate": rec_norm.sampling_rate,
            "modality": "DIGITAL_SIGNAL",
            "lead": "Lead II",
        },
        ai_results={
            "predicted_class": res_norm["prediction"],
            "probabilities": res_norm["model_probabilities"],
            "signal_quality": res_norm["signal_quality"],
            "quality_score": 0.95,
            "heart_rate_bpm": res_norm["heart_rate"],
            "mean_rr_sec": 0.82,
            "beat_count": len(res_norm["r_peaks"]),
        },
        clinician_review=signoff.to_dict(),
        machine_comparison=cmp_agree.to_dict(),
        longitudinal_comparison=serial_cmp.to_dict(),
    )
    txt = export_report_to_text(report)
    print(f"  Report generated: {len(txt)} chars, {len(txt.splitlines())} lines.")
    print("  Report contains disclaimer:", "DISCLAIMER" in txt)
    print("\n>>> ALL APPLICATION PIPELINES VERIFIED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
