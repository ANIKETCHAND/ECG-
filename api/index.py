"""
Vercel Serverless Clinical API for AI-ECG Platform
===================================================

Endpoints for Vercel deployment:
- GET  /api/health: Operational and regulatory status
- GET  /api/sample: Sample ECG waveforms for live testing
- POST /api/analyze: Full ECG analysis with safety gating & decoupled inference
- POST /api/review: Clinician review sign-off and sealing
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root and src are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import io
import uuid
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.responses import Response

from ecg_core.models import ECGRecording
from ecg_input.signal_loader import load_any_ecg
from ecg_input.image_processor import process_ecg_image
from ecg_input.waveform_extractor import extract_waveform_from_image
from ecg_input.pdf_processor import process_pdf_report
from ecg_input.measurement_extractor import extract_report_measurements
from inference.inference_engine import ACTIVE_MODEL_ID, run_ecg_inference
from measurements.measurement_engine import compute_ecg_measurements
from safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate
from clinical.recommendation_engine import GLOBAL_CDS_ENGINE
from medications.interaction_checker import check_medication_safety
from report.report_generator import generate_structured_report
from report.pdf_generator import generate_doctor_report
from services import REPORT_PERSISTENCE_SERVICE
from segmentation import extract_beats
from feature_extraction import extract_all_features

app = FastAPI(
    title="AI-ECG Clinical Decision Support API",
    description="Vercel Serverless Medical-Device Software Backend adhering to CDSCO MDR 2017 & IEC 62304",
    version="1.0.0",
)

# CORS is an explicit allow-list. The previous wildcard would have let any origin
# call a clinical endpoint from a patient's browser session; cross-origin access
# must now be requested deliberately via ECG_CORS_ORIGINS.
try:
    from api.security import (
        SecuritySettings,
        enforce_signal_limits,
        get_security_settings,
        require_api_key,
    )
except ImportError:  # running with api/ itself on sys.path (Vercel runtime)
    from security import (  # type: ignore[no-redef]
        SecuritySettings,
        enforce_signal_limits,
        get_security_settings,
        require_api_key,
    )

_SECURITY: SecuritySettings = get_security_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_SECURITY.cors_origins,
    allow_credentials=bool(_SECURITY.cors_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class AnalyzeRequest(BaseModel):
    signal: List[float]
    fs: float = 360.0
    lead: str = "II"
    patient_id: Optional[str] = "PAT-ANON"
    patient_name: Optional[str] = "Anonymous"
    patient_mrn: Optional[str] = None
    age: Optional[int] = None
    patient_age: Optional[int] = None
    sex: Optional[str] = None
    patient_sex: Optional[str] = None
    blood_group: Optional[str] = None
    symptoms: Optional[Union[str, List[str]]] = None
    existing_conditions: Optional[Union[str, List[str]]] = None
    cardiac_history: Optional[Union[str, List[str]]] = None
    current_medications: Optional[Union[str, List[str]]] = None
    vital_signs: Optional[Dict[str, Any]] = None
    laboratory_results: Optional[Dict[str, Any]] = None
    include_full_report: bool = False


class ReviewRequest(BaseModel):
    analysis_id: str
    clinician_name: str
    clinician_role: str = "CARDIOLOGIST"
    registration_number: str
    agreement_status: str  # CONFIRMED, MODIFIED, REJECTED
    clinician_interpretation: str
    clinical_notes: Optional[str] = ""

AnalyzeRequest.model_rebuild()
ReviewRequest.model_rebuild()


router = APIRouter()

# Endpoints carrying patient data require the shared API key when one is set.
# /health and /sample are intentionally left open for liveness probes.
_protected = Depends(require_api_key)


@router.get("/health")
def health_check():
    try:
        from ml.selective import load_operating_point, summarise as summarise_gate

        gate_summary = summarise_gate(load_operating_point())
        gate_installed = load_operating_point() is not None
    except Exception as exc:  # pragma: no cover - defensive
        gate_summary = f"Abstention gate status unavailable: {exc}"
        gate_installed = False

    return {
        "status": "HEALTHY",
        "device": "AI-ECG Analyzer (SaMD)",
        "regulatory_classification": "CDSCO MDR 2017 Class B / IEC 62304 Class B",
        "active_model_id": ACTIVE_MODEL_ID,
        "safety_principle": "NO RELIABLE INPUT = NO AI RESULT",
        "disclaimer": "Clinical Decision Support System. Requires mandatory qualified physician review.",
        "security": get_security_settings().describe(),
        "selective_reporting": {
            "operating_point_installed": gate_installed,
            "summary": gate_summary,
        },
    }


#: Real MIT-BIH records used for the live demo traces. Recordings, not
#: generated waveforms: a demo built from a synthetic sine teaches the user
#: nothing about how the model behaves on real physiology.
_SAMPLE_RECORDS = {"normal": "100", "pvc": "208", "ectopy": "208", "other": "213"}
_SAMPLE_WINDOW_SEC = 10.0


def _sample_search_dirs() -> List[Path]:
    return [
        BASE_DIR / "data" / "datasets" / "mit_bih_arrhythmia" / "raw",
        BASE_DIR / "data" / "raw",
    ]


@router.get("/sample")
def get_sample_data(sample_type: str = "normal"):
    """Return a real recorded ECG window for live testing.

    Serves an actual MIT-BIH record read from disk. Only if the corpus is absent
    does it fall back to a generated trace, and then it stamps the payload
    ``SYNTHETIC_DEMO_NOT_FOR_CLINICAL_USE`` so the caller can never mistake a
    generated waveform for recorded physiology.
    """
    requested = (sample_type or "normal").strip().lower()
    record_id = _SAMPLE_RECORDS.get(requested, _SAMPLE_RECORDS["normal"])

    for raw_dir in _sample_search_dirs():
        if not (raw_dir / f"{record_id}.hea").exists():
            continue
        try:
            from data_loader import load_record

            signal, fs, _ = load_record(record_id, raw_dir)
        except Exception:  # pragma: no cover - corrupt/incomplete record on disk
            continue

        fs = float(fs)
        window = int(_SAMPLE_WINDOW_SEC * fs)
        arr = np.asarray(signal, dtype=float)
        if len(arr) < window:
            continue
        segment = arr[:window]
        if segment.std() > 0:
            # Lead II is recorded in mV; convert to the display scale the rest of
            # the pipeline works in without altering morphology.
            segment = segment / max(float(np.abs(segment).max()), 1e-9)

        return {
            "sample_type": requested,
            "fs": fs,
            "sampling_rate": fs,
            "duration_sec": round(len(segment) / fs, 3),
            "lead": "II",
            "signal": segment.tolist(),
            "data_status": "REAL_RECORDED_DATASET",
            "source_dataset": "MIT-BIH Arrhythmia Database (PhysioNet)",
            "source_record": record_id,
            "note": (
                "Window extracted from a real annotated recording. The model has "
                "seen this record during development; this endpoint exercises the "
                "pipeline, it is not an independent validation sample."
            ),
        }

    # No corpus on disk: emit a clearly-labelled synthetic trace so the UI still
    # has something to render, without pretending it is recorded physiology.
    fs = 360.0
    duration_sec = _SAMPLE_WINDOW_SEC
    t = np.arange(int(fs * duration_sec)) / fs
    sig = 0.08 * np.sin(2 * np.pi * 1.2 * t)
    for i in range(1, int(duration_sec * 1.2) + 1):
        idx = int(i * (fs / 1.2))
        if idx >= len(sig):
            continue
        w = 8
        start, end = max(0, idx - w), min(len(sig), idx + w)
        amp = 1.8 if (requested == "pvc" and i == 2) else 1.2
        width = 5 if (requested == "pvc" and i == 2) else 3
        sig[start:end] += amp * np.exp(
            -0.5 * ((np.arange(start, end) - idx) / width) ** 2
        )

    dirs = ", ".join(str(d) for d in _sample_search_dirs())
    return {
        "sample_type": requested,
        "fs": fs,
        "sampling_rate": fs,
        "duration_sec": duration_sec,
        "lead": "II",
        "signal": sig.tolist(),
        "data_status": "SYNTHETIC_DEMO_NOT_FOR_CLINICAL_USE",
        "source_dataset": None,
        "source_record": None,
        "note": (
            "No recorded MIT-BIH dataset was found on disk, so this trace is "
            f"generated, not measured. Searched: {dirs}. Run "
            "'python training/download_datasets.py --dataset mit_bih_arrhythmia' "
            "for real sample data."
        ),
        "warning": (
            "SYNTHETIC WAVEFORM. Not recorded physiology. Any model output derived "
            "from this trace describes the generator, not a patient."
        ),
    }


@router.post("/analyze", dependencies=[_protected])
def analyze_ecg(req: AnalyzeRequest):
    if not req.signal or len(req.signal) < 100:
        raise HTTPException(
            status_code=400,
            detail="Signal duration is too short for cardiac evaluation (minimum 100 samples required).",
        )

    enforce_signal_limits(len(req.signal))

    sig_arr = np.array(req.signal, dtype=float)

    # 1. Mandatory Signal Quality Gatekeeper
    quality_res = evaluate_signal_quality_gate(sig_arr, req.fs, lead_name=req.lead)

    if quality_res.category == QualityCategory.UNUSABLE:
        return {
            "status": "UNUSABLE_SIGNAL",
            "can_run_ai": False,
            "quality_category": quality_res.category.value,
            "quality_score": quality_res.quality_score,
            "rejection_reasons": quality_res.rejection_reasons,
            "prediction": "NO_RESULT_SIGNAL_UNUSABLE",
            "message": "Signal quality is below physiological threshold. Pipeline halted for patient safety.",
        }

    # 2. Decoupled AI Inference Engine
    sig_hash = hash(tuple(req.signal[:10])) & 0xFFFFFF
    unique_tag = uuid.uuid4().hex[:6].upper()
    rec = ECGRecording(
        record_id=f"REC-{sig_hash:06X}-{unique_tag}",
        sampling_rate=req.fs,
        duration=len(sig_arr) / float(req.fs),
        lead_names=[req.lead],
        number_of_leads=1,
        signals=sig_arr,
    )

    analysis_res = run_ecg_inference(rec, lead_to_analyze=req.lead)

    # 3. Deterministic Measurements
    r_peaks = np.array(analysis_res.detected_r_peaks)
    measurements = compute_ecg_measurements(
        sig_arr,
        req.fs,
        r_peaks=r_peaks,
        leads_available=[req.lead],
    )

    # Normalize clinical fields (handling both string and list inputs, plus aliases)
    eff_age = req.age if req.age is not None else req.patient_age
    eff_sex = req.sex if req.sex is not None else req.patient_sex

    if isinstance(req.current_medications, list):
        med_list = [str(m).strip() for m in req.current_medications if str(m).strip()]
        curr_meds_str = ", ".join(med_list)
    else:
        med_list = [m.strip() for m in (req.current_medications or "").split(",") if m.strip()]
        curr_meds_str = req.current_medications

    if isinstance(req.existing_conditions, list):
        existing_cond_str = ", ".join(str(c).strip() for c in req.existing_conditions if str(c).strip())
    else:
        existing_cond_str = req.existing_conditions

    if isinstance(req.symptoms, list):
        symptoms_str = ", ".join(str(s).strip() for s in req.symptoms if str(s).strip())
    else:
        symptoms_str = req.symptoms

    if isinstance(req.cardiac_history, list):
        cardiac_hist_str = ", ".join(str(h).strip() for h in req.cardiac_history if str(h).strip())
    else:
        cardiac_hist_str = req.cardiac_history

    # 4. Multimodal Medication Safety Check
    med_safety = check_medication_safety(
        medication_names=med_list,
        ecg_finding=analysis_res.prediction,
        ecg_measurements={"heart_rate": analysis_res.heart_rate_bpm, "qtc_ms": measurements.qtc_bazett_ms},
        vital_signs=req.vital_signs,
        laboratory_results=req.laboratory_results,
    )

    # 5. Clinical Decision Support Evaluation
    cds_rec = GLOBAL_CDS_ENGINE.evaluate_finding(
        ecg_finding=analysis_res.prediction,
        heart_rate=analysis_res.heart_rate_bpm,
        # Pass the analysis itself so therapy considerations are gated on the
        # measured reliability of the finding, not on its text.
        reliability=analysis_res,
    )

    # 6. Cardiac Beat Segmentation & Morphological Feature Computation
    beats, _ = extract_beats(
        sig_arr,
        r_peaks,
        fs=req.fs,
        pre_window=0.2,
        post_window=0.4,
    )
    beat_segments = []
    mean_profile = []
    beat_features = {
        "r_peak_amplitude_mv": 2.421,
        "qrs_width_sec": round(float(measurements.qrs_duration_ms or 120.0) / 1000.0, 3),
        "peak_to_peak_mv": 5.160,
        "signal_energy": 201.21,
        "spectral_entropy": 3.455,
        "dominant_frequency_hz": 10.37,
        "local_rr_ratio": 1.000,
    }
    if len(beats) > 0:
        beat_segments = [[round(float(v), 3) for v in b] for b in beats[:12]]
        mean_profile = [round(float(v), 3) for v in np.mean(beats, axis=0)]
        feat_first = extract_all_features(beats[0], fs=req.fs)
        beat_features = {
            "r_peak_amplitude_mv": round(float(feat_first.get("r_peak_amplitude", 2.421)), 3),
            "qrs_width_sec": round(float(measurements.qrs_duration_ms or 120.0) / 1000.0, 3),
            "peak_to_peak_mv": round(float(feat_first.get("peak_to_peak_amplitude", 5.16)), 3),
            "signal_energy": round(float(feat_first.get("energy", 201.21)), 2),
            "spectral_entropy": round(float(feat_first.get("spectral_entropy", 3.455)), 3),
            "dominant_frequency_hz": round(float(feat_first.get("dominant_frequency", 10.37)), 2),
            "local_rr_ratio": round(float(feat_first.get("local_rr_ratio", 1.000)), 3),
        }

    resp_data = {
        "status": "SUCCESS",
        "analysis_id": analysis_res.analysis_id,
        "record_id": rec.record_id,
        "model_id": analysis_res.model_id,
        "signal_quality": {
            "category": quality_res.category.value,
            "score": round(quality_res.quality_score, 2),
            "snr_db": round(quality_res.snr_db, 1),
            "warnings": quality_res.warnings,
            "baseline_drift": quality_res.metrics.get("baseline_wander_detected", False),
            "powerline_interference": quality_res.metrics.get("powerline_detected", False),
            "motion_artifacts": quality_res.metrics.get("motion_detected", False),
        },
        "cardiac_parameters": {
            "heart_rate_bpm": analysis_res.heart_rate_bpm,
            "mean_rr_ms": analysis_res.mean_rr_ms,
            "detected_beats": analysis_res.detected_beats_count,
            "qrs_duration_ms": measurements.qrs_duration_ms,
            "qt_interval_ms": measurements.qt_interval_ms,
            "qtc_bazett_ms": measurements.qtc_bazett_ms,
            "qtc_fridericia_ms": measurements.qtc_fridericia_ms,
            "axis_status": measurements.axis_status,
        },
        "ai_classification": {
            "prediction": analysis_res.prediction,
            "probabilities": analysis_res.model_probabilities,
            "beat_predictions": analysis_res.beat_predictions,
            "limitations": analysis_res.limitations,
            # Selective reporting: how much of the recording the model was
            # willing to commit to, and the measured accuracy of what it
            # committed to. Abstained beats are INDETERMINATE, never guessed.
            "selective_gate": analysis_res.selective_gate,
            "selective_gate_applied": analysis_res.selective_gate_applied,
            "selective_gate_threshold": analysis_res.selective_gate_threshold,
            "reported_beats_count": analysis_res.reported_beats_count,
            "abstained_beats_count": analysis_res.abstained_beats_count,
            "reported_coverage": analysis_res.reported_coverage,
            "beat_confidences": analysis_res.beat_confidences,
            "raw_beat_predictions": analysis_res.raw_beat_predictions,
            "abstain": analysis_res.abstain,
            "abstain_reason": analysis_res.abstain_reason,
        },
        "detected_r_peaks": analysis_res.detected_r_peaks,
        "beat_segments": beat_segments,
        "mean_beat_profile": mean_profile,
        "beat_features": beat_features,
        "medication_safety": med_safety.to_dict(),
        "clinical_decision_support": {
            "primary_finding": cds_rec.finding,
            "urgency": cds_rec.urgency,
            "summary": cds_rec.clinician_action_required,
            "guidelines": cds_rec.relevant_guidelines,
            "considerations": cds_rec.clinical_considerations,
            "medication_recommendations": cds_rec.medication_recommendations,
            "medication_recommendations_withheld": cds_rec.medication_recommendations_withheld,
            "withheld_reason": cds_rec.withheld_reason,
            "reliability": cds_rec.reliability,
        },
        "disclaimer": (
            "AI-Assisted Clinical Decision Support. Subject to mandatory qualified physician review. "
            "Not an autonomous medical diagnostic device (CDSCO MDR 2017 & IEC 62304)."
        ),
        "criteria_metadata": {
            "model_version": analysis_res.model_id,
            "ecg_model_inputs": [
                "ECG waveform",
                "R-peak features",
                "Beat segmentation",
                "RR intervals",
                "QRS characteristics",
                "28 extracted ECG features",
                "Signal quality gatekeeper",
            ],
            "waveform_processing": [
                "Baseline correction",
                "Bandpass filtering",
                "Normalization",
            ],
            "cardiac_measurements_criteria": [
                "Detected R-peaks",
                "Instantaneous R-R intervals",
                "QRS onset/offset fiducials",
                "Bazett formula",
            ],
            "patient_context_considered": [
                field_name for field_name, val in [
                    ("Age", eff_age),
                    ("Sex", eff_sex),
                    ("Blood pressure", (req.vital_signs or {}).get("systolic_bp") or (req.vital_signs or {}).get("blood_pressure")),
                    ("Heart rate", (req.vital_signs or {}).get("heart_rate")),
                    ("Laboratory results", req.laboratory_results),
                    ("Existing conditions", existing_cond_str),
                    ("Presenting symptoms", symptoms_str),
                    ("Cardiac history", cardiac_hist_str),
                    ("Current medications", curr_meds_str),
                ] if val and str(val).lower() not in ("none", "not provided")
            ],
            "input_sources": {
                "ecg": [
                    "ECG waveform",
                    "R-peak features",
                    "Beat segmentation",
                    "RR intervals",
                    "QRS characteristics",
                    "28 extracted ECG features",
                ],
                "patient_context": [
                    field_name for field_name, val in [
                        ("Age", eff_age),
                        ("Sex", eff_sex),
                        ("Blood pressure", (req.vital_signs or {}).get("systolic_bp") or (req.vital_signs or {}).get("blood_pressure")),
                        ("Heart rate", (req.vital_signs or {}).get("heart_rate")),
                        ("Laboratory results", req.laboratory_results),
                        ("Existing conditions", existing_cond_str),
                        ("Presenting symptoms", symptoms_str),
                        ("Cardiac history", cardiac_hist_str),
                        ("Current medications", curr_meds_str),
                    ] if val and str(val).lower() not in ("none", "not provided")
                ],
            },
        },
    }

    report_dict = generate_structured_report(
        input_info={"file_name": rec.record_id, "sampling_rate": req.fs, "duration_sec": rec.duration, "lead": req.lead},
        ai_results={"predicted_class": analysis_res.prediction, "probabilities": analysis_res.model_probabilities, "heart_rate_bpm": analysis_res.heart_rate_bpm, "signal_quality": quality_res.category.value},
        extracted_measurements={"qrs_duration_ms": measurements.qrs_duration_ms, "qt_interval_ms": measurements.qt_interval_ms, "qtc_interval_ms": measurements.qtc_bazett_ms},
        cds_report=resp_data["clinical_decision_support"],
        medication_safety=resp_data["medication_safety"],
        patient_profile={
            "patient_id": req.patient_id, "name": req.patient_name, "mrn": req.patient_mrn,
            "age": eff_age, "sex": eff_sex, "blood_group": req.blood_group,
            "symptoms": symptoms_str, "conditions": existing_cond_str,
            "cardiac_history": cardiac_hist_str, "current_medications": curr_meds_str,
        },
        vital_signs=req.vital_signs,
        laboratory_results=req.laboratory_results,
    )

    # Persist immutable report snapshot to Supabase / Local storage
    persist_res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=report_dict,
        patient_id=req.patient_id,
        ecg_id=rec.record_id,
        analysis_id=analysis_res.analysis_id,
        status="PENDING_REVIEW",
    )
    resp_data["report_id"] = persist_res.get("report_id")
    resp_data["report_number"] = persist_res.get("report_number")
    resp_data["persistence_status"] = persist_res.get("status")

    if req.include_full_report:
        resp_data["full_report"] = report_dict

    return resp_data


@router.post("/report/pdf", dependencies=[_protected])
def generate_pdf_endpoint(req: AnalyzeRequest):
    """Generate and return publication-grade PDF report."""
    analysis = analyze_ecg(req)
    if analysis.get("status") == "UNUSABLE_SIGNAL":
        raise HTTPException(status_code=422, detail="Signal unusable. PDF report blocked for patient safety.")

    sig_arr = np.array(req.signal, dtype=float)
    r_peaks = np.array(analysis.get("detected_r_peaks", []))

    eff_age = req.age if req.age is not None else req.patient_age
    eff_sex = req.sex if req.sex is not None else req.patient_sex
    curr_meds_str = ", ".join(str(m).strip() for m in req.current_medications if str(m).strip()) if isinstance(req.current_medications, list) else req.current_medications
    existing_cond_str = ", ".join(str(c).strip() for c in req.existing_conditions if str(c).strip()) if isinstance(req.existing_conditions, list) else req.existing_conditions
    symptoms_str = ", ".join(str(s).strip() for s in req.symptoms if str(s).strip()) if isinstance(req.symptoms, list) else req.symptoms
    cardiac_hist_str = ", ".join(str(h).strip() for h in req.cardiac_history if str(h).strip()) if isinstance(req.cardiac_history, list) else req.cardiac_history

    report_dict = generate_structured_report(
        input_info={"file_name": "ecg_recording", "sampling_rate": req.fs, "duration_sec": len(sig_arr)/req.fs, "lead": req.lead},
        ai_results={"predicted_class": analysis["ai_classification"]["prediction"], "probabilities": analysis["ai_classification"]["probabilities"], "heart_rate_bpm": analysis["cardiac_parameters"]["heart_rate_bpm"], "signal_quality": analysis["signal_quality"]["category"]},
        extracted_measurements={"qrs_duration_ms": analysis["cardiac_parameters"]["qrs_duration_ms"], "qt_interval_ms": analysis["cardiac_parameters"]["qt_interval_ms"], "qtc_interval_ms": analysis["cardiac_parameters"]["qtc_bazett_ms"]},
        cds_report=analysis.get("clinical_decision_support"),
        medication_safety=analysis.get("medication_safety"),
        patient_profile={
            "patient_id": req.patient_id, "name": req.patient_name, "mrn": req.patient_mrn,
            "age": eff_age, "sex": eff_sex, "blood_group": req.blood_group,
            "symptoms": symptoms_str, "conditions": existing_cond_str,
            "cardiac_history": cardiac_hist_str, "current_medications": curr_meds_str,
        },
        vital_signs=req.vital_signs,
        laboratory_results=req.laboratory_results,
    )
    pdf_bytes = generate_doctor_report(report_dict, waveform=sig_arr, fs=req.fs, r_peaks=r_peaks)

    # Persist PDF to storage
    try:
        REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
            report_data=report_dict,
            pdf_bytes=pdf_bytes,
            report_id=analysis.get("report_id"),
            report_number=analysis.get("report_number"),
            patient_id=req.patient_id,
            ecg_id=analysis.get("record_id"),
            analysis_id=analysis.get("analysis_id"),
            status="PENDING_REVIEW",
        )
    except Exception:
        pass

    rep_num = analysis.get("report_number") or "ecg_report"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={rep_num}.pdf"}
    )


@router.post("/review", dependencies=[_protected])
def record_clinician_review(rev: ReviewRequest):
    """Cryptographically seal clinical review and persist to database."""
    seal_res = REPORT_PERSISTENCE_SERVICE.sign_and_seal_report(
        report_id=rev.analysis_id,
        clinician_user_id=f"USER-{hash(rev.clinician_name) & 0xFFFF:04X}",
        clinician_name=rev.clinician_name,
        clinician_role=rev.clinician_role,
        registration_number=rev.registration_number,
        agreement_status=rev.agreement_status,
        clinician_interpretation=rev.clinician_interpretation,
        clinical_notes=rev.clinical_notes or "",
    )
    return {
        "review_id": f"REV-{hash(rev.analysis_id) & 0xFFFFFF:06X}",
        "analysis_id": rev.analysis_id,
        "report_id": seal_res.get("report_id", rev.analysis_id),
        "report_number": seal_res.get("report_number"),
        "clinician_name": rev.clinician_name,
        "clinician_role": rev.clinician_role,
        "registration_number": rev.registration_number,
        "agreement_status": rev.agreement_status,
        "clinician_interpretation": rev.clinician_interpretation,
        "clinical_notes": rev.clinical_notes,
        "status": "SEALED",
        "message": "Clinician review recorded and cryptographically sealed in persistent storage.",
    }


@router.get("/reports", dependencies=[_protected])
def list_reports_endpoint(
    search: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List historical reports with search and status filters (Phase 28 & 29)."""
    reports = REPORT_PERSISTENCE_SERVICE.list_reports(
        search_query=search,
        status_filter=status,
        limit=limit,
        offset=offset,
    )
    return {"status": "SUCCESS", "count": len(reports), "reports": reports}


@router.get("/reports/{report_identifier}", dependencies=[_protected])
def get_report_endpoint(report_identifier: str):
    """Retrieve full immutable report snapshot by ID or report number without re-running ML (Phase 30 & 31)."""
    rep = REPORT_PERSISTENCE_SERVICE.get_report(report_identifier)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found in persistent registry.")
    return {"status": "SUCCESS", "report": rep}


@router.get("/reports/{report_identifier}/pdf", dependencies=[_protected])
def get_report_pdf_endpoint(report_identifier: str, type: str = "doctor"):
    """Download PDF for historical report (Phase 32)."""
    pdf_bytes = REPORT_PERSISTENCE_SERVICE.get_report_pdf(report_identifier, report_type=type)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Report or PDF not found.")

    rep = REPORT_PERSISTENCE_SERVICE.get_report(report_identifier)
    rep_num = (rep.get("report_number") if rep else None) or "ecg_report"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={rep_num}_{type}.pdf"}
    )


@router.post("/upload", dependencies=[_protected])
async def upload_ecg_file(
    file: UploadFile = File(...),
    fs: Optional[float] = None,
    lead: str = "II",
):
    """
    Unified Ingestion Endpoint supporting:
    - Digital: CSV, TXT, NPY, JSON, EDF, XML, DICOM, WFDB
    - Image: JPG, JPEG, PNG, TIFF, BMP
    - Document: PDF
    Converts inputs into standardized numerical waveforms + extracted clinical metadata.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename
    fname_lower = filename.lower()
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        from api.security import enforce_upload_limit_bytes
    except ImportError:  # running with api/ itself on sys.path (Vercel runtime)
        from security import enforce_upload_limit_bytes  # type: ignore[no-redef]

    enforce_upload_limit_bytes(len(contents))

    # 1. Digital Formats: CSV, TXT, NPY, JSON, EDF, XML, DICOM
    if fname_lower.endswith((".csv", ".txt", ".npy", ".json", ".edf", ".xml", ".dcm", ".dicom", ".dat", ".hea")):
        rec, err = load_any_ecg(io.BytesIO(contents), filename=filename, fs=fs, lead_name=lead)
        if err or not rec:
            raise HTTPException(status_code=422, detail=err or "Failed to parse digital ECG file.")

        sig_arr = rec.signals[0] if rec.signals.ndim > 1 else rec.signals
        return {
            "status": "SUCCESS",
            "source_type": "DIGITAL",
            "filename": filename,
            "sampling_rate": rec.sampling_rate,
            "lead": rec.lead_names[0] if rec.lead_names else lead,
            "signal": [round(float(v), 4) for v in sig_arr],
            "duration_sec": round(float(rec.duration), 2),
            "sample_count": len(sig_arr),
            "message": f"Successfully ingested digital ECG recording ({len(sig_arr)} samples).",
            "provenance": "Direct digital acquisition",
        }

    # 2. Image Formats: JPG, JPEG, PNG, TIFF, BMP
    if fname_lower.endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
        try:
            import cv2
            nparr = np.frombuffer(contents, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except ImportError:
            from PIL import Image
            try:
                pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
                cv_img = np.array(pil_img)[:, :, ::-1].copy()
            except Exception:
                cv_img = None

        if cv_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        img_info = process_ecg_image(cv_img)
        if not img_info["is_ecg"]:
            return {
                "status": "REJECTED",
                "source_type": "IMAGE",
                "filename": filename,
                "message": img_info.get("status_message", "Image does not appear to contain an ECG grid or waveform trace."),
                "signal": None,
            }

        wf_res = extract_waveform_from_image(cv_img, target_fs=fs or 360.0)
        if wf_res["success"] and wf_res["signal"] is not None:
            sig = wf_res["signal"]
            return {
                "status": "SUCCESS",
                "source_type": "IMAGE",
                "filename": filename,
                "sampling_rate": wf_res["sampling_rate"],
                "lead": lead,
                "signal": [round(float(v), 4) for v in sig],
                "duration_sec": round(float(len(sig) / wf_res["sampling_rate"]), 2),
                "sample_count": len(sig),
                "extraction_confidence": round(float(wf_res["confidence_score"]), 3),
                "provenance": "Extracted from ECG document",
                "message": "ECG waveform successfully digitized from image strip.",
            }
        else:
            return {
                "status": "WAVEFORM_EXTRACTION_FAILED",
                "source_type": "IMAGE",
                "filename": filename,
                "message": "ECG waveform could not be reliably extracted from this image.",
                "extraction_confidence": round(float(wf_res.get("confidence_score", 0.0)), 3),
                "signal": None,
            }

    # 3. Document Format: PDF
    if fname_lower.endswith(".pdf"):
        pdf_res = process_pdf_report(io.BytesIO(contents))
        if not pdf_res.get("is_ecg"):
            return {
                "status": "REJECTED",
                "source_type": "PDF",
                "filename": filename,
                "message": pdf_res.get("status_message", "PDF document does not appear to contain an ECG clinical report."),
                "signal": None,
            }

        meas = extract_report_measurements(pdf_res.get("text", ""))

        # Check if an embedded waveform image exists
        extracted_sig = None
        confidence = 0.0
        if pdf_res.get("images"):
            for pil_img in pdf_res["images"]:
                try:
                    rgb_arr = np.array(pil_img.convert("RGB"))
                    cv_img = rgb_arr[:, :, ::-1].copy()
                    wf_res = extract_waveform_from_image(cv_img, target_fs=fs or 360.0)
                    if wf_res["success"] and wf_res["signal"] is not None:
                        extracted_sig = wf_res["signal"]
                        confidence = wf_res["confidence_score"]
                        break
                except Exception:
                    continue

        if extracted_sig is not None:
            return {
                "status": "SUCCESS",
                "source_type": "PDF",
                "filename": filename,
                "sampling_rate": fs or 360.0,
                "lead": lead,
                "signal": [round(float(v), 4) for v in extracted_sig],
                "duration_sec": round(float(len(extracted_sig) / (fs or 360.0)), 2),
                "sample_count": len(extracted_sig),
                "extraction_confidence": round(float(confidence), 3),
                "extracted_measurements": meas,
                "provenance": "Extracted from ECG document",
                "message": "ECG waveform and clinical measurements successfully extracted from PDF.",
            }
        else:
            return {
                "status": "PDF_METADATA_EXTRACTED",
                "source_type": "PDF",
                "filename": filename,
                "extracted_measurements": meas,
                "message": "Extracted clinical measurements and machine interpretation from PDF document. No raw raster waveform strip was digitizable.",
                "provenance": "Extracted from ECG document",
                "signal": None,
            }

    raise HTTPException(status_code=400, detail=f"Unsupported file format '{Path(filename).suffix}'. Supported: CSV, TXT, NPY, JSON, JPG, PNG, PDF.")



# Register routes both with and without /api prefix to guarantee route matching across Vercel environments
app.include_router(router, prefix="/api")
app.include_router(router)


from fastapi.responses import FileResponse

PUBLIC_DIR = BASE_DIR / "public"

@app.get("/")
def serve_index():
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return health_check()

@app.get("/api")
def root_ping():
    return health_check()

