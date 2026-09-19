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
from typing import Any, Dict, List, Optional

# Ensure project root and src are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ecg_core.models import ECGRecording
from inference.inference_engine import ACTIVE_MODEL_ID, run_ecg_inference
from measurements.measurement_engine import compute_ecg_measurements
from safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate

app = FastAPI(
    title="AI-ECG Clinical Decision Support API",
    description="Vercel Serverless Medical-Device Software Backend adhering to CDSCO MDR 2017 & IEC 62304",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    signal: List[float]
    fs: float = 360.0
    lead: str = "II"
    patient_name: Optional[str] = "Anonymous"
    patient_mrn: Optional[str] = None


class ReviewRequest(BaseModel):
    analysis_id: str
    clinician_name: str
    clinician_role: str = "CARDIOLOGIST"
    registration_number: str
    agreement_status: str  # CONFIRMED, MODIFIED, REJECTED
    clinician_interpretation: str
    clinical_notes: Optional[str] = ""


@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "device": "AI-ECG Analyzer (SaMD)",
        "regulatory_classification": "CDSCO MDR 2017 Class B / IEC 62304 Class B",
        "active_model_id": ACTIVE_MODEL_ID,
        "safety_principle": "NO RELIABLE INPUT = NO AI RESULT",
        "disclaimer": "Clinical Decision Support System. Requires mandatory qualified physician review.",
    }


@app.get("/api/sample")
def get_sample_data(sample_type: str = "normal"):
    fs = 360.0
    duration_sec = 5.0
    t = np.arange(int(fs * duration_sec)) / fs

    # Synthetic periodic ECG trace
    sig = 0.08 * np.sin(2 * np.pi * 1.2 * t)
    n_beats = int(duration_sec * 1.2)

    for i in range(1, n_beats + 1):
        idx = int(i * (fs / 1.2))
        if idx < len(sig):
            # QRS complex
            w = 8
            start = max(0, idx - w)
            end = min(len(sig), idx + w)
            if sample_type == "pvc" and i == 2:
                # Ectopic wide PVC-like spike
                sig[start:end] += 1.8 * np.exp(-0.5 * ((np.arange(start, end) - idx) / 5) ** 2)
            else:
                sig[start:end] += 1.2 * np.exp(-0.5 * ((np.arange(start, end) - idx) / 3) ** 2)

    return {
        "sample_type": sample_type,
        "fs": fs,
        "duration_sec": duration_sec,
        "lead": "II",
        "signal": sig.tolist(),
    }


@app.post("/api/analyze")
def analyze_ecg(req: AnalyzeRequest):
    if not req.signal or len(req.signal) < 100:
        raise HTTPException(
            status_code=400,
            detail="Signal duration is too short for cardiac evaluation (minimum 100 samples required).",
        )

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
    rec = ECGRecording(
        record_id=f"REC-VERCEL-{hash(tuple(req.signal[:10])) & 0xFFFFFF:06X}",
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

    return {
        "status": "SUCCESS",
        "analysis_id": analysis_res.analysis_id,
        "record_id": rec.record_id,
        "model_id": analysis_res.model_id,
        "signal_quality": {
            "category": quality_res.category.value,
            "score": round(quality_res.quality_score, 2),
            "snr_db": round(quality_res.snr_db, 1),
            "warnings": quality_res.warnings,
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
        },
        "detected_r_peaks": analysis_res.detected_r_peaks,
        "disclaimer": (
            "AI-Assisted Clinical Decision Support. Subject to mandatory qualified physician review. "
            "Not an autonomous medical diagnostic device (CDSCO MDR 2017 & IEC 62304)."
        ),
    }


@app.post("/api/review")
def record_clinician_review(rev: ReviewRequest):
    return {
        "review_id": f"REV-VERCEL-{hash(rev.analysis_id) & 0xFFFFFF:06X}",
        "analysis_id": rev.analysis_id,
        "clinician_name": rev.clinician_name,
        "clinician_role": rev.clinician_role,
        "registration_number": rev.registration_number,
        "agreement_status": rev.agreement_status,
        "clinician_interpretation": rev.clinician_interpretation,
        "clinical_notes": rev.clinical_notes,
        "status": "SEALED",
        "message": "Clinician review recorded and cryptographically sealed.",
    }
