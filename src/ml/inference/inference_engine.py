"""
ML Inference Engine Module
==========================
Standalone, UI-independent machine learning inference service for ECG rhythm analysis.
Enforces the mandatory output schema and strict regulatory boundaries.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
    from src.ml.feature_extraction import extract_all_features
    from src.ml.models import load_model_artifacts
    from src.ml.peak_detection import detect_r_peaks
    from src.ml.preprocessing import preprocess_pipeline
    from src.ml.segmentation import compute_rr_intervals, extract_beats
    from src.ml.validation import validate_feature_matrix
    from src.quality.quality_gate import QualityCategory, evaluate_ecg_quality_gate
except ImportError:
    from ecg_core.models import ECGRecording
    from ml.feature_extraction import extract_all_features
    from ml.models import load_model_artifacts
    from ml.peak_detection import detect_r_peaks
    from ml.preprocessing import preprocess_pipeline
    from ml.segmentation import compute_rr_intervals, extract_beats
    from ml.validation import validate_feature_matrix
    from quality.quality_gate import QualityCategory, evaluate_ecg_quality_gate

ACTIVE_MODEL_VERSION = "ECG-RF-1.0.0"
SUPPORTED_LEADS = ["II", "MLII", "LEAD II", "LEAD_II"]


def run_ecg_ml_inference(
    recording_or_signal: Union[ECGRecording, np.ndarray, List[float]],
    fs: Optional[float] = None,
    lead_to_analyze: str = "II",
) -> Dict[str, Any]:
    """Execute complete, decoupled ECG ML inference.

    Returns:
        Structured dictionary matching Phase 4 schema.
    """
    start_time = time.perf_counter()
    analysis_id = f"ANL-{uuid.uuid4().hex[:12].upper()}"
    warnings: List[str] = []
    limitations: List[str] = [
        "Model validated strictly on Lead II for Normal Sinus Rhythm vs. PVC.",
        "Not trained to detect STEMI, AFib, blocks, or general cardiac pathology.",
        "Clinical decision support output; requires mandatory qualified physician sign-off.",
    ]

    # Resolve signal and sampling rate
    if isinstance(recording_or_signal, ECGRecording):
        sampling_rate = float(recording_or_signal.sampling_rate)
        if lead_to_analyze in recording_or_signal.lead_names:
            idx = recording_or_signal.lead_names.index(lead_to_analyze)
            sig = recording_or_signal.signals[idx]
        else:
            sig = recording_or_signal.signals[0]
            warnings.append(f"Requested lead '{lead_to_analyze}' not present; analyzed '{recording_or_signal.lead_names[0]}'.")
            lead_to_analyze = recording_or_signal.lead_names[0]
    else:
        sig = np.asarray(recording_or_signal, dtype=float)
        if fs is None or fs <= 0:
            return {
                "analysis_id": analysis_id,
                "model_version": ACTIVE_MODEL_VERSION,
                "prediction": "NO_RESULT_MISSING_SAMPLING_RATE",
                "model_probabilities": {},
                "signal_quality": "UNUSABLE",
                "heart_rate": None,
                "rr_intervals": [],
                "r_peaks": [],
                "lead_analyzed": lead_to_analyze,
                "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "warnings": ["Missing sampling rate. Cannot execute ML without verified clock frequency."],
                "limitations": limitations,
            }
        sampling_rate = float(fs)

    # 1. Lead Verification
    if lead_to_analyze.upper() not in SUPPORTED_LEADS:
        return {
            "analysis_id": analysis_id,
            "model_version": ACTIVE_MODEL_VERSION,
            "prediction": "NO_RESULT_UNSUPPORTED_LEAD",
            "model_probabilities": {},
            "signal_quality": "UNUSABLE",
            "heart_rate": None,
            "rr_intervals": [],
            "r_peaks": [],
            "lead_analyzed": lead_to_analyze,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": [f"Lead '{lead_to_analyze}' is not supported. Active model requires Lead II / MLII."],
            "limitations": limitations,
        }

    # 2. Quality Gatekeeper Safety Barrier
    gate_decision = evaluate_ecg_quality_gate(sig, sampling_rate, lead_name=lead_to_analyze)
    if not gate_decision.can_run_ai:
        return {
            "analysis_id": analysis_id,
            "model_version": ACTIVE_MODEL_VERSION,
            "prediction": f"NO_RESULT_{gate_decision.category.value}",
            "model_probabilities": {},
            "signal_quality": gate_decision.category.value,
            "heart_rate": None,
            "rr_intervals": [],
            "r_peaks": [],
            "lead_analyzed": lead_to_analyze,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": gate_decision.rejection_reasons + gate_decision.warnings,
            "limitations": limitations,
        }

    # 3. Model Loading
    try:
        clf, scaler, metadata = load_model_artifacts()
    except Exception as me:
        return {
            "analysis_id": analysis_id,
            "model_version": ACTIVE_MODEL_VERSION,
            "prediction": "NO_RESULT_MODEL_UNAVAILABLE",
            "model_probabilities": {},
            "signal_quality": gate_decision.category.value,
            "heart_rate": None,
            "rr_intervals": [],
            "r_peaks": [],
            "lead_analyzed": lead_to_analyze,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": [f"Machine learning model unavailable: {str(me)}"],
            "limitations": limitations,
        }

    # 4. Preprocessing & R-Peak Detection
    clean_sig = preprocess_pipeline(sig, sampling_rate)
    r_peaks = detect_r_peaks(clean_sig, sampling_rate)

    if len(r_peaks) < 2:
        return {
            "analysis_id": analysis_id,
            "model_version": ACTIVE_MODEL_VERSION,
            "prediction": "NO_RESULT_INSUFFICIENT_BEATS",
            "model_probabilities": {},
            "signal_quality": gate_decision.category.value,
            "heart_rate": None,
            "rr_intervals": [],
            "r_peaks": r_peaks.tolist() if hasattr(r_peaks, "tolist") else list(r_peaks),
            "lead_analyzed": lead_to_analyze,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": ["Fewer than 2 valid R-peaks detected; cardiac rhythm calculation impossible."],
            "limitations": limitations,
        }

    # 5. Beat Segmentation & Measurements
    beats, valid_peaks = extract_beats(clean_sig, r_peaks, sampling_rate)
    rr_intervals_ms, heart_rate = compute_rr_intervals(valid_peaks, sampling_rate)

    if len(beats) == 0:
        return {
            "analysis_id": analysis_id,
            "model_version": ACTIVE_MODEL_VERSION,
            "prediction": "NO_RESULT_NO_SEGMENTS",
            "model_probabilities": {},
            "signal_quality": gate_decision.category.value,
            "heart_rate": round(float(heart_rate), 1) if heart_rate else None,
            "rr_intervals": [round(float(r), 1) for r in rr_intervals_ms],
            "r_peaks": [int(p) for p in valid_peaks],
            "lead_analyzed": lead_to_analyze,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": ["Could not extract full cardiac beat windows."],
            "limitations": limitations,
        }

    # 6. Feature Extraction & Scaling
    features = extract_all_features(beats, sampling_rate, r_peaks=valid_peaks)
    is_valid_f, feat_errors = validate_feature_matrix(features, expected_dim=28)
    if not is_valid_f:
        return {
            "analysis_id": analysis_id,
            "model_version": ACTIVE_MODEL_VERSION,
            "prediction": "NO_RESULT_FEATURE_FAILURE",
            "model_probabilities": {},
            "signal_quality": gate_decision.category.value,
            "heart_rate": round(float(heart_rate), 1) if heart_rate else None,
            "rr_intervals": [round(float(r), 1) for r in rr_intervals_ms],
            "r_peaks": [int(p) for p in valid_peaks],
            "lead_analyzed": lead_to_analyze,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": feat_errors,
            "limitations": limitations,
        }

    scaled_features = scaler.transform(features)

    # 7. Model Inference
    beat_probs = clf.predict_proba(scaled_features)
    classes = list(clf.classes_)

    # Mean probability distribution
    mean_probs = np.mean(beat_probs, axis=0)
    prob_dict = {str(cls_name): round(float(p), 4) for cls_name, p in zip(classes, mean_probs)}

    # Determine window-level classification
    # If any beat has high PVC probability (>0.60), flag PVC ectopy pattern
    pvc_idx = classes.index("PVC") if "PVC" in classes else -1
    if pvc_idx != -1 and np.any(beat_probs[:, pvc_idx] > 0.60):
        final_pred = "Premature Ventricular Contraction"
    else:
        top_cls = classes[int(np.argmax(mean_probs))]
        final_pred = "Normal Sinus Rhythm" if top_cls == "Normal" else top_cls

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "analysis_id": analysis_id,
        "model_version": ACTIVE_MODEL_VERSION,
        "prediction": final_pred,
        "model_probabilities": prob_dict,
        "signal_quality": gate_decision.category.value,
        "heart_rate": round(float(heart_rate), 1) if heart_rate else None,
        "rr_intervals": [round(float(r), 1) for r in rr_intervals_ms],
        "r_peaks": [int(p) for p in valid_peaks],
        "lead_analyzed": lead_to_analyze,
        "processing_time_ms": duration_ms,
        "warnings": warnings,
        "limitations": limitations,
    }
