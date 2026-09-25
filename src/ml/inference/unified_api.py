"""
Unified Multi-Task ECG Inference Engine.
=========================================

Phases 17–26:
- Dispatches to specialised models per task (quality gate, beat arrhythmia, AF, 12-lead, ST).
- Prevents incompatible lead configurations and sampling rates from executing.
- Rule 1: multi-model architecture (never one giant model).
- Rule 3: outputs are model probabilities, not clinical diagnostic certainty.
- Rule 5: non-diagnostic research and clinical decision-support barriers.

Integrity rules added by the training-provenance work:

* A model whose registry entry is a ``SYNTHETIC_PLACEHOLDER`` is never served as
  a result. The engine returns an explicit non-clinical block instead.
* Task C (12-lead) derives its feature vector from the *actual* multi-lead signal.
  It previously fed ``numpy.zeros((1, 24))`` to the model, so its output was
  independent of the patient's ECG.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np

# Ensure the project root is importable so dataset-grounded feature builders in
# ``training/`` can be reused by the serving path.
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.ecg_core.models import ECGRecording
from src.ecg_core.standardized_record import StandardizedECGRecord
from src.ml.inference.inference_engine import run_ecg_ml_inference
from src.ml.models.registry import GLOBAL_MODEL_REGISTRY, ModelLifecycleStatus
from src.ml.tasks import (
    TASK_12LEAD_DIAGNOSTIC,
    TASK_AF_DETECTION,
    TASK_BEAT_ARRHYTHMIA,
    TASK_QUALITY_GATE,
    TASK_ST_ANALYSIS,
)
from src.quality.quality_gate import QualityCategory, evaluate_ecg_quality_gate

MODELS_DIR = PROJ_ROOT / "models"

#: Registry identifiers for the specialised task models.
TASK_MODEL_IDS = {
    "af_detection": "ECG-AF-1.0.0-candidate",
    "12lead_diagnosis": "ECG-PTBXL-1.0.0-candidate",
    "st_analysis": "ECG-ST-1.0.0-candidate",
    "quality_gate": "ECG-QUALITY-1.0.0",
}


def _extract_leads(
    recording_or_signal: Union[ECGRecording, StandardizedECGRecord, np.ndarray, List[float]],
    fs: Optional[float],
    lead: str,
) -> tuple[np.ndarray, Optional[np.ndarray], List[str], float]:
    """Return ``(signal_1d, lead_matrix_or_None, lead_names, sampling_rate)``."""
    if isinstance(recording_or_signal, StandardizedECGRecord):
        matrix = np.atleast_2d(np.asarray(recording_or_signal.signal, dtype=float))
        return matrix[0], matrix, list(recording_or_signal.leads), float(recording_or_signal.sampling_rate_hz)

    if isinstance(recording_or_signal, ECGRecording):
        leads = list(recording_or_signal.lead_names)
        signals = np.atleast_2d(np.asarray(recording_or_signal.signals, dtype=float))
        if lead in leads:
            signal = signals[leads.index(lead)]
        else:
            signal = signals[0]
        return signal, signals, leads, float(recording_or_signal.sampling_rate)

    signal = np.asarray(recording_or_signal, dtype=float)
    return signal, None, [lead], float(fs) if fs else 360.0


def _placeholder_block(
    analysis_id: str,
    task_id: str,
    model_id: str,
    entry: Dict[str, Any],
    elapsed_ms: float,
) -> Dict[str, Any]:
    """Explicit refusal to serve a model trained on fabricated data."""
    provenance = entry.get("provenance", {})
    return {
        "analysis_id": analysis_id,
        "task": task_id,
        "model_version": model_id,
        "prediction": "MODEL_UNAVAILABLE_NON_CLINICAL",
        "signal_quality": None,
        "processing_time_ms": round(elapsed_ms, 2),
        "warnings": [
            f"Model '{model_id}' is registered as {ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value}.",
            "It was fitted to fabricated or demo data and carries no clinical meaning.",
        ],
        "rejections": [
            provenance.get("reason", "Training data provenance is not real patient data."),
        ],
        "limitations": [
            "No AI result is returned for this task until a model is trained on real data.",
        ],
    }


def analyze_ecg(
    recording_or_signal: Union[ECGRecording, StandardizedECGRecord, np.ndarray, List[float]],
    fs: Optional[float] = None,
    task: str = TASK_BEAT_ARRHYTHMIA,
    model_id: Optional[str] = None,
    lead: str = "II",
    allow_placeholder: bool = False,
) -> Dict[str, Any]:
    """Unified entry point for ECG analysis across specialised task models."""
    start_time = time.perf_counter()
    analysis_id = f"ANL-{uuid.uuid4().hex[:12].upper()}"

    signal, lead_matrix, leads, sampling_rate = _extract_leads(recording_or_signal, fs, lead)

    task_id = task.task_id if hasattr(task, "task_id") else str(task)

    # 1. Mandatory signal quality gatekeeper for all clinical tasks
    gate_res = evaluate_ecg_quality_gate(signal, sampling_rate, lead_name=lead)

    if task_id == "quality_gate":
        return {
            "analysis_id": analysis_id,
            "task": "quality_gate",
            "model_version": "SIGNAL-GATE-1.0.0",
            "prediction": gate_res.category.value,
            "can_run_ai": gate_res.can_run_ai,
            "quality_score": gate_res.quality_score,
            "warnings": gate_res.warnings,
            "rejections": gate_res.rejection_reasons,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    if task_id == "beat_arrhythmia":
        res = run_ecg_ml_inference(signal, fs=sampling_rate, lead_to_analyze=lead)
        res["task"] = "beat_arrhythmia"
        return res

    if task_id in ("af_detection", "12lead_diagnosis", "st_analysis"):
        if model_id is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Provenance gate: never serve a placeholder artifact.
            if not allow_placeholder:
                registry_id = model_id or TASK_MODEL_IDS.get(task_id)
                if registry_id and GLOBAL_MODEL_REGISTRY.is_placeholder(registry_id):
                    try:
                        entry = GLOBAL_MODEL_REGISTRY.get_model_entry(registry_id)
                    except KeyError:
                        entry = {}
                    return _placeholder_block(analysis_id, task_id, registry_id, entry, elapsed_ms)

            if not gate_res.can_run_ai or gate_res.category == QualityCategory.UNUSABLE:
                return {
                    "analysis_id": analysis_id,
                    "task": task_id,
                    "model_version": "HALTED-SAFETY-GATE",
                    "prediction": "AI_ANALYSIS_BLOCKED_UNUSABLE_SIGNAL",
                    "signal_quality": gate_res.category.value,
                    "lead_analyzed": lead,
                    "processing_time_ms": round(elapsed_ms, 2),
                    "warnings": gate_res.warnings,
                    "rejections": gate_res.rejection_reasons or ["Excessive noise, flatline, or missing data."],
                    "limitations": ["Technical signal corruption precludes safe clinical evaluation."],
                }

            if task_id == "af_detection":
                return _run_af(analysis_id, model_id, signal, sampling_rate, lead, gate_res, start_time)

            if task_id == "12lead_diagnosis":
                return _run_12lead(
                    analysis_id, model_id, lead_matrix, leads, sampling_rate, gate_res, start_time
                )

            if task_id == "st_analysis":
                return _run_st(analysis_id, model_id, signal, sampling_rate, lead, gate_res, start_time)

        # Default research task safety barrier
        return {
            "analysis_id": analysis_id,
            "task": task_id,
            "model_version": f"RESEARCH-{task_id.upper()}-0.1.0",
            "prediction": "RESEARCH_TASK_EVALUATION_ONLY",
            "signal_quality": gate_res.category.value,
            "lead_analyzed": lead,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": [
                f"Task '{task_id}' is currently in research validation stage.",
                "Not approved for primary or secondary clinical triage.",
            ],
            "limitations": [
                "Investigative model architecture only.",
                "Zero patient-level diagnosis authorized.",
            ],
        }

    return {
        "analysis_id": analysis_id,
        "task": task_id,
        "model_version": "UNKNOWN",
        "prediction": "NO_RESULT_UNKNOWN_TASK",
        "signal_quality": "UNUSABLE",
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "warnings": [f"Requested task '{task_id}' is not recognized by ECG Guardian."],
        "limitations": ["Unknown task specification."],
    }


def _run_af(analysis_id, model_id, signal, sampling_rate, lead, gate_res, start_time) -> Dict[str, Any]:
    """Atrial fibrillation rhythm inference from measured RR variability."""
    model_path = MODELS_DIR / "candidate" / "af_classifier.pkl"
    scaler_path = MODELS_DIR / "candidate" / "af_scaler.pkl"
    if not (model_path.exists() and scaler_path.exists()):
        return _artifact_missing(analysis_id, "af_detection", model_id, start_time)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    from src.peak_detection import detect_r_peaks
    from training.real_features import AF_FEATURE_NAMES, af_rhythm_features, template_for

    peaks, _ = detect_r_peaks(signal, sampling_rate)
    rr = np.diff(peaks) / sampling_rate if len(peaks) > 1 else np.array([])
    features = af_rhythm_features(rr)
    if features is None:
        return {
            "analysis_id": analysis_id,
            "task": "af_detection",
            "model_version": model_id,
            "prediction": "INSUFFICIENT_RR_INTERVALS",
            "signal_quality": gate_res.category.value,
            "lead_analyzed": lead,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": ["Fewer than five RR intervals were measurable; rhythm cannot be characterised."],
            "limitations": ["Rhythm classification requires a longer recording."],
        }

    try:
        vector = template_for(AF_FEATURE_NAMES, features)
    except KeyError:
        vector = (AF_FEATURE_NAMES, [features.get(name, 0.0) for name in AF_FEATURE_NAMES])

    scaled = scaler.transform([vector[1]])
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    classes = list(model.classes_)

    return {
        "analysis_id": analysis_id,
        "task": "af_detection",
        "model_version": model_id,
        "prediction": str(prediction),
        "model_output_probabilities": {str(classes[i]): round(float(probabilities[i]), 4) for i in range(len(classes))},
        "measured_rhythm_features": features,
        "signal_quality": gate_res.category.value,
        "lead_analyzed": lead,
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "warnings": ["Model output probability is not equivalent to clinical certainty."],
        "limitations": ["Evaluates single-lead rhythm regularity."],
    }


def _run_12lead(analysis_id, model_id, lead_matrix, leads, sampling_rate, gate_res, start_time) -> Dict[str, Any]:
    """12-lead multi-label inference from real per-lead measurements."""
    model_path = MODELS_DIR / "candidate" / "ptbxl_classifier.pkl"
    scaler_path = MODELS_DIR / "candidate" / "ptbxl_scaler.pkl"
    if not (model_path.exists() and scaler_path.exists()):
        return _artifact_missing(analysis_id, "12lead_diagnosis", model_id, start_time)

    if lead_matrix is None or lead_matrix.shape[0] < 2:
        return {
            "analysis_id": analysis_id,
            "task": "12lead_diagnosis",
            "model_version": model_id,
            "prediction": "MULTI_LEAD_DATA_REQUIRED",
            "signal_quality": gate_res.category.value,
            "lead_analyzed": "requires >= 2 leads",
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": [
                "12-lead diagnosis was requested but a single-lead signal was supplied.",
                "No prediction is issued rather than feeding fabricated lead values.",
            ],
            "limitations": ["Provide a multi-lead ECGRecording with standard lead names."],
        }

    from training.real_features import PTBXL_FEATURE_NAMES, PTBXL_SUPERCLASSES, ptbxl_lead_features

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feature_dict = ptbxl_lead_features(lead_matrix, sampling_rate, leads)
    if feature_dict is None:
        return {
            "analysis_id": analysis_id,
            "task": "12lead_diagnosis",
            "model_version": model_id,
            "prediction": "NO_MEASURABLE_LEADS",
            "signal_quality": gate_res.category.value,
            "lead_analyzed": ", ".join(leads),
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": ["No standard 12-lead names could be matched, so no features could be measured."],
            "limitations": ["Lead names must follow standard nomenclature (I, II, V1..V6)."],
        }

    # Absent leads are imputed with the training-time schema default of 0.0 only
    # after being explicitly reported as missing.
    missing_leads = [name for name in PTBXL_FEATURE_NAMES if name not in feature_dict]
    vector = [feature_dict.get(name, 0.0) for name in PTBXL_FEATURE_NAMES]
    scaled = scaler.transform([vector])
    multi_predictions = model.predict(scaled)[0]

    detected = [
        PTBXL_SUPERCLASSES[i]
        for i in range(min(len(PTBXL_SUPERCLASSES), len(multi_predictions)))
        if multi_predictions[i] == 1
    ]
    positive_probs: Dict[str, float] = {}
    for idx, estimator in enumerate(getattr(model, "estimators_", [])):
        if idx >= len(PTBXL_SUPERCLASSES):
            break
        try:
            positive_probs[PTBXL_SUPERCLASSES[idx]] = round(float(estimator.predict_proba(scaled)[0][1]), 4)
        except Exception:
            continue

    return {
        "analysis_id": analysis_id,
        "task": "12lead_diagnosis",
        "model_version": model_id,
        "prediction": ", ".join(detected) if detected else "NO_SUPERCLASS_DETECTED",
        "multi_label_findings": detected,
        "model_output_probabilities": positive_probs,
        "measured_lead_features": feature_dict,
        "leads_measured": sorted({name.rsplit("_", 2)[0] for name in feature_dict}),
        "missing_lead_features": missing_leads,
        "signal_quality": gate_res.category.value,
        "lead_analyzed": "12-Lead Standard",
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "warnings": [
            "Multi-label predictions represent preliminary screening.",
            "Model output probability is not equivalent to clinical certainty.",
        ],
        "limitations": ["Requires a full standard 12-lead electrode array."],
    }


def _run_st(analysis_id, model_id, signal, sampling_rate, lead, gate_res, start_time) -> Dict[str, Any]:
    """ST-segment inference from measured J-point and ST60 values."""
    model_path = MODELS_DIR / "candidate" / "st_classifier.pkl"
    scaler_path = MODELS_DIR / "candidate" / "st_scaler.pkl"
    if not (model_path.exists() and scaler_path.exists()):
        return _artifact_missing(analysis_id, "st_analysis", model_id, start_time)

    from src.peak_detection import detect_r_peaks
    from training.real_features import ST_FEATURE_NAMES, st_features

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    peaks, _ = detect_r_peaks(signal, sampling_rate)
    measured = [
        st_features(signal, sampling_rate, int(peak))
        for peak in np.asarray(peaks, dtype=int)[:50]
    ]
    measured = [m for m in measured if m is not None]
    if not measured:
        return {
            "analysis_id": analysis_id,
            "task": "st_analysis",
            "model_version": model_id,
            "prediction": "NO_MEASURABLE_ST_SEGMENT",
            "signal_quality": gate_res.category.value,
            "lead_analyzed": lead,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": ["No beat offered a measurable ST segment; no prediction was issued."],
            "limitations": ["Requires at least one clean QRS complex with surrounding baseline."],
        }

    averaged = {
        name: float(np.mean([m[name] for m in measured])) for name in ST_FEATURE_NAMES
    }
    scaled = scaler.transform([[averaged[name] for name in ST_FEATURE_NAMES]])
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    classes = list(model.classes_)

    return {
        "analysis_id": analysis_id,
        "task": "st_analysis",
        "model_version": model_id,
        "prediction": str(prediction),
        "model_output_probabilities": {str(classes[i]): round(float(probabilities[i]), 4) for i in range(len(classes))},
        "measured_st_features": averaged,
        "beats_measured": len(measured),
        "signal_quality": gate_res.category.value,
        "lead_analyzed": lead,
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "warnings": ["ST-segment shifts must be correlated with symptoms and troponin."],
        "limitations": ["Evaluates single-lead repolarisation."],
    }


def _artifact_missing(analysis_id: str, task_id: str, model_id: Optional[str], start_time: float) -> Dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "task": task_id,
        "model_version": model_id or "UNREGISTERED",
        "prediction": "MODEL_ARTIFACT_MISSING",
        "signal_quality": None,
        "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "warnings": [f"No trained artifact is available for task '{task_id}'."],
        "limitations": ["Train this task on real data before requesting a prediction."],
    }
