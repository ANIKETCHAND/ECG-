"""
Decoupled Clinical AI Inference Engine
======================================

Executes end-to-end ECG evaluation strictly governed by safety gates:
1. Input validation & Unified ECGRecording abstraction
2. Pre-inference Signal Quality Gatekeeper (GOOD, ACCEPTABLE, POOR, UNUSABLE)
3. Hard Safety Rules (NO RELIABLE INPUT = NO AI RESULT)
4. Model compatibility check (Single-Lead II only for ECG-RF-1.0.0)
5. Preprocessing, R-peak detection, beat segmentation & 28-feature extraction
6. Random Forest inference with honest probability calibration
7. Fully traceable output matching regulatory data schema

References:
- Medical Devices Rules (MDR) 2017 (CDSCO)
- IEC 62304 Medical Device Software
- ISO 14971 Risk Management
"""

from __future__ import annotations

import json
import time
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from ecg_core.models import ECGAnalysisResult, ECGRecording
    from safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate
except ImportError:
    from src.ecg_core.models import ECGAnalysisResult, ECGRecording
    from src.safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate

from feature_extraction import extract_all_features
from peak_detection import detect_r_peaks
from preprocessing import preprocess_pipeline
from segmentation import extract_beats, calculate_rr_intervals, estimate_heart_rate

# Model Version Constants
ACTIVE_MODEL_ID = "ECG-RF-1.0.0"
PREPROCESSING_VERSION = "PREPROC-2.0-MEDIAN-BW3"
FEATURE_VERSION = "FEAT-28-AAMI"


def get_model_artifacts(models_dir: Optional[Union[str, Path]] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load serialized model artifacts with fallback resolution."""
    if models_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent / "models"
    else:
        base_dir = Path(models_dir)

    # Search production registry first, fallback to base models directory
    prod_dir = base_dir / "production"
    if (prod_dir / "classifier.pkl").exists() and (prod_dir / "scaler.pkl").exists():
        clf = joblib.load(prod_dir / "classifier.pkl")
        scaler = joblib.load(prod_dir / "scaler.pkl")
        meta_file = prod_dir / "metadata.json"
    else:
        clf = joblib.load(base_dir / "classifier.pkl")
        scaler = joblib.load(base_dir / "scaler.pkl")
        meta_file = base_dir / "metadata.json"

    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return clf, scaler, metadata


def run_ecg_inference(
    recording_or_signal: Union[ECGRecording, np.ndarray, List[float]],
    fs: Optional[float] = None,
    lead_to_analyze: str = "II",
    models_dir: Optional[Union[str, Path]] = None,
) -> ECGAnalysisResult:
    """Execute decoupled clinical AI inference engine.

    Args:
        recording_or_signal: Unified ECGRecording entity or raw voltage series
        fs: Sampling rate in Hz (mandatory if raw voltage array is provided)
        lead_to_analyze: Lead to select for inference (default: 'II')
        models_dir: Directory containing trained model artifacts

    Returns:
        ECGAnalysisResult matching regulatory data schema with traceability.
    """
    start_time = time.perf_counter()
    analysis_id = f"ANL-{uuid.uuid4().hex[:12].upper()}"

    # 1. Normalize input into ECGRecording entity
    if isinstance(recording_or_signal, ECGRecording) or type(recording_or_signal).__name__ == "ECGRecording":
        recording = recording_or_signal
    else:
        if fs is None or fs <= 0:
            return _build_failure_result(
                analysis_id=analysis_id,
                record_id="UNSPECIFIED",
                reason="Sampling frequency (fs) was not specified or is non-positive. Cannot assume missing metadata.",
                lead=lead_to_analyze,
                elapsed_ms=0.0,
            )
        raw_arr = np.asarray(recording_or_signal, dtype=float)
        recording = ECGRecording(
            record_id=f"REC-{uuid.uuid4().hex[:8].upper()}",
            sampling_rate=float(fs),
            duration=len(raw_arr) / float(fs) if len(raw_arr) > 0 else 0.0,
            lead_names=[lead_to_analyze],
            number_of_leads=1,
            signals=raw_arr,
            source_format="RAW_SIGNAL",
        )

    # 2. Technical Validation of Recording
    is_valid, val_errors = recording.validate()
    if not is_valid:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return _build_failure_result(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            reason=f"Technical recording validation failed: {'; '.join(val_errors)}",
            lead=lead_to_analyze,
            elapsed_ms=elapsed,
            patient_id=recording.patient_id,
        )

    # 3. Lead Selection & Compatibility Check
    sig_1d = recording.get_lead(lead_to_analyze)
    if sig_1d is None:
        # Check alternative common Lead II names
        for candidate in ["II", "MLII", "Lead_2", "Lead II"]:
            sig_1d = recording.get_lead(candidate)
            if sig_1d is not None:
                lead_to_analyze = candidate
                break

    if sig_1d is None:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ECGAnalysisResult(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            patient_id=recording.patient_id,
            model_id=ACTIVE_MODEL_ID,
            model_version=ACTIVE_MODEL_ID,
            preprocessing_version=PREPROCESSING_VERSION,
            lead_analyzed=lead_to_analyze,
            signal_quality="UNSUPPORTED",
            quality_score=0.0,
            quality_indicators={},
            prediction="UNSUPPORTED_LEAD_CONFIGURATION",
            model_probabilities={},
            heart_rate_bpm=None,
            mean_rr_ms=None,
            detected_beats_count=0,
            detected_r_peaks=[],
            warnings=[
                f"Model {ACTIVE_MODEL_ID} is validated strictly for Lead II / MLII representations. "
                f"The provided recording has leads ({', '.join(recording.lead_names)}) but no Lead II was found."
            ],
            limitations=["Unsupported lead configuration for active model."],
            processing_time_ms=round(elapsed, 2),
            data_hash=recording.data_hash,
        )

    # 4. Mandatory Signal Quality Gatekeeper
    quality_gate = evaluate_signal_quality_gate(
        signal=sig_1d,
        fs=recording.sampling_rate,
        lead_name=lead_to_analyze,
        min_duration_sec=1.5,
    )

    # Hard Safety Rule: If UNUSABLE -> NO AI RESULT
    if quality_gate.category == QualityCategory.UNUSABLE:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ECGAnalysisResult(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            patient_id=recording.patient_id,
            model_id=ACTIVE_MODEL_ID,
            model_version=ACTIVE_MODEL_ID,
            preprocessing_version=PREPROCESSING_VERSION,
            lead_analyzed=lead_to_analyze,
            signal_quality="UNUSABLE",
            quality_score=quality_gate.quality_score,
            quality_indicators=quality_gate.metrics,
            prediction="NO_RESULT_SIGNAL_UNUSABLE",
            model_probabilities={},
            heart_rate_bpm=None,
            mean_rr_ms=None,
            detected_beats_count=0,
            detected_r_peaks=[],
            warnings=quality_gate.rejection_reasons + quality_gate.warnings,
            limitations=["Signal quality is below minimum physiological threshold. Analysis halted for patient safety."],
            processing_time_ms=round(elapsed, 2),
            data_hash=recording.data_hash,
        )

    # 5. Preprocessing & R-Peak Detection
    processed_sig = preprocess_pipeline(sig_1d, fs=recording.sampling_rate)
    peaks, _ = detect_r_peaks(processed_sig, fs=recording.sampling_rate)

    # Compute Heart Rate & RR Dynamics
    if len(peaks) >= 2:
        rr_intervals = calculate_rr_intervals(peaks, fs=recording.sampling_rate)
        hr_bpm = estimate_heart_rate(peaks, fs=recording.sampling_rate)
        mean_rr_ms = round(float(np.mean(rr_intervals)) * 1000.0, 1) if len(rr_intervals) > 0 else None
        hr_val = round(float(hr_bpm), 1)
        rr_ms_list = [round(float(rr * 1000.0), 1) for rr in rr_intervals]
    else:
        hr_val = None
        mean_rr_ms = None
        rr_ms_list = []

    # Hard Safety Rule: If POOR -> Suppress AI Classification
    if quality_gate.category == QualityCategory.POOR or not quality_gate.can_run_ai:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ECGAnalysisResult(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            patient_id=recording.patient_id,
            model_id=ACTIVE_MODEL_ID,
            model_version=ACTIVE_MODEL_ID,
            preprocessing_version=PREPROCESSING_VERSION,
            lead_analyzed=lead_to_analyze,
            signal_quality="POOR",
            quality_score=quality_gate.quality_score,
            quality_indicators=quality_gate.metrics,
            prediction="AI_SUPPRESSED_POOR_SIGNAL",
            model_probabilities={},
            heart_rate_bpm=hr_val,
            mean_rr_ms=mean_rr_ms,
            detected_beats_count=len(peaks),
            detected_r_peaks=peaks.tolist(),
            rr_intervals_ms=rr_ms_list,
            warnings=quality_gate.warnings + [
                "Signal quality is POOR (SNR < 3 dB). AI classification is suppressed to protect patient safety."
            ],
            limitations=["Signal degraded by severe noise/drift. Requires clean re-acquisition."],
            processing_time_ms=round(elapsed, 2),
            data_hash=recording.data_hash,
        )

    # 6. Cardiac Cycle Segmentation
    beats, valid_indices = extract_beats(
        processed_sig,
        peaks,
        fs=recording.sampling_rate,
        pre_window=0.2,
        post_window=0.4,
    )

    if len(beats) == 0:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ECGAnalysisResult(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            patient_id=recording.patient_id,
            model_id=ACTIVE_MODEL_ID,
            model_version=ACTIVE_MODEL_ID,
            preprocessing_version=PREPROCESSING_VERSION,
            lead_analyzed=lead_to_analyze,
            signal_quality=quality_gate.category.value,
            quality_score=quality_gate.quality_score,
            quality_indicators=quality_gate.metrics,
            prediction="NO_BEATS_DETECTED",
            model_probabilities={},
            heart_rate_bpm=None,
            mean_rr_ms=None,
            detected_beats_count=0,
            detected_r_peaks=peaks.tolist(),
            warnings=["No complete cardiac beat windows could be segmented around detected peaks."],
            limitations=["Signal duration or peak positions did not permit beat segmentation."],
            processing_time_ms=round(elapsed, 2),
            data_hash=recording.data_hash,
        )

    # 7. 28-Feature Extraction
    feature_rows = []
    n_beats = len(beats)
    mean_rr = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8

    for i in range(n_beats):
        pre_rr = float(rr_intervals[i - 1]) if i > 0 and len(rr_intervals) > 0 else mean_rr
        post_rr = float(rr_intervals[i]) if i < len(rr_intervals) else mean_rr
        ratio = pre_rr / mean_rr if mean_rr > 0 else 1.0

        feat = extract_all_features(
            beats[i],
            fs=recording.sampling_rate,
            pre_rr=pre_rr,
            post_rr=post_rr,
            local_rr_ratio=ratio,
        )
        feature_rows.append(feat)

    feats_df = pd.DataFrame(feature_rows)

    # 8. Model Loading & Inference
    try:
        clf, scaler, metadata = get_model_artifacts(models_dir)
        feature_names = metadata.get("feature_names", [])

        if feature_names and list(feats_df.columns) != feature_names:
            # Reindex to ensure strictly identical feature column order
            missing = set(feature_names) - set(feats_df.columns)
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            feats_ordered = feats_df[feature_names]
        else:
            feats_ordered = feats_df

        # Scale features
        x_scaled = scaler.transform(feats_ordered.values if hasattr(feats_ordered, "values") else feats_ordered)

        # Beat-level predictions and probabilities
        beat_preds = clf.predict(x_scaled)
        beat_probs = clf.predict_proba(x_scaled)
        classes = list(clf.classes_)

        # Calculate overall window-level probabilities
        mean_probs = np.mean(beat_probs, axis=0)
        prob_dict = {
            cls_name: round(float(prob_val), 4)
            for cls_name, prob_val in zip(classes, mean_probs)
        }

        # Window-level classification logic
        # If any significant PVC cluster is present (>= 15% of beats or >= 2 beats), flag Ventricular Ectopy
        pvc_idx = classes.index("PVC") if "PVC" in classes else -1
        pvc_beats = int(np.sum(beat_preds == "PVC"))
        other_beats = int(np.sum(beat_preds == "Other"))
        norm_beats = int(np.sum(beat_preds == "Normal"))

        if pvc_beats > 0 and (pvc_beats >= 2 or (pvc_beats / len(beat_preds)) >= 0.15):
            window_pred = f"Ventricular Ectopy ({pvc_beats} PVC beats detected)"
        elif other_beats > (len(beat_preds) * 0.3):
            window_pred = "Other Abnormal Rhythm Pattern (Unvalidated Class)"
        elif norm_beats >= (len(beat_preds) * 0.7):
            window_pred = "Normal Sinus Rhythm"
        else:
            # Majority vote fallback
            maj_class = classes[int(np.argmax(mean_probs))]
            window_pred = f"{maj_class} Rhythm"

        limitations = [
            "Model ECG-RF-1.0.0 is validated strictly for Lead II. Precordial morphology is not evaluated.",
            "Class 'Other' has unvalidated sensitivity. Only Normal vs PVC distinction is clinically reliable.",
            "Model probability reflects statistical algorithm likelihood, not clinical diagnostic certainty.",
        ]

        warnings_out = list(quality_gate.warnings)
        if other_beats > 0:
            warnings_out.append(
                f"Detected {other_beats} beats classified as 'Other'. Note that class 'Other' has unvalidated test sensitivity."
            )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        return ECGAnalysisResult(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            patient_id=recording.patient_id,
            model_id=ACTIVE_MODEL_ID,
            model_version=ACTIVE_MODEL_ID,
            preprocessing_version=PREPROCESSING_VERSION,
            lead_analyzed=lead_to_analyze,
            signal_quality=quality_gate.category.value,
            quality_score=quality_gate.quality_score,
            quality_indicators=quality_gate.metrics,
            prediction=window_pred,
            model_probabilities=prob_dict,
            heart_rate_bpm=hr_val,
            mean_rr_ms=mean_rr_ms,
            detected_beats_count=len(beats),
            detected_r_peaks=peaks.tolist(),
            rr_intervals_ms=rr_ms_list,
            beat_predictions=beat_preds.tolist(),
            warnings=warnings_out,
            limitations=limitations,
            processing_time_ms=round(elapsed, 2),
            data_hash=recording.data_hash,
        )

    except Exception as exc:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return _build_failure_result(
            analysis_id=analysis_id,
            record_id=recording.record_id,
            reason=f"Model execution failure: {str(exc)}",
            lead=lead_to_analyze,
            elapsed_ms=elapsed,
            patient_id=recording.patient_id,
        )


def _build_failure_result(
    analysis_id: str,
    record_id: str,
    reason: str,
    lead: str,
    elapsed_ms: float,
    patient_id: Optional[str] = None,
) -> ECGAnalysisResult:
    """Build a fail-safe null result when pipeline execution fails."""
    return ECGAnalysisResult(
        analysis_id=analysis_id,
        record_id=record_id,
        patient_id=patient_id,
        model_id=ACTIVE_MODEL_ID,
        model_version=ACTIVE_MODEL_ID,
        preprocessing_version=PREPROCESSING_VERSION,
        lead_analyzed=lead,
        signal_quality="FAILED",
        quality_score=0.0,
        quality_indicators={},
        prediction="PIPELINE_EXECUTION_FAILURE",
        model_probabilities={},
        heart_rate_bpm=None,
        mean_rr_ms=None,
        detected_beats_count=0,
        detected_r_peaks=[],
        warnings=[reason],
        limitations=["Analysis could not complete due to execution exception. No clinical result generated."],
        processing_time_ms=round(elapsed_ms, 2),
        data_hash="",
    )
