"""
ECG Prediction Pipeline
=======================

Phase 15: Reusable inference pipeline connecting:
Raw ECG -> Preprocessing -> Signal Quality -> R-Peak Detection
        -> Segmentation -> Feature Extraction -> Saved Scaler & Model -> Prediction

Research/educational use only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_all_features
from peak_detection import detect_r_peaks
from preprocessing import preprocess_pipeline
from segmentation import calculate_rr_intervals, estimate_heart_rate, extract_beats
from signal_quality import assess_quality

MODELS_DIR = Path(__file__).parent.parent / "models"

# In-memory model cache to prevent reloading from disk on each inference
_MODEL_CACHE: Dict[str, Any] = {}


def get_trained_artifacts(models_dir: Optional[Path | str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load and cache trained model, scaler, and metadata.

    Returns:
        Tuple of (classifier, scaler, metadata)
    """
    global _MODEL_CACHE
    mdir = Path(models_dir) if models_dir else MODELS_DIR

    if "classifier" not in _MODEL_CACHE or _MODEL_CACHE.get("path") != str(mdir):
        clf_path = mdir / "classifier.pkl"
        scaler_path = mdir / "scaler.pkl"
        meta_path = mdir / "metadata.json"

        if not clf_path.exists() or not scaler_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {mdir}. Please run training/train_model.py first."
            )

        _MODEL_CACHE["classifier"] = joblib.load(clf_path)
        _MODEL_CACHE["scaler"] = joblib.load(scaler_path)
        with open(meta_path, "r") as f:
            _MODEL_CACHE["metadata"] = json.load(f)
        _MODEL_CACHE["path"] = str(mdir)

    return _MODEL_CACHE["classifier"], _MODEL_CACHE["scaler"], _MODEL_CACHE["metadata"]


def predict_ecg(
    signal: np.ndarray,
    fs: float = 360.0,
    models_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Complete end-to-end ECG analysis and abnormality prediction.

    Args:
        signal: 1D numpy array representing the ECG voltage series
        fs: Sampling rate in Hz (default: 360)
        models_dir: Path to directory containing saved model artifacts

    Returns:
        Structured dictionary containing predictions, probabilities, quality, peaks, and features.
    """
    # 1. Input Validation
    if signal is None or len(signal) == 0:
        raise ValueError("Input signal is empty or None")

    signal = np.asarray(signal, dtype=np.float64).flatten()

    if len(signal) < int(fs * 0.5):
        raise ValueError(f"Signal too short ({len(signal)} samples). Minimum 0.5s required.")

    # Handle NaNs in input
    if np.any(np.isnan(signal)):
        signal = pd.Series(signal).interpolate(method="linear").bfill().ffill().values

    # 2. Signal Quality Assessment
    sq_raw = assess_quality(signal, fs)
    quality_score = float(sq_raw.get("quality_score", 1.0))
    if quality_score >= 0.80:
        quality_label = "GOOD"
    elif quality_score >= 0.50:
        quality_label = "ACCEPTABLE"
    else:
        quality_label = "POOR"

    # 3. Preprocessing
    processed_signal = preprocess_pipeline(signal, fs)

    # 4. R-Peak Detection
    r_peaks, peak_info = detect_r_peaks(processed_signal, fs, prominence=0.4)

    if len(r_peaks) == 0:
        return {
            "predicted_class": "Uncertain (No R-peaks detected)",
            "probabilities": {"Normal": 0.0, "PVC": 0.0, "Other": 0.0},
            "signal_quality": quality_label,
            "quality_score": quality_score,
            "quality_indicators": sq_raw,
            "detected_peaks": [],
            "heart_rate_bpm": 0.0,
            "beat_count": 0,
            "raw_signal": signal,
            "processed_signal": processed_signal,
            "beat_predictions": [],
            "beat_probabilities": [],
            "features_df": pd.DataFrame(),
        }

    # 5. Heartbeat Segmentation
    beats, valid_indices = extract_beats(processed_signal, r_peaks, fs, pre_window=0.2, post_window=0.4)
    valid_r_peaks = r_peaks[valid_indices]

    # Calculate RR rhythm metrics
    rr_intervals = calculate_rr_intervals(valid_r_peaks, fs)
    mean_rr = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8
    hr_bpm = estimate_heart_rate(valid_r_peaks, fs)

    # 6. Feature Extraction
    clf, scaler, meta = get_trained_artifacts(models_dir)
    feature_cols = meta["feature_names"]
    classes = meta["classes"]

    feature_rows = []
    n_beats = len(beats)

    for i in range(n_beats):
        pre_rr = float(rr_intervals[i - 1]) if i > 0 and len(rr_intervals) > 0 else mean_rr
        post_rr = float(rr_intervals[i]) if i < len(rr_intervals) else mean_rr
        ratio = pre_rr / mean_rr if mean_rr > 0 else 1.0

        feat = extract_all_features(
            beats[i],
            fs,
            pre_rr=pre_rr,
            post_rr=post_rr,
            local_rr_ratio=ratio,
        )
        feature_rows.append(feat)

    features_df = pd.DataFrame(feature_rows)

    # Ensure all model feature columns exist
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].values

    # 7. Model Inference
    X_scaled = scaler.transform(X)
    beat_preds = clf.predict(X_scaled)
    beat_probs = clf.predict_proba(X_scaled)

    # Map probabilities to class names
    prob_list = []
    for p in beat_probs:
        prob_dict = {classes[idx]: float(p[idx]) for idx in range(len(classes))}
        prob_list.append(prob_dict)

    # Aggregate Overall Recording Prediction
    class_counts = pd.Series(beat_preds).value_counts().to_dict()
    mean_probs = {
        cls: float(np.mean([p[cls] for p in prob_list])) if prob_list else 0.0
        for cls in classes
    }

    # Clinical decision heuristic: if significant PVC burden (>3% of beats or >=2 PVCs), flag abnormality
    pvc_count = class_counts.get("PVC", 0)
    other_count = class_counts.get("Other", 0)
    pvc_pct = (pvc_count / n_beats) * 100 if n_beats > 0 else 0.0

    if pvc_count >= 2 or pvc_pct > 3.0:
        overall_pred = "PVC (Premature Ventricular Contraction detected)"
    elif other_count > 2:
        overall_pred = "Other (Ectopic/Abnormal rhythm pattern detected)"
    else:
        overall_pred = "Normal (Normal Sinus Rhythm)"

    return {
        "predicted_class": overall_pred,
        "probabilities": mean_probs,
        "class_counts": class_counts,
        "signal_quality": quality_label,
        "quality_score": quality_score,
        "quality_indicators": sq_raw,
        "detected_peaks": valid_r_peaks.tolist(),
        "heart_rate_bpm": float(hr_bpm),
        "mean_rr_sec": float(mean_rr),
        "beat_count": int(n_beats),
        "beats": beats,
        "beat_predictions": beat_preds.tolist(),
        "beat_probabilities": prob_list,
        "features_df": features_df,
        "raw_signal": signal,
        "processed_signal": processed_signal,
    }


