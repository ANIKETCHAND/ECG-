"""
Task G — Raw-Waveform 1-D CNN Training Pipeline
===============================================

Trains :class:`src.ml.models.deep_1d_cnn.ECGConv1DClassifier` — a genuine
convolutional network over raw single-lead beat windows — and registers it as
``ECG-CNN-1.0.0-candidate``.

This complements, rather than replaces, the feature-based models:

* ``ECG-RF-1.0.0``  — production Random Forest over 28 extracted features.
* ``ECG-MLP-1.0.0`` — feature-based MLP (no convolution).
* ``ECG-CNN-1.0.0`` — this model: convolutions over the raw waveform.

Data integrity
--------------
Two hard requirements, both enforced rather than worked around:

1. PyTorch must be installed. If it is not, the script reports
   ``TORCH_REQUIRED`` and writes no artifact. It never silently substitutes a
   different model family and labels it a CNN.
2. MIT-BIH records must be present locally. Missing data aborts training.

Splitting is patient-level: no record contributes beats to more than one
partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.data_loader import get_available_records, load_annotations, load_record
from src.label_mapping import is_beat_annotation, map_symbol_to_class
from src.ml.models.deep_1d_cnn import ECGConv1DClassifier, torch_available
from src.ml.models.registry import GLOBAL_MODEL_REGISTRY
from src.segmentation import extract_beats
from src.preprocessing import preprocess_pipeline
from training.guards import real_provenance, record_provenance, require_real_dataset
from training.splitting.patient_splitter import split_records_by_patient

MODELS_DIR = PROJ_DIR / "models"
DATA_DIR = PROJ_DIR / "data" / "raw"
MODEL_ID = "ECG-CNN-1.0.0-candidate"
FS = 360.0
WINDOW_SAMPLES = 216


def _beats_for_record(record_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (beat_windows, class_labels) for one MIT-BIH record."""
    try:
        signal_df = load_record(str(DATA_DIR), record_id)
        annotations = load_annotations(str(DATA_DIR), record_id)
    except Exception:
        return None, None

    if signal_df is None or annotations is None:
        return None, None

    raw = np.asarray(signal_df.iloc[:, 0], dtype=float) if hasattr(signal_df, "iloc") else np.asarray(signal_df, dtype=float)
    filtered = preprocess_pipeline(raw, fs=FS)

    samples = np.asarray(annotations.get("sample", []), dtype=int)
    symbols = list(annotations.get("symbol", []))

    beat_samples: List[int] = []
    labels: List[str] = []
    for sample, symbol in zip(samples, symbols):
        if not is_beat_annotation(str(symbol)):
            continue
        beat_samples.append(int(sample))
        labels.append(map_symbol_to_class(str(symbol), mode="3class"))

    if len(beat_samples) < 5:
        return None, None

    beats, valid_peaks = extract_beats(filtered, np.asarray(beat_samples), fs=FS)
    if len(beats) == 0:
        return None, None

    # ``extract_beats`` may drop peaks near the record edges; re-align labels.
    peak_to_label = dict(zip(beat_samples, labels))
    aligned = [peak_to_label.get(int(peak)) for peak in np.asarray(valid_peaks)]
    keep = [i for i, label in enumerate(aligned) if label is not None]
    if not keep:
        return None, None

    windows = np.asarray(beats, dtype=float)[keep]
    aligned_labels = np.asarray([aligned[i] for i in keep])

    # Standardise the window length expected by the CNN.
    if windows.shape[1] != WINDOW_SAMPLES:
        windows = np.asarray([np.interp(np.linspace(0, 1, WINDOW_SAMPLES), np.linspace(0, 1, w.shape[0]), w) for w in windows])

    return windows, aligned_labels


def train_waveform_cnn(allow_placeholder: Optional[bool] = None) -> Dict[str, Any]:
    """Train the raw-waveform 1-D CNN on patient-partitioned MIT-BIH data."""
    if not torch_available():
        return {
            "status": "TORCH_REQUIRED",
            "task": "waveform_cnn",
            "reason": (
                "ECGConv1DClassifier requires PyTorch, which is not installed. "
                "No artifact was written and no substitute model was substituted."
            ),
            "remediation": "pip install torch   then re-run this script.",
        }

    records = sorted(get_available_records(str(DATA_DIR))) if DATA_DIR.exists() else []
    require_real_dataset(
        dataset_id="mit_bih_arrhythmia",
        task="waveform_cnn",
        records=records,
        location=DATA_DIR,
        allow_placeholder=allow_placeholder,
        download_hint="python training/download_datasets.py --dataset mit_bih_arrhythmia",
    )

    train_records, _val_records, test_records = split_records_by_patient(
        list(records), test_ratio=0.25, val_ratio=0.0, random_seed=42
    )

    def collect(record_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        used: List[str] = []
        for record_id in record_ids:
            windows, labels = _beats_for_record(record_id)
            if windows is None or labels is None:
                continue
            xs.append(windows)
            ys.append(labels)
            used.append(record_id)
        if not xs:
            return np.empty((0, WINDOW_SAMPLES)), np.empty((0,)), []
        return np.vstack(xs), np.concatenate(ys), used

    X_train, y_train, train_used = collect(train_records)
    X_test, y_test, test_used = collect(test_records)

    if len(X_train) < 100 or len(X_test) < 20:
        return {
            "status": "INSUFFICIENT_DERIVED_SAMPLES",
            "task": "waveform_cnn",
            "reason": "Too few beats were derived to train a convolutional model.",
            "train_beats": int(len(X_train)),
            "test_beats": int(len(X_test)),
        }

    classes = sorted(set(y_train.tolist()))
    if len(classes) < 2:
        return {
            "status": "SINGLE_CLASS_TRAINING_DATA",
            "task": "waveform_cnn",
            "reason": "Training partition contained only one beat class.",
            "classes": classes,
        }

    model = ECGConv1DClassifier(
        input_length=WINDOW_SAMPLES,
        num_classes=len(classes),
        class_names=classes,
        epochs=25,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    from sklearn.metrics import f1_score

    metrics = {
        "accuracy": float(np.mean(predictions == y_test)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "n_train_beats": int(len(y_train)),
        "n_test_beats": int(len(y_test)),
        "classes": classes,
    }

    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    artifact = cand_dir / "waveform_cnn.pkl"
    joblib.dump(model, artifact)

    provenance = real_provenance(
        dataset_id="mit_bih_arrhythmia",
        task="waveform_cnn",
        records=records,
        train_records=train_used,
        test_records=test_used,
        n_train_samples=int(len(y_train)),
        n_test_samples=int(len(y_test)),
    )
    provenance["architecture"] = "CONV1D (raw waveform, 3 conv blocks + adaptive pooling)"
    provenance["input_representation"] = f"raw single-lead beat window, {WINDOW_SAMPLES} samples"
    provenance["metrics"] = metrics
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="CANDIDATE",
        artifact_path="candidate/waveform_cnn.pkl",
        description="1-D CNN over raw single-lead beat waveforms (real MIT-BIH data)",
    )

    return {"status": "SUCCESS", "task": "waveform_cnn", "metrics": metrics, "provenance": provenance}


def register_if_present() -> Optional[Dict[str, Any]]:
    """Expose the registry entry for the CNN if it has been trained."""
    try:
        return GLOBAL_MODEL_REGISTRY.get_model_entry(MODEL_ID)
    except KeyError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the raw-waveform 1-D CNN")
    parser.add_argument("--allow-placeholder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = train_waveform_cnn(allow_placeholder=args.allow_placeholder or None)
    print("Train Waveform CNN Result:", json.dumps(result, indent=2, default=str))
