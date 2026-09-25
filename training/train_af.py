"""
Task B — Atrial Fibrillation Rhythm Detection Training Pipeline
==============================================================

Trains a specialised rhythm-level classifier for Atrial Fibrillation / Flutter
from RR-interval irregularity, computed from real annotated recordings.

Data integrity
--------------
This script NEVER fabricates training samples. When the MIT-BIH Atrial
Fibrillation Database is absent, training aborts with
:class:`training.guards.PlaceholderTrainingNotAllowed` unless placeholder
generation was explicitly authorised (``ECG_ALLOW_PLACEHOLDER_TRAINING=1`` or
``--allow-placeholder``). Placeholder artifacts are stamped
``SYNTHETIC_PLACEHOLDER`` in the model registry and can never be served as the
active production model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.datasets.registry import DATASET_REGISTRY
from training.guards import (
    MissingDatasetError,
    placeholder_training_allowed,
    real_provenance,
    record_provenance,
    require_real_dataset,
    synthetic_provenance,
)
from training.real_features import AF_FEATURE_NAMES, af_label_from_rhythm_aux, af_rhythm_features, template_for

MODELS_DIR = PROJ_DIR / "models"
DATASET_ID = "mit_bih_afdb"
MODEL_ID = "ECG-AF-1.0.0-candidate"


def extract_rhythm_af_features(rr_intervals: np.ndarray, signal: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Backwards-compatible wrapper returning zeroed features for short input.

    Kept for callers that expect a dict rather than ``None``. New code should
    prefer :func:`training.real_features.af_rhythm_features`, which signals
    "not enough data" explicitly instead of returning fabricated zeros.
    """
    features = af_rhythm_features(rr_intervals)
    if features is None:
        return {name: 0.0 for name in AF_FEATURE_NAMES}
    return features


def _windowed_rhythm_samples(
    rhythm_changes: List[Tuple[int, str]],
    rr_samples: np.ndarray,
    fs: float,
    *,
    window_s: float = 30.0,
) -> List[Tuple[np.ndarray, str]]:
    """Split a record into labelled rhythm windows from real annotations.

    Args:
        rhythm_changes: ``(sample_index, label)`` pairs of rhythm-change
            annotations, sorted by sample index.
        rr_samples: RR intervals in *samples* for the whole record.
        fs: Sampling rate in Hz.
        window_s: Window length used to slice the RR series.

    Returns:
        List of ``(rr_intervals_seconds, label)`` per window, skipping windows
        whose label could not be resolved.
    """
    if not rhythm_changes or rr_samples.size < 5:
        return []

    beat_samples = np.concatenate([[0], np.cumsum(rr_samples)])
    window_samples = window_s * fs
    n_windows = max(1, int(beat_samples[-1] // window_samples))
    samples: List[Tuple[np.ndarray, str]] = []

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples

        label: Optional[str] = None
        for change_sample, change_label in rhythm_changes:
            if change_sample <= start:
                label = change_label
            else:
                break
        if label is None:
            continue

        mask = (beat_samples[:-1] >= start) & (beat_samples[:-1] < end)
        rr_window = rr_samples[mask]
        samples.append((rr_window / fs, label))

    return samples


def _load_afdb_record(record_path: Path) -> Optional[Tuple[List[Tuple[np.ndarray, str]], float]]:
    """Read one AFDB record and return labelled rhythm windows plus its fs."""
    try:
        import wfdb
    except ImportError:  # pragma: no cover - wfdb is a declared dependency
        return None

    try:
        record = wfdb.rdrecord(str(record_path.with_suffix("")))
        annotation = wfdb.rdann(str(record_path.with_suffix("")), "atr")
    except Exception:
        return None

    fs = float(record.fs)
    beat_samples = np.asarray(annotation.sample, dtype=float)
    if beat_samples.size < 6:
        return None
    rr_samples = np.diff(beat_samples)

    rhythm_changes: List[Tuple[int, str]] = []
    if annotation.aux_note:
        for sample, aux in zip(annotation.sample, annotation.aux_note):
            mapped = af_label_from_rhythm_aux(aux)
            if mapped is not None:
                rhythm_changes.append((float(sample), mapped))
    rhythm_changes.sort(key=lambda item: item[0])

    return _windowed_rhythm_samples(rhythm_changes, rr_samples, fs), fs


def _train_from_records(record_paths: List[Path]) -> Dict[str, Any]:
    """Train the AF model on real AFDB records using a patient-level split."""
    records: List[Dict[str, Any]] = []
    for path in record_paths:
        loaded = _load_afdb_record(path)
        if loaded is None:
            continue
        windows, _fs = loaded
        for rr_seconds, label in windows:
            features = af_rhythm_features(rr_seconds)
            if features is None:
                continue
            records.append({"record": path.stem, "label": label, "features": features})

    if len(records) < 20:
        return {
            "status": "INSUFFICIENT_DERIVED_SAMPLES",
            "task": "af_detection",
            "reason": (
                "AFDB records were found but yielded too few labelled rhythm windows. "
                "No model was trained and no artifact was written."
            ),
            "derived_samples": len(records),
        }

    # Patient-level split: never place windows from one record in both partitions.
    unique_records = sorted({r["record"] for r in records})
    rng = np.random.default_rng(42)
    shuffled = list(unique_records)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * 0.25)))
    test_records = set(shuffled[:n_test])
    train_records = [r for r in unique_records if r not in test_records]

    train_rows = [r for r in records if r["record"] not in test_records]
    test_rows = [r for r in records if r["record"] in test_records]

    if len(train_rows) < 10 or len(test_rows) < 5:
        return {
            "status": "INSUFFICIENT_DERIVED_SAMPLES",
            "task": "af_detection",
            "reason": "Patient-level split produced an unusable partition size.",
            "derived_samples": len(records),
        }

    X_train, names = np.array(
        [template_for(AF_FEATURE_NAMES, r["features"])[1] for r in train_rows], dtype=float
    ), AF_FEATURE_NAMES
    y_train = np.array([r["label"] for r in train_rows])
    X_test = np.array([template_for(AF_FEATURE_NAMES, r["features"])[1] for r in test_rows], dtype=float)
    y_test = np.array([r["label"] for r in test_rows])

    if len(set(y_train)) < 2:
        return {
            "status": "SINGLE_CLASS_TRAINING_DATA",
            "task": "af_detection",
            "reason": "Derived training windows contained only one rhythm class.",
        }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(y_pred == y_test)),
        "n_train_windows": int(len(y_train)),
        "n_test_windows": int(len(y_test)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    try:
        probs = model.predict_proba(X_test_scaled)
        if len(set(y_test)) > 1 and probs.shape[1] > 1:
            metrics["auroc"] = float(roc_auc_score((y_test == "AFib").astype(int), probs[:, list(model.classes_).index("AFib")]))
    except Exception:
        pass

    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cand_dir / "af_classifier.pkl")
    joblib.dump(scaler, cand_dir / "af_scaler.pkl")

    provenance = real_provenance(
        dataset_id=DATASET_ID,
        task="af_detection",
        records=record_paths,
        train_records=train_records,
        test_records=sorted(test_records),
        n_train_samples=len(y_train),
        n_test_samples=len(y_test),
    )
    provenance["feature_names"] = names
    provenance["label_source"] = "AFDB rhythm-change annotations (WFDB aux_note)"
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="CANDIDATE",
        artifact_path="candidate/af_classifier.pkl",
        scaler_path="candidate/af_scaler.pkl",
        description="RR irregularity & entropy Random Forest for Atrial Fibrillation (real AFDB data)",
    )

    return {"status": "SUCCESS", "task": "af_detection", "metrics": metrics, "provenance": provenance}


def _build_placeholder() -> Dict[str, Any]:
    """Build a clearly-labelled non-clinical placeholder artifact."""
    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_samples = 400
    half = n_samples // 2
    X = np.vstack(
        [
            np.column_stack(
                [
                    rng.normal(0.06, 0.02, half).clip(0.01, 0.15),
                    rng.normal(0.04, 0.015, half).clip(0.01, 0.10),
                    rng.normal(1.2, 0.2, half).clip(0.5, 1.8),
                    rng.normal(5.0, 2.0, half).clip(0.0, 20.0),
                    rng.normal(0.8, 0.05, half).clip(0.5, 1.2),
                ]
            ),
            np.column_stack(
                [
                    rng.normal(0.24, 0.06, half).clip(0.12, 0.50),
                    rng.normal(0.16, 0.05, half).clip(0.08, 0.35),
                    rng.normal(2.6, 0.3, half).clip(1.9, 3.5),
                    rng.normal(45.0, 10.0, half).clip(10.0, 80.0),
                    rng.normal(0.7, 0.12, half).clip(0.4, 1.1),
                ]
            ),
        ]
    )
    y = np.array(["Non-AFib"] * half + ["AFib"] * half)

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(scaler.fit_transform(X), y)

    joblib.dump(model, cand_dir / "af_classifier.pkl")
    joblib.dump(scaler, cand_dir / "af_scaler.pkl")

    provenance = synthetic_provenance(
        dataset_id=DATASET_ID,
        task="af_detection",
        reason="Dataset absent; placeholder generated under explicit opt-in.",
        n_samples=n_samples,
    )
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="SYNTHETIC_PLACEHOLDER",
        artifact_path="candidate/af_classifier.pkl",
        scaler_path="candidate/af_scaler.pkl",
        description="NON-CLINICAL placeholder for AF rhythm plumbing only",
    )

    return {
        "status": "SYNTHETIC_PLACEHOLDER_SAVED",
        "task": "af_detection",
        "model_path": str(cand_dir / "af_classifier.pkl"),
        "classes": ["AFib", "Non-AFib"],
        "clinical_claim": "NONE",
        "provenance": provenance,
    }


def train_af_models(allow_placeholder: Optional[bool] = None) -> dict:
    """Train the Task B Atrial Fibrillation rhythm classifier."""
    af_path = DATASET_REGISTRY.get_local_path(DATASET_ID) / "raw"
    record_paths = sorted(af_path.glob("*.hea"))

    use_placeholder = require_real_dataset(
        dataset_id=DATASET_ID,
        task="af_detection",
        records=record_paths,
        location=af_path,
        allow_placeholder=allow_placeholder,
    )

    if use_placeholder:
        return _build_placeholder()

    return _train_from_records(record_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task B AF detection model")
    parser.add_argument("--allow-placeholder", action="store_true", help="Build a non-clinical placeholder artifact")
    args = parser.parse_args()
    try:
        res = train_af_models(allow_placeholder=args.allow_placeholder or None)
    except MissingDatasetError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    print("Train AF Result:", res)
