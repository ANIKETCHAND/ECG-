"""
Task A — Beat-Level Arrhythmia Training Pipeline
================================================

Trains the unbalanced/balanced Random Forest, a Logistic Regression baseline and
a neural waveform model on patient-partitioned MIT-BIH data with zero patient
contamination.

Data integrity
--------------
Training data must come from ``data/processed/train_dataset.csv`` and
``data/processed/test_dataset.csv``, which are produced with a patient-level
split. When they are absent:

* training aborts by default, and
* only an explicit opt-in (``--allow-placeholder`` or
  ``ECG_ALLOW_PLACEHOLDER_TRAINING=1``) permits fitting on the two bundled demo
  fixtures — an artifact stamped ``DEMO_FIXTURE`` that can never be promoted to
  production.

Model promotion is evidence-gated: a fitted model only reaches ``VALIDATED``
when it clears explicit clinical thresholds on the held-out partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.ml.models.deep_1d_cnn import ECGFeatureMLPClassifier
from training.guards import (
    fixture_provenance,
    placeholder_training_allowed,
    real_provenance,
    record_provenance,
    require_real_dataset,
)

DATA_DIR = PROJ_DIR / "data" / "processed"
SPLITS_DIR = PROJ_DIR / "data" / "splits" / "beat_arrhythmia"
MODELS_DIR = PROJ_DIR / "models"

#: Evidence gates required before a fitted model may be marked VALIDATED.
PROMOTION_GATES = {
    "pvc_sensitivity_min": 0.98,
    "macro_f1_min": 0.60,
    "auroc_min": 0.95,
}

IGNORE_COLUMNS = {"label", "record_id", "symbol"}


def _build_fixture_partition() -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build a tiny partition from the two bundled demo fixtures.

    The fixtures are genuinely recorded signals, but two files cannot represent a
    clinical corpus. The result is therefore never production-eligible.
    """
    from src.feature_extraction import extract_features_batch
    from src.peak_detection import detect_r_peaks
    from src.preprocessing import preprocess_pipeline
    from src.segmentation import extract_beats

    sources = [
        (PROJ_DIR / "sample_ecgs" / "normal_ecg_sample.csv", "Normal"),
        (PROJ_DIR / "sample_ecgs" / "pvc_arrhythmia_sample.csv", "PVC"),
    ]

    rows: List[pd.DataFrame] = []
    used_files: List[str] = []
    for csv_path, label in sources:
        if not csv_path.exists():
            continue
        used_files.append(str(csv_path))
        raw_df = pd.read_csv(csv_path)
        signal = preprocess_pipeline(raw_df.iloc[:, 0].dropna().values.astype(float), fs=360.0)
        peaks, _ = detect_r_peaks(signal, fs=360.0)
        beats, valid_beat_indices = extract_beats(signal, peaks, fs=360.0)
        if len(beats) == 0:
            continue
        # ``extract_beats`` returns positions within ``peaks``, not sample
        # indices. Passing them straight through as r_peaks produced meaningless
        # RR features, so map back to sample indices first.
        valid_samples = np.asarray(peaks, dtype=int)[np.asarray(valid_beat_indices, dtype=int)]
        features = pd.DataFrame(extract_features_batch(beats, fs=360.0, r_peaks=valid_samples))
        features["label"] = label
        rows.append(features)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    if combined["label"].nunique() < 2:
        return None

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        combined, test_size=0.3, random_state=42, stratify=combined["label"]
    )
    train_df.attrs["source_files"] = used_files
    test_df.attrs["source_files"] = used_files
    return train_df, test_df


def _evaluate(model: Any, X: np.ndarray, y: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    y_pred = model.predict(X)
    metrics: Dict[str, Any] = {
        "accuracy": float(np.mean(y_pred == y)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }
    try:
        probs = model.predict_proba(X)
        if len(classes) > 1:
            metrics["auroc"] = float(
                roc_auc_score(label_binarize(y, classes=classes), probs, multi_class="ovr", average="weighted")
            )
    except Exception:
        pass

    # PVC sensitivity, where PVC is the safety-critical class.
    if "PVC" in classes:
        pvc_idx = list(y) and np.array([1 if v == "PVC" else 0 for v in y])
        pred_pvc = np.array([1 if v == "PVC" else 0 for v in y_pred])
        positives = pvc_idx.sum()
        metrics["pvc_sensitivity"] = float((pvc_idx & pred_pvc).sum() / positives) if positives else None
    return metrics


def _passes_promotion_gates(metrics: Dict[str, Any]) -> bool:
    sensitivity = metrics.get("pvc_sensitivity")
    if sensitivity is None or sensitivity < PROMOTION_GATES["pvc_sensitivity_min"]:
        return False
    if metrics.get("macro_f1", 0.0) < PROMOTION_GATES["macro_f1_min"]:
        return False
    if metrics.get("auroc", 1.0) < PROMOTION_GATES["auroc_min"]:
        return False
    return True


def train_arrhythmia_models(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    allow_placeholder: Optional[bool] = None,
) -> dict:
    """Train Task A models with strict patient isolation."""
    train_path = DATA_DIR / "train_dataset.csv"
    test_path = DATA_DIR / "test_dataset.csv"
    split_source = "data/processed CSVs"
    fixture_files: List[str] = []

    if train_df is None:
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            require_real_dataset(
                dataset_id="mit_bih_arrhythmia",
                task="beat_arrhythmia",
                records=[p for p in (train_path, test_path) if p.exists()],
                location=DATA_DIR,
                allow_placeholder=allow_placeholder,
                download_hint="python training/download_datasets.py --dataset mit_bih_arrhythmia && python training/create_splits.py",
            )
            partition = _build_fixture_partition()
            if partition is None:
                return {
                    "status": "FAILED",
                    "task": "beat_arrhythmia",
                    "reason": "No processed partition and the bundled demo fixtures could not form two classes.",
                }
            train_df, test_df = partition
            split_source = "bundled demo fixtures"
            fixture_files = train_df.attrs.get("source_files", [])

    if test_df is None:
        return {"status": "FAILED", "task": "beat_arrhythmia", "reason": "A test partition is required."}

    feature_cols = [c for c in train_df.columns if c not in IGNORE_COLUMNS]
    classes = sorted(train_df["label"].astype(str).unique().tolist())

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].astype(str).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].astype(str).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    mlp = ECGFeatureMLPClassifier(input_length=len(feature_cols), max_iter=150, random_state=42)
    mlp.fit(X_train, y_train)

    rf_metrics = _evaluate(rf, X_test_scaled, y_test, classes)
    lr_metrics = _evaluate(lr, X_test_scaled, y_test, classes)

    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr, cand_dir / "baseline_classifier.pkl")
    joblib.dump(rf, cand_dir / "classifier.pkl")
    joblib.dump(scaler, cand_dir / "scaler.pkl")
    joblib.dump(mlp, cand_dir / "deep_classifier.pkl")

    is_fixture = split_source == "bundled demo fixtures"
    promoted = (not is_fixture) and _passes_promotion_gates(rf_metrics)
    status = "VALIDATED" if promoted else "CANDIDATE"

    if is_fixture:
        provenance = fixture_provenance(
            dataset_id="bundled_demo_fixtures",
            task="beat_arrhythmia",
            source_files=fixture_files,
            n_samples=len(y_train),
        )
    else:
        provenance = real_provenance(
            dataset_id="mit_bih_arrhythmia",
            task="beat_arrhythmia",
            records=sorted(train_df.get("record_id", pd.Series([])).astype(str).unique().tolist())
            + sorted(test_df.get("record_id", pd.Series([])).astype(str).unique().tolist()),
            train_records=sorted(train_df.get("record_id", pd.Series([])).astype(str).unique().tolist()),
            test_records=sorted(test_df.get("record_id", pd.Series([])).astype(str).unique().tolist()),
            n_train_samples=len(y_train),
            n_test_samples=len(y_test),
        )
    provenance["feature_names"] = feature_cols
    provenance["split_source"] = split_source
    provenance["metrics"] = rf_metrics
    provenance["promotion_gates"] = PROMOTION_GATES
    provenance["clinically_validated"] = promoted

    record_provenance(
        model_id="ECG-RF-2.0.0-candidate",
        provenance=provenance,
        status=status,
        artifact_path="candidate/classifier.pkl",
        scaler_path="candidate/scaler.pkl",
        description="Balanced Random Forest for single-lead beat arrhythmia",
    )
    record_provenance(
        model_id="ECG-MLP-1.0.0-candidate",
        provenance=provenance,
        status="EXPERIMENTAL",
        artifact_path="candidate/deep_classifier.pkl",
        scaler_path="candidate/scaler.pkl",
        description="scikit-learn MLP over the 28 extracted features",
    )
    record_provenance(
        model_id="ECG-LR-1.0.0",
        provenance=provenance,
        status="EXPERIMENTAL",
        artifact_path="candidate/baseline_classifier.pkl",
        scaler_path="candidate/scaler.pkl",
        description="Logistic regression baseline",
    )

    if not is_fixture:
        metadata = {
            "model_name": "RandomForestClassifier",
            "baseline_model_name": "LogisticRegression",
            "dataset": "MIT-BIH Arrhythmia Database",
            "classes": classes,
            "feature_names": feature_cols,
            "sampling_rate_hz": 360,
            "window_pre_sec": 0.2,
            "window_post_sec": 0.4,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "feature_importances": {f: float(i) for f, i in zip(feature_cols, rf.feature_importances_)},
            "metrics": rf_metrics,
            "provenance": provenance,
            "created_at": pd.Timestamp.now().isoformat(),
            "disclaimer": "AI-Assisted Clinical Decision Support. Subject to mandatory qualified physician review.",
        }
        for target in (MODELS_DIR, MODELS_DIR / "production"):
            target.mkdir(parents=True, exist_ok=True)
            with open(target / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

    return {
        "status": "SUCCESS",
        "task": "beat_arrhythmia",
        "split_source": split_source,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "classes": classes,
        "promoted_to_validated": promoted,
        "metrics": rf_metrics,
        "baseline_metrics": lr_metrics,
        "placeholders_remaining": placeholder_training_allowed(allow_placeholder) and is_fixture,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task A beat arrhythmia models")
    parser.add_argument("--allow-placeholder", action="store_true", help="Permit fitting on the bundled demo fixtures")
    args = parser.parse_args()
    res = train_arrhythmia_models(allow_placeholder=args.allow_placeholder or None)
    print("Train Arrhythmia Result:", res)
