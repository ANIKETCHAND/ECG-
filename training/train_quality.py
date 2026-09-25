"""
Task E — Signal Quality Gatekeeper Model Training Pipeline
==========================================================

Trains a statistical classifier over *measured* technical quality descriptors:

    SNR (dB), clipping ratio, baseline-wander ratio, powerline ratio,
    motion-spike ratio

Output categories: GOOD, ACCEPTABLE, POOR, UNUSABLE.

Data integrity
--------------
This task previously invented 600 synthetic rows with hand-tuned means and
standard deviations, then reported the fit as a success. That produced a
classifier whose accuracy measured nothing about real ECG acquisition.

Now the task requires an actual labelled corpus at
``data/processed/quality_dataset.csv`` with columns:

    snr_db, clipping_ratio, baseline_wander_ratio, powerline_ratio,
    motion_spike_ratio, label, record_id

The production quality decision remains rule-based and deterministic in
``src/quality/quality_gate.py``; a learned model is an optional refinement only
where a labelled corpus exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from training.guards import record_provenance, real_provenance, synthetic_provenance
from training.real_features import QUALITY_FEATURE_NAMES

MODELS_DIR = PROJ_DIR / "models"
MODEL_ID = "ECG-QUALITY-1.0.0"
CLASSES = ["GOOD", "ACCEPTABLE", "POOR", "UNUSABLE"]


def labelled_corpus_path() -> Path:
    return PROJ_DIR / "data" / "processed" / "quality_dataset.csv"


def _train_from_labelled_corpus(csv_path: Path) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = set(QUALITY_FEATURE_NAMES) | {"label", "record_id"}
    missing = required - set(df.columns)
    if missing:
        return {
            "status": "INVALID_LABELLED_CORPUS",
            "task": "quality_gate",
            "reason": f"Corpus is missing required columns: {sorted(missing)}",
        }

    records = sorted(df["record_id"].astype(str).unique())
    if len(records) < 4:
        return {
            "status": "INSUFFICIENT_LABELLED_RECORDS",
            "task": "quality_gate",
            "reason": "A patient/record-level split needs at least four distinct records.",
            "records_found": len(records),
        }

    rng = np.random.default_rng(42)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * 0.25)))
    test_records = set(shuffled[:n_test])

    train_df = df[~df["record_id"].astype(str).isin(test_records)]
    test_df = df[df["record_id"].astype(str).isin(test_records)]

    X_train = train_df[QUALITY_FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[QUALITY_FEATURE_NAMES].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy()

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)

    y_pred = model.predict(scaler.transform(X_test))
    metrics = {
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(y_pred == y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cand_dir / "quality_classifier.pkl")
    joblib.dump(scaler, cand_dir / "quality_scaler.pkl")

    provenance = real_provenance(
        dataset_id="quality_labelled_corpus",
        task="quality_gate",
        records=records,
        train_records=sorted(set(train_df["record_id"].astype(str))),
        test_records=sorted(test_records),
        n_train_samples=len(y_train),
        n_test_samples=len(y_test),
    )
    provenance["feature_names"] = QUALITY_FEATURE_NAMES
    provenance["label_source"] = f"human/expert labels in {csv_path.name}"
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="VALIDATED",
        artifact_path="candidate/quality_classifier.pkl",
        scaler_path="candidate/quality_scaler.pkl",
        description="Learned signal-quality classifier trained on a labelled acquisition corpus",
    )

    return {"status": "SUCCESS", "task": "quality_gate", "metrics": metrics, "provenance": provenance}


def _build_placeholder() -> Dict[str, Any]:
    """Build a clearly-labelled non-clinical placeholder artifact."""
    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_per_class = 150
    X = np.vstack(
        [
            np.column_stack(
                [
                    rng.normal(24.0, 3.0, n_per_class).clip(18.0, 40.0),
                    rng.uniform(0.0, 0.001, n_per_class),
                    rng.uniform(0.01, 0.08, n_per_class),
                    rng.uniform(0.01, 0.05, n_per_class),
                    rng.uniform(0.0, 0.02, n_per_class),
                ]
            ),
            np.column_stack(
                [
                    rng.normal(14.0, 2.0, n_per_class).clip(10.0, 18.0),
                    rng.uniform(0.001, 0.01, n_per_class),
                    rng.uniform(0.08, 0.18, n_per_class),
                    rng.uniform(0.05, 0.15, n_per_class),
                    rng.uniform(0.02, 0.06, n_per_class),
                ]
            ),
            np.column_stack(
                [
                    rng.normal(7.0, 2.0, n_per_class).clip(3.0, 10.0),
                    rng.uniform(0.01, 0.04, n_per_class),
                    rng.uniform(0.18, 0.35, n_per_class),
                    rng.uniform(0.15, 0.30, n_per_class),
                    rng.uniform(0.06, 0.15, n_per_class),
                ]
            ),
            np.column_stack(
                [
                    rng.normal(1.0, 1.5, n_per_class).clip(-5.0, 3.0),
                    rng.uniform(0.05, 0.40, n_per_class),
                    rng.uniform(0.35, 0.80, n_per_class),
                    rng.uniform(0.30, 0.70, n_per_class),
                    rng.uniform(0.15, 0.40, n_per_class),
                ]
            ),
        ]
    )
    y = np.array(["GOOD"] * n_per_class + ["ACCEPTABLE"] * n_per_class + ["POOR"] * n_per_class + ["UNUSABLE"] * n_per_class)

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(scaler.fit_transform(X), y)

    joblib.dump(model, cand_dir / "quality_classifier.pkl")
    joblib.dump(scaler, cand_dir / "quality_scaler.pkl")

    provenance = synthetic_provenance(
        dataset_id="quality_labelled_corpus",
        task="quality_gate",
        reason="No labelled quality corpus present; placeholder generated under explicit opt-in.",
        n_samples=len(X),
    )
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="SYNTHETIC_PLACEHOLDER",
        artifact_path="candidate/quality_classifier.pkl",
        scaler_path="candidate/quality_scaler.pkl",
        description="NON-CLINICAL placeholder for quality-gate plumbing only",
    )

    return {
        "status": "SYNTHETIC_PLACEHOLDER_SAVED",
        "task": "quality_gate",
        "model_path": str(cand_dir / "quality_classifier.pkl"),
        "classes": CLASSES,
        "n_samples": len(X),
        "clinical_claim": "NONE",
        "provenance": provenance,
    }


def train_quality_model(allow_placeholder: Optional[bool] = None) -> dict:
    """Train the Task E quality-gate model."""
    corpus = labelled_corpus_path()
    if corpus.exists():
        return _train_from_labelled_corpus(corpus)

    from training.guards import placeholder_training_allowed

    if placeholder_training_allowed(allow_placeholder):
        return _build_placeholder()

    return {
        "status": "NO_LABELLED_CORPUS",
        "task": "quality_gate",
        "expected_corpus": str(corpus),
        "reason": (
            "No labelled signal-quality corpus is present. The production quality decision is "
            "rule-based (src/quality/quality_gate.py) and requires no learned model. To train one, "
            "supply a corpus with columns "
            f"{QUALITY_FEATURE_NAMES + ['label', 'record_id']}, or pass --allow-placeholder to build "
            "a clearly-marked non-clinical placeholder."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task E quality-gate model")
    parser.add_argument("--allow-placeholder", action="store_true", help="Build a non-clinical placeholder artifact")
    args = parser.parse_args()
    res = train_quality_model(allow_placeholder=args.allow_placeholder or None)
    print("Train Quality Model Result:", res)
