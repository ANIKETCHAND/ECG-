"""
Task D — ST-Segment & Ischemia Analysis Training Pipeline
=========================================================

Trains models for detecting ischaemic ST-elevation and ST-depression from
J-point displacement, ST60 amplitude and ST slope.

Data integrity and label validity
---------------------------------
Two distinct failure modes are refused here:

1. **Fabricated data.** When the European ST-T Database is absent, training
   aborts unless placeholder generation was explicitly authorised. Placeholders
   are stamped ``SYNTHETIC_PLACEHOLDER`` and can never be production models.

2. **Circular labels.** Labelling each beat by thresholding the very ST
   amplitudes the model is then asked to predict teaches the classifier to
   rediscover a cut-off, not to recognise ischaemia. Such weak labels are only
   produced when the caller explicitly opts in with ``--allow-weak-labels``, and
   the provenance records ``label_source = weak_threshold_derived`` so the
   limitation travels with the artifact.
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

from src.datasets.registry import DATASET_REGISTRY
from training.guards import (
    MissingDatasetError,
    real_provenance,
    record_provenance,
    require_real_dataset,
    synthetic_provenance,
)
from training.real_features import ST_FEATURE_NAMES, st_features, st_label_from_features, template_for

MODELS_DIR = PROJ_DIR / "models"
DATASET_ID = "european_st_t"
MODEL_ID = "ECG-ST-1.0.0-candidate"
CLASSES = ["Normal", "ST_Elevation", "ST_Depression"]


def _labelled_dataset_path() -> Path:
    return PROJ_DIR / "data" / "processed" / "st_dataset.csv"


def _train_from_labelled_csv(csv_path: Path) -> Dict[str, Any]:
    """Train from an externally labelled ST corpus (preferred path)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = set(ST_FEATURE_NAMES) | {"label", "record_id"}
    missing = required - set(df.columns)
    if missing:
        return {
            "status": "INVALID_LABELLED_CORPUS",
            "task": "st_analysis",
            "reason": f"Labelled corpus is missing required columns: {sorted(missing)}",
        }

    records = sorted(df["record_id"].astype(str).unique())
    rng = np.random.default_rng(42)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * 0.25)))
    test_records = set(shuffled[:n_test])

    train_df = df[~df["record_id"].astype(str).isin(test_records)]
    test_df = df[df["record_id"].astype(str).isin(test_records)]

    X_train = train_df[ST_FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[ST_FEATURE_NAMES].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy()

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
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
    joblib.dump(model, cand_dir / "st_classifier.pkl")
    joblib.dump(scaler, cand_dir / "st_scaler.pkl")

    provenance = real_provenance(
        dataset_id=DATASET_ID,
        task="st_analysis",
        records=records,
        train_records=sorted(set(train_df["record_id"].astype(str))),
        test_records=sorted(test_records),
        n_train_samples=len(y_train),
        n_test_samples=len(y_test),
    )
    provenance["feature_names"] = ST_FEATURE_NAMES
    provenance["label_source"] = f"externally labelled corpus: {csv_path.name}"
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="CANDIDATE",
        artifact_path="candidate/st_classifier.pkl",
        scaler_path="candidate/st_scaler.pkl",
        description="ST-segment ischaemia classifier trained on a labelled corpus",
    )

    return {"status": "SUCCESS", "task": "st_analysis", "metrics": metrics, "provenance": provenance}


def _weak_label_from_records(record_paths: List[Path]) -> Dict[str, Any]:
    """Build a weak-labelled dataset by thresholding measured ST amplitudes.

    Only reachable through explicit opt-in; the resulting limitation is recorded
    in the artifact provenance.
    """
    try:
        import wfdb
    except ImportError:  # pragma: no cover - wfdb is a declared dependency
        return {"status": "NO_WFDB", "task": "st_analysis", "reason": "wfdb is required to read EDB records."}

    rows: List[Dict[str, Any]] = []
    for path in record_paths:
        try:
            record = wfdb.rdrecord(str(path.with_suffix("")))
            annotation = wfdb.rdann(str(path.with_suffix("")), "atr")
        except Exception:
            continue

        fs = float(record.fs)
        signal = np.asarray(record.p_signal[:, 0], dtype=float)
        for beat_index in np.asarray(getattr(annotation, "sample", []), dtype=int):
            features = st_features(signal, fs, int(beat_index))
            if features is None:
                continue
            row = {"record_id": path.stem, **features}
            row["label"] = st_label_from_features(features)
            rows.append(row)

    if len(rows) < 50:
        return {
            "status": "INSUFFICIENT_DERIVED_SAMPLES",
            "task": "st_analysis",
            "reason": "EDB records yielded too few measurable beats to train.",
            "derived_samples": len(rows),
        }

    import pandas as pd

    out_path = _labelled_dataset_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    result = _train_from_labelled_csv(out_path)
    if result.get("status") == "SUCCESS":
        result["provenance"]["label_source"] = "weak_threshold_derived (CIRCULAR: labels derived from the same ST amplitudes used as features)"
        result["provenance"]["clinical_claim"] = (
            "NONE - model can only rediscover the labelling threshold; not evidence of ischaemia detection"
        )
        record_provenance(
            model_id=MODEL_ID,
            provenance=result["provenance"],
            status="SYNTHETIC_PLACEHOLDER",
            artifact_path="candidate/st_classifier.pkl",
            scaler_path="candidate/st_scaler.pkl",
            description="ST placeholder trained on threshold-derived weak labels",
        )
    return result


def _build_placeholder() -> Dict[str, Any]:
    """Build a clearly-labelled non-clinical placeholder artifact."""
    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            np.column_stack([rng.normal(0.01, 0.03, 100), rng.normal(0.02, 0.03, 100), rng.normal(12.0, 5.0, 100)]),
            np.column_stack(
                [
                    rng.normal(0.22, 0.06, 100).clip(0.10, 0.60),
                    rng.normal(0.25, 0.08, 100).clip(0.12, 0.70),
                    rng.normal(2.0, 6.0, 100),
                ]
            ),
            np.column_stack(
                [
                    rng.normal(-0.18, 0.05, 100).clip(-0.50, -0.06),
                    rng.normal(-0.20, 0.06, 100).clip(-0.60, -0.08),
                    rng.normal(-8.0, 5.0, 100),
                ]
            ),
        ]
    )
    y = np.array(["Normal"] * 100 + ["ST_Elevation"] * 100 + ["ST_Depression"] * 100)

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(scaler.fit_transform(X), y)

    joblib.dump(model, cand_dir / "st_classifier.pkl")
    joblib.dump(scaler, cand_dir / "st_scaler.pkl")

    provenance = synthetic_provenance(
        dataset_id=DATASET_ID,
        task="st_analysis",
        reason="Dataset absent; placeholder generated under explicit opt-in.",
        n_samples=len(X),
    )
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="SYNTHETIC_PLACEHOLDER",
        artifact_path="candidate/st_classifier.pkl",
        scaler_path="candidate/st_scaler.pkl",
        description="NON-CLINICAL placeholder for ST plumbing only",
    )

    return {
        "status": "SYNTHETIC_PLACEHOLDER_SAVED",
        "task": "st_analysis",
        "model_path": str(cand_dir / "st_classifier.pkl"),
        "classes": CLASSES,
        "clinical_claim": "NONE",
        "provenance": provenance,
    }


def train_st_models(allow_placeholder: Optional[bool] = None, allow_weak_labels: bool = False) -> dict:
    """Train the Task D ST-segment ischaemia classifier."""
    labelled = _labelled_dataset_path()
    if labelled.exists():
        return _train_from_labelled_csv(labelled)

    st_root = DATASET_REGISTRY.get_local_path(DATASET_ID) / "raw"
    record_paths = sorted(st_root.glob("*.hea"))

    use_placeholder = require_real_dataset(
        dataset_id=DATASET_ID,
        task="st_analysis",
        records=record_paths,
        location=st_root,
        allow_placeholder=allow_placeholder,
    )

    if use_placeholder:
        return _build_placeholder()

    if not allow_weak_labels:
        return {
            "status": "NO_VERIFIED_LABELS",
            "task": "st_analysis",
            "records_found": len(record_paths),
            "reason": (
                "EDB waveforms are present but no verified ischaemia label source is configured. "
                "Deriving labels by thresholding the same ST amplitudes the model predicts would be "
                "circular. Provide data/processed/st_dataset.csv with expert labels, or pass "
                "--allow-weak-labels to build a clearly-marked placeholder."
            ),
        }

    return _weak_label_from_records(record_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task D ST-segment ischaemia model")
    parser.add_argument("--allow-placeholder", action="store_true", help="Build a non-clinical placeholder artifact")
    parser.add_argument("--allow-weak-labels", action="store_true", help="Permit circular threshold-derived labels")
    args = parser.parse_args()
    try:
        res = train_st_models(
            allow_placeholder=args.allow_placeholder or None,
            allow_weak_labels=args.allow_weak_labels,
        )
    except MissingDatasetError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    print("Train ST Result:", res)
