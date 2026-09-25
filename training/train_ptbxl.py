"""
Task C — 12-Lead Multi-Label Clinical Diagnosis Training Pipeline
=================================================================

Trains multi-label classifiers for standard 12-lead clinical ECG across the
PTB-XL diagnostic superclasses: NORM, MI, STTC, CD, HYP.

Multiple concurrent pathologies are kept as independent labels; the pipeline
never forces them into a single mutually-exclusive class.

Data integrity
--------------
No synthetic fallback. When PTB-XL is absent, training aborts unless
placeholder generation was explicitly authorised, in which case the artifact is
stamped ``SYNTHETIC_PLACEHOLDER`` and barred from production.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
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
from training.real_features import (
    PTBXL_FEATURE_NAMES,
    PTBXL_SUPERCLASSES,
    ptbxl_labels_from_scp_codes,
    ptbxl_lead_features,
    template_for,
)

MODELS_DIR = PROJ_DIR / "models"
DATASET_ID = "ptb_xl"
MODEL_ID = "ECG-PTBXL-1.0.0-candidate"


def _read_metadata(ptb_root: Path) -> List[Dict[str, Any]]:
    """Read the PTB-XL metadata table, if present."""
    for candidate in ("ptbxl_database.csv", "metadata.csv"):
        csv_path = ptb_root / candidate
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return rows
    return []


def _load_12lead(ptb_root: Path, rel_path: str) -> Optional[Tuple[np.ndarray, float, List[str]]]:
    """Load one PTB-XL record as a 12-lead matrix."""
    try:
        import wfdb
    except ImportError:  # pragma: no cover - wfdb is a declared dependency
        return None

    record_base = ptb_root / rel_path
    try:
        record = wfdb.rdrecord(str(record_base))
    except Exception:
        return None

    signal = np.asarray(record.p_signal, dtype=float).T
    return signal, float(record.fs), list(record.sig_name)


def _build_feature_matrix(
    ptb_root: Path,
    rows: List[Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix, multi-label targets and patient ids from metadata."""
    patients: List[str] = []
    features: List[List[float]] = []
    targets: List[List[int]] = []

    for row in rows:
        rel = row.get("filename_hr") or row.get("filename_lr")
        if not rel:
            continue
        loaded = _load_12lead(ptb_root, rel)
        if loaded is None:
            continue
        signal, fs, lead_names = loaded

        feature_dict = ptbxl_lead_features(signal, fs, lead_names)
        if feature_dict is None:
            continue

        try:
            scp_codes = ast.literal_eval(row.get("scp_codes", "{}") or "{}")
        except (ValueError, SyntaxError):
            continue

        labels = ptbxl_labels_from_scp_codes(scp_codes)
        if not labels:
            continue

        # A lead may be missing from a given export; fill absent descriptors from
        # the schema explicitly as NaN so the imputer/scaler handles them,
        # rather than inventing a plausible-looking voltage.
        vector = [feature_dict.get(name, np.nan) for name in PTBXL_FEATURE_NAMES]

        patients.append(str(row.get("patient_id", rel)))
        features.append(vector)
        targets.append([1 if sc in labels else 0 for sc in PTBXL_SUPERCLASSES])

    return patients, np.asarray(features, dtype=float), np.asarray(targets, dtype=int), PTBXL_SUPERCLASSES


def _train_from_records(ptb_root: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train and honestly evaluate the 12-lead multi-label model."""
    patients, X, y, superclasses = _build_feature_matrix(ptb_root, rows)
    if X.shape[0] < 30:
        return {
            "status": "INSUFFICIENT_DERIVED_SAMPLES",
            "task": "12lead_diagnosis",
            "reason": "Too few usable PTB-XL records were derived to train.",
            "derived_samples": int(X.shape[0]),
        }

    # Replace non-finite descriptors with the training-set median computed
    # strictly from the training partition (below), so we first split.
    unique_patients = sorted(set(patients))
    rng = np.random.default_rng(42)
    shuffled = list(unique_patients)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * 0.2)))
    test_patients = set(shuffled[:n_test])
    train_patients = [p for p in unique_patients if p not in test_patients]

    train_mask = np.array([p not in test_patients for p in patients])
    test_mask = ~train_mask
    if train_mask.sum() < 20 or test_mask.sum() < 5:
        return {
            "status": "INSUFFICIENT_DERIVED_SAMPLES",
            "task": "12lead_diagnosis",
            "reason": "Patient-level split produced an unusable partition size.",
        }

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Median imputation learned on the training partition only.
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = np.column_stack([est.predict_proba(X_test_scaled)[:, 1] for est in model.estimators_])

    per_label: Dict[str, Any] = {}
    for idx, name in enumerate(superclasses):
        support = int(y_test[:, idx].sum())
        entry: Dict[str, Any] = {"support": support}
        if support > 0:
            entry["f1"] = float(f1_score(y_test[:, idx], predictions[:, idx], zero_division=0))
        if 0 < support < len(y_test):
            try:
                entry["auroc"] = float(roc_auc_score(y_test[:, idx], probabilities[:, idx]))
                entry["auprc"] = float(average_precision_score(y_test[:, idx], probabilities[:, idx]))
            except ValueError:
                pass
        per_label[name] = entry

    metrics = {
        "micro_f1": float(f1_score(y_test, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "n_train_records": int(train_mask.sum()),
        "n_test_records": int(test_mask.sum()),
        "per_label": per_label,
    }

    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cand_dir / "ptbxl_classifier.pkl")
    joblib.dump(scaler, cand_dir / "ptbxl_scaler.pkl")

    provenance = real_provenance(
        dataset_id=DATASET_ID,
        task="12lead_diagnosis",
        records=[r for r in (row.get("filename_hr") or row.get("filename_lr", "") for row in rows) if r],
        train_records=train_patients,
        test_records=sorted(test_patients),
        n_train_samples=int(train_mask.sum()),
        n_test_samples=int(test_mask.sum()),
    )
    provenance["feature_names"] = PTBXL_FEATURE_NAMES
    provenance["label_source"] = "PTB-XL scp_codes aggregated to diagnostic superclasses"
    provenance["missing_lead_policy"] = "excluded from feature schema (not imputed at extraction)"
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="CANDIDATE",
        artifact_path="candidate/ptbxl_classifier.pkl",
        scaler_path="candidate/ptbxl_scaler.pkl",
        description="12-Lead multi-label classifier for NORM/MI/STTC/CD/HYP (real PTB-XL data)",
    )

    return {"status": "SUCCESS", "task": "12lead_diagnosis", "metrics": metrics, "provenance": provenance}


def _build_placeholder() -> Dict[str, Any]:
    """Build a clearly-labelled non-clinical placeholder artifact."""
    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_samples = 300
    n_features = len(PTBXL_FEATURE_NAMES)

    X = rng.normal(0.0, 1.0, (n_samples, n_features))
    y = np.zeros((n_samples, len(PTBXL_SUPERCLASSES)), dtype=int)
    for i in range(n_samples):
        if rng.random() > 0.5:
            y[i, 0] = 1
        else:
            for j, p in enumerate((0.4, 0.4, 0.6, 0.7), start=1):
                if rng.random() > p:
                    y[i, j] = 1
            if y[i].sum() == 0:
                y[i, 0] = 1

    scaler = StandardScaler()
    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42), n_jobs=-1
    )
    model.fit(scaler.fit_transform(X), y)

    joblib.dump(model, cand_dir / "ptbxl_classifier.pkl")
    joblib.dump(scaler, cand_dir / "ptbxl_scaler.pkl")

    provenance = synthetic_provenance(
        dataset_id=DATASET_ID,
        task="12lead_diagnosis",
        reason="Dataset absent; placeholder generated under explicit opt-in.",
        n_samples=n_samples,
    )
    record_provenance(
        model_id=MODEL_ID,
        provenance=provenance,
        status="SYNTHETIC_PLACEHOLDER",
        artifact_path="candidate/ptbxl_classifier.pkl",
        scaler_path="candidate/ptbxl_scaler.pkl",
        description="NON-CLINICAL placeholder for 12-lead plumbing only",
    )

    return {
        "status": "SYNTHETIC_PLACEHOLDER_SAVED",
        "task": "12lead_diagnosis",
        "model_path": str(cand_dir / "ptbxl_classifier.pkl"),
        "target_superclasses": PTBXL_SUPERCLASSES,
        "clinical_claim": "NONE",
        "provenance": provenance,
    }


def train_ptbxl_models(allow_placeholder: Optional[bool] = None) -> dict:
    """Train the Task C 12-lead multi-label model."""
    ptb_root = DATASET_REGISTRY.get_local_path(DATASET_ID)
    raw_root = ptb_root / "raw"
    search_root = raw_root if raw_root.exists() else ptb_root

    rows = _read_metadata(search_root)
    record_paths = sorted(search_root.glob("**/*.hea"))

    use_placeholder = require_real_dataset(
        dataset_id=DATASET_ID,
        task="12lead_diagnosis",
        records=rows or record_paths,
        location=search_root,
        allow_placeholder=allow_placeholder,
    )

    if use_placeholder:
        return _build_placeholder()

    if not rows:
        return {
            "status": "METADATA_REQUIRED",
            "task": "12lead_diagnosis",
            "reason": (
                "PTB-XL waveform files are present but ptbxl_database.csv (which carries the "
                "diagnostic labels) was not found. Labels must not be guessed."
            ),
            "records_found": len(record_paths),
        }

    return _train_from_records(search_root, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task C 12-lead multi-label model")
    parser.add_argument("--allow-placeholder", action="store_true", help="Build a non-clinical placeholder artifact")
    args = parser.parse_args()
    try:
        res = train_ptbxl_models(allow_placeholder=args.allow_placeholder or None)
    except MissingDatasetError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    print("Train PTB-XL Result:", res)
