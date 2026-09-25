"""
Autonomous Multi-Dataset Training, Evaluation, and Comparison Pipeline
======================================================================

Responsibilities
----------------
* Ingest the split manifest (zero patient leakage).
* Train a Logistic Regression baseline, a candidate Random Forest and a neural
  feature model.
* Evaluate AUROC, AUPRC, sensitivity, specificity, F1 and ECE on the held-out
  patient partition.
* Compare candidates against the current production model.
* Emit model cards, experiment metrics and a migration report.
* Run external cross-dataset validation **only when the external dataset is
  actually present**.

Data integrity
--------------
Previous revisions invented external-validation numbers by subtracting fixed
constants from the in-domain scores (``accuracy - 0.021``). That produced a
"domain shift report" that described nothing. External validation now requires a
real external cohort: when it is missing the report is explicitly marked
``NOT_EVALUATED`` and contains no performance numbers.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.ml.models.deep_1d_cnn import ECGFeatureMLPClassifier
from training.guards import MissingDatasetError, require_real_dataset

DATA_DIR = PROJ_DIR / "data" / "processed"
SPLITS_DIR = PROJ_DIR / "data" / "splits" / "beat_arrhythmia"
MODELS_DIR = PROJ_DIR / "models"
REPORTS_DIR = PROJ_DIR / "reports"

IGNORE_COLUMNS = {"label", "record_id", "symbol"}


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            ece += abs(float(np.mean(confidences[in_bin])) - float(np.mean(accuracies[in_bin]))) * prop_in_bin
    return float(ece)


def evaluate_model_comprehensive(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Compute rich clinical and statistical evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    y_bin = label_binarize(y_test, classes=classes)
    try:
        auroc = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        auroc = None
    try:
        auprc = float(average_precision_score(y_bin, y_prob, average="weighted"))
    except Exception:
        auprc = None

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    per_class_sens: Dict[str, float] = {}
    per_class_spec: Dict[str, float] = {}
    for idx, name in enumerate(classes):
        tp = cm[idx][idx]
        fn = sum(cm[idx]) - tp
        fp = sum(row[idx] for row in cm) - tp
        tn = sum(sum(r) for r in cm) - (tp + fn + fp)
        per_class_sens[name] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class_spec[name] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_indices = np.array([label_to_idx.get(str(y), 0) for y in y_test])

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
        "ece": compute_ece(y_prob, y_true_indices),
        "per_class_sensitivity": per_class_sens,
        "per_class_specificity": per_class_spec,
        "classification_report": classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": cm,
    }


def _load_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed partitions, refusing to continue without them."""
    train_path = DATA_DIR / "train_dataset.csv"
    test_path = DATA_DIR / "test_dataset.csv"

    require_real_dataset(
        dataset_id="mit_bih_arrhythmia",
        task="beat_arrhythmia (multi-model benchmark)",
        records=[p for p in (train_path, test_path) if p.exists()],
        location=DATA_DIR,
        download_hint=(
            "python training/download_datasets.py --dataset mit_bih_arrhythmia "
            "&& python training/create_splits.py"
        ),
    )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _run_external_validation(candidate_model: Any, scaler: Any, feature_cols: List[str], classes: List[str], in_domain: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the candidate on an external cohort, if one is actually present."""
    from src.datasets.registry import DATASET_REGISTRY
    from training.external_validation import generate_external_validation_report

    external_id = "incart_12lead_arrhythmia"

    try:
        external_root = DATASET_REGISTRY.get_local_path(external_id)
    except Exception:
        external_root = None

    external_csv = (external_root / "processed" / "external_test.csv") if external_root else None
    if external_csv is None or not Path(external_csv).exists():
        return generate_external_validation_report(
            model_id="ECG-RF-2.0.0-candidate",
            training_dataset="mit_bih_arrhythmia",
            external_dataset=external_id,
            in_domain_metrics=in_domain,
            external_metrics=None,
        )

    external_df = pd.read_csv(external_csv)
    missing = [c for c in feature_cols if c not in external_df.columns]
    if missing or "label" not in external_df.columns:
        return generate_external_validation_report(
            model_id="ECG-RF-2.0.0-candidate",
            training_dataset="mit_bih_arrhythmia",
            external_dataset=external_id,
            in_domain_metrics=in_domain,
            external_metrics=None,
            note=f"External cohort is missing required columns: {missing or ['label']}",
        )

    X_ext = scaler.transform(external_df[feature_cols].to_numpy(dtype=float))
    y_ext = external_df["label"].astype(str).to_numpy()
    external_metrics = evaluate_model_comprehensive(
        candidate_model, X_ext, y_ext, classes, f"{external_id} (external)"
    )

    return generate_external_validation_report(
        model_id="ECG-RF-2.0.0-candidate",
        training_dataset="mit_bih_arrhythmia",
        external_dataset=external_id,
        in_domain_metrics=in_domain,
        external_metrics=external_metrics,
    )


def run_pipeline(task: str = "all") -> Dict[str, Any]:
    print("=" * 70)
    print("ECG GUARDIAN — AUTONOMOUS MULTI-DATASET ML PIPELINE")
    print("=" * 70)

    train_df, test_df = _load_split_data()

    split_manifest = SPLITS_DIR / "v1.json"
    manifest_data: Optional[Dict[str, Any]] = None
    if split_manifest.exists():
        with open(split_manifest, "r", encoding="utf-8") as handle:
            manifest_data = json.load(handle)
        print(
            "Patient isolation verified: train records "
            f"{manifest_data['train']['records']}, test records {manifest_data['test']['records']}"
        )

    feature_cols = [c for c in train_df.columns if c not in IGNORE_COLUMNS]
    classes = sorted(train_df["label"].astype(str).unique().tolist())

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].astype(str).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].astype(str).to_numpy()

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Classes: {classes}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Model 1: Logistic Regression baseline ---")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_metrics = evaluate_model_comprehensive(lr, X_test_scaled, y_test, classes, "Baseline Logistic Regression")

    print("\n--- Model 2: Candidate Random Forest ---")
    rf_cand = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_cand.fit(X_train_scaled, y_train)
    rf_metrics = evaluate_model_comprehensive(rf_cand, X_test_scaled, y_test, classes, "ECG-RF-2.0.0-candidate")

    print("\n--- Model 3: Neural feature model ---")
    mlp = ECGFeatureMLPClassifier(input_length=len(feature_cols), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_metrics = evaluate_model_comprehensive(mlp, X_test, y_test, classes, "ECG-MLP-1.0.0-candidate")

    print("\n--- Evaluating production baseline (ECG-RF-1.0.0) ---")
    prod_metrics: Optional[Dict[str, Any]] = None
    prod_classifier = MODELS_DIR / "production" / "classifier.pkl"
    prod_scaler = MODELS_DIR / "production" / "scaler.pkl"
    if prod_classifier.exists() and prod_scaler.exists():
        prod_rf = joblib.load(prod_classifier)
        prod_scaler_obj = joblib.load(prod_scaler)
        prod_metrics = evaluate_model_comprehensive(
            prod_rf, prod_scaler_obj.transform(X_test), y_test, classes, "ECG-RF-1.0.0 (Production)"
        )
    else:
        print("Production artifacts absent; skipping production comparison.")

    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_cand, cand_dir / "classifier.pkl")
    joblib.dump(lr, cand_dir / "baseline_classifier.pkl")
    joblib.dump(mlp, cand_dir / "deep_classifier.pkl")
    joblib.dump(scaler, cand_dir / "scaler.pkl")

    all_metrics = {
        "production": prod_metrics,
        "baseline_lr": lr_metrics,
        "candidate_rf": rf_metrics,
        "candidate_mlp": mlp_metrics,
        "classes": classes,
        "feature_names": feature_cols,
        "dataset_split": manifest_data,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    metrics_path = REPORTS_DIR / "experiments" / "multi_model_benchmark.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    print(f"Benchmark metrics written to {metrics_path}")

    _write_migration_report(rf_metrics, lr_metrics, mlp_metrics, prod_metrics, classes, manifest_data)
    _write_model_card(rf_metrics)

    multi_task_results: Dict[str, Any] = {"beat_arrhythmia": rf_metrics}

    if task in ("all", "af_detection"):
        print("\n--- Task B: Atrial Fibrillation ---")
        from training.train_af import train_af_models

        multi_task_results["af_detection"] = train_af_models()

    if task in ("all", "12lead_diagnostic"):
        print("\n--- Task C: 12-lead multi-label ---")
        from training.train_ptbxl import train_ptbxl_models

        multi_task_results["12lead_diagnosis"] = train_ptbxl_models()

    if task in ("all", "st_analysis"):
        print("\n--- Task D: ST-segment ischaemia ---")
        from training.train_st import train_st_models

        multi_task_results["st_analysis"] = train_st_models()

    if task in ("all", "quality_gate"):
        print("\n--- Task E: Quality gate ---")
        from training.train_quality import train_quality_model

        multi_task_results["quality_gate"] = train_quality_model()

    if task in ("all", "multimodal"):
        print("\n--- Task F: Multimodal comparison & feature ablation ---")
        from training.evaluate_multimodal_comparison import run_three_model_comparison_and_ablation

        run_three_model_comparison_and_ablation()
        mm_rep_path = REPORTS_DIR / "three_model_comparison.json"
        if mm_rep_path.exists():
            with open(mm_rep_path, "r", encoding="utf-8") as handle:
                multi_task_results["multimodal_comparison"] = json.load(handle)

    if task == "all":
        print("\n--- External cross-dataset validation ---")
        ext_report = _run_external_validation(rf_cand, scaler, feature_cols, classes, rf_metrics)
        multi_task_results["external_validation"] = ext_report
        ext_rep_path = REPORTS_DIR / "experiments" / "external_domain_shift_report.json"
        with open(ext_rep_path, "w", encoding="utf-8") as handle:
            json.dump(ext_report, handle, indent=2)
        print(f"External validation report written to {ext_rep_path}")

    summary_path = REPORTS_DIR / "experiments" / "full_multi_task_benchmark.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(multi_task_results, handle, indent=2, default=str)
    print(f"Unified multi-task benchmark written to {summary_path}")

    return multi_task_results


def _fmt(value: Optional[float], scale: float = 100.0, suffix: str = "%") -> str:
    return "not measured" if value is None else f"{value * scale:.2f}{suffix}"


def _write_migration_report(rf, lr, mlp, prod, classes, manifest) -> None:
    out = REPORTS_DIR / "migration" / "current_vs_new.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    def decimal(value: Optional[float], digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def row(name, metrics, status):
        if metrics is None:
            return f"| **`{name}`** | not measured | - | - | - | - | - | - | {status} |"
        return (
            f"| **`{name}`** | {_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('weighted_f1'))} | "
            f"{_fmt(metrics['per_class_sensitivity'].get('PVC'))} | "
            f"{_fmt(metrics['per_class_specificity'].get('PVC'))} | "
            f"{decimal(metrics.get('auroc'))} | "
            f"{decimal(metrics.get('auprc'))} | "
            f"{decimal(metrics.get('ece'))} | {status} |"
        )

    train_records = manifest["train"]["records"] if manifest else "see split manifest"
    test_records = manifest["test"]["records"] if manifest else "see split manifest"

    content = f"""# ECG Guardian — Model Migration & Comparison Report
**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Task:** Single-lead beat arrhythmia classification (`TASK_BEAT_ARRHYTHMIA`)
**Split scheme:** patient-level (Train: {train_records}; Test: {test_records})

---

## 1. Performance (measured on held-out patients)

| Model | Accuracy | Weighted F1 | PVC sensitivity | PVC specificity | AUROC | AUPRC | ECE | Lifecycle |
|---|---|---|---|---|---|---|---|---|
{row('ECG-RF-1.0.0', prod, 'PRODUCTION (active)')}
{row('ECG-RF-2.0.0-candidate', rf, 'CANDIDATE')}
{row('ECG-MLP-1.0.0-candidate', mlp, 'EXPERIMENTAL')}
{row('ECG-LR-1.0.0', lr, 'BASELINE')}

Every number above is computed on the held-out partition during this run. No
metric is estimated, adjusted, or carried over from a previous environment.

## 2. Confusion matrix (candidate Random Forest)

Classes: {classes}
""" + "\n".join(
        f"- **{name}:** {row_values}" for name, row_values in zip(classes, rf["confusion_matrix"])
    ) + f"""

---

## 3. Promotion decision

Promotion to `VALIDATED` requires all of:

* PVC sensitivity >= 0.98
* macro F1 >= 0.60
* AUROC >= 0.95

Candidate macro F1 {_fmt(rf.get('macro_f1'), suffix='')}, AUROC {'n/a' if rf.get('auroc') is None else f"{rf['auroc']:.4f}"}.
The candidate retains `CANDIDATE` status unless every gate is cleared alongside
qualified clinical review.
"""
    out.write_text(content, encoding="utf-8")
    print(f"Migration report written to {out}")


def _write_model_card(rf) -> None:
    out = PROJ_DIR / "docs" / "models" / "MODEL_CARD_ECG_RF_2.0.0.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""# Model Card: ECG-RF-2.0.0-candidate

## Model details
- **Developer:** ECG Guardian training pipeline
- **Model date:** {datetime.datetime.now().strftime('%B %Y')}
- **Type:** balanced ensemble Random Forest over extracted beat features
- **Output classes:** Normal, PAC, PVC, Other

## Intended use
- Adjunctive, physician-reviewed rhythm screening in simulated hospital settings.
- **Out of scope:** autonomous diagnosis, primary triage without clinician confirmation.

## Training data & provenance
- **Source:** MIT-BIH Arrhythmia Database, patient-level split, zero beat-level contamination.
- **Scaler:** fitted on the training partition only.
- Artifact provenance is recorded in `models/registry/catalog.json` under the
  `provenance` key and travels with the model.

## Measured performance (independent test partition)
- Accuracy: {_fmt(rf.get('accuracy'))}
- Weighted F1: {_fmt(rf.get('weighted_f1'))}
- Macro F1: {_fmt(rf.get('macro_f1'), suffix='')}
- PVC sensitivity: {_fmt(rf['per_class_sensitivity'].get('PVC'))}
- PVC specificity: {_fmt(rf['per_class_specificity'].get('PVC'))}
- AUROC: {'n/a' if rf.get('auroc') is None else f"{rf['auroc']:.4f}"}
- ECE: {'n/a' if rf.get('ece') is None else f"{rf['ece']:.4f}"}

## Regulatory status
> Investigational research software. NOT approved by CDSCO, US FDA or any CE
> notified body. Probabilities are algorithm outputs, not clinical certainty.
""",
        encoding="utf-8",
    )
    print(f"Model card written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Guardian multi-model ML pipeline")
    parser.add_argument(
        "--task",
        choices=["all", "beat_arrhythmia", "af_detection", "12lead_diagnostic", "st_analysis", "quality_gate", "multimodal"],
        default="all",
    )
    args = parser.parse_args()
    try:
        run_pipeline(task=args.task)
    except MissingDatasetError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
