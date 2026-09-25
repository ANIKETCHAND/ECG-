"""
Calibration & Conformal Fitting
===============================

Fits the two post-hoc layers used at inference time:

* a per-class calibration map (isotonic by default), and
* a Mondrian conformal predictor that yields prediction sets and abstentions.

Method (and why it is not leakage)
----------------------------------
The held-out test partition is split **by record** into two disjoint halves:

* ``calibration`` — used to fit the calibration map and conformal quantiles,
* ``evaluation`` — used to measure the effect, and never seen by the fit.

Reported ECE-before/ECE-after and empirical coverage therefore come from data
that had no influence on the fitted artifacts. Calibrating on the same rows used
to report the improvement would produce a number that means nothing.

Usage::

    python training/fit_calibration.py [--alpha 0.10] [--method isotonic]
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

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.ml.calibration import (
    MondrianConformalClassifier,
    MulticlassCalibrator,
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_brier_score,
    reliability_bins,
    save_calibration,
)
from training.guards import MissingDatasetError, require_real_dataset

DATA_DIR = PROJ_DIR / "data" / "processed"
MODELS_DIR = PROJ_DIR / "models"
IGNORE_COLUMNS = {"label", "record_id", "symbol"}
DEFAULT_ALPHA = 0.10


def _split_records_for_calibration(records: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministically partition records into calibration and evaluation halves."""
    ordered = sorted(records)
    if len(ordered) < 2:
        raise ValueError(
            "Calibration needs at least two distinct records so that the fitting and "
            "evaluation halves stay disjoint."
        )
    n_calibration = max(1, len(ordered) // 2)
    return ordered[:n_calibration], ordered[n_calibration:]


def _load_artifacts() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the production classifier, scaler, and metadata."""
    for classifier_path, scaler_path, metadata_path in (
        (MODELS_DIR / "production" / "classifier.pkl", MODELS_DIR / "production" / "scaler.pkl", MODELS_DIR / "production" / "metadata.json"),
        (MODELS_DIR / "classifier.pkl", MODELS_DIR / "scaler.pkl", MODELS_DIR / "metadata.json"),
    ):
        if classifier_path.exists() and scaler_path.exists():
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            return joblib.load(classifier_path), joblib.load(scaler_path), metadata
    raise FileNotFoundError("No trained classifier/scaler artifacts were found.")


def _select_calibration_method(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: List[str],
    *,
    seed: int = 42,
) -> Tuple[str, Dict[str, Any]]:
    """Choose between isotonic and Platt scaling using an inner hold-out.

    The selection split is drawn from the calibration partition only, so the outer
    evaluation partition still has no influence on which method is chosen.
    Isotonic needs far more data than Platt to avoid overfitting, and with a small
    calibration set it can make calibration *worse* — so the choice is measured
    rather than assumed.
    """
    index_of = {name: i for i, name in enumerate(classes)}
    y_index = np.array([index_of.get(label, 0) for label in labels])

    n = len(labels)
    if n < 40:
        return "platt", {"selection": "insufficient_samples_defaulted_to_platt", "n": int(n)}

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_selection = max(10, int(round(n * 0.3)))
    selection_idx, fit_idx = order[:n_selection], order[n_selection:]

    scores: Dict[str, Optional[float]] = {}
    for candidate in ("isotonic", "platt"):
        try:
            calibrator = MulticlassCalibrator(method=candidate, classes=classes)
            calibrator.fit(probabilities[fit_idx], labels[fit_idx])
            scores[candidate] = float(
                expected_calibration_error(calibrator.transform(probabilities[selection_idx]), y_index[selection_idx])
            )
        except Exception:
            scores[candidate] = None

    comparable = {k: v for k, v in scores.items() if v is not None}
    if not comparable:
        return "platt", {"selection": "all_candidates_failed_defaulted_to_platt", "selection_ece": scores}

    best = min(comparable, key=comparable.get)
    return best, {
        "selection_split": "row-level hold-out drawn from the calibration partition only",
        "selection_ece": scores,
        "n_selection_rows": int(n_selection),
    }


def fit_calibration(alpha: float = DEFAULT_ALPHA, method: str = "auto") -> Dict[str, Any]:
    """Fit and persist calibration and conformal artifacts."""
    test_path = DATA_DIR / "test_dataset.csv"
    require_real_dataset(
        dataset_id="mit_bih_arrhythmia",
        task="probability_calibration",
        records=[p for p in (test_path,) if p.exists()],
        location=DATA_DIR,
        download_hint=(
            "python training/download_datasets.py --dataset mit_bih_arrhythmia "
            "&& python training/create_splits.py"
        ),
    )

    frame = pd.read_csv(test_path)
    if "record_id" not in frame.columns or "label" not in frame.columns:
        raise ValueError("The processed test partition must contain 'record_id' and 'label' columns.")

    records = sorted(frame["record_id"].astype(str).unique().tolist())
    calibration_records, evaluation_records = _split_records_for_calibration(records)

    classifier, scaler, metadata = _load_artifacts()
    feature_names = metadata.get("feature_names") or [c for c in frame.columns if c not in IGNORE_COLUMNS]
    classes = list(classifier.classes_)

    def predict(frame_subset: pd.DataFrame) -> np.ndarray:
        features = frame_subset[feature_names].to_numpy(dtype=float)
        return classifier.predict_proba(scaler.transform(features))

    calibration_frame = frame[frame["record_id"].astype(str).isin(calibration_records)]
    evaluation_frame = frame[frame["record_id"].astype(str).isin(evaluation_records)]

    if calibration_frame.empty or evaluation_frame.empty:
        raise ValueError("Record-level split produced an empty calibration or evaluation partition.")

    calibration_probs = predict(calibration_frame)
    calibration_labels = calibration_frame["label"].astype(str).to_numpy()
    evaluation_probs = predict(evaluation_frame)
    evaluation_labels = evaluation_frame["label"].astype(str).to_numpy()

    selection_diagnostics: Dict[str, Any] = {"selection": "explicit_method_requested", "method": method}
    if method == "auto":
        chosen_method, selection_diagnostics = _select_calibration_method(
            calibration_probs, calibration_labels, classes
        )
    else:
        chosen_method = method

    calibrator = MulticlassCalibrator(method=chosen_method, classes=classes).fit(
        calibration_probs, calibration_labels
    )

    # Conformal is fitted on *calibrated* scores so that served probabilities and
    # the conformal guarantee describe the same quantity.
    calibrated_calibration_probs = calibrator.transform(calibration_probs)
    conformal = MondrianConformalClassifier(alpha=alpha, classes=classes).fit(
        calibrated_calibration_probs, calibration_labels
    )

    # Measure the effect, strictly on the untouched evaluation half.
    index_of = {name: i for i, name in enumerate(classes)}
    evaluation_index = np.array([index_of.get(label, 0) for label in evaluation_labels])

    calibrated_evaluation = calibrator.transform(evaluation_probs)

    prediction_sets = conformal.predict_set(calibrated_evaluation)
    set_sizes = [len(s) for s in prediction_sets]
    covered = [evaluation_labels[i] in prediction_sets[i] for i in range(len(evaluation_labels))]

    metrics = {
        "calibration_records": calibration_records,
        "evaluation_records": evaluation_records,
        "selected_method": chosen_method,
        "method_selection": selection_diagnostics,
        "conformal_pooled_classes": conformal.pooled_classes,
        "conformal_class_counts": conformal.counts,
        "n_calibration_samples": int(len(calibration_labels)),
        "n_evaluation_samples": int(len(evaluation_labels)),
        "ece_before": float(expected_calibration_error(evaluation_probs, evaluation_index)),
        "ece_after": float(expected_calibration_error(calibrated_evaluation, evaluation_index)),
        "mce_before": float(maximum_calibration_error(evaluation_probs, evaluation_index)),
        "mce_after": float(maximum_calibration_error(calibrated_evaluation, evaluation_index)),
        "brier_before": float(multiclass_brier_score(evaluation_probs, evaluation_index)),
        "brier_after": float(multiclass_brier_score(calibrated_evaluation, evaluation_index)),
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": float(np.mean(covered)),
        "mean_prediction_set_size": float(np.mean(set_sizes)),
        "abstention_rate": float(np.mean([size > 1 for size in set_sizes])),
        "reliability_bins_after": reliability_bins(calibrated_evaluation, evaluation_index),
    }

    # Deployment gate: only recommend applying the artifact when it measurably
    # improved calibration (and conformal coverage is not materially below target)
    # on the untouched evaluation partition.
    ece_improved = metrics["ece_after"] <= metrics["ece_before"]
    coverage_ok = metrics["empirical_coverage"] >= (1.0 - alpha) - 0.05
    deployment_recommended = bool(ece_improved and coverage_ok)

    if not deployment_recommended:
        reasons = []
        if not ece_improved:
            reasons.append(
                f"ECE worsened on the evaluation partition ({metrics['ece_before']:.4f} -> {metrics['ece_after']:.4f})"
            )
        if not coverage_ok:
            reasons.append(
                f"empirical conformal coverage {metrics['empirical_coverage']:.3f} is materially below the "
                f"{1.0 - alpha:.2f} target"
            )
        recommendation_reason = (
            "; ".join(reasons)
            + ". This typically means the calibration partition is too small or not exchangeable with the "
            "evaluation partition. Artifact retained for inspection; inference serves raw probabilities."
        )
    else:
        recommendation_reason = (
            "Calibration and conformal coverage both met their criteria on a record-disjoint evaluation "
            "partition."
        )

    metrics["deployment_recommended"] = deployment_recommended
    metrics["deployment_recommendation_reason"] = recommendation_reason

    path = save_calibration(
        calibrator,
        conformal,
        extra={
            "deployment_recommended": deployment_recommended,
            "deployment_recommendation_reason": recommendation_reason,
            "model_id": metadata.get("model_name", "unknown"),
            "classes": classes,
            "feature_names": feature_names,
            "method": chosen_method,
            "alpha": alpha,
            "metrics": metrics,
            "methodology_note": (
                "Calibration and conformal quantiles were fitted on calibration records and "
                "evaluated on disjoint record-level evaluation partitions. No evaluation row "
                "influenced the fitted artifacts."
            ),
            "clinical_note": (
                "Probability calibration makes reported probabilities more honest; it does not "
                "make the model diagnostic. Conformal abstention flags cases the model cannot separate."
            ),
        },
    )

    return {"status": "SUCCESS", "task": "calibration", "artifact": str(path), "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit calibration and conformal artifacts")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Conformal miscoverage rate (default 0.10)")
    parser.add_argument(
        "--method",
        choices=["auto", "isotonic", "platt"],
        default="auto",
        help="Calibration family; 'auto' selects by measured ECE on an inner hold-out.",
    )
    args = parser.parse_args()
    try:
        result = fit_calibration(alpha=args.alpha, method=args.method)
    except MissingDatasetError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    print("Fit Calibration Result:", json.dumps(result, indent=2, default=str))
