"""
Comprehensive Statistical & Clinical Model Evaluator
===================================================
Calculates:
- Overall Accuracy & Balanced Accuracy
- Sensitivity (Recall) & Specificity per class
- Positive Predictive Value (PPV) & Negative Predictive Value (NPV)
- Weighted & Macro F1-Score
- Multi-class AUROC & AUPRC
- Expected Calibration Error (ECE)
- Confusion Matrix
Fulfills Phase 23 mandates without hiding poor-performing classes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def compute_expected_calibration_error(probabilities: np.ndarray, y_true_indices: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            ece += np.abs(avg_conf - avg_acc) * prop_in_bin

    return float(ece)


def evaluate_clinical_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    model_name: str = "ECG-Classifier",
) -> Dict[str, Any]:
    """Compute rich multi-class clinical and statistical performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Per-class sensitivity, specificity, PPV, NPV
    per_class_sens: Dict[str, float] = {}
    per_class_spec: Dict[str, float] = {}
    per_class_ppv: Dict[str, float] = {}
    per_class_npv: Dict[str, float] = {}

    for idx, c in enumerate(classes):
        tp = cm[idx][idx]
        fn = sum(cm[idx]) - tp
        fp = sum(row[idx] for row in cm) - tp
        tn = sum(sum(r) for r in cm) - (tp + fn + fp)

        per_class_sens[c] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class_spec[c] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        per_class_ppv[c] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        per_class_npv[c] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # AUROC / AUPRC
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1:
            auroc = float(roc_auc_score(y_true == classes[1], y_prob[:, 1]))
            auprc = float(average_precision_score(y_true == classes[1], y_prob[:, 1]))
        else:
            auroc = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted"))
            auprc = float(average_precision_score(y_bin, y_prob, average="weighted"))
    except Exception:
        auroc, auprc = 0.0, 0.0

    # Calibration
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_indices = np.array([label_to_idx.get(y, 0) for y in y_true])
    ece = compute_expected_calibration_error(y_prob, y_true_indices)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "expected_calibration_error": round(ece, 4),
        "sensitivity": per_class_sens,
        "specificity": per_class_spec,
        "positive_predictive_value": per_class_ppv,
        "negative_predictive_value": per_class_npv,
        "confusion_matrix": cm,
        "classes": classes,
    }
