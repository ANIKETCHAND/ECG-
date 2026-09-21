"""
Clinical Performance Metrics Engine
===================================
Rigorous medical-device performance metrics conforming to clinical validation principles:
- Diagnostic Sensitivity (Recall, True Positive Rate)
- Diagnostic Specificity (True Negative Rate)
- Positive Predictive Value (PPV, Precision)
- Negative Predictive Value (NPV)
- F1-Score
- Area Under ROC Curve (AUROC)
- Area Under Precision-Recall Curve (AUPRC)
- Expected Calibration Error (ECE)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_binary_clinical_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """Calculate clinical metrics for binary classification (e.g. PVC vs Normal).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probability for the positive class (optional).
        pos_label: Positive class label (default: 1).

    Returns:
        Structured dictionary of clinical metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    # Confusion matrix elements
    # [[TN, FP], [FN, TP]]
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tp = int(np.sum((y_t == 1) & (y_p == 1)))

    total = len(y_t)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # Sensitivity / Recall
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Positive Predictive Value (Precision)
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Negative Predictive Value
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # F1-Score
    f1 = float(2 * (ppv * sensitivity) / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "total_samples": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "positive_predictive_value": round(ppv, 4),
        "negative_predictive_value": round(npv, 4),
        "f1_score": round(f1, 4),
        "auroc": None,
        "auprc": None,
    }

    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=float)
        try:
            if len(np.unique(y_t)) > 1:
                metrics["auroc"] = round(float(roc_auc_score(y_t, probs)), 4)
                prec, rec, _ = precision_recall_curve(y_t, probs)
                metrics["auprc"] = round(float(auc(rec, prec)), 4)
        except Exception:
            pass

    return metrics


def compute_expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) for probabilistic outputs."""
    y_t = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = float(np.mean(y_t[in_bin]))
            confidence_in_bin = float(np.mean(probs[in_bin]))
            diff = abs(accuracy_in_bin - confidence_in_bin)
            ece += diff * prop_in_bin
            bin_details.append({
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": int(np.sum(in_bin)),
                "accuracy": round(accuracy_in_bin, 4),
                "confidence": round(confidence_in_bin, 4),
                "gap": round(diff, 4),
            })

    return {
        "expected_calibration_error": round(float(ece), 4),
        "number_of_bins": n_bins,
        "bin_details": bin_details,
    }


def calculate_multiclass_clinical_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str],
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate clinical metrics for multi-class ECG classification."""
    y_t = np.asarray(y_true, dtype=str)
    y_p = np.asarray(y_pred, dtype=str)
    cls_list = list(classes)

    cm = confusion_matrix(y_t, y_p, labels=cls_list)
    total = len(y_t)
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class = {}
    for i, c in enumerate(cls_list):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total - tp - fn - fp)

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float(2 * (ppv * sens) / (ppv + sens)) if (ppv + sens) > 0 else 0.0

        per_class[c] = {
            "support": int(tp + fn),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "f1_score": round(f1, 4),
        }

    return {
        "overall_accuracy": round(accuracy, 4),
        "total_eval_samples": total,
        "classes": cls_list,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class,
    }


def evaluate_diagnostic_performance(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str],
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute complete diagnostic evaluation report."""
    results = calculate_multiclass_clinical_metrics(y_true, y_pred, classes, y_probs)
    return results
