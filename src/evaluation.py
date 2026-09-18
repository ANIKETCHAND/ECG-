"""
Model Evaluation Module
=======================

Computes classification metrics, confusion matrices, and generates
evaluation plots for ECG abnormality detection.

Research/educational use only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray | List[str],
    y_pred: np.ndarray | List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Ordered list of class labels

    Returns:
        Dictionary with overall and per-class metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class = {}
    for idx, cls_name in enumerate(labels):
        support = int(np.sum(y_true == cls_name))
        per_class[cls_name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
            "support": support,
        }

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": labels,
    }


def plot_confusion_matrix(
    cm: np.ndarray | List[List[int]],
    class_names: List[str],
    save_path: Optional[Path | str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> plt.Figure:
    """Plot confusion matrix with counts and optional percentages."""
    cm = np.asarray(cm)
    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = None

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 and cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize and cm_norm is not None:
                text = f"{val}\n({cm_norm[i, j]*100:.1f}%)"
            else:
                text = f"{val}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_class_distribution(
    y: np.ndarray | pd.Series | List[str],
    save_path: Optional[Path | str] = None,
    title: str = "Class Distribution",
) -> plt.Figure:
    """Plot bar chart of class counts."""
    series = pd.Series(y)
    counts = series.value_counts()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(counts.index, counts.values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(counts)])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Beat Class")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig
