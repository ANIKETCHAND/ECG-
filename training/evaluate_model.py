"""
Model Evaluation Script
=======================

Phase 12 & 13: Comprehensive evaluation of Random Forest vs Baseline Logistic Regression
on unseen test patient records.

Generates:
- reports/evaluation_report.json
- reports/figures/confusion_matrix.png
- reports/figures/confusion_matrix_baseline.png
- reports/figures/class_distribution.png
- reports/figures/feature_importance.png

Research/educational use only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import compute_metrics, plot_class_distribution, plot_confusion_matrix

PROJ_DIR = Path(__file__).parent.parent
DATA_DIR = PROJ_DIR / "data" / "processed"
MODELS_DIR = PROJ_DIR / "models"
REPORTS_DIR = PROJ_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate() -> None:
    print("Loading test data and trained models...")
    test_df = pd.read_csv(DATA_DIR / "test_dataset.csv")
    with open(MODELS_DIR / "metadata.json", "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_names"]
    classes = meta["classes"]

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    rf_clf = joblib.load(MODELS_DIR / "classifier.pkl")
    lr_clf = joblib.load(MODELS_DIR / "baseline_classifier.pkl")

    X_test_scaled = scaler.transform(X_test)

    # 1. Random Forest Predictions
    y_pred_rf = rf_clf.predict(X_test_scaled)
    metrics_rf = compute_metrics(y_test, y_pred_rf, labels=classes)

    # 2. Baseline Logistic Regression Predictions
    y_pred_lr = lr_clf.predict(X_test_scaled)
    metrics_lr = compute_metrics(y_test, y_pred_lr, labels=classes)

    print("\n" + "=" * 60)
    print("MODEL 1: RANDOM FOREST (Primary Classifier)")
    print("=" * 60)
    print(f"Accuracy         : {metrics_rf['accuracy'] * 100:.2f}%")
    print(f"Macro Precision  : {metrics_rf['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall     : {metrics_rf['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score   : {metrics_rf['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1-Score: {metrics_rf['f1_weighted'] * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cls, vals in metrics_rf["per_class"].items():
        print(f"  {cls:8s} | Prec: {vals['precision']*100:5.1f}% | Rec: {vals['recall']*100:5.1f}% | F1: {vals['f1_score']*100:5.1f}% | Support: {vals['support']}")

    print("\n" + "=" * 60)
    print("MODEL 2: LOGISTIC REGRESSION (Baseline Classifier)")
    print("=" * 60)
    print(f"Accuracy         : {metrics_lr['accuracy'] * 100:.2f}%")
    print(f"Macro Precision  : {metrics_lr['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall     : {metrics_lr['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score   : {metrics_lr['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1-Score: {metrics_lr['f1_weighted'] * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cls, vals in metrics_lr["per_class"].items():
        print(f"  {cls:8s} | Prec: {vals['precision']*100:5.1f}% | Rec: {vals['recall']*100:5.1f}% | F1: {vals['f1_score']*100:5.1f}% | Support: {vals['support']}")
    print("=" * 60)

    # Save figures
    print("\nSaving evaluation plots...")
    plot_confusion_matrix(
        metrics_rf["confusion_matrix"],
        classes,
        save_path=FIGURES_DIR / "confusion_matrix.png",
        title="Random Forest Confusion Matrix",
        normalize=True,
    )
    plot_confusion_matrix(
        metrics_lr["confusion_matrix"],
        classes,
        save_path=FIGURES_DIR / "confusion_matrix_baseline.png",
        title="Logistic Regression Confusion Matrix",
        normalize=True,
    )
    plot_class_distribution(
        y_test,
        save_path=FIGURES_DIR / "class_distribution.png",
        title="Test Set Class Distribution (Unseen Records)",
    )

    # Feature importance plot
    importances = meta.get("feature_importances", {})
    sorted_imps = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_imps = sorted_imps[::-1]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.barh([x[0] for x in sorted_imps], [x[1] for x in sorted_imps], color="#2b5c8f")
    ax.set_title("Random Forest Top 10 Feature Importances")
    ax.set_xlabel("Gini Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    # Save complete evaluation report JSON
    report = {
        "random_forest": metrics_rf,
        "logistic_regression": metrics_lr,
        "test_records": meta.get("test_records", []),
        "test_samples": len(test_df),
        "test_distribution": pd.Series(y_test).value_counts().to_dict(),
    }
    with open(REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"All evaluation results and figures saved to {REPORTS_DIR}")


if __name__ == "__main__":
    evaluate()
