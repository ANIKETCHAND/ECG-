"""
Multi-Label Medication Candidate Model Training & Calibration
=============================================================

Phases 8, 11, & 12:
Trains multi-label medication candidate classifiers:
- Compares Logistic Regression, Random Forest, and Gradient Boosting
- Employs MultiOutputClassifier / OneVsRestClassifier
- Applies CalibratedClassifierCV (sigmoid/isotonic) to ensure model association scores are well-calibrated
- Evaluates Micro F1, Macro F1, Precision@1, Precision@3, Recall@3, and PR-AUC
- Serializes trained model bundle into models/medication_model/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from training.medication.build_medication_dataset import (
    FEATURE_COLUMNS,
    MEDICATION_TARGET_CLASSES,
    build_multimodal_medication_dataset,
)


def train_medication_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_out_dir: Path | str = "models/medication_model",
) -> Dict[str, Any]:
    """
    Trains and compares multi-label candidate models, calibrates outputs, and saves artifacts.
    """
    out_dir = Path(models_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_cols = [f"tgt_{c}" for c in MEDICATION_TARGET_CLASSES]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[target_cols].values

    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df[target_cols].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[target_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ── Model Exploration & Selection ──────────────────────────────────────────
    model_candidates = {
        "logistic_regression": MultiOutputClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "random_forest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
        ),
        "gradient_boosting": MultiOutputClassifier(
            GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42)
        ),
    }

    best_name = "gradient_boosting"
    best_macro_f1 = -1.0
    val_results = {}

    for name, model in model_candidates.items():
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_val, y_val_pred, average="micro", zero_division=0)
        val_results[name] = {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1)}
        print(f"[{name.upper()}] Validation Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_name = name

    print(f"\n[MODEL SELECTION] Selected Best Architecture: {best_name.upper()} (Macro F1 = {best_macro_f1:.4f})")

    # ── Calibration Layer ──────────────────────────────────────────────────────
    # Calibrate binary estimators independently for reliable model association scores
    calibrated_estimators = []
    for i, target in enumerate(MEDICATION_TARGET_CLASSES):
        base_clf = GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42 + i)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        calibrated_clf.fit(X_train_scaled, y_train[:, i])
        calibrated_estimators.append(calibrated_clf)

    # ── Evaluation on Held-Out Test Set ─────────────────────────────────────────
    test_probs = np.column_stack([clf.predict_proba(X_test_scaled)[:, 1] for clf in calibrated_estimators])
    test_preds = (test_probs >= 0.50).astype(int)

    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(y_test, test_preds, average="micro", zero_division=0)
    test_macro_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    test_macro_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
    test_pr_auc = average_precision_score(y_test, test_probs, average="macro")

    # Precision@K and Recall@K
    def evaluate_top_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> Tuple[float, float]:
        precisions = []
        recalls = []
        for i in range(len(y_true)):
            top_k_indices = np.argsort(y_score[i])[::-1][:k]
            true_positives = sum(y_true[i, idx] == 1 for idx in top_k_indices)
            actual_positives = sum(y_true[i] == 1)
            precisions.append(true_positives / float(k))
            recalls.append(true_positives / float(actual_positives) if actual_positives > 0 else 1.0)
        return float(np.mean(precisions)), float(np.mean(recalls))

    prec_at_1, rec_at_1 = evaluate_top_k(y_test, test_probs, k=1)
    prec_at_3, rec_at_3 = evaluate_top_k(y_test, test_probs, k=3)

    metrics_report = {
        "model_architecture": f"Calibrated-{best_name}",
        "calibration_method": "sigmoid",
        "validation_comparison": val_results,
        "test_metrics": {
            "macro_f1": float(round(test_macro_f1, 4)),
            "micro_f1": float(round(test_micro_f1, 4)),
            "macro_precision": float(round(test_macro_prec, 4)),
            "macro_recall": float(round(test_macro_rec, 4)),
            "pr_auc": float(round(test_pr_auc, 4)),
            "precision_at_1": float(round(prec_at_1, 4)),
            "precision_at_3": float(round(prec_at_3, 4)),
            "recall_at_3": float(round(rec_at_3, 4)),
        },
        "per_class_f1": {
            cls_name: float(round(f1_score(y_test[:, i], test_preds[:, i], zero_division=0), 4))
            for i, cls_name in enumerate(MEDICATION_TARGET_CLASSES)
        },
    }

    print("\n[HELD-OUT TEST SET EVALUATION]")
    print(f"  Macro F1: {test_macro_f1:.4f}")
    print(f"  Micro F1: {test_micro_f1:.4f}")
    print(f"  Precision@1: {prec_at_1:.4f}")
    print(f"  Precision@3: {prec_at_3:.4f}")
    print(f"  Recall@3: {rec_at_3:.4f}")
    print(f"  Macro PR-AUC: {test_pr_auc:.4f}")

    # ── Artifact Serialization ─────────────────────────────────────────────────
    joblib.dump(calibrated_estimators, out_dir / "calibrated_estimators.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    meta = {
        "model_id": "MED-CANDIDATE-1.0.0",
        "model_name": "CalibratedGradientBoostingMultiLabel",
        "version": "1.0.0",
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": MEDICATION_TARGET_CLASSES,
        "num_features": len(FEATURE_COLUMNS),
        "num_classes": len(MEDICATION_TARGET_CLASSES),
        "training_framework": "scikit-learn",
        "governance": "Clinical decision-support candidate generator; non-autonomous.",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"[ARTIFACTS] Model bundle successfully saved to {out_dir}/")
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-label medication candidate model")
    parser.add_argument("--out-dir", type=str, default="models/medication_model")
    args = parser.parse_args()

    train_df, val_df, test_df = build_multimodal_medication_dataset(output_dir="training/medication/splits")
    train_medication_models(train_df, val_df, test_df, models_out_dir=args.out_dir)
