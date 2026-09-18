"""
Model Training Module
=====================

Phase 11 & 12: Train Random Forest and Baseline (Logistic Regression) models
on record-level split ECG dataset with zero data leakage.

Saves:
- models/classifier.pkl (Random Forest)
- models/baseline_classifier.pkl (Logistic Regression)
- models/scaler.pkl (StandardScaler fitted on train data only)
- models/metadata.json (Configuration, classes, feature names, record splits)

Research/educational use only.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJ_DIR = Path(__file__).parent.parent
DATA_DIR = PROJ_DIR / "data" / "processed"
MODELS_DIR = PROJ_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_models(
    train_path: Path = DATA_DIR / "train_dataset.csv",
    test_path: Path = DATA_DIR / "test_dataset.csv",
    split_info_path: Path = DATA_DIR / "split_info.json",
) -> None:
    """Train Random Forest and baseline models, saving artifacts to models/."""
    print("Loading train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    # Separate features and labels
    ignore_cols = {"label", "record_id", "symbol"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    classes = sorted(list(np.unique(y_train)))
    print(f"Features ({len(feature_cols)}): {feature_cols[:5]} ...")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Standard Scaler (fitted ONLY on train set)
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Random Forest Classifier
    print("\nTraining Random Forest Classifier (n_estimators=100, class_weight='balanced')...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)

    # 2. Baseline Logistic Regression Classifier
    print("Training Baseline Logistic Regression Classifier...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train_scaled, y_train)

    # Feature importances from Random Forest
    importances = dict(zip(feature_cols, [float(v) for v in rf_model.feature_importances_]))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Features by Random Forest Importance:")
    for feat_name, imp in top_features:
        print(f"  {feat_name:25s}: {imp:.4f}")

    # Save artifacts
    print("\nSaving model artifacts to models/...")
    joblib.dump(rf_model, MODELS_DIR / "classifier.pkl")
    joblib.dump(lr_model, MODELS_DIR / "baseline_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    metadata = {
        "model_name": "RandomForestClassifier",
        "baseline_model_name": "LogisticRegression",
        "dataset": "MIT-BIH Arrhythmia Database",
        "classes": classes,
        "feature_names": feature_cols,
        "sampling_rate_hz": 360,
        "window_pre_sec": 0.2,
        "window_post_sec": 0.4,
        "train_records": split_info.get("train", {}).get("records", []),
        "test_records": split_info.get("test", {}).get("records", []),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "feature_importances": importances,
        "created_at": datetime.datetime.now().isoformat(),
        "disclaimer": "Educational and research purposes only. Not a medical diagnostic device.",
    }

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model training and artifact serialization complete!")


if __name__ == "__main__":
    train_models()
