"""
ML Model Management Subsystem
=============================
Loads serialized Random Forest and Baseline Logistic Regression model artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_model_artifacts() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load production classifier, scaler, and metadata.

    Returns:
        (classifier, scaler, metadata)
    """
    clf_path = MODELS_DIR / "classifier.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path = MODELS_DIR / "metadata.json"

    if not clf_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Model artifacts missing from {MODELS_DIR}.")

    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)

    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return clf, scaler, metadata


__all__ = ["load_model_artifacts", "MODELS_DIR"]
