"""
Task E — Signal Quality Gatekeeper Model Training Pipeline
==========================================================
Trains statistical classification model for pre-inference technical quality gating.
Output categories: GOOD, ACCEPTABLE, POOR, UNUSABLE.
Prevents corrupted or saturated signals from producing misleading diagnostic predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

MODELS_DIR = PROJ_DIR / "models"


def train_quality_model() -> dict:
    """Train Task E quality assessment model."""
    cand_dir = MODELS_DIR / "candidate"
    cand_dir.mkdir(parents=True, exist_ok=True)

    classes = ["GOOD", "ACCEPTABLE", "POOR", "UNUSABLE"]

    # Feature space: SNR (dB), clipping ratio, baseline wander ratio, powerline ratio, motion spikes
    np.random.seed(42)
    n_per_class = 150

    # GOOD: High SNR (>18dB), near-zero clipping, low wander, low powerline
    g_snr = np.random.normal(24.0, 3.0, n_per_class).clip(18.0, 40.0)
    g_clip = np.random.uniform(0.0, 0.001, n_per_class)
    g_wander = np.random.uniform(0.01, 0.08, n_per_class)
    g_powerline = np.random.uniform(0.01, 0.05, n_per_class)

    # ACCEPTABLE: Moderate SNR (10-18dB), minimal clipping, moderate wander
    a_snr = np.random.normal(14.0, 2.0, n_per_class).clip(10.0, 18.0)
    a_clip = np.random.uniform(0.001, 0.01, n_per_class)
    a_wander = np.random.uniform(0.08, 0.18, n_per_class)
    a_powerline = np.random.uniform(0.05, 0.15, n_per_class)

    # POOR: Low SNR (4-10dB), noticeable clipping, high wander or powerline
    p_snr = np.random.normal(7.0, 2.0, n_per_class).clip(3.0, 10.0)
    p_clip = np.random.uniform(0.01, 0.04, n_per_class)
    p_wander = np.random.uniform(0.18, 0.35, n_per_class)
    p_powerline = np.random.uniform(0.15, 0.30, n_per_class)

    # UNUSABLE: Severe clipping (>5%), flatline (SNR < 2dB), or extreme noise
    u_snr = np.random.normal(1.0, 1.5, n_per_class).clip(-5.0, 3.0)
    u_clip = np.random.uniform(0.05, 0.40, n_per_class)
    u_wander = np.random.uniform(0.35, 0.80, n_per_class)
    u_powerline = np.random.uniform(0.30, 0.70, n_per_class)

    X = np.vstack([
        np.column_stack([g_snr, g_clip, g_wander, g_powerline]),
        np.column_stack([a_snr, a_clip, a_wander, a_powerline]),
        np.column_stack([p_snr, p_clip, p_wander, p_powerline]),
        np.column_stack([u_snr, u_clip, u_wander, u_powerline]),
    ])
    y = np.array(
        ["GOOD"] * n_per_class +
        ["ACCEPTABLE"] * n_per_class +
        ["POOR"] * n_per_class +
        ["UNUSABLE"] * n_per_class
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_quality = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_quality.fit(X_scaled, y)

    joblib.dump(rf_quality, cand_dir / "quality_classifier.pkl")
    joblib.dump(scaler, cand_dir / "quality_scaler.pkl")

    return {
        "status": "SUCCESS",
        "task": "quality_gate",
        "model_path": str(cand_dir / "quality_classifier.pkl"),
        "classes": classes,
        "n_samples": len(X),
    }


if __name__ == "__main__":
    res = train_quality_model()
    print("Train Quality Model Result:", res)
