"""Unit tests for End-to-End Prediction Pipeline."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prediction import predict_ecg


def test_predict_ecg_valid_synthetic():
    fs = 360.0
    t = np.linspace(0, 5, int(5 * fs), endpoint=False)
    # Simulated ECG rhythm with R-peaks
    sig = np.sin(2 * np.pi * 1.2 * t)
    # Add sharp spikes for R-peaks every 0.8s
    for p in range(int(0.4 * fs), len(sig), int(0.8 * fs)):
        sig[p] = 3.0

    res = predict_ecg(sig, fs)

    assert "predicted_class" in res
    assert "probabilities" in res
    assert "signal_quality" in res
    assert res["signal_quality"] in ["GOOD", "ACCEPTABLE", "POOR"]
    assert "detected_peaks" in res
    assert "heart_rate_bpm" in res
    assert res["beat_count"] > 0


def test_predict_ecg_empty():
    with pytest.raises(ValueError):
        predict_ecg(np.array([]), 360.0)


def test_predict_ecg_too_short():
    with pytest.raises(ValueError):
        predict_ecg(np.ones(50), 360.0)  # Only 50 samples (< 0.5s)


def test_predict_ecg_with_nans():
    fs = 360.0
    t = np.linspace(0, 4, int(4 * fs), endpoint=False)
    sig = np.sin(2 * np.pi * 1.2 * t)
    for p in range(int(0.4 * fs), len(sig), int(0.8 * fs)):
        sig[p] = 3.0

    # Inject sporadic NaNs
    sig[100:105] = np.nan
    sig[300] = np.nan

    res = predict_ecg(sig, fs)
    assert "predicted_class" in res
    assert not np.isnan(res["quality_score"])
