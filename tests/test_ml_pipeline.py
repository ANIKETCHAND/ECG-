"""
Unit Tests for Phase 4: Refactored ML Pipeline
==============================================
Tests independent, UI-decoupled inference engine, schema compliance,
and fail-safe traps for unsupported leads, missing sampling rates, and poor signals.
"""

import numpy as np
import pytest

from src.ecg_core.models import ECGRecording
from src.ml.inference.inference_engine import ACTIVE_MODEL_VERSION, run_ecg_ml_inference


def test_ml_inference_valid_synthetic_signal():
    fs = 360.0
    duration = 5.0
    t = np.arange(int(fs * duration)) / fs

    # Synthetic periodic ECG trace
    sig = 0.08 * np.sin(2 * np.pi * 1.2 * t)
    for i in range(1, int(duration * 1.2)):
        idx = int(i * (fs / 1.2))
        w = 6
        start = max(0, idx - w)
        end = min(len(sig), idx + w)
        sig[start:end] += 1.2 * np.exp(-0.5 * ((np.arange(start, end) - idx) / 2.5) ** 2)

    res = run_ecg_ml_inference(sig, fs=fs, lead_to_analyze="II")
    assert isinstance(res, dict)
    assert res["model_version"] == ACTIVE_MODEL_VERSION
    assert res["prediction"] in ["Normal Sinus Rhythm", "Premature Ventricular Contraction"]
    assert "Normal" in res["model_probabilities"] or "PVC" in res["model_probabilities"]
    assert res["signal_quality"] in ["GOOD", "ACCEPTABLE"]
    assert res["heart_rate"] is not None
    assert len(res["r_peaks"]) >= 2
    assert len(res["rr_intervals"]) >= 1
    assert res["processing_time_ms"] > 0
    assert len(res["limitations"]) > 0


def test_ml_inference_unsupported_lead():
    # Lead V5 is not validated for active single-lead model
    sig = np.sin(np.linspace(0, 10, 500))
    res = run_ecg_ml_inference(sig, fs=360.0, lead_to_analyze="V5")
    assert "UNSUPPORTED_LEAD" in res["prediction"]
    assert res["model_probabilities"] == {}
    assert res["heart_rate"] is None


def test_ml_inference_missing_sampling_rate():
    # Rule 4: Must fail safe if fs is None or <= 0
    sig = np.sin(np.linspace(0, 10, 500))
    res = run_ecg_ml_inference(sig, fs=None, lead_to_analyze="II")
    assert "MISSING_SAMPLING_RATE" in res["prediction"]
    assert res["heart_rate"] is None


def test_ml_inference_flatline_halted_by_quality_gate():
    # Quality gatekeeper must halt inference on unusable input
    flatline = np.zeros(1000)
    res = run_ecg_ml_inference(flatline, fs=360.0, lead_to_analyze="II")
    assert "NO_RESULT" in res["prediction"]
    assert res["signal_quality"] == "UNUSABLE"
    assert res["model_probabilities"] == {}


def test_ml_inference_ecg_recording_object():
    fs = 360.0
    sig = np.sin(2 * np.pi * 1.2 * np.arange(int(fs * 4)) / fs)
    rec = ECGRecording(
        record_id="REC-TEST-ML",
        sampling_rate=fs,
        duration=4.0,
        lead_names=["II"],
        number_of_leads=1,
        signals=sig,
    )
    res = run_ecg_ml_inference(rec)
    assert res["analysis_id"].startswith("ANL-")
    assert res["lead_analyzed"] == "II"
