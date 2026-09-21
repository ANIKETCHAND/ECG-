"""
Comprehensive Failure-Mode and Boundary Trapping Test Suite
===========================================================
Mandatory regulatory testing verifying the 10 Global Safety Rules:
1. NEVER convert failure into "Normal" (if error: prediction = "Normal" is FORBIDDEN).
2. NEVER generate synthetic ECG to replace missing patient data.
3. NEVER assume missing metadata (sampling rate, duration, leads).
4. NEVER run ML inference on unusable signals (flatline, severe clipping, excessive noise).
5. Unsupported leads MUST produce NO_RESULT_UNSUPPORTED_LEAD.
"""

import numpy as np
import pytest

from src.ecg_core.models import ECGRecording
from src.ml.inference.inference_engine import run_ecg_ml_inference
from src.quality.quality_gate import QualityCategory, evaluate_ecg_quality_gate


def test_failure_mode_missing_sampling_rate():
    sig = np.sin(np.linspace(0, 10, 500))
    # None sampling rate
    res = run_ecg_ml_inference(sig, fs=None, lead_to_analyze="II")
    assert "NO_RESULT" in res["prediction"]
    assert "MISSING_SAMPLING_RATE" in res["prediction"]
    assert res["prediction"] != "Normal Sinus Rhythm"
    assert res["model_probabilities"] == {}

    # Zero or negative sampling rate
    res_zero = run_ecg_ml_inference(sig, fs=0.0, lead_to_analyze="II")
    assert "NO_RESULT" in res_zero["prediction"]
    assert res_zero["prediction"] != "Normal Sinus Rhythm"


def test_failure_mode_unsupported_lead():
    sig = np.sin(np.linspace(0, 10, 500))
    res = run_ecg_ml_inference(sig, fs=360.0, lead_to_analyze="V4")
    assert "NO_RESULT" in res["prediction"]
    assert "UNSUPPORTED_LEAD" in res["prediction"]
    assert res["prediction"] != "Normal Sinus Rhythm"
    assert res["model_probabilities"] == {}


def test_failure_mode_flatline_electrode_disconnect():
    flatline = np.zeros(2000)
    res = run_ecg_ml_inference(flatline, fs=360.0, lead_to_analyze="II")
    assert "NO_RESULT" in res["prediction"]
    assert res["prediction"] != "Normal Sinus Rhythm"
    assert res["signal_quality"] == "UNUSABLE"


def test_failure_mode_adc_clipping_rail_saturation():
    # Signal saturated at 3.3V rail for 80% of window
    fs = 360.0
    sig = np.full(int(fs * 4), 3.3)
    sig[:100] = np.random.randn(100)  # slight variance at start
    res = run_ecg_ml_inference(sig, fs=fs, lead_to_analyze="II")
    assert "NO_RESULT" in res["prediction"]
    assert res["prediction"] != "Normal Sinus Rhythm"


def test_failure_mode_insufficient_signal_duration():
    # Only 0.2 seconds (72 samples at 360Hz)
    short_sig = np.random.randn(72)
    gate = evaluate_ecg_quality_gate(short_sig, fs=360.0)
    assert not gate.can_run_ai
    assert gate.category == QualityCategory.UNUSABLE
    assert any("duration" in r.lower() for r in gate.rejection_reasons)


def test_failure_mode_all_nans_and_infs():
    nan_sig = np.full(1000, np.nan)
    gate = evaluate_ecg_quality_gate(nan_sig, fs=360.0)
    assert not gate.can_run_ai
    assert gate.category == QualityCategory.UNUSABLE


def test_failure_mode_insufficient_r_peaks():
    # A single spike in 5 seconds cannot produce rhythm metrics
    fs = 360.0
    sig = np.zeros(int(fs * 5))
    sig[500] = 2.0  # Only 1 peak
    res = run_ecg_ml_inference(sig, fs=fs, lead_to_analyze="II")
    assert "NO_RESULT" in res["prediction"]
    assert "INSUFFICIENT_BEATS" in res["prediction"] or "UNUSABLE" in res["prediction"]
    assert res["prediction"] != "Normal Sinus Rhythm"
