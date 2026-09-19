"""
Unit Tests for Decoupled Clinical AI Inference Engine
"""

from pathlib import Path
import numpy as np
import pytest
from src.ecg_core.models import ECGRecording
from src.inference.inference_engine import (
    ACTIVE_MODEL_ID,
    get_model_artifacts,
    run_ecg_inference,
)


def test_get_model_artifacts():
    clf, scaler, meta = get_model_artifacts()
    assert clf is not None
    assert scaler is not None
    assert hasattr(clf, "predict")
    assert hasattr(scaler, "transform")


def test_inference_engine_missing_fs():
    sig = np.sin(np.linspace(0, 10, 1000))
    res = run_ecg_inference(sig, fs=None)
    assert res.prediction == "PIPELINE_EXECUTION_FAILURE"
    assert any("sampling frequency" in w.lower() for w in res.warnings)


def test_inference_engine_unusable_signal():
    # Empty signal
    res = run_ecg_inference(np.array([]), fs=360.0)
    assert res.prediction == "PIPELINE_EXECUTION_FAILURE"
    assert any("empty" in w.lower() for w in res.warnings)


def test_inference_engine_synthetic_clean_ecg():
    fs = 360.0
    t = np.arange(fs * 5) / fs
    sig = 0.1 * np.sin(2 * np.pi * 1.2 * t)
    for i in range(1, 5):
        peak_idx = int(i * fs)
        sig[peak_idx - 5 : peak_idx + 5] += 1.2

    rec = ECGRecording(
        record_id="REC-TEST-INFER",
        sampling_rate=fs,
        duration=5.0,
        lead_names=["II"],
        number_of_leads=1,
        signals=sig,
    )
    res = run_ecg_inference(rec)
    assert res.analysis_id.startswith("ANL-")
    assert res.model_id == ACTIVE_MODEL_ID
    assert res.record_id == "REC-TEST-INFER"
    assert any(c in res.prediction for c in ("Normal", "PVC", "Other", "INDETERMINATE"))
    assert "Normal" in res.model_probabilities
    assert "PVC" in res.model_probabilities
    assert res.processing_time_ms >= 0.0


def test_inference_engine_unsupported_lead():
    fs = 360.0
    sig = np.ones((1, 1800))
    rec = ECGRecording(
        record_id="REC-NO-LEAD-II",
        sampling_rate=fs,
        duration=5.0,
        lead_names=["V5"],
        number_of_leads=1,
        signals=sig,
    )
    res = run_ecg_inference(rec, lead_to_analyze="II")
    assert res.prediction == "UNSUPPORTED_LEAD_CONFIGURATION"
    assert any("lead" in w.lower() for w in res.warnings)
