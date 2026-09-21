"""
Unit Tests for Phase 3: ECG Quality Copilot & Artifact Analyzers
================================================================
Validates artifact, clipping, drift, and multi-lead copilot UI formatting.
"""

import numpy as np
import pytest

from src.ecg_core.models import ECGRecording
from src.quality.artifact_detection import detect_motion_and_muscle_artifacts
from src.quality.baseline_wander import analyze_baseline_wander
from src.quality.clipping_detection import detect_clipping_and_flatline
from src.quality.lead_quality import evaluate_single_lead_quality
from src.quality.noise_detection import analyze_noise_and_powerline
from src.quality.signal_quality import assess_ecg_copilot_quality


def test_clipping_detector():
    sig = np.sin(np.linspace(0, 10, 500))
    # Clip peaks to emulate ADC saturation
    sig[sig > 0.8] = 0.8
    sig[sig < -0.8] = -0.8
    is_clipped, is_flat, metrics = detect_clipping_and_flatline(sig)
    assert is_clipped is True
    assert metrics["clipping_ratio"] > 0.05


def test_baseline_wander_detector():
    fs = 360.0
    t = np.arange(int(fs * 5)) / fs
    # Inject large 0.1 Hz baseline drift
    sig = np.sin(2 * np.pi * 1.2 * t) + 3.0 * np.sin(2 * np.pi * 0.1 * t)
    has_drift, drift_ratio, metrics = analyze_baseline_wander(sig, fs)
    assert has_drift is True
    assert drift_ratio > 0.35


def test_powerline_detector():
    fs = 500.0
    t = np.arange(int(fs * 4)) / fs
    # Inject strong 50 Hz powerline hum
    sig = np.sin(2 * np.pi * 1.2 * t) + 1.5 * np.sin(2 * np.pi * 50.0 * t)
    snr, has_powerline, metrics = analyze_noise_and_powerline(sig, fs)
    assert has_powerline is True
    assert metrics["powerline_50hz_ratio"] > 0.15


def test_motion_artifact_detector():
    fs = 360.0
    sig = np.sin(np.linspace(0, 10, 500))
    # Inject extreme spike
    sig[100] = 50.0
    sig[101] = -50.0
    has_muscle, has_spikes, metrics = detect_motion_and_muscle_artifacts(sig, fs)
    assert has_spikes is True


def test_multilead_copilot_ui_formatter():
    fs = 360.0
    n = int(fs * 3)
    t = np.arange(n) / fs

    lead_ii = np.sin(2 * np.pi * 1.2 * t)  # Good
    lead_v1 = np.sin(2 * np.pi * 1.2 * t)  # Good
    lead_v2 = np.zeros(n)                  # Unusable flatline

    multilead_matrix = np.array([lead_ii, lead_v1, lead_v2])
    rec = ECGRecording(
        record_id="REC-COPILOT-001",
        sampling_rate=fs,
        duration=3.0,
        lead_names=["II", "V1", "V2"],
        number_of_leads=3,
        signals=multilead_matrix,
    )

    res = assess_ecg_copilot_quality(rec)
    assert res["overall_status"] == "UNUSABLE"
    assert res["can_run_ai"] is False
    assert "V2" in res["ui_text"]
    assert "Recommendation" in res["ui_text"]
    assert "Overall: UNUSABLE" in res["ui_text"]
