"""
Unit Tests for Phase 3: ECG Quality Gatekeeper
==============================================
Validates the hard safety interlock: NO RELIABLE INPUT = NO AI RESULT.
Verifies that UNUSABLE and POOR signals halt or suppress AI execution.
"""

import numpy as np
import pytest

from src.quality.quality_gate import QualityCategory, evaluate_ecg_quality_gate


def test_quality_gate_clean_signal_allows_ai():
    fs = 360.0
    t = np.arange(int(fs * 4)) / fs
    clean_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
    
    decision = evaluate_ecg_quality_gate(clean_ecg, fs, lead_name="II")
    assert decision.category in [QualityCategory.GOOD, QualityCategory.ACCEPTABLE]
    assert decision.can_run_ai is True
    assert decision.can_compute_measurements is True
    assert len(decision.rejection_reasons) == 0


def test_quality_gate_empty_signal_halts():
    decision = evaluate_ecg_quality_gate(np.array([]), 360.0)
    assert decision.category == QualityCategory.UNUSABLE
    assert decision.can_run_ai is False
    assert decision.can_compute_measurements is False
    assert any("empty" in r.lower() for r in decision.rejection_reasons)


def test_quality_gate_flatline_disconnect_halts():
    fs = 360.0
    flatline = np.zeros(int(fs * 5))  # 5 seconds of absolute 0V
    decision = evaluate_ecg_quality_gate(flatline, fs, lead_name="V1")
    assert decision.category == QualityCategory.UNUSABLE
    assert decision.can_run_ai is False
    assert decision.can_compute_measurements is False
    assert any("flatline" in r.lower() or "disconnected" in r.lower() for r in decision.rejection_reasons)


def test_quality_gate_too_short_halts():
    fs = 360.0
    too_short = np.sin(np.arange(100))  # 100 samples / 360 = 0.27s (< 1.5s)
    decision = evaluate_ecg_quality_gate(too_short, fs, min_duration_sec=1.5)
    assert decision.category == QualityCategory.UNUSABLE
    assert decision.can_run_ai is False
    assert "below minimum required" in decision.rejection_reasons[0]


def test_quality_gate_extreme_noise_suppresses_ai():
    fs = 360.0
    # Random white noise with zero physiological structure
    pure_noise = np.random.normal(0, 5.0, int(fs * 4))
    decision = evaluate_ecg_quality_gate(pure_noise, fs)
    assert decision.can_run_ai is False
    assert decision.category in [QualityCategory.POOR, QualityCategory.UNUSABLE]
