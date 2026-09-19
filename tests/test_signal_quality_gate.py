"""
Unit Tests for Mandatory Signal Quality Safety Gatekeeper
"""

import numpy as np
import pytest
from src.safety.signal_quality_gate import (
    QualityCategory,
    QualityGateResult,
    evaluate_signal_quality_gate,
)


def test_quality_gate_null_and_empty():
    res = evaluate_signal_quality_gate(None, 360.0)
    assert res.category == QualityCategory.UNUSABLE
    assert not res.can_run_ai
    assert not res.can_compute_measurements
    assert len(res.rejection_reasons) > 0

    res_empty = evaluate_signal_quality_gate(np.array([]), 360.0)
    assert res_empty.category == QualityCategory.UNUSABLE
    assert not res_empty.can_run_ai


def test_quality_gate_too_short():
    sig = np.sin(np.linspace(0, 10, 100))  # < 1 second at 360 Hz
    res = evaluate_signal_quality_gate(sig, 360.0, min_duration_sec=1.5)
    assert res.category == QualityCategory.UNUSABLE
    assert not res.can_run_ai
    assert any("duration" in r.lower() for r in res.rejection_reasons)


def test_quality_gate_nans_and_infinities():
    sig = np.ones(3600)
    sig[10:100] = np.nan
    res = evaluate_signal_quality_gate(sig, 360.0)
    assert res.category == QualityCategory.UNUSABLE
    assert not res.can_run_ai
    assert any("nan" in r.lower() for r in res.rejection_reasons)


def test_quality_gate_flatline():
    sig = np.ones(3600) * 2.5
    res = evaluate_signal_quality_gate(sig, 360.0)
    assert res.category == QualityCategory.UNUSABLE
    assert not res.can_run_ai
    assert any("amplitude" in r.lower() or "variance" in r.lower() or "flat" in r.lower() for r in res.rejection_reasons)


def test_quality_gate_clean_synthetic_signal():
    fs = 360.0
    t = np.arange(fs * 5) / fs
    # Clean simulated periodic signal with QRS-like spikes
    sig = 0.1 * np.sin(2 * np.pi * 1.2 * t)
    for i in range(1, 5):
        peak_idx = int(i * fs)
        sig[peak_idx - 5 : peak_idx + 5] += 1.2

    res = evaluate_signal_quality_gate(sig, fs)
    assert res.category in (QualityCategory.GOOD, QualityCategory.ACCEPTABLE)
    assert res.can_run_ai
    assert res.quality_score > 0.4
