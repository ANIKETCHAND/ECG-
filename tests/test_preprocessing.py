"""Unit tests for ECG Preprocessing module."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    bandpass_filter,
    normalize_signal,
    preprocess_pipeline,
    remove_baseline_wander,
)


@pytest.fixture
def synthetic_ecg():
    """Generate a clean synthetic ECG-like test signal."""
    fs = 360.0
    t = np.linspace(0, 5, int(5 * fs), endpoint=False)
    # Sine wave + harmonics mimicking ECG
    sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 10 * t)
    return sig, fs


def test_remove_baseline_wander(synthetic_ecg):
    sig, fs = synthetic_ecg
    # Add strong low frequency drift (0.2 Hz)
    t = np.linspace(0, 5, len(sig))
    drift = 2.0 * np.sin(2 * np.pi * 0.2 * t)
    corrupted = sig + drift

    corrected = remove_baseline_wander(corrupted, kernel_size=121)

    assert len(corrected) == len(corrupted)
    assert not np.isnan(corrected).any()
    assert not np.isinf(corrected).any()
    # Corrected signal should have much lower low-frequency energy
    assert np.std(corrected) < np.std(corrupted)


def test_bandpass_filter(synthetic_ecg):
    sig, fs = synthetic_ecg
    # Add high frequency noise (50 Hz)
    t = np.linspace(0, 5, len(sig))
    noise = 0.5 * np.sin(2 * np.pi * 50 * t)
    noisy = sig + noise

    filtered = bandpass_filter(noisy, fs, lowcut=5, highcut=15, order=4)

    assert len(filtered) == len(noisy)
    assert not np.isnan(filtered).any()
    assert not np.isinf(filtered).any()


def test_normalize_signal(synthetic_ecg):
    sig, _ = synthetic_ecg

    # Z-score normalization
    norm_z = normalize_signal(sig, method="zscore")
    assert np.isclose(np.mean(norm_z), 0.0, atol=1e-3)
    assert np.isclose(np.std(norm_z), 1.0, atol=1e-3)

    # Min-max normalization
    norm_mm = normalize_signal(sig, method="minmax")
    assert np.isclose(np.min(norm_mm), 0.0, atol=1e-3)
    assert np.isclose(np.max(norm_mm), 1.0, atol=1e-3)


def test_preprocess_pipeline(synthetic_ecg):
    sig, fs = synthetic_ecg
    processed = preprocess_pipeline(sig, fs)

    assert len(processed) == len(sig)
    assert not np.isnan(processed).any()
    assert not np.isinf(processed).any()
    assert np.isclose(np.mean(processed), 0.0, atol=1e-2)
