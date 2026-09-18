"""Unit tests for R-Peak Detection module."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peak_detection import detect_r_peaks, validate_peaks


@pytest.fixture
def pulse_train():
    """Create a periodic pulse train simulating sharp R-peaks."""
    fs = 360.0
    duration = 5.0
    n_samples = int(duration * fs)
    sig = np.zeros(n_samples)

    # Place sharp spikes every 0.8s (~75 bpm)
    peak_locations = np.arange(int(0.5 * fs), n_samples, int(0.8 * fs))
    for p in peak_locations:
        sig[p] = 2.5
        sig[p - 1] = 1.0
        sig[p + 1] = 1.0

    return sig, fs, peak_locations


def test_detect_r_peaks(pulse_train):
    sig, fs, true_peaks = pulse_train

    detected_peaks, info = detect_r_peaks(sig, fs, prominence=0.5)

    assert len(detected_peaks) > 0
    assert len(detected_peaks) == len(true_peaks)
    assert np.allclose(detected_peaks, true_peaks, atol=2)
    assert "prominences" in info


def test_validate_peaks_refractory_period():
    fs = 360.0
    # Two peaks spaced by only 50ms (< refractory period of 300ms)
    false_peaks = np.array([100, 118, 500])

    validated = validate_peaks(false_peaks, None, fs, refractory_period=0.3)

    assert len(validated) == 2
    assert validated[0] == 100
    assert validated[1] == 500
