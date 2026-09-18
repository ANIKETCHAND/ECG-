"""Unit tests for Feature Extraction module."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_extraction import (
    extract_all_features,
    extract_ecg_specific_features,
    extract_frequency_domain_features,
    extract_time_domain_features,
)


@pytest.fixture
def sample_beat():
    fs = 360.0
    win_len = int(0.6 * fs) + 1  # 217 samples
    t = np.linspace(-0.2, 0.4, win_len)
    # Synthetic QRS: negative dip (Q), tall spike (R), negative dip (S), rounded bump (T)
    beat = 1.5 * np.exp(- (t / 0.03) ** 2) + 0.3 * np.exp(- ((t - 0.2) / 0.08) ** 2)
    return beat, fs


def test_extract_time_domain_features(sample_beat):
    beat, fs = sample_beat
    feats = extract_time_domain_features(beat, fs)

    assert "mean" in feats
    assert "std" in feats
    assert "rms" in feats
    assert "energy" in feats
    assert np.isfinite(feats["energy"])
    assert not np.isnan(feats["mean"])


def test_extract_ecg_specific_features(sample_beat):
    beat, fs = sample_beat
    feats = extract_ecg_specific_features(beat, fs)

    assert "r_peak_amplitude" in feats
    assert "peak_to_peak_amplitude" in feats
    assert "max_slope" in feats
    assert feats["r_peak_amplitude"] > 0


def test_extract_frequency_domain_features(sample_beat):
    beat, fs = sample_beat
    feats = extract_frequency_domain_features(beat, fs)

    assert "total_power" in feats
    assert "dominant_frequency" in feats
    assert "spectral_entropy" in feats
    assert feats["total_power"] > 0


def test_extract_all_features_robustness():
    fs = 360.0
    # Flat zero signal
    flat_beat = np.zeros(217)
    feats_flat = extract_all_features(flat_beat, fs)

    for k, v in feats_flat.items():
        assert not np.isnan(v), f"Feature {k} was NaN"
        assert not np.isinf(v), f"Feature {k} was Inf"

    # Constant signal
    const_beat = np.ones(217) * 5.0
    feats_const = extract_all_features(const_beat, fs)
    for k, v in feats_const.items():
        assert not np.isnan(v), f"Feature {k} was NaN"
        assert not np.isinf(v), f"Feature {k} was Inf"
