"""Unit tests for Heartbeat Segmentation module."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from segmentation import (
    calculate_rr_intervals,
    estimate_heart_rate,
    extract_beats,
)


def test_extract_beats():
    fs = 360.0
    sig = np.random.randn(3600)  # 10s of signal
    peaks = np.array([200, 500, 800, 3550])  # Includes an edge peak near end

    pre_win = 0.2   # 72 samples
    post_win = 0.4  # 144 samples
    expected_len = int(pre_win * fs) + int(post_win * fs) + 1  # 217 samples

    beats, valid_indices = extract_beats(sig, peaks, fs, pre_window=pre_win, post_window=post_win)

    assert len(beats) == len(peaks)
    assert beats.shape[1] == expected_len
    assert len(valid_indices) == len(peaks)
    # Check no NaNs
    assert not np.isnan(beats).any()


def test_rr_intervals_and_heart_rate():
    fs = 360.0
    # Regular 1-second intervals (60 BPM)
    peaks = np.array([360, 720, 1080, 1440])

    rr = calculate_rr_intervals(peaks, fs)
    assert len(rr) == 3
    assert np.allclose(rr, [1.0, 1.0, 1.0])

    hr = estimate_heart_rate(peaks, fs)
    assert np.isclose(hr, 60.0, atol=0.1)


def test_empty_peaks_handling():
    fs = 360.0
    peaks = np.array([])

    rr = calculate_rr_intervals(peaks, fs)
    assert len(rr) == 0

    hr = estimate_heart_rate(peaks, fs)
    assert hr == 0.0
