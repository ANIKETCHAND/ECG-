"""
Baseline Wander Analysis Module
===============================
Quantifies low-frequency baseline drift caused by patient respiration or perspiration.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy import signal as scipy_signal


def analyze_baseline_wander(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float = 0.5,
) -> Tuple[bool, float, Dict[str, float]]:
    """Evaluate baseline wander severity using low-pass residual energy.

    Returns:
        (has_excessive_drift: bool, drift_ratio: float, metrics: dict)
    """
    if len(signal) < int(fs):
        return False, 0.0, {"drift_ratio": 0.0, "low_freq_power": 0.0}

    # Extract baseline drift via low-pass filter
    nyq = 0.5 * fs
    norm_cutoff = min(cutoff_hz / nyq, 0.49)

    b, a = scipy_signal.butter(2, norm_cutoff, btype="low")
    drift = scipy_signal.filtfilt(b, a, signal)

    total_var = float(np.var(signal)) + 1e-9
    drift_var = float(np.var(drift))

    drift_ratio = float(drift_var / total_var)
    has_excessive_drift = drift_ratio > 0.35

    metrics = {
        "drift_ratio": round(drift_ratio, 4),
        "drift_amplitude_peak_to_peak": round(float(np.ptp(drift)), 4),
        "drift_variance": round(drift_var, 6),
    }

    return has_excessive_drift, drift_ratio, metrics
