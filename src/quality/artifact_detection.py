"""
Artifact & Motion Spike Detection Module
========================================
Detects muscle tremor (EMG) bursts, electrode motion artifacts, and sudden baseline jumps.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def detect_motion_and_muscle_artifacts(
    signal: np.ndarray,
    fs: float,
) -> Tuple[bool, bool, Dict[str, float]]:
    """Inspect signal for electromyographic (EMG) noise and electrode motion artifacts.

    Returns:
        (has_muscle_artifact: bool, has_motion_spikes: bool, metrics: dict)
    """
    if len(signal) < 20:
        return False, False, {"emg_energy_ratio": 0.0, "max_z_spike": 0.0}

    # 1. Motion spikes: extreme non-physiological sample-to-sample derivatives
    # Note: Normal QRS complexes have steep slopes (z ~ 5-10). Motion artifacts create huge outliers (z > 15).
    diffs = np.abs(np.diff(signal))
    std_diff = float(np.std(diffs)) + 1e-9
    max_diff = float(np.max(diffs))
    sig_std = float(np.std(signal)) + 1e-9

    z_diff = (diffs - np.mean(diffs)) / std_diff
    extreme_spikes = int(np.sum(z_diff > 18.0))
    has_motion_spikes = (extreme_spikes > 1) or (max_diff > 10.0 * sig_std)

    # 2. High-frequency muscle tremor (EMG): difference signal variance in >35 Hz band
    second_diff = np.abs(np.diff(diffs))
    emg_proxy = float(np.mean(second_diff))
    total_amp = float(np.std(signal)) + 1e-9

    emg_ratio = emg_proxy / total_amp
    has_muscle_artifact = emg_ratio > 0.45

    metrics = {
        "outlier_derivative_spikes": extreme_spikes,
        "max_derivative_jump": round(max_diff, 4),
        "emg_roughness_ratio": round(emg_ratio, 4),
    }

    return has_muscle_artifact, has_motion_spikes, metrics
