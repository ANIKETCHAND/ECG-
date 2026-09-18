"""
Signal Quality Assessment Module
=====================================

Assesses ECG signal quality using multiple metrics:
- Signal-to-noise ratio (SNR)
- Baseline wander detection
- Powerline interference detection (50/60 Hz)
- Motion artifact detection
- Overall quality score

Research/educational use only.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from typing import Dict, List


def compute_snr(signal: np.ndarray, fs: float, noise_band: tuple[float, float] | None = None) -> float:
    """Estimate SNR using frequency band power ratio.

    Args:
        signal: Preprocessed ECG signal
        fs: Sampling frequency
        noise_band: Tuple of (low, high) Hz for noise band.
                    Default: (0-2 Hz) for baseline wander

    Returns:
        SNR estimate in dB
    """
    if noise_band is None:
        noise_band = (0, 2.0)

    # Compute PSD using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))

    # Signal band: 5-15 Hz (ECG main content)
    signal_mask = (freqs >= 5) & (freqs <= 15)
    signal_power = np.mean(psd[signal_mask]) if np.any(signal_mask) else 0.0

    # Noise band
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) and np.mean(psd[noise_mask]) > 0 else 1e-10

    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float("inf")

    return snr_db


def detect_baseline_wander(signal: np.ndarray, fs: float, threshold: float = 0.5) -> Dict[str, float]:
    """Detect baseline wander using low-frequency power.

    Args:
        signal: ECG signal
        fs: Sampling frequency
        threshold: PSD threshold for baseline wander detection

    Returns:
        Dict with 'has_baseline_wander', 'lf_power', 'quality_score'
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))

    # Low frequency power (0-2 Hz)
    lf_mask = freqs < 2.0
    lf_power = float(np.mean(psd[lf_mask])) if np.any(lf_mask) else 0.0

    has_wander = lf_power > threshold

    return {
        "has_baseline_wander": has_wander,
        "lf_power": lf_power,
    }


def detect_powerline_interference(signal: np.ndarray, fs: float,
                                   frequencies: list[float] | None = None) -> Dict[str, float]:
    """Detect powerline interference (50 or 60 Hz).

    Args:
        signal: ECG signal
        fs: Sampling frequency
        frequencies: Frequencies to check (default: [50, 60])

    Returns:
        Dict with interference detection results
    """
    if frequencies is None:
        frequencies = [50.0, 60.0]

    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))

    results = {}
    for freq in frequencies:
        # Find nearest frequency bin
        idx = np.argmin(np.abs(freqs - freq))
        power = float(psd[idx])
        results[f"power_{freq}hz"] = power
        results[f"has_{freq}hz_interference"] = power > 0.01

    return results


def detect_motion_artifacts(signal: np.ndarray, fs: float,
                              threshold: float = 3.0) -> Dict[str, float]:
    """Detect motion artifacts using signal variance analysis.

    Args:
        signal: ECG signal
        fs: Sampling frequency
        threshold: Z-score threshold for outlier detection

    Returns:
        Dict with artifact detection results
    """
    # Compute short-term variance
    window = int(0.5 * fs)  # 500ms windows
    n_windows = len(signal) // window

    variances = []
    for i in range(n_windows):
        segment = signal[i * window: (i + 1) * window]
        variances.append(np.var(segment))

    variances = np.array(variances)

    if len(variances) > 0 and np.std(variances) > 0:
        z_scores = np.abs((variances - np.mean(variances)) / np.std(variances))
        outlier_ratio = np.mean(z_scores > threshold)
    else:
        outlier_ratio = 0.0

    return {
        "outlier_ratio": outlier_ratio,
        "has_artifacts": outlier_ratio > 0.1,
    }


def assess_quality(signal: np.ndarray, fs: float = 360.0) -> Dict[str, float]:
    """Comprehensive signal quality assessment.

    Args:
        signal: Raw ECG signal
        fs: Sampling frequency (default 360 Hz for MIT-BIH)

    Returns:
        Dict with quality metrics and overall score (0-1)
    """
    # SNR
    snr = compute_snr(signal, fs)

    # Baseline wander
    bw_result = detect_baseline_wander(signal, fs)

    # Powerline interference
    pl_result = detect_powerline_interference(signal, fs)

    # Motion artifacts
    ma_result = detect_motion_artifacts(signal, fs)

    # Compute quality score (0-1)
    score = 1.0

    # SNR penalty
    if snr < 0:
        score *= 0.3
    elif snr < 10:
        score *= 0.6
    elif snr < 20:
        score *= 0.85

    # Baseline wander penalty
    if bw_result["has_baseline_wander"]:
        score *= 0.7

    # Powerline penalty
    if pl_result.get("has_50hz_interference", False) or pl_result.get("has_60hz_interference", False):
        score *= 0.8

    # Motion artifact penalty
    if ma_result["has_artifacts"]:
        score *= 0.6

    return {
        "snr_db": snr,
        "baseline_wander": bw_result["has_baseline_wander"],
        "lf_power": bw_result["lf_power"],
        "powerline_50hz_power": pl_result.get("power_50hz", 0.0),
        "powerline_60hz_power": pl_result.get("power_60hz", 0.0),
        "has_powerline_interference": pl_result.get("has_50hz_interference", False) or pl_result.get("has_60hz_interference", False),
        "artifact_outlier_ratio": ma_result["outlier_ratio"],
        "has_motion_artifacts": ma_result["has_artifacts"],
        "quality_score": max(0.0, min(1.0, score)),
    }


def assess_batch_quality(signals: list[np.ndarray], fs: float = 360.0) -> pd.DataFrame:
    """Assess quality for multiple signals.

    Args:
        signals: List of ECG signals
        fs: Sampling frequency

    Returns:
        DataFrame with quality metrics for each signal
    """
    results = [assess_quality(signal, fs) for signal in signals]
    return pd.DataFrame(results)