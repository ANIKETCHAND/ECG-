"""
Noise & Powerline Interference Detection Module
===============================================
Computes quantitative Signal-to-Noise Ratio (SNR), high-frequency noise,
and 50 Hz / 60 Hz electrical powerline interference harmonics.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy import signal as scipy_signal


def analyze_noise_and_powerline(
    signal: np.ndarray,
    fs: float,
) -> Tuple[float, bool, Dict[str, float]]:
    """Compute SNR (dB) and evaluate 50/60 Hz powerline interference.

    Returns:
        (snr_db: float, has_powerline: bool, metrics: dict)
    """
    n_samples = len(signal)
    if n_samples < int(fs):
        return 0.0, False, {"snr_db": 0.0, "powerline_50hz_ratio": 0.0, "powerline_60hz_ratio": 0.0}

    # Bandpass filter signal between 0.5 and 40 Hz as cardiac signal component
    nyq = 0.5 * fs
    low = max(0.5 / nyq, 0.001)
    high = min(40.0 / nyq, 0.99)

    b, a = scipy_signal.butter(3, [low, high], btype="bandpass")
    filtered = scipy_signal.filtfilt(b, a, signal)

    noise = signal - filtered
    sig_power = float(np.mean(filtered**2))
    noise_power = float(np.mean(noise**2)) + 1e-9

    snr_db = 10.0 * np.log10(sig_power / noise_power) if sig_power > 0 else -30.0

    # Power Spectral Density for powerline frequencies (50 Hz, 60 Hz)
    f, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))

    def get_band_power(f_center: float, half_bw: float = 1.0) -> float:
        mask = (f >= f_center - half_bw) & (f <= f_center + half_bw)
        return float(np.sum(psd[mask])) if np.any(mask) else 0.0

    total_power = float(np.sum(psd)) + 1e-9
    power_50 = get_band_power(50.0)
    power_60 = get_band_power(60.0)

    ratio_50 = power_50 / total_power
    ratio_60 = power_60 / total_power

    has_powerline = (ratio_50 > 0.15) or (ratio_60 > 0.15)

    metrics = {
        "snr_db": round(float(snr_db), 2),
        "powerline_50hz_ratio": round(ratio_50, 4),
        "powerline_60hz_ratio": round(ratio_60, 4),
        "high_freq_noise_power": round(noise_power, 6),
    }

    return float(snr_db), has_powerline, metrics
