"""
ECG Preprocessing Module
=========================

Implements preprocessing pipeline for ECG signals:
- Baseline drift correction (median filter)
- Bandpass filtering (Butterworth)
- Z-score normalization

Research/educational use only.
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def remove_baseline_wander(signal, kernel_size=121):
    """Remove baseline wander using median filter.

    Args:
        signal: 1D ECG signal
        kernel_size: Size of median filter kernel (default 121 samples ~337ms at 360Hz)

    Returns:
        Baseline-corrected signal
    """
    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline


def bandpass_filter(signal, fs, lowcut=5, highcut=15, order=4):
    """Apply Butterworth bandpass filter.

    Args:
        signal: 1D ECG signal
        fs: Sampling frequency (Hz)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)

    return filtered


def normalize_signal(signal, method="zscore"):
    """Normalize ECG signal.

    Args:
        signal: 1D ECG signal
        method: Normalization method ('zscore', 'minmax', 'mean')

    Returns:
        Normalized signal
    """
    if method == "zscore":
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            return (signal - mean) / std
        return signal - mean
    elif method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val > min_val:
            return (signal - min_val) / (max_val - min_val)
        return signal
    elif method == "mean":
        return signal - np.mean(signal)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_pipeline(signal, fs, lowcut=5, highcut=15, norm_method="zscore"):
    """Full preprocessing pipeline.

    Args:
        signal: Raw ECG signal (1D array)
        fs: Sampling frequency
        lowcut: Bandpass low cutoff (Hz)
        highcut: Bandpass high cutoff (Hz)
        norm_method: Normalization method

    Returns:
        Preprocessed signal
    """
    # Step 1: Remove baseline wander
    signal = remove_baseline_wander(signal)

    # Step 2: Bandpass filter
    signal = bandpass_filter(signal, fs, lowcut, highcut)

    # Step 3: Normalize
    signal = normalize_signal(signal, method=norm_method)

    return signal