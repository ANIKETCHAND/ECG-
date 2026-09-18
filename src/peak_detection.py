"""
R-Peak Detection Module
=======================

Implements R-peak detection using SciPy's find_peaks with adaptive thresholds.

Key Features:
- Adaptive threshold based on signal statistics
- Configurable distance parameter (refractory period)
- Prominence-based peak selection
- Validation using refractory period constraint

Research/educational use only.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(signal, fs, prominence=0.5, distance=None,
                   width=None, threshold=None):
    """Detect R-peaks in ECG signal.

    Uses SciPy's find_peaks with adaptive parameters.

    Args:
        signal: 1D ECG signal
        fs: Sampling frequency (Hz)
        prominence: Peak prominence for detection
        distance: Minimum distance between peaks (refractory period in samples)
                   Default: 0.5 * fs (~0.5s at 360Hz)
        width: Minimum peak width (not commonly used for R-peaks)
        threshold: Amplitude threshold

    Returns:
        Tuple of (peak_indices, peak_info_dict)
    """
    if distance is None:
        # Default: 0.5 seconds at given sampling rate
        distance = int(0.5 * fs)

    # Find peaks using prominence and distance
    peaks, properties = find_peaks(
        signal,
        prominence=prominence,
        distance=distance,
        width=width,
        height=threshold,
    )

    # Validate peaks: ensure minimum refractory period
    validated_peaks = validate_peaks(peaks, signal, fs)

    # Collect peak properties
    peak_info = {
        "indices": validated_peaks,
        "prominences": properties.get("prominences", np.array([])),
        "heights": properties.get("heights", np.array([])),
        "widths": properties.get("widths", np.array([])),
    }

    return validated_peaks, peak_info


def validate_peaks(peak_indices, signal, fs, refractory_period=None):
    """Validate R-peaks using refractory period constraint.

    Prevents double-detection by ensuring minimum time between peaks.

    Args:
        peak_indices: Initial peak detections
        signal: ECG signal
        fs: Sampling frequency
        refractory_period: Minimum time between beats in seconds

    Returns:
        Validated peak indices
    """
    if refractory_period is None:
        refractory_period = 0.3  # 300ms default

    if len(peak_indices) <= 1:
        return peak_indices

    valid = [peak_indices[0]]

    for peak in peak_indices[1:]:
        # Check distance from last valid peak
        if peak - valid[-1] >= refractory_period * fs:
            valid.append(peak)

    return np.array(valid)


def detect_r_peaks_multilead(ecg_data, fs, leads=None, **kwargs):
    """Detect R-peaks in multi-lead ECG.

    Args:
        ecg_data: 2D array of shape (n_leads, n_samples)
        fs: Sampling frequency
        leads: List of lead indices to process (default: all)
        **kwargs: Additional arguments passed to detect_r_peaks

    Returns:
        Dictionary mapping lead names to detected peaks
    """
    if leads is None:
        leads = list(range(ecg_data.shape[0]))

    results = {}
    for lead_idx in leads:
        lead_peaks, lead_info = detect_r_peaks(
            ecg_data[lead_idx, :], fs, **kwargs
        )
        results[f"lead_{lead_idx}"] = {
            "indices": lead_peaks,
            "info": lead_info,
        }

    return results