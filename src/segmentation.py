"""
Heartbeat Segmentation Module
=================================

Extracts heartbeat windows around detected R-peaks.

Handles edge cases (first/last beats) by padding with signal edges.

Research/educational use only.
"""

import numpy as np
from typing import Optional, Tuple, List


def extract_beats(signal: np.ndarray, r_peaks: np.ndarray,
                    fs: float, pre_window: float = 0.2,
                    post_window: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """Extract heartbeat windows around R-peaks.

    Args:
        signal: Preprocessed ECG signal
        r_peaks: Array of R-peak indices
        fs: Sampling frequency (Hz)
        pre_window: Seconds before R-peak to include (default 0.2s = 200ms)
        post_window: Seconds after R-peak to include (default 0.4s = 400ms)

    Returns:
        Tuple of (beats_array, valid_indices)
        beats_array: Shape (n_beats, window_size)
        valid_indices: Indices of successfully extracted beats
    """
    pre_samples = int(pre_window * fs)
    post_samples = int(post_window * fs)
    window_size = pre_samples + post_samples + 1  # +1 for the R-peak itself

    beats = []
    valid_indices = []

    for i, peak_idx in enumerate(r_peaks):
        start = peak_idx - pre_samples
        end = peak_idx + post_samples

        # Handle edge cases
        if start < 0:
            # Pad at beginning with first sample value
            padded = np.zeros(window_size)
            offset = abs(start)
            padded[offset:offset + len(signal[:end])] = signal[:end]
            beats.append(padded)
            valid_indices.append(i)
            continue

        if end >= len(signal):
            # Pad at end with last sample value
            padded = np.zeros(window_size)
            available = len(signal[start:])
            padded[:available] = signal[start:]
            beats.append(padded)
            valid_indices.append(i)
            continue

        # Normal case: extract window
        beat = signal[start:end + 1]
        beats.append(beat)
        valid_indices.append(i)

    return np.array(beats), np.array(valid_indices)


def extract_beat_segments(signal: np.ndarray, r_peaks: np.ndarray,
                            fs: float, window_size: int) -> np.ndarray:
    """Extract fixed-size beat segments.

    Args:
        signal: Preprocessed ECG signal
        r_peaks: Array of R-peak indices
        fs: Sampling frequency
        window_size: Size of segment in samples

    Returns:
        Array of shape (n_beats, window_size)
    """
    half_window = window_size // 2
    segments = []

    for peak_idx in r_peaks:
        start = max(0, peak_idx - half_window)
        end = min(len(signal), peak_idx + half_window)

        segment = signal[start:end]
        segments.append(segment)

    return np.array(segments)


def calculate_rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """Calculate RR intervals from R-peak indices.

    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency

    Returns:
        Array of RR intervals in seconds
    """
    if len(r_peaks) < 2:
        return np.array([])

    rr_samples = np.diff(r_peaks)
    rr_intervals = rr_samples / fs

    return rr_intervals


def estimate_heart_rate(r_peaks: np.ndarray, fs: float) -> float:
    """Estimate heart rate from R-peaks.

    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency

    Returns:
        Estimated heart rate in BPM
    """
    rr_intervals = calculate_rr_intervals(r_peaks, fs)

    if len(rr_intervals) == 0:
        return 0.0

    # Average RR interval
    avg_rr = np.mean(rr_intervals)

    if avg_rr > 0:
        return 60.0 / avg_rr

    return 0.0