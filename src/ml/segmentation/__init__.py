"""
ML Segmentation Subsystem
=========================
Segments individual cardiac cycles and calculates dynamic R-R metrics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

try:
    from src.segmentation import calculate_rr_intervals, estimate_heart_rate, extract_beats as _extract_beats
except ImportError:
    from segmentation import calculate_rr_intervals, estimate_heart_rate, extract_beats as _extract_beats


def extract_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    pre_window: float = 0.2,
    post_window: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract beats and return (beats_array, valid_r_peaks)."""
    beats, valid_indices = _extract_beats(
        signal, r_peaks, fs, pre_window=pre_window, post_window=post_window
    )
    if len(valid_indices) > 0 and len(r_peaks) >= len(valid_indices):
        valid_peaks = r_peaks[valid_indices]
    else:
        valid_peaks = r_peaks
    return beats, valid_peaks


def compute_rr_intervals(r_peaks: np.ndarray, fs: float) -> Tuple[List[float], Optional[float]]:
    """Compute RR intervals in ms and estimated heart rate in bpm."""
    rr_sec = calculate_rr_intervals(r_peaks, fs)
    rr_ms = [float(r * 1000.0) for r in rr_sec]
    hr = estimate_heart_rate(r_peaks, fs) if len(rr_sec) > 0 else None
    return rr_ms, hr


__all__ = ["extract_beats", "compute_rr_intervals"]
