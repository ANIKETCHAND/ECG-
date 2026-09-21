"""
ML Peak Detection Subsystem
===========================
Detects QRS R-peaks using physiologically constrained refractory periods.
"""

from __future__ import annotations

import numpy as np

try:
    from src.peak_detection import detect_r_peaks as _detect_peaks, validate_peaks
except ImportError:
    from peak_detection import detect_r_peaks as _detect_peaks, validate_peaks


def detect_r_peaks(signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
    """Detect R-peaks and return array of peak indices."""
    res = _detect_peaks(signal, fs, **kwargs)
    if isinstance(res, tuple):
        return np.asarray(res[0], dtype=int)
    return np.asarray(res, dtype=int)


__all__ = ["detect_r_peaks", "validate_peaks"]
