"""
Waveform Extraction Validation Module
=====================================

Validates whether an extracted waveform from an ECG image or PDF
has sufficient fidelity to be passed to the ML classifier.

Strictly protects against hallucinated or corrupted waveform inference.

Research/educational use only.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def validate_extracted_signal(
    signal: np.ndarray,
    fs: float,
    confidence_score: float,
    min_duration_sec: float = 1.5,
) -> Tuple[bool, str]:
    """Validate whether an extracted signal is clinically and mathematically usable.

    Args:
        signal: 1D extracted voltage array
        fs: Sampling frequency in Hz
        confidence_score: Confidence returned by waveform extractor
        min_duration_sec: Minimum required signal duration

    Returns:
        Tuple of (is_valid: bool, explanation: str)
    """
    if signal is None or len(signal) == 0:
        return False, "No waveform signal was extracted."

    if confidence_score < 0.60:
        return (
            False,
            "Waveform extraction confidence is too low for reliable AI analysis. "
            "The image trace appears blurry, fragmented, or obstructed by text/grid lines."
        )

    duration = len(signal) / float(fs)
    if duration < min_duration_sec:
        return (
            False,
            f"Extracted signal duration ({duration:.1f}s) is below the minimum required ({min_duration_sec}s)."
        )

    # Check for NaNs or Infs
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return False, "Extracted signal contains non-numeric or infinite artifacts."

    # Check dynamic range
    sig_std = float(np.std(signal))
    sig_range = float(np.ptp(signal))

    if sig_std < 0.05 or sig_range < 0.2:
        return (
            False,
            "Extracted signal lacks sufficient electrical variance (appears flat or silent)."
        )

    # Check for extreme outlier spikes (e.g. border artifacts)
    z_scores = np.abs((signal - np.mean(signal)) / (sig_std + 1e-9))
    if np.max(z_scores) > 25.0:
        return (
            False,
            "Extracted signal contains extreme synthetic border or clipping spikes."
        )

    return True, "Signal passed all fidelity and mathematical validation gates."
