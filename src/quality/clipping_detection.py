"""
Clipping & Flatline Detection Module
====================================
Detects ADC rail saturation, amplifier clipping, flatline disconnection, and zero-variance segments.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def detect_clipping_and_flatline(
    signal: np.ndarray,
    clipping_margin: float = 0.02,
    max_flatline_samples: int = 50,
) -> Tuple[bool, bool, Dict[str, float]]:
    """Inspect signal for amplifier clipping and flatline disconnections.

    Returns:
        (is_clipped: bool, is_flatline: bool, metrics: dict)
    """
    if len(signal) == 0:
        return False, True, {"flatline_ratio": 1.0, "clipping_ratio": 0.0}

    sig_min = float(np.min(signal))
    sig_max = float(np.max(signal))
    sig_range = sig_max - sig_min

    if sig_range < 1e-4:
        return False, True, {"flatline_ratio": 1.0, "clipping_ratio": 0.0, "consecutive_flat_samples": len(signal)}

    # Flatline detection: count longest run of consecutive identical/near-identical samples
    diffs = np.abs(np.diff(signal))
    zero_diffs = diffs < (sig_range * 1e-4)

    max_consecutive_flat = 0
    current_flat = 0
    for zd in zero_diffs:
        if zd:
            current_flat += 1
            if current_flat > max_consecutive_flat:
                max_consecutive_flat = current_flat
        else:
            current_flat = 0

    is_flatline = max_consecutive_flat >= max_flatline_samples
    flatline_ratio = float(np.mean(zero_diffs))

    # Clipping detection: count percentage of samples near min or max rail
    threshold_low = sig_min + (sig_range * clipping_margin)
    threshold_high = sig_max - (sig_range * clipping_margin)

    clipped_low = np.sum(signal <= threshold_low)
    clipped_high = np.sum(signal >= threshold_high)
    clipping_ratio = float(clipped_low + clipped_high) / float(len(signal))

    is_clipped = clipping_ratio > 0.05

    metrics = {
        "sig_min": sig_min,
        "sig_max": sig_max,
        "sig_range": sig_range,
        "clipping_ratio": round(clipping_ratio, 4),
        "flatline_ratio": round(flatline_ratio, 4),
        "max_consecutive_flat": max_consecutive_flat,
    }

    return is_clipped, is_flatline, metrics
