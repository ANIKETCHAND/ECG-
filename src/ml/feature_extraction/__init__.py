"""
ML Feature Extraction Subsystem
===============================
Extracts 28 time-domain, morphological, spectral, and R-R interval features.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from src.feature_extraction import (
        extract_all_features as _extract_single,
        extract_ecg_specific_features,
        extract_features_batch,
        extract_frequency_domain_features,
        extract_time_domain_features,
    )
except ImportError:
    from feature_extraction import (
        extract_all_features as _extract_single,
        extract_ecg_specific_features,
        extract_features_batch,
        extract_frequency_domain_features,
        extract_time_domain_features,
    )

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "energy",
    "rms",
    "mav",
    "snr",
    "zero_crossing_rate",
    "autocorr_first_peak",
    "r_peak_amplitude",
    "p_wave_amplitude",
    "t_wave_amplitude",
    "peak_to_peak_amplitude",
    "max_slope",
    "qrs_width_samples",
    "total_power",
    "lf_power",
    "hf_power",
    "vhf_power",
    "dominant_frequency",
    "max_power",
    "spectral_entropy",
    "pre_rr",
    "post_rr",
    "local_rr_ratio",
]


def extract_all_features(
    beats_or_beat: np.ndarray,
    fs: float,
    r_peaks: Optional[np.ndarray] = None,
    pre_rr: Optional[float] = None,
    post_rr: Optional[float] = None,
    local_rr_ratio: Optional[float] = None,
) -> Union[np.ndarray, Dict[str, float]]:
    """Extract features for single beat (returns dict) or multiple beats (returns 2D numpy array)."""
    arr = np.asarray(beats_or_beat)
    if arr.ndim == 1:
        return _extract_single(
            arr,
            fs,
            pre_rr=pre_rr,
            post_rr=post_rr,
            local_rr_ratio=local_rr_ratio,
        )

    if len(arr) == 0:
        return np.empty((0, len(FEATURE_NAMES)))

    batch_dict = extract_features_batch(arr, fs, r_peaks=r_peaks)
    cols = []
    for col in FEATURE_NAMES:
        if col in batch_dict:
            cols.append(batch_dict[col])
        else:
            cols.append(np.zeros(len(arr)))
    return np.column_stack(cols)


__all__ = [
    "FEATURE_NAMES",
    "extract_all_features",
    "extract_time_domain_features",
    "extract_ecg_specific_features",
    "extract_frequency_domain_features",
    "extract_features_batch",
]
