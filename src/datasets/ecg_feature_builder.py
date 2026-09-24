"""
ECG Feature Builder
===================

Phase 7:
Builds standardized 28-dimensional electrophysiological feature vectors
from raw or segmented ECG waveforms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from src.feature_extraction import (
    extract_all_features,
    extract_ecg_specific_features,
    extract_frequency_domain_features,
    extract_time_domain_features,
)
from src.peak_detection import detect_r_peaks
from src.segmentation import extract_beats

ECG_FEATURE_NAMES = [
    "mean", "std", "min", "max", "range", "median", "energy", "rms", "mav", "snr",
    "zero_crossing_rate", "autocorr_first_peak", "r_peak_amplitude", "p_wave_amplitude",
    "t_wave_amplitude", "peak_to_peak_amplitude", "max_slope", "qrs_width_samples",
    "total_power", "lf_power", "hf_power", "vhf_power", "dominant_frequency", "max_power",
    "spectral_entropy", "pre_rr", "post_rr", "local_rr_ratio",
]


class ECGFeatureBuilder:
    """Extracts standardized 28-dimensional feature representations from ECG data."""

    def __init__(self, fs: float = 360.0):
        self.fs = fs
        self.feature_names = list(ECG_FEATURE_NAMES)

    def extract_features_from_beat(
        self,
        beat: np.ndarray,
        pre_rr: Optional[float] = None,
        post_rr: Optional[float] = None,
        local_rr_ratio: Optional[float] = None,
    ) -> Dict[str, float]:
        return extract_all_features(
            beat,
            self.fs,
            pre_rr=pre_rr,
            post_rr=post_rr,
            local_rr_ratio=local_rr_ratio,
        )

    def extract_features_from_signal(
        self,
        signal: np.ndarray,
        r_peaks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Processes continuous signal, detects peaks, segments beats, and extracts feature matrix."""
        if r_peaks is None:
            r_peaks = detect_r_peaks(signal, self.fs)

        if len(r_peaks) == 0:
            return {
                "r_peaks": [],
                "feature_matrix": np.empty((0, len(self.feature_names))),
                "feature_names": self.feature_names,
            }

        beats, valid_indices = extract_beats(signal, r_peaks, self.fs)
        valid_peaks = r_peaks[valid_indices] if len(valid_indices) > 0 else np.array([], dtype=int)
        if len(valid_peaks) == 0:
            return {
                "r_peaks": list(r_peaks),
                "feature_matrix": np.empty((0, len(self.feature_names))),
                "feature_names": self.feature_names,
            }

        # Calculate RR intervals
        r_peak_times = valid_peaks / self.fs
        rr_intervals = np.diff(r_peak_times)
        mean_rr = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8

        feature_rows = []
        for i, (beat, peak) in enumerate(zip(beats, valid_peaks)):
            pre_rr = float(rr_intervals[i - 1]) if i > 0 else mean_rr
            post_rr = float(rr_intervals[i]) if i < len(rr_intervals) else mean_rr
            
            ratio = pre_rr / mean_rr if mean_rr > 0 else 1.0
            f_dict = self.extract_features_from_beat(
                beat,
                pre_rr=pre_rr,
                post_rr=post_rr,
                local_rr_ratio=ratio,
            )
            feature_rows.append([f_dict[k] for k in self.feature_names])

        return {
            "r_peaks": list(valid_peaks),
            "feature_matrix": np.array(feature_rows, dtype=np.float32),
            "feature_names": self.feature_names,
        }


GLOBAL_ECG_FEATURE_BUILDER = ECGFeatureBuilder()
