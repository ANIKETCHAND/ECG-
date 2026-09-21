"""
Configurable Preprocessing Pipeline for Multi-Dataset ECG Machine Learning.
Supports resampling, highpass/bandpass filtering, notch filtering,
and patient-leakage-free scaling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

from src.ecg_core.standardized_record import StandardizedECGRecord


class ECGPreprocessingPipeline:
    """Config-driven preprocessing pipeline operating on StandardizedECGRecord or raw ndarrays."""

    def __init__(self, config: Union[Dict[str, Any], str, Path]):
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = dict(config)

        self.pipeline_id = self.config.get("pipeline_id", "custom_pipeline")
        self.target_fs = float(self.config.get("target_fs", 360.0))
        self.resample_enabled = bool(self.config.get("resample", True))
        self.baseline_filter = self.config.get("baseline_filter")
        self.bandpass_filter_cfg = self.config.get("bandpass_filter")
        self.notch_filter_cfg = self.config.get("notch_filter")
        self.norm_method = self.config.get("normalization", "zscore")
        self.fitted_scaler_params: Optional[Dict[str, float]] = None

    def resample_signal(self, signal: np.ndarray, orig_fs: float) -> Tuple[np.ndarray, float]:
        """Resample 1D or multi-channel signal to target_fs."""
        if not self.resample_enabled or np.isclose(orig_fs, self.target_fs, atol=0.1):
            return signal.copy(), orig_fs

        num_samples = int(np.round(len(signal) * (self.target_fs / orig_fs)))
        if signal.ndim == 1:
            resampled = resample(signal, num_samples)
        else:
            resampled = resample(signal, num_samples, axis=0)
        return resampled, self.target_fs

    def apply_baseline_removal(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Butterworth highpass baseline wander removal."""
        if not self.baseline_filter:
            return signal
        cutoff = float(self.baseline_filter.get("cutoff_hz", 0.5))
        order = int(self.baseline_filter.get("order", 3))
        nyquist = 0.5 * fs
        normal_cutoff = max(1e-4, min(cutoff / nyquist, 0.99))
        b, a = butter(order, normal_cutoff, btype="highpass", analog=False)
        return filtfilt(b, a, signal, axis=0)

    def apply_bandpass(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Bandpass filter for muscular / high freq noise."""
        if not self.bandpass_filter_cfg:
            return signal
        low = float(self.bandpass_filter_cfg.get("low_cutoff_hz", 0.5))
        high = float(self.bandpass_filter_cfg.get("high_cutoff_hz", 40.0))
        order = int(self.bandpass_filter_cfg.get("order", 3))
        nyquist = 0.5 * fs
        low_norm = max(1e-4, low / nyquist)
        high_norm = min(0.99, high / nyquist)
        if low_norm >= high_norm:
            return signal
        b, a = butter(order, [low_norm, high_norm], btype="bandpass", analog=False)
        return filtfilt(b, a, signal, axis=0)

    def apply_notch(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Mains interference notch filter (50/60 Hz)."""
        if not self.notch_filter_cfg:
            return signal
        freq = float(self.notch_filter_cfg.get("freq_hz", 50.0))
        q = float(self.notch_filter_cfg.get("q", 30.0))
        nyquist = 0.5 * fs
        if freq >= nyquist:
            return signal
        b, a = iirnotch(freq / nyquist, q)
        return filtfilt(b, a, signal, axis=0)

    def fit_scaler(self, train_signals: List[np.ndarray]) -> None:
        """Fit scaler statistics exclusively on training cohort to prevent data leakage."""
        all_vals = np.concatenate([s.flatten() for s in train_signals if len(s) > 0])
        self.fitted_scaler_params = {
            "mean": float(np.mean(all_vals)),
            "std": float(np.std(all_vals)) if float(np.std(all_vals)) > 1e-8 else 1.0,
            "min": float(np.min(all_vals)),
            "max": float(np.max(all_vals)) if float(np.max(all_vals)) > float(np.min(all_vals)) else float(np.min(all_vals)) + 1.0,
        }

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal using configured method."""
        if self.norm_method == "zscore" or self.norm_method == "lead_wise_zscore":
            mean = np.mean(signal, axis=0)
            std = np.std(signal, axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            return (signal - mean) / std
        elif self.norm_method == "fitted_zscore" and self.fitted_scaler_params:
            return (signal - self.fitted_scaler_params["mean"]) / self.fitted_scaler_params["std"]
        elif self.norm_method == "minmax":
            smin = np.min(signal, axis=0)
            smax = np.max(signal, axis=0)
            denom = np.where((smax - smin) < 1e-8, 1.0, smax - smin)
            return (signal - smin) / denom
        return signal

    def transform_array(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
        """Process raw numpy array through the pipeline."""
        sig = signal.astype(np.float64)
        sig, current_fs = self.resample_signal(sig, fs)
        sig = self.apply_baseline_removal(sig, current_fs)
        sig = self.apply_notch(sig, current_fs)
        sig = self.apply_bandpass(sig, current_fs)
        sig = self.normalize(sig)
        return sig, current_fs

    def process_record(self, record: StandardizedECGRecord) -> StandardizedECGRecord:
        """Process a StandardizedECGRecord preserving patient metadata and provenance."""
        transformed_sig, final_fs = self.transform_array(record.signal, record.sampling_rate_hz)

        ratio = final_fs / record.sampling_rate_hz
        new_annotations = []
        for ann in record.annotations:
            new_sample = int(np.round(ann.sample_index * ratio))
            new_ann = ann.model_copy(update={"sample_index": new_sample, "time_seconds": float(new_sample / final_fs)})
            new_annotations.append(new_ann)

        return StandardizedECGRecord(
            record_id=f"{record.record_id}_prep",
            patient_id=record.patient_id,
            dataset_source=record.dataset_source,
            sampling_rate_hz=final_fs,
            leads=list(record.leads),
            signal=transformed_sig,
            annotations=new_annotations,
            metadata={
                **record.metadata,
                "preprocessing_pipeline": self.pipeline_id,
                "orig_sampling_rate_hz": record.sampling_rate_hz,
            },
        )
