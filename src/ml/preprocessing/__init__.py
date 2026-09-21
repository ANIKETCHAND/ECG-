"""
ML Preprocessing Subsystem
==========================
Filters baseline drift and high-frequency noise.
"""

from __future__ import annotations

try:
    from src.preprocessing import (
        bandpass_filter,
        normalize_signal,
        preprocess_pipeline,
        remove_baseline_wander,
    )
except ImportError:
    from preprocessing import (
        bandpass_filter,
        normalize_signal,
        preprocess_pipeline,
        remove_baseline_wander,
    )

from src.ml.preprocessing.pipeline import ECGPreprocessingPipeline

__all__ = [
    "preprocess_pipeline",
    "remove_baseline_wander",
    "bandpass_filter",
    "normalize_signal",
    "ECGPreprocessingPipeline",
]
