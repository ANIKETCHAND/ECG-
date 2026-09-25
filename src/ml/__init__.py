"""
ECG GUARDIAN Machine Learning Package
=====================================
Modular decoupled ML pipeline:
- preprocessing
- peak_detection
- segmentation
- feature_extraction
- models
- inference
- validation
"""

from .calibration import (
    MondrianConformalClassifier,
    MulticlassCalibrator,
    apply_calibration,
    load_calibration,
    save_calibration,
)
from .inference.inference_engine import ACTIVE_MODEL_VERSION, run_ecg_ml_inference
from .selective import (
    INDETERMINATE,
    OperatingPoint,
    apply_gate,
    load_operating_point,
    operating_curve,
    record_wise_jackknife,
    save_operating_point,
    select_operating_point,
    summarise as summarise_operating_point,
    ungated,
)
from .shadow import ShadowComparisonReport, run_shadow_comparison

__all__ = [
    "run_ecg_ml_inference",
    "ACTIVE_MODEL_VERSION",
    "MulticlassCalibrator",
    "MondrianConformalClassifier",
    "apply_calibration",
    "save_calibration",
    "load_calibration",
    "run_shadow_comparison",
    "ShadowComparisonReport",
    # Selective prediction / abstention
    "INDETERMINATE",
    "OperatingPoint",
    "apply_gate",
    "ungated",
    "operating_curve",
    "select_operating_point",
    "record_wise_jackknife",
    "save_operating_point",
    "load_operating_point",
    "summarise_operating_point",
]
