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

from .inference.inference_engine import ACTIVE_MODEL_VERSION, run_ecg_ml_inference

__all__ = ["run_ecg_ml_inference", "ACTIVE_MODEL_VERSION"]
