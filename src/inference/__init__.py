"""
Inference Engine Package
========================

Provides decoupled, clinical-grade AI inference:
- run_ecg_inference
- get_model_artifacts
- ACTIVE_MODEL_ID
- PREPROCESSING_VERSION
"""

try:
    from .inference_engine import (
        ACTIVE_MODEL_ID,
        PREPROCESSING_VERSION,
        get_model_artifacts,
        run_ecg_inference,
    )
except ImportError:
    from inference.inference_engine import (
        ACTIVE_MODEL_ID,
        PREPROCESSING_VERSION,
        get_model_artifacts,
        run_ecg_inference,
    )

__all__ = [
    "run_ecg_inference",
    "get_model_artifacts",
    "ACTIVE_MODEL_ID",
    "PREPROCESSING_VERSION",
]
