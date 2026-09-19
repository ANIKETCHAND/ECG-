"""
ECG Measurements Package
========================

Provides deterministic biomedical measurement algorithms:
- ClinicalECGMeasurements
- compute_ecg_measurements
"""

try:
    from .measurement_engine import (
        ClinicalECGMeasurements,
        compute_ecg_measurements,
    )
except ImportError:
    from measurements.measurement_engine import (
        ClinicalECGMeasurements,
        compute_ecg_measurements,
    )

__all__ = [
    "ClinicalECGMeasurements",
    "compute_ecg_measurements",
]
