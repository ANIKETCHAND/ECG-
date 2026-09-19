"""
ECG Core Package
================

Provides unified data models and foundational clinical abstractions:
- ECGRecording: Unified physiological ECG matrix
- ECGAnalysisResult: Structured analytical output
- ClinicianReview: Medical sign-off and review entity
"""

try:
    from .models import (
        STANDARD_12_LEADS,
        ClinicianReview,
        ECGAnalysisResult,
        ECGRecording,
    )
except ImportError:
    from ecg_core.models import (
        STANDARD_12_LEADS,
        ClinicianReview,
        ECGAnalysisResult,
        ECGRecording,
    )

__all__ = [
    "ECGRecording",
    "ECGAnalysisResult",
    "ClinicianReview",
    "STANDARD_12_LEADS",
]
