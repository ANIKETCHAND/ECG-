"""
Extraction Validator Module
===========================
Exports waveform extraction validation gate.
Guarantees anti-hallucination barriers across digitized images and PDF reports.
"""

from __future__ import annotations

try:
    from src.ecg_input.extraction_validation import validate_extracted_signal
except ImportError:
    from ecg_input.extraction_validation import validate_extracted_signal


__all__ = ["validate_extracted_signal"]
