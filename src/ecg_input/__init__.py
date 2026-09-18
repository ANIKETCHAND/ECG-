"""
ECG Input Processing Package
============================

Unified ingestion pipeline supporting:
- Digital ECG formats: CSV, TXT, NPY
- ECG report images: JPG, JPEG, PNG, TIFF
- Clinical ECG documents: PDF

Research/educational use only.
"""

from ecg_input.input_detector import InputModality, detect_input_modality
from ecg_input.signal_loader import load_digital_signal, STANDARD_SAMPLING_RATES
from ecg_input.measurement_extractor import extract_report_measurements
from ecg_input.pdf_processor import process_pdf_report
from ecg_input.image_processor import process_ecg_image
from ecg_input.waveform_extractor import extract_waveform_from_image
from ecg_input.extraction_validation import validate_extracted_signal

__all__ = [
    "InputModality",
    "detect_input_modality",
    "load_digital_signal",
    "STANDARD_SAMPLING_RATES",
    "extract_report_measurements",
    "process_pdf_report",
    "process_ecg_image",
    "extract_waveform_from_image",
    "validate_extracted_signal",
]
