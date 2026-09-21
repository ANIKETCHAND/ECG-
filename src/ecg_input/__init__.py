"""
ECG Input Processing Package
============================

Unified ingestion pipeline supporting:
- Digital ECG formats: CSV, TXT, NPY, EDF, XML, JSON, DICOM, WFDB
- ECG report images: JPG, JPEG, PNG, TIFF, BMP
- Clinical ECG documents: PDF
"""

from .input_detector import InputModality, detect_input_modality
from .signal_loader import load_digital_signal, STANDARD_SAMPLING_RATES, load_any_ecg
from .csv_loader import load_csv_ecg
from .txt_loader import load_txt_ecg
from .npy_loader import load_npy_ecg
from .json_loader import load_json_ecg
from .edf_loader import load_edf_ecg
from .xml_loader import load_xml_ecg
from .dicom_loader import load_dicom_ecg
from .wfdb_loader import load_wfdb_record
from .measurement_extractor import extract_report_measurements
from .pdf_processor import process_pdf_report
from .image_processor import process_ecg_image
from .waveform_extractor import extract_waveform_from_image
from .extraction_validation import validate_extracted_signal
from .metadata_extractor import (
    extract_patient_demographics,
    extract_technical_calibrations,
    extract_machine_statement,
)

__all__ = [
    "InputModality",
    "detect_input_modality",
    "load_digital_signal",
    "load_any_ecg",
    "load_csv_ecg",
    "load_txt_ecg",
    "load_npy_ecg",
    "load_json_ecg",
    "load_edf_ecg",
    "load_xml_ecg",
    "load_dicom_ecg",
    "load_wfdb_record",
    "STANDARD_SAMPLING_RATES",
    "extract_report_measurements",
    "process_pdf_report",
    "process_ecg_image",
    "extract_waveform_from_image",
    "validate_extracted_signal",
    "extract_patient_demographics",
    "extract_technical_calibrations",
    "extract_machine_statement",
]
