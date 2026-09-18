"""
Unit tests for input modality and format detection.
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_input.input_detector import InputModality, detect_input_modality


def test_detect_digital_csv():
    modality, fmt = detect_input_modality("patient_ecg.csv")
    assert modality == InputModality.DIGITAL_SIGNAL
    assert fmt == "csv"


def test_detect_digital_npy():
    modality, fmt = detect_input_modality("recording.npy")
    assert modality == InputModality.DIGITAL_SIGNAL
    assert fmt == "npy"


def test_detect_digital_txt():
    modality, fmt = detect_input_modality("lead_ii.txt")
    assert modality == InputModality.DIGITAL_SIGNAL
    assert fmt == "txt"


def test_detect_report_pdf():
    modality, fmt = detect_input_modality("clinical_ecg.pdf")
    assert modality == InputModality.REPORT_PDF
    assert fmt == "pdf"


def test_detect_report_images():
    for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        modality, fmt = detect_input_modality(f"ecg_scan.{ext}")
        assert modality == InputModality.REPORT_IMAGE
        assert fmt == ext


def test_detect_unsupported():
    modality, fmt = detect_input_modality("document.docx")
    assert modality == InputModality.UNSUPPORTED
    assert fmt == "docx"


def test_detect_by_magic_bytes_pdf():
    pdf_bytes = b"%PDF-1.4 header dummy content"
    modality, fmt = detect_input_modality("unknown_file", file_bytes=pdf_bytes)
    assert modality == InputModality.REPORT_PDF
    assert fmt == "pdf"


def test_detect_by_magic_bytes_png():
    png_bytes = b"\x89PNG\r\n\x1a\n"
    modality, fmt = detect_input_modality("blob", file_bytes=png_bytes)
    assert modality == InputModality.REPORT_IMAGE
    assert fmt == "png"


def test_detect_by_magic_bytes_jpeg():
    jpg_bytes = b"\xff\xd8\xff\xe0"
    modality, fmt = detect_input_modality("blob", file_bytes=jpg_bytes)
    assert modality == InputModality.REPORT_IMAGE
    assert fmt == "jpeg"
