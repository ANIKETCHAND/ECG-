"""
Unit tests for PDF ECG report parsing and clinical measurement extraction.
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_input.pdf_processor import process_pdf_report
from ecg_input.measurement_extractor import extract_report_measurements


SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "sample_ecgs" / "sample_clinical_ecg_report.pdf"


def test_pdf_report_exists_and_parses():
    assert SAMPLE_PDF_PATH.exists(), f"Sample PDF not found at {SAMPLE_PDF_PATH}"
    result = process_pdf_report(SAMPLE_PDF_PATH)

    assert result["is_ecg"] is True
    assert result["page_count"] >= 1
    assert "CLINICAL" in result["text"] or "ELECTROCARDIOGRAM" in result["text"] or "ECG" in result["text"].upper()


def test_pdf_measurement_extraction():
    result = process_pdf_report(SAMPLE_PDF_PATH)
    meas = result["measurements"]

    assert meas.get("heart_rate_printed") == 74.0
    assert meas.get("pr_interval_ms") == 164.0
    assert meas.get("qrs_duration_ms") == 88.0
    assert meas.get("qt_interval_ms") == 392.0
    assert meas.get("qtc_interval_ms") == 418.0
    assert int(meas.get("patient_age")) == 58
    assert meas.get("patient_sex") == "Male"
    assert any("sinus rhythm" in item.lower() for item in meas.get("machine_interpretation", []))


def test_measurement_extractor_isolated():
    raw_sample_text = """
    HOSPITAL CARDIOLOGY SERVICE
    Name: Jane Smith   Age: 64 yr   Sex: Female
    Vent Rate: 84 BPM
    PR Int: 172 ms
    QRS Dur: 88 ms
    QT/QTc: 382/412 ms
    P-QRS-T Axes: 55 45 40
    Diagnosis: Normal Sinus Rhythm. Unremarkable 12-lead ECG.
    """
    meas = extract_report_measurements(raw_sample_text)
    assert meas["heart_rate_printed"] == 84.0
    assert meas["pr_interval_ms"] == 172.0
    assert meas["qrs_duration_ms"] == 88.0
    assert meas["qt_interval_ms"] == 382.0
    assert meas["qtc_interval_ms"] == 412.0
    assert int(meas["patient_age"]) == 64
    assert meas["patient_sex"] == "Female"
    assert any("sinus rhythm" in item.lower() for item in meas["machine_interpretation"])


def test_invalid_pdf():
    result = process_pdf_report("non_existent_file.pdf")
    assert result["is_ecg"] is False
    assert result["page_count"] == 0
