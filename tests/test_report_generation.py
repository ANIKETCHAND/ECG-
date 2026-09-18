"""
Unit tests for structured report generation and PDF/JSON/TXT export.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from report.report_generator import (
    generate_structured_report,
    export_report_to_text,
    export_report_to_json,
)
from report.pdf_generator import generate_pdf_report


@pytest.fixture
def mock_report_inputs():
    input_info = {
        "file_name": "patient_lead_ii.csv",
        "file_modality": "DIGITAL_SIGNAL",
        "format": "csv",
        "sampling_rate": 360.0,
        "duration_sec": 10.0,
        "total_samples": 3600,
        "lead": "Lead II",
    }
    ai_results = {
        "predicted_class": "Normal Rhythm (Normal Sinus)",
        "probabilities": {"Normal": 0.94, "PVC": 0.04, "Other": 0.02},
        "signal_quality": "GOOD",
        "quality_score": 0.95,
        "heart_rate_bpm": 72.5,
        "mean_rr_sec": 0.828,
        "beat_count": 12,
        "quality_indicators": {
            "snr_db": 22.4,
            "baseline_wander": False,
            "has_powerline_interference": False,
            "has_motion_artifacts": False,
        },
        "class_counts": {"Normal": 12, "PVC": 0, "Other": 0},
    }
    extracted_measurements = {
        "patient_name": "Doe, Jane",
        "patient_age": "45",
        "patient_sex": "Female",
        "recording_date": "2026-05-10",
        "pr_interval_ms": 160.0,
        "qrs_duration_ms": 86.0,
        "qt_interval_ms": 390.0,
        "qtc_interval_ms": 415.0,
        "machine_interpretation": ["Normal sinus rhythm"],
    }
    waveform_status = {
        "is_extracted": False,
        "message": "Direct digital signal ingestion",
    }
    return input_info, ai_results, extracted_measurements, waveform_status


def test_generate_structured_report(mock_report_inputs):
    inp, ai_res, meas, wf_stat = mock_report_inputs
    report = generate_structured_report(inp, ai_res, meas, wf_stat)

    assert "report_title" in report
    assert "generated_at" in report
    assert report["patient_info"]["patient_name"] == "Doe, Jane"
    assert report["signal_quality"]["category"] == "GOOD"
    assert report["cardiac_parameters"]["heart_rate_bpm"] == 72.5
    assert report["ai_analysis"]["primary_pattern"] == "Normal Rhythm (Normal Sinus)"
    assert len(report["findings"]) > 0
    assert "disclaimer" in report


def test_export_report_to_text(mock_report_inputs):
    inp, ai_res, meas, wf_stat = mock_report_inputs
    report = generate_structured_report(inp, ai_res, meas, wf_stat)
    txt = export_report_to_text(report)

    assert "AI ECG SCREENING REPORT" in txt
    assert "PATIENT & INPUT INFORMATION" in txt
    assert "CARDIAC PARAMETERS" in txt
    assert "AI ABNORMALITY ANALYSIS" in txt
    assert "MEDICAL DISCLAIMER" in txt


def test_export_report_to_json(mock_report_inputs):
    inp, ai_res, meas, wf_stat = mock_report_inputs
    report = generate_structured_report(inp, ai_res, meas, wf_stat)
    json_str = export_report_to_json(report)

    parsed = json.loads(json_str)
    assert parsed["signal_quality"]["quality_score"] == 0.95
    assert parsed["ai_analysis"]["probabilities"]["Normal"] == 94.0


def test_generate_pdf_report(mock_report_inputs):
    inp, ai_res, meas, wf_stat = mock_report_inputs
    report = generate_structured_report(inp, ai_res, meas, wf_stat)

    # Simulated waveform
    fs = 360.0
    t = np.linspace(0, 5.0, int(5.0 * fs))
    waveform = np.sin(2 * np.pi * 1.2 * t)
    r_peaks = np.array([360, 720, 1080, 1440])

    pdf_bytes = generate_pdf_report(report, waveform=waveform, fs=fs, r_peaks=r_peaks)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
