"""
Tests for scanned-report OCR and image digitisation round-trip accuracy.

OCR is exercised for its *reporting contract* rather than its accuracy, because
the Tesseract engine is an optional dependency: whether it is installed here or
not, the module must never return empty text as if it had successfully read a
blank document.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ecg_input.ocr import (
    MIN_USEFUL_CHARACTERS,
    extract_text_from_image,
    is_ocr_available,
    ocr_pdf_report,
    ocr_report_measurements,
    ocr_unavailable_reason,
)
from training.validate_digitization import (
    render_strip_image,
    synthesise_ecg,
    validate_round_trip,
)


def _blank_document_image() -> np.ndarray:
    image = np.full((300, 900, 3), 255, dtype=np.uint8)
    return image


# ------------------------------------------------------------------------ OCR
def test_availability_and_reason_are_consistent():
    available = is_ocr_available()
    reason = ocr_unavailable_reason()
    assert available is (reason is None)
    if not available:
        assert reason


def test_unavailable_reason_names_the_package_and_fix():
    reason = ocr_unavailable_reason()
    if reason:
        # The operator must be told what to install.
        assert "pytesseract" in reason or "tesseract" in reason.lower()


def test_extract_text_never_silently_returns_empty_on_failure():
    result = extract_text_from_image(_blank_document_image())
    assert result["status"] in {
        "SUCCESS",
        "OCR_UNAVAILABLE",
        "EMPTY_TEXT",
        "LOW_CONFIDENCE",
        "PROCESSING_FAILED",
    }
    assert "text" in result
    assert result["duration_ms"] >= 0.0

    if not is_ocr_available():
        assert result["status"] == "OCR_UNAVAILABLE"
        assert result["reason"]
        assert result["characters"] == 0


def test_blank_image_reports_empty_text_not_success_when_ocr_available():
    result = extract_text_from_image(_blank_document_image())
    if result["status"] == "SUCCESS":
        assert result["characters"] >= MIN_USEFUL_CHARACTERS
    elif is_ocr_available():
        assert result["status"] in {"EMPTY_TEXT", "LOW_CONFIDENCE"}


def test_measurements_key_present_even_when_ocr_unavailable():
    result = ocr_report_measurements(_blank_document_image())
    assert "measurements" in result
    if not is_ocr_available():
        assert result["measurements"] == {}
        assert result["status"] == "OCR_UNAVAILABLE"


def test_ocr_pdf_reports_unavailable_without_crashing():
    result = ocr_pdf_report(b"not really a pdf")
    assert result["status"] in {"OCR_UNAVAILABLE", "RASTERISER_UNAVAILABLE", "PROCESSING_FAILED", "EMPTY_TEXT"}
    assert result["reason"]
    assert result["text"] == ""


# ---------------------------------------------------------------- digitisation
def test_render_produces_a_grid_and_trace():
    signal, _time, peaks = synthesise_ecg(duration_s=2.5, heart_rate_bpm=75)
    image = render_strip_image(signal, 360.0)
    assert image.ndim == 3 and image.shape[2] == 3
    # The trace is dark, the grid is not.
    assert image.min() == 0
    assert image.max() == 255
    assert len(peaks) >= 2


def test_round_trip_reports_amplitude_as_not_calibrated():
    report = validate_round_trip(duration_s=5.0, heart_rate_bpm=75)
    assert report["amplitude_recovery"]["status"] == "NOT_CALIBRATED"
    assert "not recoverable" in report["amplitude_recovery"]["reason"].lower()


def test_round_trip_extracts_a_waveform_and_measures_timing():
    report = validate_round_trip(duration_s=10.0, heart_rate_bpm=75)
    assert report["extraction"]["success"] is True

    timing = report["timing"]
    assert timing["status"] == "MEASURED"
    assert timing["matched_beats"] > 0
    assert timing["mean_absolute_error_ms"] is not None
    assert timing["mean_absolute_error_ms"] >= 0.0


def test_round_trip_beat_count_is_in_the_right_ballpark():
    """The digitised trace must contain roughly the right number of beats."""
    duration, heart_rate = 10.0, 75.0
    report = validate_round_trip(duration_s=duration, heart_rate_bpm=heart_rate)
    expected = duration * heart_rate / 60.0
    recovered = report["timing"]["recovered_beat_count"]
    assert abs(recovered - expected) <= max(2, 0.25 * expected)


def test_round_trip_morphology_correlation_is_bounded():
    report = validate_round_trip(duration_s=5.0, heart_rate_bpm=90)
    correlation = report["morphology_correlation"]
    if correlation is not None:
        assert -1.0 <= correlation <= 1.0


def test_round_trip_reports_ground_truth_for_audit():
    report = validate_round_trip(duration_s=5.0, heart_rate_bpm=60)
    truth = report["ground_truth"]
    assert truth["heart_rate_bpm"] == 60
    assert truth["true_beats"] == len(synthesise_ecg(5.0, 60)[2])
