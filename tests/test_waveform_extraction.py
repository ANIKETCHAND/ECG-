"""
Unit tests for waveform extraction from image traces and extraction validation.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_input.waveform_extractor import extract_waveform_from_image
from ecg_input.extraction_validation import validate_extracted_signal


def test_extract_waveform_from_synthetic_image():
    # 800 wide by 200 high image
    w, h = 800, 200
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Draw a clean sinusoidal waveform
    for x in range(w - 1):
        y1 = int(100 - 50 * np.sin(2 * np.pi * x / 100))
        y2 = int(100 - 50 * np.sin(2 * np.pi * (x + 1) / 100))
        cv2.line(img, (x, y1), (x + 1, y2), (0, 0, 0), 2)

    res = extract_waveform_from_image(img, target_fs=360.0, assumed_duration_sec=2.5)
    assert res["success"] is True
    assert res["signal"] is not None
    assert len(res["signal"]) > 0
    assert res["confidence_score"] > 0.70


def test_extract_waveform_empty_or_too_small():
    empty_img = np.array([])
    res = extract_waveform_from_image(empty_img)
    assert res["success"] is False

    small_img = np.full((30, 30, 3), 255, dtype=np.uint8)
    res_small = extract_waveform_from_image(small_img)
    assert res_small["success"] is False


def test_validation_gate_valid_signal():
    fs = 360.0
    t = np.linspace(0, 3.0, int(3.0 * fs))
    # Synthetic normal wave with adequate variance
    signal = 0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.2 * np.sin(2 * np.pi * 5.0 * t)
    valid, msg = validate_extracted_signal(signal, fs=fs, confidence_score=0.85)
    assert valid is True
    assert "passed" in msg.lower()


def test_validation_gate_low_confidence():
    fs = 360.0
    signal = np.sin(np.linspace(0, 5, 1000))
    valid, msg = validate_extracted_signal(signal, fs=fs, confidence_score=0.45)
    assert valid is False
    assert "confidence is too low" in msg.lower()


def test_validation_gate_too_short():
    fs = 360.0
    short_signal = np.sin(np.linspace(0, 1, int(0.5 * fs)))  # 0.5s duration
    valid, msg = validate_extracted_signal(short_signal, fs=fs, confidence_score=0.90, min_duration_sec=1.5)
    assert valid is False
    assert "below the minimum" in msg.lower()


def test_validation_gate_flat_signal():
    fs = 360.0
    flat_signal = np.full(int(3.0 * fs), 0.1)
    valid, msg = validate_extracted_signal(flat_signal, fs=fs, confidence_score=0.90)
    assert valid is False
    assert "flat or silent" in msg.lower()
