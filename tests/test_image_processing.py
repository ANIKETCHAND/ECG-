"""
Unit tests for ECG image processing and grid detection.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_input.image_processor import process_ecg_image


def test_process_valid_ecg_image_array():
    # Create a 600x200 simulated ECG strip with a white background and dark line
    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    # Draw pink grid lines
    for x in range(0, 600, 20):
        cv2.line(img, (x, 0), (x, 200), (200, 180, 255), 1)
    for y in range(0, 200, 20):
        cv2.line(img, (0, y), (600, y), (200, 180, 255), 1)
    # Draw a simulated ECG wave
    for x in range(599):
        y1 = int(100 + 40 * np.sin(2 * np.pi * x / 100))
        y2 = int(100 + 40 * np.sin(2 * np.pi * (x + 1) / 100))
        cv2.line(img, (x, y1), (x + 1, y2), (0, 0, 0), 2)

    res = process_ecg_image(img)
    assert res["is_ecg"] is True
    assert res["cv_image"] is not None
    assert res["gray_image"] is not None
    assert res["pil_image"] is not None
    assert res["dimensions"] == (600, 200)


def test_process_too_small_image():
    small_img = np.full((30, 30, 3), 255, dtype=np.uint8)
    res = process_ecg_image(small_img)
    assert res["is_ecg"] is False
    assert "too small" in res["status_message"].lower()


def test_process_invalid_input():
    res = process_ecg_image(b"not_an_image_content")
    assert res["is_ecg"] is False
    assert res["cv_image"] is None
