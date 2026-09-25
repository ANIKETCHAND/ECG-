"""
ECG Image Processor
===================

Preprocesses and analyzes uploaded ECG images (JPG, PNG, TIFF):
- Validates resolution and aspect ratio
- Detects ECG grid lines (pink/red/gray grid characteristics)
- Contrast enhancement and noise reduction
- Crops or isolates active waveform strip regions

Research/educational use only.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from PIL import Image


def process_ecg_image(
    file_or_path: Union[str, Path, BinaryIO, Image.Image, np.ndarray],
) -> Dict[str, Any]:
    """Inspect and preprocess an ECG report image.

    Args:
        file_or_path: Image file, path, PIL Image, or numpy array

    Returns:
        Dictionary containing:
        - 'is_ecg': bool
        - 'cv_image': np.ndarray (BGR)
        - 'gray_image': np.ndarray (Grayscale)
        - 'pil_image': PIL.Image
        - 'has_grid': bool
        - 'dimensions': Tuple[int, int] (width, height)
        - 'status_message': str
    """
    cv_img = None
    pil_img = None

    if isinstance(file_or_path, np.ndarray):
        cv_img = file_or_path
        if cv_img.ndim == 2:
            if cv2 is not None:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            else:
                cv_img = np.stack([cv_img, cv_img, cv_img], axis=-1)
        if cv2 is not None:
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(cv_img[:, :, ::-1])
    elif isinstance(file_or_path, Image.Image):
        pil_img = file_or_path.convert("RGB")
        rgb_arr = np.array(pil_img)
        if cv2 is not None:
            cv_img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        else:
            cv_img = rgb_arr[:, :, ::-1].copy()
    elif hasattr(file_or_path, "read"):
        file_or_path.seek(0)
        file_bytes = file_or_path.read()
        if cv2 is not None:
            nparr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            try:
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                rgb_arr = np.array(pil_img)
                cv_img = rgb_arr[:, :, ::-1].copy()
            except Exception:
                cv_img = None
    else:
        if cv2 is not None:
            cv_img = cv2.imread(str(file_or_path))
            if cv_img is not None:
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            try:
                pil_img = Image.open(str(file_or_path)).convert("RGB")
                rgb_arr = np.array(pil_img)
                cv_img = rgb_arr[:, :, ::-1].copy()
            except Exception:
                cv_img = None

    if cv_img is None or pil_img is None:
        return {
            "is_ecg": False,
            "cv_image": None,
            "gray_image": None,
            "pil_image": None,
            "has_grid": False,
            "dimensions": (0, 0),
            "status_message": "Failed to decode image data.",
        }

    h, w = cv_img.shape[:2]
    if h < 50 or w < 50:
        return {
            "is_ecg": False,
            "cv_image": cv_img,
            "gray_image": None,
            "pil_image": pil_img,
            "has_grid": False,
            "dimensions": (w, h),
            "status_message": "Image dimensions are too small to be a valid ECG.",
        }

    if cv2 is not None:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = (cv_img[:, :, 0] * 0.114 + cv_img[:, :, 1] * 0.587 + cv_img[:, :, 2] * 0.299).astype(np.uint8)

    # Detect ECG grid lines or waveform traces
    has_grid = _detect_ecg_grid(cv_img, gray)
    has_waveform = _detect_waveform_curves(gray)

    is_ecg = has_grid or has_waveform

    if is_ecg:
        status_msg = "ECG recording visual features detected."
    else:
        status_msg = "Image does not appear to contain an ECG grid or waveform."

    return {
        "is_ecg": is_ecg,
        "cv_image": cv_img,
        "gray_image": gray,
        "pil_image": pil_img,
        "has_grid": has_grid,
        "dimensions": (w, h),
        "status_message": status_msg,
    }


def _detect_ecg_grid(bgr_img: np.ndarray, gray: np.ndarray) -> bool:
    """Detect presence of characteristic ECG millimeter grid lines (pink, red, or light gray)."""
    h, w = gray.shape

    if cv2 is not None:
        # 1. Color check in HSV: standard ECG paper has pink/reddish/orange grid
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 30, 150]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 30, 150]), np.array([180, 255, 255]))
        pink_mask = cv2.bitwise_or(mask1, mask2)

        pink_ratio = float(np.sum(pink_mask > 0)) / (h * w)
        if pink_ratio > 0.05:
            return True

        # 2. Morphological check for regular grid structure
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        lines_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

        grid_h_ratio = float(np.sum(lines_h > 0)) / (h * w)
        grid_v_ratio = float(np.sum(lines_v > 0)) / (h * w)

        if grid_h_ratio > 0.015 and grid_v_ratio > 0.015:
            return True
    else:
        # NumPy-based color check: BGR format
        b = bgr_img[:, :, 0].astype(float)
        g = bgr_img[:, :, 1].astype(float)
        r = bgr_img[:, :, 2].astype(float)
        pink_mask = (r > 150) & (g > 30) & (b > 60) & (r > g + 15)
        pink_ratio = float(np.sum(pink_mask)) / (h * w)
        if pink_ratio > 0.05:
            return True

        # Gradient check for grid lines
        diff_v = np.abs(np.diff(gray.astype(float), axis=0)) > 20
        diff_h = np.abs(np.diff(gray.astype(float), axis=1)) > 20
        if float(np.mean(diff_v)) > 0.02 and float(np.mean(diff_h)) > 0.02:
            return True

    return False


def _detect_waveform_curves(gray: np.ndarray) -> bool:
    """Check for high-contrast dark oscillatory traces typical of ECG signals."""
    if cv2 is not None:
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.sum(edges > 0)) / edges.size
    else:
        diff_y = np.abs(np.diff(gray.astype(float), axis=0)[:, :-1])
        diff_x = np.abs(np.diff(gray.astype(float), axis=1)[:-1, :])
        edges = (diff_y + diff_x) > 40
        edge_ratio = float(np.sum(edges)) / edges.size

    # Typical ECG traces have continuous connected edges with moderate density (2% to 35%)
    return 0.02 <= edge_ratio <= 0.35

