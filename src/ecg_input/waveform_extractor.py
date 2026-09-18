"""
ECG Waveform Extractor Module
=============================

Extracts 1D numerical voltage signals from 2D ECG strip images:
- Color segmentation / grid suppression
- Dark trace extraction via column-wise center of mass
- Continuity check and gap interpolation
- Coordinate inversion (upward R-peaks -> positive voltage)
- Strictly avoids fabricating data if trace is occluded or corrupted

Research/educational use only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


def extract_waveform_from_image(
    bgr_image: np.ndarray,
    target_fs: float = 360.0,
    assumed_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract a 1D numerical ECG waveform from an image.

    Args:
        bgr_image: 3D BGR image array (OpenCV format)
        target_fs: Target sampling rate in Hz (default 360 Hz for model compatibility)
        assumed_duration_sec: Known or estimated duration in seconds (if standard 10s or 2.5s lead strip)

    Returns:
        Dictionary containing:
        - 'success': bool
        - 'signal': Optional[np.ndarray] (1D float array)
        - 'sampling_rate': float
        - 'confidence_score': float (0.0 to 1.0)
        - 'message': str
        - 'trace_mask': Optional[np.ndarray]
    """
    if bgr_image is None or bgr_image.size == 0:
        return {
            "success": False,
            "signal": None,
            "sampling_rate": target_fs,
            "confidence_score": 0.0,
            "message": "Input image is empty.",
            "trace_mask": None,
        }

    h, w = bgr_image.shape[:2]
    if w < 200 or h < 50:
        return {
            "success": False,
            "signal": None,
            "sampling_rate": target_fs,
            "confidence_score": 0.0,
            "message": "Image resolution too low for waveform extraction.",
            "trace_mask": None,
        }

    # 1. Color filtering: eliminate pink/red ECG grid
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Grid mask (pink/red)
    mask1 = cv2.inRange(hsv, np.array([0, 20, 100]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 20, 100]), np.array([180, 255, 255]))
    grid_mask = cv2.bitwise_or(mask1, mask2)

    # If grid present, neutralize it in grayscale
    suppressed_gray = gray.copy()
    suppressed_gray[grid_mask > 0] = 255

    # 2. Extract dark trace pixels
    # Blur slightly to connect antialiased pixels
    blurred = cv2.GaussianBlur(suppressed_gray, (3, 3), 0)
    # Adaptive or Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove isolated specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean_trace = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Column-wise extraction
    raw_y = np.full(w, np.nan)
    valid_cols = 0

    for x in range(w):
        col_pixels = np.where(clean_trace[:, x] > 0)[0]
        if len(col_pixels) > 0:
            # Center of mass of the trace in this column
            raw_y[x] = np.mean(col_pixels)
            valid_cols += 1

    continuity_ratio = float(valid_cols) / float(w)

    # If trace is missing on more than 35% of the columns, extraction is unreliable
    if continuity_ratio < 0.65:
        return {
            "success": False,
            "signal": None,
            "sampling_rate": target_fs,
            "confidence_score": round(continuity_ratio, 2),
            "message": "Waveform extraction confidence is too low for reliable AI analysis. Trace discontinuity is high.",
            "trace_mask": clean_trace,
        }

    # Interpolate NaN gaps
    valid_idx = np.where(~np.isnan(raw_y))[0]
    if len(valid_idx) < 20:
        return {
            "success": False,
            "signal": None,
            "sampling_rate": target_fs,
            "confidence_score": 0.1,
            "message": "Insufficient contiguous trace samples identified.",
            "trace_mask": clean_trace,
        }

    all_x = np.arange(w)
    interp_y = np.interp(all_x, valid_idx, raw_y[valid_idx])

    # 4. Invert coordinates (image y increases downward, electrical voltage increases upward)
    baseline = np.median(interp_y)
    ecg_inverted = -(interp_y - baseline)

    # Check signal variance: if it's completely flat (e.g. straight line border), reject
    sig_std = np.std(ecg_inverted)
    if sig_std < 1.0 or np.ptp(ecg_inverted) < 5.0:
        return {
            "success": False,
            "signal": None,
            "sampling_rate": target_fs,
            "confidence_score": 0.2,
            "message": "Extracted line lacks cardiac oscillation features.",
            "trace_mask": clean_trace,
        }

    # 5. Resampling to target sampling frequency
    # Standard ECG strip length is typically 2.5s (in 4x3 12-lead layout) or 10s (rhythm strip)
    if assumed_duration_sec is None:
        # Default assume 10.0s if wide image (w > 1200), else 2.5s
        assumed_duration_sec = 10.0 if w >= 1000 else 2.5

    target_n_samples = int(assumed_duration_sec * target_fs)
    resampled_x = np.linspace(0, w - 1, target_n_samples)
    final_signal = np.interp(resampled_x, all_x, ecg_inverted)

    # Normalize to approximate mV range
    # 1 large box (5mm) is 0.5 mV. Assume standard gain where QRS peak is ~1.0-2.0 units
    max_peak = np.max(np.abs(final_signal))
    if max_peak > 0:
        final_signal = (final_signal / max_peak) * 1.5

    confidence = min(1.0, round(continuity_ratio * 0.95, 2))

    return {
        "success": True,
        "signal": final_signal,
        "sampling_rate": target_fs,
        "confidence_score": confidence,
        "message": "Waveform successfully extracted from image.",
        "trace_mask": clean_trace,
    }
