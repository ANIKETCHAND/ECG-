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

try:
    import cv2
except ImportError:
    cv2 = None

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
    if cv2 is not None:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        mask1 = cv2.inRange(hsv, np.array([0, 20, 100]), np.array([20, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 20, 100]), np.array([180, 255, 255]))
        grid_mask = cv2.bitwise_or(mask1, mask2)
        suppressed_gray = gray.copy()
        suppressed_gray[grid_mask > 0] = 255

        # 2. Extract dark trace pixels
        blurred = cv2.GaussianBlur(suppressed_gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean_trace = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:
        # Pure NumPy fallback
        b = bgr_image[:, :, 0].astype(float)
        g = bgr_image[:, :, 1].astype(float)
        r = bgr_image[:, :, 2].astype(float)
        gray = (b * 0.114 + g * 0.587 + r * 0.299).astype(np.uint8)

        pink_mask = (r > 150) & (g > 30) & (b > 60) & (r > g + 15)
        suppressed_gray = gray.copy()
        suppressed_gray[pink_mask] = 255

        # Threshold dark trace (trace pixels are darker than background)
        p15 = np.percentile(suppressed_gray, 20)
        clean_trace = np.zeros_like(suppressed_gray, dtype=np.uint8)
        clean_trace[suppressed_gray <= max(p15, 120)] = 255


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
    # Standard ECG rhythm strip default duration is 10.0 seconds when calibration is unmeasured
    is_calibrated = False
    if assumed_duration_sec is None:
        assumed_duration_sec = 10.0
        calibration_status = "UNCALIBRATED_OPTICAL_EXTRACT"
    else:
        calibration_status = "USER_SPECIFIED_DURATION"

    target_n_samples = int(assumed_duration_sec * target_fs)
    resampled_x = np.linspace(0, w - 1, target_n_samples)
    final_signal = np.interp(resampled_x, all_x, ecg_inverted)

    # 6. Medical-Grade Signal Smoothing & Quantization Denoising
    # Pixel quantization produces high-frequency saw-tooth jitter (> 50 Hz).
    # We apply a zero-phase 0.5 - 45.0 Hz Butterworth bandpass filter and median baseline removal.
    try:
        from scipy import signal as scipy_signal
        nyq = 0.5 * target_fs
        low = max(0.5 / nyq, 0.001)
        high = min(45.0 / nyq, 0.99)
        b, a = scipy_signal.butter(3, [low, high], btype="bandpass")
        final_signal = scipy_signal.filtfilt(b, a, final_signal)
    except Exception:
        # Robust 5-point moving average smoothing fallback
        kernel = np.ones(5) / 5.0
        final_signal = np.convolve(final_signal, kernel, mode="same")

    # Baseline center without arbitrary 1.5 mV scaling
    # Forcing peak amplitude to 1.5 mV destroys ST elevation, ST depression, and LVH criteria.
    baseline_val = float(np.median(final_signal))
    final_signal = final_signal - baseline_val
    std_val = float(np.std(final_signal))
    if std_val > 1e-4:
        # Standardized signal preserving true relative morphology
        final_signal = final_signal / std_val

    confidence = min(1.0, round(continuity_ratio * 0.95, 2))

    return {
        "success": True,
        "signal": final_signal,
        "sampling_rate": target_fs,
        "confidence_score": confidence,
        "duration_sec": assumed_duration_sec,
        "calibration_status": calibration_status,
        "is_calibrated": is_calibrated,
        "can_evaluate_st": False,
        "clinical_warning": "ST-segment elevation and amplitude criteria withheld: Missing physical grid calibration.",
        "message": "Waveform successfully extracted and filtered from image.",
        "trace_mask": clean_trace,
    }
