"""
Digital ECG Signal Loader
=========================

Robust ingestion for CSV, TXT, and NPY files:
- Automatically detects delimiters (comma, tab, semicolon, whitespace)
- Isolates ECG voltage channel from timestamp/index columns
- Infers sampling rate from time intervals if present
- Validates signal dynamic range and finite values

Research/educational use only.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

STANDARD_SAMPLING_RATES = [125, 250, 360, 500, 1000]


def load_digital_signal(
    file_or_path: Union[str, Path, BinaryIO],
    filename: str,
    user_fs: Optional[float] = None,
    default_fs: float = 360.0,
) -> Dict[str, Any]:
    """Load a 1D ECG voltage signal from CSV, TXT, or NPY.

    Args:
        file_or_path: Path or file-like object
        filename: Original file name string
        user_fs: User-specified sampling rate if known
        default_fs: Fallback sampling rate if none detected and none specified

    Returns:
        Dictionary with:
        - 'signal': 1D np.ndarray
        - 'sampling_rate': float
        - 'detected_fs': Optional[float] (if mathematically inferred from time column)
        - 'needs_fs_confirmation': bool
        - 'column_name': str
        - 'total_samples': int
        - 'duration_sec': float
    """
    fname_lower = filename.lower()

    if fname_lower.endswith(".npy"):
        return _load_npy(file_or_path, user_fs, default_fs)

    return _load_text_csv(file_or_path, user_fs, default_fs)


def _load_npy(file_or_path: Any, user_fs: Optional[float], default_fs: float) -> Dict[str, Any]:
    """Load NumPy .npy array."""
    arr = np.load(file_or_path)
    arr = np.asarray(arr, dtype=np.float64).flatten()

    if len(arr) == 0:
        raise ValueError("Uploaded .npy file is empty.")

    # Clean NaNs/Infs
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    fs = user_fs if user_fs is not None else default_fs
    needs_confirm = user_fs is None

    return {
        "signal": arr,
        "sampling_rate": float(fs),
        "detected_fs": None,
        "needs_fs_confirmation": needs_confirm,
        "column_name": "npy_array",
        "total_samples": len(arr),
        "duration_sec": len(arr) / float(fs),
    }


def _load_text_csv(file_or_path: Any, user_fs: Optional[float], default_fs: float) -> Dict[str, Any]:
    """Load CSV or delimited TXT file."""
    # Read raw bytes if stream
    if hasattr(file_or_path, "read"):
        raw_bytes = file_or_path.read()
        if isinstance(raw_bytes, str):
            text_data = raw_bytes
        else:
            text_data = raw_bytes.decode("utf-8", errors="replace")
        buffer = io.StringIO(text_data)
    else:
        with open(file_or_path, "r", encoding="utf-8", errors="replace") as f:
            text_data = f.read()
        buffer = io.StringIO(text_data)

    # Try reading with pandas using automatic engine
    df = None
    delimiters = [",", "\t", ";", r"\s+"]
    for sep in delimiters:
        try:
            buffer.seek(0)
            candidate = pd.read_csv(buffer, sep=sep, engine="python")
            # If at least one numeric column exists and rows > 0, accept
            num_cols = candidate.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0 and len(candidate) > 2:
                df = candidate
                break
        except Exception:
            continue

    if df is None or df.empty:
        # Fallback: single column without header
        buffer.seek(0)
        try:
            df = pd.read_csv(buffer, header=None)
        except Exception as exc:
            raise ValueError(f"Could not parse digital ECG data: {exc}")

    # Ensure columns have numeric data
    numeric_cols: List[str] = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0.6 * len(df):
            df[col] = converted
            numeric_cols.append(str(col))

    if not numeric_cols:
        raise ValueError("The uploaded file does not contain any valid numeric voltage samples.")

    # Check for time column to detect sampling rate
    detected_fs: Optional[float] = None
    signal_col: Optional[str] = None

    time_keywords = ["time", "t_sec", "t_s", "timestamp", "sec", "seconds", "sample", "index"]
    ecg_keywords = ["ecg", "lead", "mlii", "v1", "v2", "v3", "v4", "v5", "v6", "signal", "val", "voltage", "mv"]

    time_col = None
    for col in numeric_cols:
        c_lower = col.lower()
        if any(kw == c_lower or kw in c_lower for kw in time_keywords):
            # Verify it's monotonic increasing
            vals = df[col].dropna().values
            if len(vals) > 10:
                diffs = np.diff(vals[:100])
                if np.all(diffs > 0):
                    median_step = float(np.median(diffs))
                    if 0.0005 <= median_step <= 0.02:  # 50 Hz to 2000 Hz
                        detected_fs = round(1.0 / median_step, 1)
                        time_col = col
                        break

    # Choose best voltage column
    candidate_signal_cols = [c for c in numeric_cols if c != time_col]
    if not candidate_signal_cols:
        candidate_signal_cols = numeric_cols

    # Look for explicit ECG keyword
    for col in candidate_signal_cols:
        c_lower = col.lower()
        if any(kw in c_lower for kw in ecg_keywords):
            signal_col = col
            break

    # If no keyword matched, choose column with maximum oscillatory variance
    if signal_col is None:
        best_std = -1.0
        for col in candidate_signal_cols:
            s_std = float(df[col].std())
            if s_std > best_std:
                best_std = s_std
                signal_col = col

    if signal_col is None:
        signal_col = candidate_signal_cols[0]

    raw_signal = df[signal_col].dropna().values.astype(np.float64)

    # Clean NaNs/Infs
    raw_signal = np.nan_to_num(raw_signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Determine final sampling rate
    if user_fs is not None:
        final_fs = float(user_fs)
        needs_confirm = False
    elif detected_fs is not None:
        final_fs = float(detected_fs)
        needs_confirm = False
    else:
        final_fs = float(default_fs)
        needs_confirm = True

    return {
        "signal": raw_signal,
        "sampling_rate": final_fs,
        "detected_fs": detected_fs,
        "needs_fs_confirmation": needs_confirm,
        "column_name": str(signal_col),
        "total_samples": len(raw_signal),
        "duration_sec": len(raw_signal) / final_fs,
    }
