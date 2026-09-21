"""
Digital ECG Signal Loader
=========================

Robust ingestion for CSV, TXT, NPY, EDF, XML, JSON, DICOM, and WFDB formats:
- Automatically detects delimiters (comma, tab, semicolon, whitespace)
- Isolates ECG voltage channel from timestamp/index columns
- Infers sampling rate from time intervals if present
- Converts into unified ECGRecording instances
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.ecg_core.models import ECGRecording
    from src.ecg_input.csv_loader import load_csv_ecg
    from src.ecg_input.dicom_loader import load_dicom_ecg
    from src.ecg_input.edf_loader import load_edf_ecg
    from src.ecg_input.input_detector import InputModality, detect_input_modality
    from src.ecg_input.json_loader import load_json_ecg
    from src.ecg_input.npy_loader import load_npy_ecg
    from src.ecg_input.txt_loader import load_txt_ecg
    from src.ecg_input.wfdb_loader import load_wfdb_record
    from src.ecg_input.xml_loader import load_xml_ecg
except ImportError:
    from ecg_core.models import ECGRecording
    from ecg_input.csv_loader import load_csv_ecg
    from ecg_input.dicom_loader import load_dicom_ecg
    from ecg_input.edf_loader import load_edf_ecg
    from ecg_input.input_detector import InputModality, detect_input_modality
    from ecg_input.json_loader import load_json_ecg
    from ecg_input.npy_loader import load_npy_ecg
    from ecg_input.txt_loader import load_txt_ecg
    from ecg_input.wfdb_loader import load_wfdb_record
    from ecg_input.xml_loader import load_xml_ecg


STANDARD_SAMPLING_RATES = [125, 250, 360, 500, 1000]


def load_any_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    filename: str,
    fs: Optional[float] = None,
    lead_name: str = "II",
    patient_id: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Universal dispatcher: ingests any supported ECG digital format into an ECGRecording.

    Returns:
        (ECGRecording, None) if successful, or (None, error_message) on failure.
    """
    fname_lower = filename.lower()

    if fname_lower.endswith(".csv"):
        return load_csv_ecg(file_or_path, fs=fs, lead_name=lead_name, patient_id=patient_id)

    if fname_lower.endswith(".txt"):
        return load_txt_ecg(file_or_path, fs=fs, lead_name=lead_name, patient_id=patient_id)

    if fname_lower.endswith(".npy"):
        return load_npy_ecg(file_or_path, fs=fs, lead_names=[lead_name], patient_id=patient_id)

    if fname_lower.endswith((".edf", ".rec")):
        return load_edf_ecg(file_or_path, lead_name=lead_name, patient_id=patient_id)

    if fname_lower.endswith((".xml", ".hl7")):
        return load_xml_ecg(file_or_path, fs=fs, lead_name=lead_name, patient_id=patient_id)

    if fname_lower.endswith(".json"):
        return load_json_ecg(file_or_path, fs=fs, lead_name=lead_name, patient_id=patient_id)

    if fname_lower.endswith((".dcm", ".dicom")):
        return load_dicom_ecg(file_or_path, lead_name=lead_name, patient_id=patient_id)

    if fname_lower.endswith((".dat", ".hea")):
        stem = str(Path(filename).with_suffix(""))
        return load_wfdb_record(stem, lead_name=lead_name, patient_id=patient_id)

    return None, f"NO RESULT: Unsupported digital format extension '{Path(filename).suffix}'."


def load_digital_signal(
    file_or_path: Union[str, Path, BinaryIO],
    filename: str,
    user_fs: Optional[float] = None,
    default_fs: float = 360.0,
) -> Dict[str, Any]:
    """Backward-compatible loader returning dictionary representation for legacy pipelines."""
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

    df = None
    delimiters = [",", "\t", ";", r"\s+"]

    for sep in delimiters:
        try:
            buffer.seek(0)
            candidate = pd.read_csv(buffer, sep=sep, engine="python", nrows=20)
            if candidate.shape[1] >= 1 and candidate.shape[0] >= 5:
                # Check numeric columns
                numeric_cols = candidate.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, sep=sep, engine="python")
                    break
        except Exception:
            continue

    if df is None or df.empty:
        raise ValueError("Could not parse file as delimited numeric table.")

    # Drop non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric data columns found in the file.")

    detected_fs = None
    signal_col = None

    # Check for time column
    time_candidates = [col for col in numeric_df.columns if any(kw in str(col).lower() for kw in ["time", "t_sec", "t_ms", "timestamp", "sec"])]
    if time_candidates:
        time_col = time_candidates[0]
        t_vals = numeric_df[time_col].dropna().values
        if len(t_vals) > 10:
            diffs = np.diff(t_vals)
            positive_diffs = diffs[diffs > 0]
            if len(positive_diffs) > 5:
                median_dt = float(np.median(positive_diffs))
                if median_dt > 0:
                    # If in milliseconds
                    if median_dt >= 1.0 and np.max(t_vals) > 100:
                        candidate_fs = 1000.0 / median_dt
                    else:
                        candidate_fs = 1.0 / median_dt
                    # Snap to standard frequency if close
                    for std_fs in STANDARD_SAMPLING_RATES:
                        if abs(candidate_fs - std_fs) / std_fs < 0.05:
                            candidate_fs = float(std_fs)
                            break
                    detected_fs = round(candidate_fs, 2)

        # Voltage candidates excluding time column
        voltage_candidates = [col for col in numeric_df.columns if col != time_col]
    else:
        voltage_candidates = list(numeric_df.columns)

    if not voltage_candidates:
        raise ValueError("No voltage signal column could be isolated.")

    # Select best voltage candidate
    ecg_name_candidates = [col for col in voltage_candidates if any(kw in str(col).lower() for kw in ["ecg", "lead", "ii", "mlii", "voltage", "mv"])]
    signal_col = ecg_name_candidates[0] if ecg_name_candidates else voltage_candidates[0]

    raw_signal = numeric_df[signal_col].dropna().values.astype(np.float64)

    if len(raw_signal) == 0:
        raise ValueError(f"Signal column '{signal_col}' contains no valid numeric data.")

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
