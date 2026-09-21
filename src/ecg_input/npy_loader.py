"""
NPY ECG Signal Loader
=====================
Loads serialized NumPy array files (.npy) into unified ECGRecording instances.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def load_npy_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    fs: Optional[float] = None,
    lead_names: Optional[List[str]] = None,
    patient_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Load ECG from NumPy .npy file."""
    try:
        if isinstance(file_or_path, (str, Path)):
            arr = np.load(str(file_or_path), allow_pickle=False)
        elif isinstance(file_or_path, bytes):
            buf = io.BytesIO(file_or_path)
            arr = np.load(buf, allow_pickle=False)
        elif hasattr(file_or_path, "read"):
            arr = np.load(file_or_path, allow_pickle=False)
        else:
            return None, "NO RESULT: Invalid input type for NPY loader."

        if arr.size == 0:
            return None, "NO RESULT: NPY array is empty."

        if not np.issubdtype(arr.dtype, np.number):
            return None, "NO RESULT: NPY array does not contain numeric data."

        if fs is None or fs <= 0:
            return None, "NO RESULT: Missing sampling rate. Cannot assume sampling frequency (Rule 4)."

        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            signals = arr
            leads = lead_names if lead_names else ["II"]
        elif arr.ndim == 2:
            # Ensure shape is (leads, samples). If columns > rows and rows <= 12, assume (leads, samples)
            if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 12:
                signals = arr.T
            else:
                signals = arr
            n_leads = signals.shape[0]
            if lead_names and len(lead_names) == n_leads:
                leads = lead_names
            else:
                leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"][:n_leads] if n_leads == 12 else [f"Lead_{i+1}" for i in range(n_leads)]
        else:
            return None, f"NO RESULT: NPY array has unsupported dimension: {arr.ndim}. Must be 1D or 2D."

        duration = (signals.shape[1] if signals.ndim > 1 else len(signals)) / float(fs)

        rec = ECGRecording(
            record_id=f"REC-NPY-{hash(tuple(signals.flatten()[:5])) & 0xFFFFFF:06X}",
            sampling_rate=float(fs),
            duration=duration,
            lead_names=leads,
            number_of_leads=len(leads),
            signals=signals,
            units="mV",
            patient_id=patient_id,
            device=device,
            source_format="NPY",
            metadata={"original_shape": list(arr.shape)},
        )
        is_valid, errs = rec.validate()
        if not is_valid:
            return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"

        return rec, None

    except Exception as e:
        return None, f"NO RESULT: Failed to parse NPY: {str(e)}"
