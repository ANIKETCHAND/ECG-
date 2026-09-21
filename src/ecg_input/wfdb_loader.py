"""
WFDB ECG Record Loader
======================
Loads PhysioNet WFDB format records (.dat, .hea, .atr) into unified ECGRecording instances.
Uses the verified Python `wfdb` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def load_wfdb_record(
    record_path_without_ext: str,
    lead_name: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Load WFDB format record using wfdb library.

    Args:
        record_path_without_ext: Path to the WFDB record without extension (e.g. 'data/raw/100').
    """
    try:
        import wfdb
    except ImportError:
        return None, "NO RESULT: Python 'wfdb' package is not installed."

    try:
        path = Path(record_path_without_ext)
        rec = wfdb.rdrecord(str(path))

        signals = rec.p_signal  # (samples, leads)
        if signals is None or signals.size == 0:
            return None, "NO RESULT: WFDB record contains no signal data."

        fs = float(rec.fs)
        if fs <= 0:
            return None, f"NO RESULT: Invalid sampling frequency in WFDB header ({fs} Hz)."

        lead_names = list(rec.sig_name) if rec.sig_name else [f"Lead_{i+1}" for i in range(signals.shape[1])]

        # If a specific lead is requested, isolate it
        if lead_name is not None:
            if lead_name in lead_names:
                idx = lead_names.index(lead_name)
                sig_data = signals[:, idx]
                selected_leads = [lead_name]
            else:
                # Fallback to first channel or return error
                return None, f"NO RESULT: Requested lead '{lead_name}' not in WFDB record ({', '.join(lead_names)})."
        else:
            sig_data = signals.T  # (leads, samples)
            selected_leads = lead_names

        recording = ECGRecording(
            record_id=f"REC-WFDB-{path.stem}",
            sampling_rate=fs,
            duration=len(signals) / fs,
            lead_names=selected_leads,
            number_of_leads=len(selected_leads),
            signals=sig_data,
            units=rec.units[0] if rec.units else "mV",
            patient_id=patient_id or path.stem,
            source_format="WFDB",
            metadata={
                "comments": rec.comments,
                "base_date": str(rec.base_date) if hasattr(rec, "base_date") else None,
                "base_time": str(rec.base_time) if hasattr(rec, "base_time") else None,
            },
        )
        is_valid, errs = recording.validate()
        if not is_valid:
            return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"

        return recording, None

    except Exception as e:
        return None, f"NO RESULT: Failed to read WFDB record: {str(e)}"
