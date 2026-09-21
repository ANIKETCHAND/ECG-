"""
MIT-BIH to Standardized Record Converter
========================================
Converts native MIT-BIH WFDB files into universal StandardizedECGRecord entities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import wfdb

from src.datasets.patient_index import resolve_patient_id
from src.ecg_core.standardized_record import StandardizedECGRecord


def convert_mit_record_to_standard(
    record_path_or_name: str | Path,
    dataset_id: str = "mit_bih_arrhythmia",
) -> StandardizedECGRecord:
    """Read a PhysioNet WFDB record and convert into StandardizedECGRecord."""
    rec_str = str(record_path_or_name)
    # Strip extension if passed
    for ext in [".hea", ".dat", ".atr"]:
        if rec_str.endswith(ext):
            rec_str = rec_str[:-len(ext)]

    record = wfdb.rdrecord(rec_str)
    rec_name = Path(rec_str).name
    patient_id = resolve_patient_id(dataset_id, rec_name)

    annotations = {}
    try:
        ann = wfdb.rdann(rec_str, "atr")
        annotations = {
            "sample_indices": ann.sample.tolist(),
            "symbols": list(ann.symbol),
            "subtypes": ann.subtype.tolist() if hasattr(ann, "subtype") else [],
            "chan": ann.chan.tolist() if hasattr(ann, "chan") else [],
        }
    except Exception:
        pass

    # Shape: (n_samples, n_leads) -> transpose to (n_leads, n_samples)
    signals = record.p_signal.T if record.p_signal is not None else np.empty((0, 0))

    return StandardizedECGRecord(
        record_id=rec_name,
        patient_id=patient_id,
        dataset_id=dataset_id,
        sampling_rate=float(record.fs),
        duration=round(record.sig_len / record.fs, 3) if record.fs > 0 else 0.0,
        lead_names=list(record.sig_name) if record.sig_name else ["MLII", "V1"],
        number_of_leads=len(record.sig_name) if record.sig_name else 1,
        signals=signals,
        units=record.units[0] if record.units else "mV",
        annotations=annotations,
        patient_metadata={"record_name": rec_name},
        device_metadata={"comments": record.comments},
    )
