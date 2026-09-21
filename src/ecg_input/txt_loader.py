"""
TXT ECG Signal Loader
=====================
Loads raw whitespace/tab-delimited text files into unified ECGRecording instances.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
    from src.ecg_input.csv_loader import load_csv_ecg
except ImportError:
    from ecg_core.models import ECGRecording
    from ecg_input.csv_loader import load_csv_ecg



def load_txt_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    fs: Optional[float] = None,
    lead_name: str = "II",
    patient_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Load ECG from plain-text file.
    Reuses robust delimiter-sniffing engine with TXT format metadata tag.
    """
    rec, err = load_csv_ecg(
        file_or_path=file_or_path,
        fs=fs,
        lead_name=lead_name,
        patient_id=patient_id,
        device=device,
    )
    if rec is not None:
        rec.source_format = "TXT"
        return rec, None
    return None, err
