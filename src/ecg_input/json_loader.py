"""
JSON ECG Signal Loader
======================
Loads structured JSON representations of ECG recordings into unified ECGRecording instances.
Supports both single-lead arrays, dictionary records, and standard FHIR/JSON schemas.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def load_json_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    fs: Optional[float] = None,
    lead_name: str = "II",
    patient_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Load ECG from JSON file or JSON string."""
    try:
        if isinstance(file_or_path, (str, Path)):
            with open(file_or_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        elif isinstance(file_or_path, bytes):
            content = json.loads(file_or_path.decode("utf-8"))
        elif hasattr(file_or_path, "read"):
            raw = file_or_path.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            content = json.loads(text)
        else:
            return None, "NO RESULT: Invalid input type for JSON loader."

        # Case 1: Plain list of floats: [0.1, 0.2, ...]
        if isinstance(content, list):
            if not content:
                return None, "NO RESULT: JSON array is empty."
            signals = np.array(content, dtype=float)
            if fs is None or fs <= 0:
                return None, "NO RESULT: Missing sampling rate. Cannot assume sampling frequency (Rule 4)."
            rec = ECGRecording(
                record_id=f"REC-JSON-{hash(tuple(signals[:5])) & 0xFFFFFF:06X}",
                sampling_rate=float(fs),
                duration=len(signals) / float(fs),
                lead_names=[lead_name],
                number_of_leads=1,
                signals=signals,
                units="mV",
                patient_id=patient_id,
                device=device,
                source_format="JSON",
            )
            return rec, None

        # Case 2: Structured dictionary
        if isinstance(content, dict):
            # Extract signal array
            signal_raw = content.get("signal") or content.get("signals") or content.get("data")
            if signal_raw is None:
                return None, "NO RESULT: JSON object does not contain 'signal' or 'data' field."

            signals = np.array(signal_raw, dtype=float)
            if signals.size == 0:
                return None, "NO RESULT: JSON signal array is empty."

            # Sampling rate resolution
            detected_fs = content.get("sampling_rate") or content.get("fs") or fs
            if detected_fs is None or float(detected_fs) <= 0:
                return None, "NO RESULT: Missing sampling rate. Cannot assume sampling frequency (Rule 4)."

            detected_leads = content.get("lead_names") or content.get("leads")
            if not detected_leads:
                detected_leads = [content.get("lead", lead_name)]

            rec_id = content.get("record_id") or f"REC-JSON-{hash(tuple(signals.flatten()[:5])) & 0xFFFFFF:06X}"

            rec = ECGRecording(
                record_id=str(rec_id),
                sampling_rate=float(detected_fs),
                duration=len(signals.T if signals.ndim > 1 else signals) / float(detected_fs),
                lead_names=detected_leads,
                number_of_leads=len(detected_leads),
                signals=signals,
                units=content.get("units", "mV"),
                patient_id=content.get("patient_id") or patient_id,
                device=content.get("device") or device,
                source_format="JSON",
                metadata=content.get("metadata", {}),
            )
            is_valid, errs = rec.validate()
            if not is_valid:
                return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"
            return rec, None

        return None, "NO RESULT: Unrecognized JSON structure for ECG signal."

    except Exception as e:
        return None, f"NO RESULT: Failed to parse JSON: {str(e)}"
