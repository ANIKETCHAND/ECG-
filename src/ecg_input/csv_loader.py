"""
CSV ECG Signal Loader
=====================
Loads delimited CSV files into unified ECGRecording instances.
Performs header sniffing, delimiter detection, and numeric column isolation.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def load_csv_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    fs: Optional[float] = None,
    lead_name: str = "II",
    patient_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Load ECG from CSV file.

    Returns:
        (ECGRecording, None) if successful, or (None, error_message) on failure.
    """
    try:
        # Read content into string
        if isinstance(file_or_path, (str, Path)):
            with open(file_or_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        elif isinstance(file_or_path, bytes):
            content = file_or_path.decode("utf-8", errors="replace")
        elif hasattr(file_or_path, "read"):
            raw = file_or_path.read()
            content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        else:
            return None, "NO RESULT: Invalid file input type for CSV loader."

        if not content.strip():
            return None, "NO RESULT: CSV file is empty."

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return None, "NO RESULT: CSV file contains no valid data lines."

        # Detect delimiter (strictly excluding decimal points)
        sample_text = "\n".join(lines[:10])
        delimiter = ","
        try:
            sniffer = csv.Sniffer()
            detected = sniffer.sniff(sample_text).delimiter
            if detected in [",", "\t", ";", "|", " "]:
                delimiter = detected
        except Exception:
            delimiter = "," if "," in sample_text else ("\t" if "\t" in sample_text else (";" if ";" in sample_text else " "))


        # Parse rows
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        raw_rows = list(reader)

        # Separate headers and numeric rows
        numeric_data: List[List[float]] = []
        headers: List[str] = []

        for row in raw_rows:
            cleaned = [c.strip() for c in row if c.strip()]
            if not cleaned:
                continue
            # Check if row is numeric
            try:
                vals = [float(c) for c in cleaned]
                numeric_data.append(vals)
            except ValueError:
                # Treat as header if not already recorded
                if not headers and not numeric_data:
                    headers = cleaned

        if not numeric_data:
            return None, "NO RESULT: No valid numeric ECG voltage series found in CSV."

        arr = np.array(numeric_data, dtype=float)
        n_rows, n_cols = arr.shape

        # Column assignment:
        # If 1 column -> voltage
        # If 2 columns -> check if first is monotonic time
        if n_cols == 1:
            signals = arr[:, 0]
            leads = [lead_name]
        elif n_cols == 2:
            # Check if col 0 is time (strictly increasing)
            diffs = np.diff(arr[:, 0])
            if np.all(diffs > 0) and np.std(diffs) < 0.01:
                # Time column found; derive fs if not provided
                if fs is None or fs <= 0:
                    dt = np.median(diffs)
                    fs = round(1.0 / dt, 2) if dt > 0 else 360.0
                signals = arr[:, 1]
                leads = [lead_name]
            else:
                # Multi-lead or both voltage
                signals = arr.T
                leads = headers[:n_cols] if len(headers) >= n_cols else [f"Lead_{i+1}" for i in range(n_cols)]
        else:
            # Multi-channel ECG matrix
            signals = arr.T
            leads = headers[:n_cols] if len(headers) >= n_cols else [f"Lead_{i+1}" for i in range(n_cols)]

        if fs is None or fs <= 0:
            return None, "NO RESULT: Missing sampling rate. Cannot assume sampling frequency (Rule 4)."

        rec = ECGRecording(
            record_id=f"REC-CSV-{hash(tuple(signals.flatten()[:5])) & 0xFFFFFF:06X}",
            sampling_rate=float(fs),
            duration=len(signals.T) / float(fs) if signals.ndim > 1 else len(signals) / float(fs),
            lead_names=leads,
            number_of_leads=len(leads),
            signals=signals,
            units="mV",
            patient_id=patient_id,
            device=device,
            source_format="CSV",
            metadata={"headers": headers, "detected_delimiter": delimiter},
        )
        is_valid, errs = rec.validate()
        if not is_valid:
            return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"

        return rec, None

    except Exception as e:
        return None, f"NO RESULT: Failed to parse CSV: {str(e)}"
