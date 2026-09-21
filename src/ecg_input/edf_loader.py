"""
EDF / EDF+ European Data Format ECG Signal Loader
=================================================
Pure-Python native parser for European Data Format (EDF / EDF+) medical recordings.
Converts 16-bit digitized data records into calibrated physical millivolt (mV) signals.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def load_edf_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    lead_name: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Parse and load EDF/EDF+ format file.

    Returns:
        (ECGRecording, None) if successful, or (None, error_message) on failure.
    """
    try:
        if isinstance(file_or_path, (str, Path)):
            with open(file_or_path, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_or_path, bytes):
            raw_bytes = file_or_path
        elif hasattr(file_or_path, "read"):
            data = file_or_path.read()
            raw_bytes = data if isinstance(data, bytes) else data.encode("latin-1")
        else:
            return None, "NO RESULT: Invalid input type for EDF loader."

        if len(raw_bytes) < 256:
            return None, "NO RESULT: File too small to contain valid EDF header (minimum 256 bytes)."

        # Parse main 256-byte header
        header_text = raw_bytes[:256].decode("latin-1", errors="replace")
        version = header_text[:8].strip()
        patient_info = header_text[8:88].strip()
        record_info = header_text[88:168].strip()
        start_date = header_text[168:176].strip()
        start_time = header_text[176:184].strip()

        try:
            header_bytes = int(header_text[184:192].strip())
            n_records = int(header_text[236:244].strip())
            record_duration = float(header_text[244:252].strip())
            n_signals = int(header_text[252:256].strip())
        except ValueError as ve:
            return None, f"NO RESULT: Corrupted numeric field in EDF header: {str(ve)}"

        if n_signals <= 0:
            return None, f"NO RESULT: Invalid signal count in EDF header ({n_signals})."

        expected_header_len = 256 + (n_signals * 256)
        if len(raw_bytes) < expected_header_len:
            return None, "NO RESULT: Incomplete EDF signal header section."

        # Parse signal-specific headers
        offset = 256
        labels = [raw_bytes[offset + i * 16 : offset + (i + 1) * 16].decode("latin-1").strip() for i in range(n_signals)]
        offset += n_signals * 16
        offset += n_signals * 80  # Transducer types
        dimensions = [raw_bytes[offset + i * 8 : offset + (i + 1) * 8].decode("latin-1").strip() for i in range(n_signals)]
        offset += n_signals * 8
        phys_min = [float(raw_bytes[offset + i * 8 : offset + (i + 1) * 8].decode("latin-1").strip() or 0.0) for i in range(n_signals)]
        offset += n_signals * 8
        phys_max = [float(raw_bytes[offset + i * 8 : offset + (i + 1) * 8].decode("latin-1").strip() or 1.0) for i in range(n_signals)]
        offset += n_signals * 8
        dig_min = [float(raw_bytes[offset + i * 8 : offset + (i + 1) * 8].decode("latin-1").strip() or -32768) for i in range(n_signals)]
        offset += n_signals * 8
        dig_max = [float(raw_bytes[offset + i * 8 : offset + (i + 1) * 8].decode("latin-1").strip() or 32767) for i in range(n_signals)]
        offset += n_signals * 8
        offset += n_signals * 80  # Prefiltering
        n_samples_per_record = [int(raw_bytes[offset + i * 8 : offset + (i + 1) * 8].decode("latin-1").strip() or 0) for i in range(n_signals)]
        offset += n_signals * 8
        offset += n_signals * 32  # Reserved

        if any(s <= 0 for s in n_samples_per_record):
            return None, "NO RESULT: Zero or negative samples per record in EDF header."

        if record_duration <= 0:
            return None, "NO RESULT: Invalid record duration in EDF header."

        # Calculate sampling rates per signal
        sampling_rates = [round(float(s) / record_duration, 2) for s in n_samples_per_record]
        primary_fs = sampling_rates[0]

        # Read signal data
        data_bytes = raw_bytes[header_bytes:]
        bytes_per_record = sum(s * 2 for s in n_samples_per_record)

        if n_records < 0:
            n_records = len(data_bytes) // bytes_per_record if bytes_per_record > 0 else 0

        if n_records == 0 or len(data_bytes) < bytes_per_record:
            return None, "NO RESULT: EDF file contains zero complete data records."

        # Unpack 16-bit integers
        parsed_signals: List[List[float]] = [[] for _ in range(n_signals)]
        rec_offset = 0

        for _ in range(n_records):
            if rec_offset + bytes_per_record > len(data_bytes):
                break
            for sig_idx in range(n_signals):
                count = n_samples_per_record[sig_idx]
                chunk = data_bytes[rec_offset : rec_offset + count * 2]
                rec_offset += count * 2
                ints = struct.unpack(f"<{count}h", chunk)
                
                # Calibrate to physical units
                d_min = dig_min[sig_idx]
                d_max = dig_max[sig_idx]
                p_min = phys_min[sig_idx]
                p_max = phys_max[sig_idx]
                scale = (p_max - p_min) / (d_max - d_min) if d_max != d_min else 1.0

                unit = dimensions[sig_idx].lower()
                # If units are microvolts (uV), scale to mV
                unit_scale = 0.001 if "uv" in unit or "µv" in unit else (1000.0 if "v" == unit else 1.0)

                calibrated = [(float(val) - d_min) * scale * unit_scale + p_min * unit_scale for val in ints]
                parsed_signals[sig_idx].extend(calibrated)

        sig_arrays = np.array(parsed_signals, dtype=float)

        # Isolate requested lead if specified
        if lead_name is not None:
            # Check matching label
            matched_indices = [idx for idx, lbl in enumerate(labels) if lead_name.lower() in lbl.lower()]
            if matched_indices:
                chosen_idx = matched_indices[0]
                signals = sig_arrays[chosen_idx]
                leads = [labels[chosen_idx]]
            else:
                signals = sig_arrays[0]
                leads = [labels[0]]
        else:
            signals = sig_arrays
            leads = labels

        rec = ECGRecording(
            record_id=f"REC-EDF-{hash(tuple(raw_bytes[:30])) & 0xFFFFFF:06X}",
            sampling_rate=primary_fs,
            duration=len(sig_arrays[0]) / primary_fs,
            lead_names=leads,
            number_of_leads=len(leads),
            signals=signals,
            units="mV",
            patient_id=patient_id or patient_info or "Anonymous",
            source_format="EDF",
            metadata={
                "patient_info": patient_info,
                "record_info": record_info,
                "start_date": start_date,
                "start_time": start_time,
                "n_records": n_records,
                "original_labels": labels,
            },
        )
        is_valid, errs = rec.validate()
        if not is_valid:
            return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"

        return rec, None

    except Exception as e:
        return None, f"NO RESULT: Failed to parse EDF: {str(e)}"
