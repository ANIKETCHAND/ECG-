"""
DICOM Waveform ECG Signal Loader
================================
Loads DICOM 12-Lead ECG Waveform Storage objects (SOP Class 1.2.840.10008.5.1.4.1.1.9.1.1)
and General ECG Waveform Storage objects.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def is_dicom_file(file_bytes: bytes) -> bool:
    """Check if byte stream contains standard DICOM 'DICM' preamble at offset 128."""
    if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
        return True
    return False


def load_dicom_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    lead_name: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Parse and load DICOM ECG waveform.

    Returns:
        (ECGRecording, None) if successful, or (None, error_message) on failure.
    """
    # Read raw bytes for preamble validation
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_or_path, bytes):
        raw_bytes = file_or_path
    elif hasattr(file_or_path, "read"):
        raw_bytes = file_or_path.read()
    else:
        return None, "NO RESULT: Invalid input type for DICOM loader."

    if not is_dicom_file(raw_bytes):
        return None, "NO RESULT: File lacks valid DICOM preamble ('DICM' marker at offset 128)."

    # Try importing pydicom
    try:
        import pydicom
    except ImportError:
        return None, (
            "NOT IMPLEMENTED: Native DICOM Waveform parsing requires 'pydicom' package. "
            "Please install pydicom or export ECG as digital CSV/EDF/WFDB."
        )

    try:
        ds = pydicom.dcmread(io.BytesIO(raw_bytes))

        # Verify Waveform Sequence presence
        if "WaveformSequence" not in ds or len(ds.WaveformSequence) == 0:
            return None, "NO RESULT: DICOM file is an image or report, not an ECG Waveform SOP Class."

        wf_item = ds.WaveformSequence[0]
        n_channels = getattr(wf_item, "NumberOfWaveformChannels", 1)
        n_samples = getattr(wf_item, "NumberOfWaveformSamples", 0)
        fs = getattr(wf_item, "SamplingFrequency", None)

        if fs is None or float(fs) <= 0:
            return None, "NO RESULT: DICOM Waveform lacks valid SamplingFrequency."

        raw_data = wf_item.WaveformData
        # Unpack based on WaveformBitsAllocated (typically 16-bit signed)
        bits = getattr(wf_item, "WaveformBitsAllocated", 16)
        if bits == 16:
            sig_arr = np.frombuffer(raw_data, dtype="<i2")
        elif bits == 8:
            sig_arr = np.frombuffer(raw_data, dtype="<i1")
        else:
            return None, f"NO RESULT: Unsupported DICOM waveform bit allocation ({bits}-bit)."

        # Reshape to (samples, channels)
        if len(sig_arr) == n_channels * n_samples:
            sig_matrix = sig_arr.reshape((n_samples, n_channels)).T  # (channels, samples)
        else:
            sig_matrix = sig_arr.reshape((n_channels, -1))

        # Channel labels from ChannelDefinitionSequence
        lead_names = []
        if "ChannelDefinitionSequence" in wf_item:
            for ch in wf_item.ChannelDefinitionSequence:
                # Source: ChannelSourceSequence -> CodeMeaning
                name = None
                if "ChannelSourceSequence" in ch and len(ch.ChannelSourceSequence) > 0:
                    name = getattr(ch.ChannelSourceSequence[0], "CodeMeaning", None)
                lead_names.append(name if name else f"Lead_{len(lead_names)+1}")
        else:
            lead_names = [f"Lead_{i+1}" for i in range(sig_matrix.shape[0])]

        pat_id = getattr(ds, "PatientID", patient_id or "Anonymous")

        rec = ECGRecording(
            record_id=f"REC-DCM-{getattr(ds, 'SOPInstanceUID', hash(raw_bytes[:20]))}",
            sampling_rate=float(fs),
            duration=sig_matrix.shape[1] / float(fs),
            lead_names=lead_names,
            number_of_leads=sig_matrix.shape[0],
            signals=sig_matrix,
            units="mV",
            patient_id=str(pat_id),
            device=str(getattr(ds, "ManufacturerModelName", "DICOM Device")),
            manufacturer=str(getattr(ds, "Manufacturer", "Unknown")),
            source_format="DICOM",
            metadata={
                "SOPClassUID": str(getattr(ds, "SOPClassUID", "")),
                "StudyDate": str(getattr(ds, "StudyDate", "")),
            },
        )
        is_valid, errs = rec.validate()
        if not is_valid:
            return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"

        return rec, None

    except Exception as e:
        return None, f"NO RESULT: Failed to parse DICOM: {str(e)}"
