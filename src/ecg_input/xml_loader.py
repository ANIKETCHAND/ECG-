"""
XML / HL7 aECG ECG Signal Loader
================================
Parses XML format electrocardiograms, including HL7 aECG and vendor-structured XML files.
Extracts sequence digits, sampling rates, leads, and patient metadata.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
except ImportError:
    from ecg_core.models import ECGRecording



def load_xml_ecg(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    fs: Optional[float] = None,
    lead_name: str = "II",
    patient_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[ECGRecording], Optional[str]]:
    """Parse and load ECG from XML document."""
    try:
        if isinstance(file_or_path, (str, Path)):
            with open(file_or_path, "rb") as f:
                content = f.read()
        elif isinstance(file_or_path, bytes):
            content = file_or_path
        elif hasattr(file_or_path, "read"):
            data = file_or_path.read()
            content = data if isinstance(data, bytes) else data.encode("utf-8")
        else:
            return None, "NO RESULT: Invalid input type for XML loader."

        if not content.strip():
            return None, "NO RESULT: XML file is empty."

        try:
            root = ET.fromstring(content)
        except ET.ParseError as pe:
            return None, f"NO RESULT: Corrupted XML structure: {str(pe)}"

        detected_leads: List[str] = []
        signal_series: List[List[float]] = []
        detected_fs = fs

        # 1. Look for HL7 aECG / Sierra format: elements containing digits or sequence
        # Check standard <sequence>, <digits>, <lead>, or <waveform> elements
        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower()  # Strip XML namespace
            if tag in ["digits", "sequence", "waveformdata", "data", "points"]:
                text = (elem.text or "").strip()
                if not text:
                    continue
                # Digits can be space, comma, or semicolon delimited
                tokens = text.replace(",", " ").replace(";", " ").split()
                try:
                    vals = [float(t) for t in tokens]
                    if len(vals) > 50:
                        # Determine lead name from parent or attribute
                        lead_lbl = (
                            elem.attrib.get("lead")
                            or elem.attrib.get("name")
                            or (elem.attrib.get("code") if "code" in elem.attrib else None)
                            or lead_name
                        )
                        detected_leads.append(lead_lbl)
                        signal_series.append(vals)
                except ValueError:
                    continue

            # Check sampling frequency attributes
            if detected_fs is None:
                for attr_key, attr_val in elem.attrib.items():
                    if "rate" in attr_key.lower() or "frequency" in attr_key.lower() or "fs" in attr_key.lower():
                        try:
                            val = float(attr_val)
                            if val > 0:
                                detected_fs = val
                        except ValueError:
                            pass

        if not signal_series:
            return None, "NO RESULT: No valid numeric ECG sequence data found in XML."

        if detected_fs is None or detected_fs <= 0:
            return None, "NO RESULT: Missing sampling rate. Cannot assume sampling frequency (Rule 4)."

        signals = np.array(signal_series, dtype=float)
        if signals.ndim == 1:
            signals = np.expand_dims(signals, axis=0)

        rec = ECGRecording(
            record_id=f"REC-XML-{hash(tuple(content[:20])) & 0xFFFFFF:06X}",
            sampling_rate=float(detected_fs),
            duration=signals.shape[1] / float(detected_fs),
            lead_names=detected_leads if detected_leads else [lead_name],
            number_of_leads=signals.shape[0],
            signals=signals if signals.shape[0] > 1 else signals[0],
            units="mV",
            patient_id=patient_id,
            device=device,
            source_format="XML",
            metadata={"xml_root_tag": root.tag},
        )
        is_valid, errs = rec.validate()
        if not is_valid:
            return None, f"NO RESULT: Validation failed: {'; '.join(errs)}"

        return rec, None

    except Exception as e:
        return None, f"NO RESULT: Failed to parse XML: {str(e)}"
