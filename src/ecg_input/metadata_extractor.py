"""
Clinical Metadata Extractor
===========================
Extracts patient demographics, acquisition device metadata, technical calibrations,
machine-printed measurements, and diagnostic statements from ECG files and text reports.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def extract_patient_demographics(text: str) -> Dict[str, Optional[str]]:
    """Extract patient demographics from report text using regex patterns."""
    demographics: Dict[str, Optional[str]] = {
        "patient_id": None,
        "patient_name": None,
        "age": None,
        "sex": None,
        "date_of_birth": None,
    }

    if not text:
        return demographics

    # Patient ID / MRN
    id_match = re.search(r"(?:ID|MRN|Patient ID|Record No|File No)[:\s#]+([A-Za-z0-9\-_]+)", text, re.I)
    if id_match:
        demographics["patient_id"] = id_match.group(1).strip()

    # Patient Name
    name_match = re.search(r"(?:Name|Patient Name|Pt Name)[:\s]+([A-Za-z\s,\.]+?)(?:\s{2,}|\n|Age|DOB|Sex|ID|$)", text, re.I)
    if name_match:
        cand = name_match.group(1).strip()
        if len(cand) > 1 and not any(w in cand.lower() for w in ["unconfirmed", "normal", "abnormal", "ecg"]):
            demographics["patient_name"] = cand

    # Age
    age_match = re.search(r"(?:Age|Yr)[:\s]+(\d{1,3})\s*(?:yr|yo|years)?", text, re.I)
    if age_match:
        demographics["age"] = age_match.group(1)

    # Sex / Gender
    sex_match = re.search(r"(?:Sex|Gender)[:\s]+(Male|Female|M|F)\b", text, re.I)
    if sex_match:
        val = sex_match.group(1).upper()
        demographics["sex"] = "MALE" if val in ["M", "MALE"] else "FEMALE"

    return demographics


def extract_technical_calibrations(text: str) -> Dict[str, Optional[float]]:
    """Extract technical recording parameters (Paper speed mm/s, Voltage gain mm/mV, Sampling rate Hz)."""
    calibrations: Dict[str, Optional[float]] = {
        "paper_speed_mm_s": None,
        "voltage_gain_mm_mv": None,
        "sampling_rate_hz": None,
    }
    if not text:
        return calibrations

    # Speed: 25 mm/s, 50 mm/s
    speed_match = re.search(r"(\d+(?:\.\d+)?)\s*mm/s", text, re.I)
    if speed_match:
        calibrations["paper_speed_mm_s"] = float(speed_match.group(1))

    # Gain: 10 mm/mV, 5 mm/mV, 20 mm/mV
    gain_match = re.search(r"(\d+(?:\.\d+)?)\s*mm/mV", text, re.I)
    if gain_match:
        calibrations["voltage_gain_mm_mv"] = float(gain_match.group(1))

    # Sampling frequency: e.g. 500 Hz, 360 Hz
    fs_match = re.search(r"(?:fs|sampling rate|sample rate|frequency)[:\s]*(\d+(?:\.\d+)?)\s*Hz", text, re.I)
    if fs_match:
        calibrations["sampling_rate_hz"] = float(fs_match.group(1))

    return calibrations


def extract_machine_statement(text: str) -> Optional[str]:
    """Extract machine-printed diagnostic interpretation statement from report text."""
    if not text:
        return None

    # Common report heading markers for machine interpretation
    patterns = [
        r"(?:UNCONFIRMED DIAGNOSIS|INTERPRETATION|ANALYSIS|FINDINGS|DIAGNOSTIC STATEMENT)[:\s\n]+(.*?)(?=\n{2,}|\bTechnician\b|\bMD\b|\bDoctor\b|\bReferred\b|$)",
        r"(?:Normal Sinus Rhythm.*)",
        r"(?:Sinus Rhythm.*)",
        r"(?:Sinus Bradycardia.*)",
        r"(?:Sinus Tachycardia.*)",
        r"(?:Ventricular Premature.*)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.I | re.DOTALL)
        if m:
            stmt = m.group(0 if "(" not in pat else 1).strip()
            # Clean up multi-line statement
            cleaned = " ".join(stmt.split()[:40])
            if len(cleaned) > 5:
                return cleaned

    return None
