"""
ECG Report Measurement Extractor
================================

Extracts printed clinical measurements and patient information from ECG report text:
- Patient Name, Age, Sex, Recording Date
- Heart Rate / Ventricular Rate (BPM)
- PR interval, QRS duration, QT/QTc interval (ms)
- P, QRS, T electrical axes (degrees)
- Machine-printed interpretations (explicitly tagged as SOURCE/MACHINE, not AI diagnosis)

Research/educational use only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_report_measurements(text: str) -> Dict[str, Any]:
    """Parse printed text from an ECG report and extract standard parameters.

    Args:
        text: Raw text string extracted from PDF or OCR

    Returns:
        Dictionary with extracted parameters and source tags.
    """
    if not text or not text.strip():
        return _empty_measurements()

    clean_text = " ".join(text.split())

    measurements: Dict[str, Any] = {
        "patient_name": _extract_regex(clean_text, r"(?:Patient|Name|PT|Pt Name)\s*[:\-]\s*([A-Za-z,\s]+?)(?=\s+(?:ID|Age|Sex|DOB|Date|\d))"),
        "patient_age": _extract_regex(clean_text, r"(?:Age|Yr|Yrs)\s*[:\-]?\s*(\d{1,3})\s*(?:Y|Yr|Yrs|Years)?"),
        "patient_sex": _extract_sex(clean_text),
        "recording_date": _extract_regex(clean_text, r"(?:Date|Recorded|Time)\s*[:\-]?\s*(\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?)"),
        
        # Clinical electrical measurements
        "heart_rate_printed": _extract_numeric(clean_text, r"(?:Vent\.?\s*Rate|Heart\s*Rate|HR|Rate)\s*[:\-=]?\s*(\d{2,3})\s*(?:BPM|bpm)?"),
        "pr_interval_ms": _extract_numeric(clean_text, r"(?:PR|P\-R|PQ)\s*(?:interval|int)?\s*[:\-=]?\s*(\d{2,3})\s*(?:ms)?"),
        "qrs_duration_ms": _extract_numeric(clean_text, r"(?:QRS|QRS\s*dur|QRS\s*duration)\s*[:\-=]?\s*(\d{2,3})\s*(?:ms)?"),
        "qt_interval_ms": None,
        "qtc_interval_ms": None,
        
        # Axes
        "p_axis_deg": None,
        "qrs_axis_deg": None,
        "t_axis_deg": None,
        
        # Machine printed diagnosis/interpretation
        "machine_interpretation": _extract_machine_interpretation(text),
        "extraction_confidence": "Not Assessed",
        "has_extracted_data": False,
    }

    # Extract combined QT / QTc (e.g. "QT/QTc: 382/418 ms" or "QT/QTc 390/420")
    qt_qtc_match = re.search(r"(?:QT\s*/\s*QTc|QT\s*-\s*QTc)\s*[:\-=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", clean_text, re.IGNORECASE)
    if qt_qtc_match:
        measurements["qt_interval_ms"] = float(qt_qtc_match.group(1))
        measurements["qtc_interval_ms"] = float(qt_qtc_match.group(2))
    else:
        # Separate extraction
        measurements["qt_interval_ms"] = _extract_numeric(clean_text, r"\bQT\b\s*[:\-=]?\s*(\d{2,3})\s*(?:ms)?")
        measurements["qtc_interval_ms"] = _extract_numeric(clean_text, r"\bQTc\b(?:\s*\(Bazett\))?\s*[:\-=]?\s*(\d{2,3})\s*(?:ms)?")

    # Extract Axes e.g. "P-R-T axes: 48 65 32" or "P/QRS/T: 45/60/30"
    axes_match = re.search(r"(?:P[\s\-/]QRS[\s\-/]T|P[\s\-/]R[\s\-/]T)\s*axes?\s*[:\-=]?\s*([\-+]?\d{1,3})\s*[\s/]\s*([\-+]?\d{1,3})\s*[\s/]\s*([\-+]?\d{1,3})", clean_text, re.IGNORECASE)
    if axes_match:
        measurements["p_axis_deg"] = float(axes_match.group(1))
        measurements["qrs_axis_deg"] = float(axes_match.group(2))
        measurements["t_axis_deg"] = float(axes_match.group(3))

    # Evaluate if any valid clinical measurements were extracted
    valid_count = sum(
        1 for k in ["heart_rate_printed", "pr_interval_ms", "qrs_duration_ms", "qt_interval_ms", "qtc_interval_ms", "p_axis_deg"]
        if measurements[k] is not None
    )
    if valid_count >= 3:
        measurements["extraction_confidence"] = "High"
        measurements["has_extracted_data"] = True
    elif valid_count >= 1:
        measurements["extraction_confidence"] = "Moderate"
        measurements["has_extracted_data"] = True
    else:
        measurements["extraction_confidence"] = "Low"
        measurements["has_extracted_data"] = False

    return measurements


def _empty_measurements() -> Dict[str, Any]:
    return {
        "patient_name": None,
        "patient_age": None,
        "patient_sex": None,
        "recording_date": None,
        "heart_rate_printed": None,
        "pr_interval_ms": None,
        "qrs_duration_ms": None,
        "qt_interval_ms": None,
        "qtc_interval_ms": None,
        "p_axis_deg": None,
        "qrs_axis_deg": None,
        "t_axis_deg": None,
        "machine_interpretation": None,
        "extraction_confidence": "Low",
        "has_extracted_data": False,
    }


def _extract_regex(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        return val if len(val) > 0 else None
    return None


def _extract_numeric(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_sex(text: str) -> Optional[str]:
    match = re.search(r"(?:Sex|Gender)\s*[:\-]?\s*(Male|Female|M|F)\b", text, re.IGNORECASE)
    if match:
        raw = match.group(1).upper()
        if raw in ("M", "MALE"):
            return "Male"
        if raw in ("F", "FEMALE"):
            return "Female"
    return None


def _extract_machine_interpretation(text: str) -> Optional[List[str]]:
    """Extract printed diagnostic sentences generated by the ECG machine."""
    common_phrases = [
        "Normal sinus rhythm",
        "Sinus rhythm",
        "Sinus tachycardia",
        "Sinus bradycardia",
        "Atrial fibrillation",
        "Premature ventricular complexes",
        "Non-specific ST-T changes",
        "Left ventricular hypertrophy",
        "Right bundle branch block",
        "Left bundle branch block",
        "Normal ECG",
        "Abnormal ECG",
        "Borderline ECG",
        "ST elevation",
        "ST depression",
        "T wave inversion",
    ]
    found = []
    text_lower = text.lower()
    for phrase in common_phrases:
        if phrase.lower() in text_lower:
            found.append(phrase)

    return found if found else None
