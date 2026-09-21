"""
ECG Machine Interpretation Parser
=================================
Parses automated diagnosis and measurement statements printed on ECG machine headers
(GE Marquette 12SL, Philips DXL, Mortara VERITAS, Schiller, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedMachineInterpretation:
    raw_statement: str
    normalized_category: str  # NORMAL_SINUS, VENTRICULAR_ECTOPY, ATRIAL_FIBRILLATION, ISCHEMIA_INFARCT, CONDUCTION_BLOCK, BRADYCARDIA, TACHYCARDIA, UNSPECIFIED
    reported_heart_rate: Optional[float] = None
    reported_pr_interval_ms: Optional[float] = None
    reported_qrs_duration_ms: Optional[float] = None
    reported_qtc_ms: Optional[float] = None
    is_unconfirmed: bool = True
    parsed_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_machine_interpretation(
    raw_text: Optional[str],
    measurements: Optional[Dict[str, Any]] = None,
) -> ParsedMachineInterpretation:
    """Parse raw machine interpretation text into structured clinical categories.

    Args:
        raw_text: Free-text string extracted from machine report or DICOM/XML tag.
        measurements: Optional numerical measurements dict from ECG header.

    Returns:
        ParsedMachineInterpretation object.
    """
    if not raw_text or not raw_text.strip():
        return ParsedMachineInterpretation(
            raw_statement="NO STATEMENT PROVIDED",
            normalized_category="UNSPECIFIED",
            is_unconfirmed=True,
            parsed_findings=["No machine statement available"],
        )

    text = raw_text.strip()
    upper = text.upper()

    # Unconfirmed detection
    is_unconfirmed = bool(
        "UNCONFIRMED" in upper
        or "UNVERIFIED" in upper
        or "AUTOMATED" in upper
        or "COMPUTER INTERPRETATION" in upper
    )

    findings = []
    category = "UNSPECIFIED"

    # Category matching
    if any(k in upper for k in ["PVC", "VENTRICULAR PREMATURE", "VENTRICULAR ECTOP", "VPB"]):
        category = "VENTRICULAR_ECTOPY"
        findings.append("Ventricular ectopy / PVC reported by machine")
    elif any(k in upper for k in ["INFARCT", "STEMI", "ST ELEVATION", "ACUTE MI", "ISCHEMI"]):
        category = "ISCHEMIA_INFARCT"
        findings.append("Acute ischemia / infarction pattern reported by machine")
    elif any(k in upper for k in ["ATRIAL FIBRILLATION", "AFIB", "A-FIB", "A. FIB"]):
        category = "ATRIAL_FIBRILLATION"
        findings.append("Atrial fibrillation / flutter reported by machine")
    elif any(k in upper for k in ["BLOCK", "BUNDLE BRANCH", "LBBB", "RBBB", "AV BLOCK"]):
        category = "CONDUCTION_BLOCK"
        findings.append("Intraventricular or atrioventricular conduction delay reported")
    elif any(k in upper for k in ["NORMAL SINUS", "NORMAL ECG", "WITHIN NORMAL LIMITS", "NSR"]):
        category = "NORMAL_SINUS"
        findings.append("Normal sinus rhythm reported by machine")
    elif "BRADYCARDIA" in upper:
        category = "BRADYCARDIA"
        findings.append("Sinus bradycardia reported by machine")
    elif "TACHYCARDIA" in upper:
        category = "TACHYCARDIA"
        findings.append("Sinus tachycardia reported by machine")

    # Extract numerical parameters if embedded in text
    hr: Optional[float] = None
    pr: Optional[float] = None
    qrs: Optional[float] = None
    qtc: Optional[float] = None

    hr_match = re.search(r"(?:HR|HEART RATE)[:\s=]+(\d+)", upper)
    if hr_match:
        hr = float(hr_match.group(1))

    pr_match = re.search(r"PR[:\s=]+(\d+)", upper)
    if pr_match:
        pr = float(pr_match.group(1))

    qrs_match = re.search(r"QRS[:\s=]+(\d+)", upper)
    if qrs_match:
        qrs = float(qrs_match.group(1))

    qtc_match = re.search(r"QTC[:\s=]+(\d+)", upper)
    if qtc_match:
        qtc = float(qtc_match.group(1))

    # Supplement from explicit measurements dict if available
    if measurements:
        if hr is None and "heart_rate" in measurements:
            hr = float(measurements["heart_rate"])
        if pr is None and "pr_interval" in measurements:
            pr = float(measurements["pr_interval"])
        if qrs is None and "qrs_duration" in measurements:
            qrs = float(measurements["qrs_duration"])
        if qtc is None and "qtc" in measurements:
            qtc = float(measurements["qtc"])

    return ParsedMachineInterpretation(
        raw_statement=text,
        normalized_category=category,
        reported_heart_rate=hr,
        reported_pr_interval_ms=pr,
        reported_qrs_duration_ms=qrs,
        reported_qtc_ms=qtc,
        is_unconfirmed=is_unconfirmed,
        parsed_findings=findings,
    )
