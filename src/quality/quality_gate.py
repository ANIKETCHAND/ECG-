"""
ECG Quality Gatekeeper
======================
Enforces the inviolable safety interlock:
NO RELIABLE INPUT = NO AI RESULT

Critical Rule:
If UNUSABLE -> NO AI ANALYSIS (Pipeline halted).
If POOR -> AI classification suppressed (Basic measurements only).
If ACCEPTABLE / GOOD -> AI analysis permitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from src.quality.lead_quality import evaluate_single_lead_quality
except ImportError:
    from quality.lead_quality import evaluate_single_lead_quality


class QualityCategory(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNUSABLE = "UNUSABLE"


@dataclass
class QualityGateDecision:
    """Immutable decision record from the ECG Quality Copilot."""
    category: QualityCategory
    can_run_ai: bool
    can_compute_measurements: bool
    quality_score: float
    snr_db: float
    lead_name: str
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


def evaluate_ecg_quality_gate(
    signal: Optional[np.ndarray],
    fs: float,
    lead_name: str = "II",
    min_duration_sec: float = 1.5,
) -> QualityGateDecision:
    """Execute pre-inference quality gatekeeper on an ECG signal."""
    # 1. Null / Empty Check
    if signal is None or len(signal) == 0:
        return QualityGateDecision(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-99.0,
            lead_name=lead_name,
            rejection_reasons=["No physiological signal data provided (empty input)."],
            recommendation="Provide a valid, non-empty ECG digital recording.",
        )

    sig = np.asarray(signal, dtype=float)

    # 2. Duration Check
    duration = len(sig) / float(fs) if fs > 0 else 0.0
    if duration < min_duration_sec:
        return QualityGateDecision(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-99.0,
            lead_name=lead_name,
            rejection_reasons=[f"Recording duration ({duration:.2f}s) is below minimum required ({min_duration_sec}s)."],
            recommendation=f"Acquire at least {min_duration_sec} seconds of continuous ECG recording.",
        )

    # 3. Sampling frequency validation
    if fs <= 0 or fs > 10000:
        return QualityGateDecision(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-99.0,
            lead_name=lead_name,
            rejection_reasons=[f"Sampling frequency ({fs} Hz) is outside physiological range."],
            recommendation="Verify digitizer clock or provide standard 250-500 Hz ECG.",
        )

    # 4. Lead-level quality evaluation
    res = evaluate_single_lead_quality(sig, fs, lead_name=lead_name)
    status_str = res["status"]
    cat = QualityCategory(status_str)

    rejection_reasons: List[str] = []
    warnings: List[str] = []

    if cat == QualityCategory.UNUSABLE:
        can_ai = False
        can_measure = False
        rejection_reasons.extend(res["issues"])
    elif cat == QualityCategory.POOR:
        can_ai = False
        can_measure = True
        warnings.extend(res["issues"])
        rejection_reasons.append("Signal quality is POOR: AI rhythm inference suppressed for patient safety.")
    else:  # GOOD or ACCEPTABLE
        can_ai = True
        can_measure = True
        warnings.extend(res["issues"])

    return QualityGateDecision(
        category=cat,
        can_run_ai=can_ai,
        can_compute_measurements=can_measure,
        quality_score=res["quality_score"],
        snr_db=res["snr_db"],
        lead_name=lead_name,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
        recommendation=res["recommendation"],
        metrics=res["metrics"],
    )
