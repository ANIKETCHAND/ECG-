"""
Longitudinal ECG Comparison Module
==================================
Performs pair-wise comparative analysis between a current ECG and a patient's prior baseline.
Strictly calculates objective metric deltas without clinical speculation or diagnostic hallucinations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LongitudinalChangeStatus(str, Enum):
    STABLE = "STABLE"
    SIGNIFICANT_CHANGE = "SIGNIFICANT_CHANGE"
    INCOMPATIBLE_COMPARISON = "INCOMPATIBLE_COMPARISON"


@dataclass
class LongitudinalComparisonResult:
    current_analysis_id: str
    prior_analysis_id: str
    patient_id: str
    status: LongitudinalChangeStatus
    delta_heart_rate_bpm: Optional[float]
    delta_mean_rr_ms: Optional[float]
    delta_qrs_duration_ms: Optional[float]
    delta_qtc_ms: Optional[float]
    prior_rhythm: str
    current_rhythm: str
    rhythm_transition_detected: bool
    objective_change_summary: str
    notable_deltas: List[str] = field(default_factory=list)
    clinician_action_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def compare_serial_ecgs(
    current_analysis: Dict[str, Any],
    prior_analysis: Dict[str, Any],
    patient_id: str = "UNKNOWN-PT",
) -> LongitudinalComparisonResult:
    """Compare a current ECG analysis against a prior ECG analysis for the same patient.

    Args:
        current_analysis: Dict from run_ecg_ml_inference / measurement engine.
        prior_analysis: Dict from previous ECG analysis.
        patient_id: Patient identifier.

    Returns:
        LongitudinalComparisonResult with objective deltas.
    """
    cur_lead = current_analysis.get("lead_analyzed", "II")
    prior_lead = prior_analysis.get("lead_analyzed", "II")

    # Lead compatibility check
    if cur_lead != prior_lead:
        return LongitudinalComparisonResult(
            current_analysis_id=current_analysis.get("analysis_id", "CUR"),
            prior_analysis_id=prior_analysis.get("analysis_id", "PRIOR"),
            patient_id=patient_id,
            status=LongitudinalChangeStatus.INCOMPATIBLE_COMPARISON,
            delta_heart_rate_bpm=None,
            delta_mean_rr_ms=None,
            delta_qrs_duration_ms=None,
            delta_qtc_ms=None,
            prior_rhythm=prior_analysis.get("prediction", "UNKNOWN"),
            current_rhythm=current_analysis.get("prediction", "UNKNOWN"),
            rhythm_transition_detected=False,
            objective_change_summary=f"Incompatible leads: Current lead '{cur_lead}' vs. Prior lead '{prior_lead}'. Direct interval comparison suppressed.",
            clinician_action_note="Compare identical leads to obtain valid serial interval measurements.",
        )

    # Rhythms
    cur_rhythm = current_analysis.get("prediction", "NO_RESULT")
    prior_rhythm = prior_analysis.get("prediction", "NO_RESULT")

    # Metrics
    cur_hr = current_analysis.get("heart_rate")
    prior_hr = prior_analysis.get("heart_rate")
    d_hr = round(float(cur_hr) - float(prior_hr), 1) if (cur_hr is not None and prior_hr is not None) else None

    # RR intervals
    cur_rrs = current_analysis.get("rr_intervals", [])
    prior_rrs = prior_analysis.get("rr_intervals", [])
    cur_mean_rr = float(sum(cur_rrs) / len(cur_rrs)) if cur_rrs else None
    prior_mean_rr = float(sum(prior_rrs) / len(prior_rrs)) if prior_rrs else None
    d_rr = round(cur_mean_rr - prior_mean_rr, 1) if (cur_mean_rr is not None and prior_mean_rr is not None) else None

    # QRS duration
    cur_qrs = current_analysis.get("qrs_duration_ms")
    prior_qrs = prior_analysis.get("qrs_duration_ms")
    d_qrs = round(float(cur_qrs) - float(prior_qrs), 1) if (cur_qrs is not None and prior_qrs is not None) else None

    # QTc
    cur_qtc = current_analysis.get("qtc_ms")
    prior_qtc = prior_analysis.get("qtc_ms")
    d_qtc = round(float(cur_qtc) - float(prior_qtc), 1) if (cur_qtc is not None and prior_qtc is not None) else None

    notable_deltas = []
    is_significant = False

    # Check rhythm shift
    rhythm_transition = bool(cur_rhythm != prior_rhythm and "NO_RESULT" not in cur_rhythm and "NO_RESULT" not in prior_rhythm)
    if rhythm_transition:
        is_significant = True
        notable_deltas.append(f"Rhythm shift detected: '{prior_rhythm}' -> '{cur_rhythm}'")

    # Check HR change (> 20 bpm)
    if d_hr is not None and abs(d_hr) >= 20.0:
        is_significant = True
        direction = "increase" if d_hr > 0 else "decrease"
        notable_deltas.append(f"Heart rate {direction} of {abs(d_hr)} bpm (from {prior_hr} to {cur_hr} bpm)")

    # Check QRS widening (>= 20 ms)
    if d_qrs is not None and d_qrs >= 20.0:
        is_significant = True
        notable_deltas.append(f"QRS widening of +{d_qrs} ms (from {prior_qrs} to {cur_qrs} ms)")

    # Check QTc prolongation (>= 30 ms)
    if d_qtc is not None and d_qtc >= 30.0:
        is_significant = True
        notable_deltas.append(f"QTc prolongation of +{d_qtc} ms (from {prior_qtc} to {cur_qtc} ms)")

    status = LongitudinalChangeStatus.SIGNIFICANT_CHANGE if is_significant else LongitudinalChangeStatus.STABLE

    if is_significant:
        summary = f"CHANGE DETECTED: {len(notable_deltas)} notable metric delta(s) observed relative to baseline."
        advisory = "Review current tracing against baseline to verify electrophysiological changes and evaluate patient clinical course."
    else:
        summary = "STABLE: Serial measurements consistent with prior baseline tracing within standard biological variation."
        advisory = "No urgent longitudinal interval deviations detected."

    return LongitudinalComparisonResult(
        current_analysis_id=current_analysis.get("analysis_id", "CUR"),
        prior_analysis_id=prior_analysis.get("analysis_id", "PRIOR"),
        patient_id=patient_id,
        status=status,
        delta_heart_rate_bpm=d_hr,
        delta_mean_rr_ms=d_rr,
        delta_qrs_duration_ms=d_qrs,
        delta_qtc_ms=d_qtc,
        prior_rhythm=prior_rhythm,
        current_rhythm=cur_rhythm,
        rhythm_transition_detected=rhythm_transition,
        objective_change_summary=summary,
        notable_deltas=notable_deltas,
        clinician_action_note=advisory,
    )
