"""
Medication x QTc Temporal Association
=====================================

Neither the medication-safety module nor the longitudinal module can answer this
question alone:

    "The patient's QTc has risen — does that track the drug we started?"

This module joins the two: it pairs serial QTc measurements with medication start
dates and reports whether a rise is *temporally consistent* with a QT-prolonging
agent being introduced.

What it does and does not claim
-------------------------------
It reports **temporal association only**. Drug-induced QT prolongation is a
diagnosis requiring electrocardiographic review, electrolyte assessment and
clinical judgement; a rising trend after a start date is a hypothesis worth
raising with a clinician, not a causal finding. The report says so in every
output, and refuses to produce a comparison when either side of the drug start
date is unobserved.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from medications.medication_database import GLOBAL_MEDICATION_DB, MedicationEntry
except (ImportError, ValueError):
    from src.medications.medication_database import GLOBAL_MEDICATION_DB, MedicationEntry


#: QTc change thresholds, in ms and percent, used to describe a rise.
SIGNIFICANT_ABSOLUTE_DELTA_MS = 20.0
SIGNIFICANT_RELATIVE_DELTA_PERCENT = 8.0

#: Minimum recordings required on each side of a medication start date.
MIN_RECORDINGS_PER_SIDE = 1

RISK_HIGH = {"HIGH", "MODERATE"}


@dataclass
class MedicationQTcAssociation:
    medication: str
    qt_prolongation_risk: str
    medication_start: Optional[str]
    pre_start_qtc_ms: Optional[float]
    pre_start_recordings: int
    post_start_qtc_ms: Optional[float]
    post_start_recordings: int
    qtc_delta_ms: Optional[float]
    qtc_delta_percent: Optional[float]
    association: str  # SUPPORTED, NOT_SUPPORTED, INSUFFICIENT_DATA
    confidence: str  # HIGH, MODERATE, LOW
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MedicationQTcReport:
    patient_id: str
    recordings_considered: int
    associations: List[MedicationQTcAssociation] = field(default_factory=list)
    review_required: bool = False
    summary: str = ""
    clinical_note: str = (
        "Temporal association only. A QTc rise following a medication start is a hypothesis for "
        "clinician review, not evidence of causation. Confirm with serial ECGs, electrolytes "
        "(K+, Mg2+, Ca2+) and the patient's clinical picture."
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["associations"] = [a.to_dict() for a in self.associations]
        return data


def _parse_date(value: Any) -> Optional[datetime.date]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _get_qtc(recording: Dict[str, Any]) -> Optional[float]:
    """Extract a QTc value (ms) from several tolerated shapes of history entry."""
    for key in ("qtc_ms", "qtc_interval", "qtc"):
        value = recording.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    measurements = recording.get("ecg_measurements") or recording.get("measurements") or {}
    if isinstance(measurements, dict):
        for key in ("qtc_interval", "qtc_ms", "qtc"):
            value = measurements.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


def _lookup_medication(name: str) -> Optional[MedicationEntry]:
    try:
        return GLOBAL_MEDICATION_DB.get_by_name(name)
    except Exception:
        return None


def assess_qtc_medication_association(
    history: Sequence[Dict[str, Any]],
    medications: Sequence[Dict[str, Any]],
    patient_id: str = "UNKNOWN-PT",
) -> MedicationQTcReport:
    """Pair serial QTc records with medication start dates.

    Args:
        history: Chronological ECG history entries. Each may carry
            ``timestamp`` and a QTc value under ``qtc_ms``/``qtc_interval`` or a
            nested ``ecg_measurements`` dict.
        medications: Medication entries with ``medication_name`` (or
            ``generic_name``) and an optional ``start_date``.
        patient_id: Patient identifier for the report header.

    Returns:
        A :class:`MedicationQTcReport`. Associations are reported only where both
        a pre-start and post-start QTc observation exist.
    """
    report = MedicationQTcReport(patient_id=patient_id, recordings_considered=len(history))

    # Build a dated QTc series, dropping entries we cannot place in time.
    dated_qtc: List[tuple[datetime.date, float]] = []
    for recording in history:
        date = _parse_date(recording.get("timestamp") or recording.get("recording_timestamp") or recording.get("date"))
        qtc = _get_qtc(recording)
        if date is not None and qtc is not None:
            dated_qtc.append((date, qtc))
    dated_qtc.sort(key=lambda item: item[0])

    if not dated_qtc or not medications:
        report.summary = (
            "Insufficient data for a medication-QTc association: "
            f"{len(dated_qtc)} dated QTc value(s) and {len(medications)} medication(s) available."
        )
        return report

    for medication in medications:
        name = (
            medication.get("medication_name")
            or medication.get("generic_name")
            or medication.get("name")
            or "UNKNOWN"
        )
        entry = _lookup_medication(str(name))
        risk = entry.qt_prolongation_risk if entry else "UNKNOWN"
        start = _parse_date(medication.get("start_date") or medication.get("started_at"))

        if start is None:
            report.associations.append(
                MedicationQTcAssociation(
                    medication=str(name),
                    qt_prolongation_risk=risk,
                    medication_start=None,
                    pre_start_qtc_ms=None,
                    pre_start_recordings=0,
                    post_start_qtc_ms=None,
                    post_start_recordings=0,
                    qtc_delta_ms=None,
                    qtc_delta_percent=None,
                    association="INSUFFICIENT_DATA",
                    confidence="LOW",
                    note="This medication has no recorded start date, so no temporal pairing is possible.",
                )
            )
            continue

        pre = [qtc for date, qtc in dated_qtc if date < start]
        post = [qtc for date, qtc in dated_qtc if date >= start]

        if len(pre) < MIN_RECORDINGS_PER_SIDE or len(post) < MIN_RECORDINGS_PER_SIDE:
            report.associations.append(
                MedicationQTcAssociation(
                    medication=str(name),
                    qt_prolongation_risk=risk,
                    medication_start=start.isoformat(),
                    pre_start_qtc_ms=float(max(pre)) if pre else None,
                    pre_start_recordings=len(pre),
                    post_start_qtc_ms=float(max(post)) if post else None,
                    post_start_recordings=len(post),
                    qtc_delta_ms=None,
                    qtc_delta_percent=None,
                    association="INSUFFICIENT_DATA",
                    confidence="LOW",
                    note=(
                        f"Needs at least {MIN_RECORDINGS_PER_SIDE} ECG with a QTc value on each side of "
                        f"{start.isoformat()}; found {len(pre)} before and {len(post)} after."
                    ),
                )
            )
            continue

        pre_qtc = float(max(pre))
        post_qtc = float(max(post))
        delta = post_qtc - pre_qtc
        delta_percent = (delta / pre_qtc * 100.0) if pre_qtc > 0 else None

        meets_threshold = delta >= SIGNIFICANT_ABSOLUTE_DELTA_MS or (
            delta_percent is not None and delta_percent >= SIGNIFICANT_RELATIVE_DELTA_PERCENT
        )
        is_prolonging = risk in RISK_HIGH

        if meets_threshold and is_prolonging:
            association = "SUPPORTED"
            confidence = "MODERATE" if (len(pre) + len(post)) >= 4 else "LOW"
            note = (
                f"QTc rose {delta:+.1f} ms after {name} ({risk} QT-prolongation risk) was started. "
                "Temporal association is consistent with the medication; clinical confirmation required."
            )
            report.review_required = True
        elif meets_threshold and not is_prolonging:
            association = "NOT_SUPPORTED"
            confidence = "LOW"
            note = (
                f"QTc rose {delta:+.1f} ms after {name} started, but this drug is not classified as "
                "QT-prolonging. Look for another explanation (electrolytes, ischaemia, other agents)."
            )
            report.review_required = True
        else:
            association = "NOT_SUPPORTED"
            confidence = "MODERATE" if (len(pre) + len(post)) >= 4 else "LOW"
            note = f"No significant QTc change ({delta:+.1f} ms) across the {name} start date."

        report.associations.append(
            MedicationQTcAssociation(
                medication=str(name),
                qt_prolongation_risk=risk,
                medication_start=start.isoformat(),
                pre_start_qtc_ms=round(pre_qtc, 1),
                pre_start_recordings=len(pre),
                post_start_qtc_ms=round(post_qtc, 1),
                post_start_recordings=len(post),
                qtc_delta_ms=round(delta, 1),
                qtc_delta_percent=round(delta_percent, 2) if delta_percent is not None else None,
                association=association,
                confidence=confidence,
                note=note,
            )
        )

    supported = [a for a in report.associations if a.association == "SUPPORTED"]
    if supported:
        names = ", ".join(a.medication for a in supported)
        report.summary = (
            f"{len(supported)} medication(s) show a temporal association with a QTc rise ({names}). "
            "Clinician review advised."
        )
    elif report.associations:
        report.summary = "No medication showed a temporal association with a significant QTc rise."
    else:
        report.summary = "No assessable medications were provided."

    return report
