"""
Longitudinal Trend Analysis Subsystem
=====================================
Computes trajectories and rate progressions across serial historical ECG recordings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class LongitudinalTrendReport:
    patient_id: str
    total_recordings: int
    timestamps: List[str]
    heart_rate_trajectory: List[Optional[float]]
    qrs_duration_trajectory: List[Optional[float]]
    qtc_trajectory: List[Optional[float]]
    rhythm_progression: List[str]
    hr_trend_slope: Optional[float]  # bpm per recording or day
    overall_stability: str  # STABLE, EVOLVING_RHYTHM, PROGRESSIVE_TACHYCARDIA, PROGRESSIVE_BRADYCARDIA
    trend_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_serial_trends(
    history: List[Dict[str, Any]],
    patient_id: str = "UNKNOWN-PT",
) -> LongitudinalTrendReport:
    """Analyze trend progression across chronologically ordered ECG analyses."""
    if not history:
        return LongitudinalTrendReport(
            patient_id=patient_id,
            total_recordings=0,
            timestamps=[],
            heart_rate_trajectory=[],
            qrs_duration_trajectory=[],
            qtc_trajectory=[],
            rhythm_progression=[],
            hr_trend_slope=None,
            overall_stability="INSUFFICIENT_DATA",
            trend_notes=["No historical ECG analyses available for trend evaluation."],
        )

    timestamps = [h.get("timestamp", f"Record-{i+1}") for i, h in enumerate(history)]
    hrs = [h.get("heart_rate") for h in history]
    qrs = [h.get("qrs_duration_ms") for h in history]
    qtcs = [h.get("qtc_ms") for h in history]
    rhythms = [str(h.get("prediction", "Unknown")) for h in history]

    # Calculate HR trend slope if >= 2 valid HR values
    valid_hrs = [(i, float(v)) for i, v in enumerate(hrs) if v is not None]
    notes = []
    hr_slope: Optional[float] = None
    stability = "STABLE"

    if len(valid_hrs) >= 2:
        xs = [p[0] for p in valid_hrs]
        ys = [p[1] for p in valid_hrs]
        poly = np.polyfit(xs, ys, 1)
        hr_slope = round(float(poly[0]), 2)

        if hr_slope >= 5.0:
            stability = "PROGRESSIVE_TACHYCARDIA"
            notes.append(f"Progressive upward heart rate trend (+{hr_slope} bpm/recording)")
        elif hr_slope <= -5.0:
            stability = "PROGRESSIVE_BRADYCARDIA"
            notes.append(f"Progressive downward heart rate trend ({hr_slope} bpm/recording)")

    # Check rhythm progression
    unique_rhythms = set(rhythms)
    if len(unique_rhythms) > 1:
        if stability == "STABLE":
            stability = "EVOLVING_RHYTHM"
        notes.append(f"Rhythm variations noted across timeline: {sorted(list(unique_rhythms))}")

    return LongitudinalTrendReport(
        patient_id=patient_id,
        total_recordings=len(history),
        timestamps=timestamps,
        heart_rate_trajectory=hrs,
        qrs_duration_trajectory=qrs,
        qtc_trajectory=qtcs,
        rhythm_progression=rhythms,
        hr_trend_slope=hr_slope,
        overall_stability=stability,
        trend_notes=notes,
    )
