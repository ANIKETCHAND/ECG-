"""
Beat-Level Clinical Evidence Subsystem
======================================
Identifies specific aberrant beats with physiological coupling intervals,
prematurity indices, compensatory pauses, and QRS morphologic widths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class BeatEvidence:
    beat_number: int  # 1-indexed for physician communication
    sample_index: int
    timestamp_seconds: float
    is_aberrant: bool
    predicted_class: str
    pvc_probability: float
    coupling_interval_ms: float
    compensatory_pause_ms: Optional[float]
    prematurity_index: float  # pre_rr / mean_rr
    compensatory_ratio: Optional[float]  # post_rr / mean_rr
    qrs_duration_ms: float
    is_wide_qrs: bool
    morphology_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_beats_for_evidence(
    beats: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    beat_probs: np.ndarray,
    classes: List[str],
    pvc_prob_threshold: float = 0.50,
) -> List[BeatEvidence]:
    """Analyze every segmented beat and collect physiological evidence.

    Args:
        beats: 2D array of heartbeat waveforms (n_beats, window_size).
        r_peaks: 1D array of R-peak sample indices.
        fs: Sampling rate in Hz.
        beat_probs: 2D array of class probabilities (n_beats, n_classes).
        classes: Ordered list of class names (e.g. ['Normal', 'Other', 'PVC']).
        pvc_prob_threshold: Probability threshold above which a beat is considered aberrant.

    Returns:
        List of BeatEvidence objects for all beats.
    """
    n_beats = len(beats)
    if n_beats == 0:
        return []

    pvc_idx = classes.index("PVC") if "PVC" in classes else -1
    normal_idx = classes.index("Normal") if "Normal" in classes else -1

    # Calculate RR intervals
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8
    else:
        rr_intervals = np.array([])
        mean_rr = 0.8

    evidences: List[BeatEvidence] = []

    for i in range(n_beats):
        peak_sample = int(r_peaks[i]) if i < len(r_peaks) else 0
        t_sec = round(float(peak_sample / fs), 3)

        # RR metrics
        pre_rr_sec = float(rr_intervals[i - 1]) if i > 0 and len(rr_intervals) > 0 else mean_rr
        post_rr_sec = float(rr_intervals[i]) if i < len(rr_intervals) else None

        coupling_ms = round(pre_rr_sec * 1000.0, 1)
        pause_ms = round(post_rr_sec * 1000.0, 1) if post_rr_sec is not None else None

        prematurity_index = round(pre_rr_sec / mean_rr, 3) if mean_rr > 0 else 1.0
        compensatory_ratio = round(post_rr_sec / mean_rr, 3) if (post_rr_sec is not None and mean_rr > 0) else None

        # Probabilities
        probs = beat_probs[i] if i < len(beat_probs) else np.zeros(len(classes))
        pvc_prob = float(probs[pvc_idx]) if pvc_idx >= 0 and pvc_idx < len(probs) else 0.0
        top_cls_idx = int(np.argmax(probs)) if len(probs) > 0 else 0
        predicted_class = classes[top_cls_idx] if top_cls_idx < len(classes) else "Unknown"

        # Morphological QRS estimate
        beat_wf = beats[i]
        slopes = np.abs(np.diff(beat_wf))
        max_slope = np.max(slopes) if len(slopes) > 0 else 1.0
        high_slope_samples = int(np.sum(slopes > 0.12 * max_slope)) if max_slope > 0 else 0
        qrs_duration_ms = round((high_slope_samples / fs) * 1000.0, 1)
        is_wide_qrs = qrs_duration_ms > 120.0

        morphology_notes = []
        is_aberrant = False

        if pvc_prob >= pvc_prob_threshold or predicted_class == "PVC":
            is_aberrant = True
            morphology_notes.append("Elevated PVC probability from ensemble classifier")

        if prematurity_index < 0.85:
            morphology_notes.append(f"Premature coupling interval ({coupling_ms} ms, {round((1-prematurity_index)*100, 1)}% early)")
            if pvc_prob > 0.35:
                is_aberrant = True

        if compensatory_ratio is not None and compensatory_ratio > 1.15:
            morphology_notes.append(f"Compensatory post-ectopic pause ({pause_ms} ms, {round((compensatory_ratio-1)*100, 1)}% extended)")

        if is_wide_qrs:
            morphology_notes.append(f"Broadened QRS complex duration ({qrs_duration_ms} ms > 120 ms threshold)")

        evidences.append(
            BeatEvidence(
                beat_number=i + 1,
                sample_index=peak_sample,
                timestamp_seconds=t_sec,
                is_aberrant=is_aberrant,
                predicted_class=predicted_class,
                pvc_probability=round(pvc_prob, 4),
                coupling_interval_ms=coupling_ms,
                compensatory_pause_ms=pause_ms,
                prematurity_index=prematurity_index,
                compensatory_ratio=compensatory_ratio,
                qrs_duration_ms=qrs_duration_ms,
                is_wide_qrs=is_wide_qrs,
                morphology_notes=morphology_notes,
            )
        )

    return evidences
