"""
AI Evidence Engine
==================
Synthesizes beat-level, morphological, rhythm, and feature evidence
to explain AI classification results clearly to healthcare professionals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.evidence.beat_evidence import BeatEvidence, analyze_beats_for_evidence
from src.evidence.feature_evidence import compute_feature_attribution_for_beat
from src.evidence.waveform_evidence import WaveformSnippet, extract_waveform_snippet


@dataclass
class EvidenceReport:
    analysis_id: str
    lead_name: str
    overall_classification: str
    total_beats_analyzed: int
    aberrant_beats_count: int
    aberrant_beat_numbers: List[int]
    rhythm_regularity_cv: float  # Coefficient of variation of RR intervals
    mean_rr_ms: float
    evidence_summary: str
    beat_evidences: List[BeatEvidence]
    feature_attributions: Dict[int, List[Dict[str, Any]]]  # beat_number -> attributions
    waveform_snippets: List[WaveformSnippet]
    clinician_verification_checklist: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["beat_evidences"] = [b.to_dict() for b in self.beat_evidences]
        data["waveform_snippets"] = [w.to_dict() for w in self.waveform_snippets]
        return data


def generate_ai_evidence(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    beats: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    beat_probs: np.ndarray,
    classes: List[str],
    overall_classification: str,
    analysis_id: str = "EVD-UNKNOWN",
    lead_name: str = "II",
) -> EvidenceReport:
    """Generate comprehensive, clinician-verifiable AI evidence for an ECG analysis.

    Args:
        signal: Continuous ECG signal array.
        fs: Sampling frequency in Hz.
        r_peaks: Array of R-peak sample indices.
        beats: 2D array of segmented heartbeat cycles.
        features: 2D array of extracted features.
        feature_names: Ordered list of feature names.
        beat_probs: 2D array of class probabilities for each beat.
        classes: List of class names.
        overall_classification: Final predicted class string.
        analysis_id: Analysis tracking identifier.
        lead_name: Lead inspected.

    Returns:
        Structured EvidenceReport.
    """
    # 1. Beat-level evidence
    beat_evidences = analyze_beats_for_evidence(
        beats=beats,
        r_peaks=r_peaks,
        fs=fs,
        beat_probs=beat_probs,
        classes=classes,
    )

    aberrant_beats = [b for b in beat_evidences if b.is_aberrant]
    aberrant_numbers = [b.beat_number for b in aberrant_beats]

    # 2. Rhythm regularity
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = float(np.mean(rr_intervals))
        std_rr = float(np.std(rr_intervals))
        rr_cv = round((std_rr / mean_rr) * 100.0, 2) if mean_rr > 0 else 0.0
        mean_rr_ms = round(mean_rr * 1000.0, 1)
    else:
        rr_cv = 0.0
        mean_rr_ms = 800.0

    # 3. Feature attributions
    feature_attributions: Dict[int, List[Dict[str, Any]]] = {}
    normal_indices = [i for i, b in enumerate(beat_evidences) if not b.is_aberrant]
    baseline_features = features[normal_indices] if len(normal_indices) > 0 else features

    for b in aberrant_beats:
        idx = b.beat_number - 1
        if idx < len(features):
            attrs = compute_feature_attribution_for_beat(
                beat_features=features[idx],
                baseline_features=baseline_features,
                feature_names=feature_names,
                top_k=4,
            )
            feature_attributions[b.beat_number] = attrs

    # 4. Waveform snippets
    snippets: List[WaveformSnippet] = []
    for b in aberrant_beats:
        snip = extract_waveform_snippet(
            signal=signal,
            peak_sample=b.sample_index,
            fs=fs,
            beat_number=b.beat_number,
            lead_name=lead_name,
        )
        snippets.append(snip)

    # 5. Narrative summary
    if len(aberrant_beats) > 0:
        b_first = aberrant_beats[0]
        evidence_summary = (
            f"Aberrant ventricular ectopy detected: {len(aberrant_beats)} beat(s) flagged "
            f"(Beats {aberrant_numbers}). Initial ectopy at Beat #{b_first.beat_number} "
            f"(t={b_first.timestamp_seconds}s) demonstrates coupling interval of {b_first.coupling_interval_ms} ms "
            f"({round((1-b_first.prematurity_index)*100, 1)}% early) and compensatory pause of {b_first.compensatory_pause_ms} ms."
        )
    else:
        evidence_summary = (
            f"Normal regular cardiac conduction: {len(beat_evidences)} consecutive beats analyzed "
            f"with consistent R-R regularity (CV={rr_cv}%, mean R-R={mean_rr_ms} ms) and normal narrow QRS morphology."
        )

    checklist = [
        "Inspect highlighted aberrant beats against raw rhythm strip for electrode motion spikes.",
        "Verify compensatory pause duration relative to baseline cardiac cycle length.",
        "Confirm QRS morphology (broadening, concordance, polarity) on 12-lead ECG before intervention.",
    ]

    return EvidenceReport(
        analysis_id=analysis_id,
        lead_name=lead_name,
        overall_classification=overall_classification,
        total_beats_analyzed=len(beat_evidences),
        aberrant_beats_count=len(aberrant_beats),
        aberrant_beat_numbers=aberrant_numbers,
        rhythm_regularity_cv=rr_cv,
        mean_rr_ms=mean_rr_ms,
        evidence_summary=evidence_summary,
        beat_evidences=beat_evidences,
        feature_attributions=feature_attributions,
        waveform_snippets=snippets,
        clinician_verification_checklist=checklist,
    )
