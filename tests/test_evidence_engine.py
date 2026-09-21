"""
Unit Tests for Phase 6: ECG Evidence Engine
===========================================
Validates beat-level clinical evidence, feature attribution, waveform snippet extraction,
and comprehensive evidence reporting for physician review.
"""

import numpy as np
import pytest

from src.evidence.beat_evidence import analyze_beats_for_evidence
from src.evidence.evidence_engine import generate_ai_evidence
from src.evidence.feature_evidence import compute_feature_attribution_for_beat
from src.evidence.waveform_evidence import extract_waveform_snippet


def test_beat_evidence_detection():
    fs = 360.0
    n_beats = 5
    window_size = 217
    beats = np.zeros((n_beats, window_size))

    # Peaks at regular 1-second intervals, except beat #3 is premature (0.6s after beat #2)
    # Beat 1: 360 (1.0s)
    # Beat 2: 720 (2.0s)
    # Beat 3: 936 (2.6s) -> premature! pre_rr = 0.6s
    # Beat 4: 1440 (4.0s) -> compensatory pause! post_rr = 1.4s
    # Beat 5: 1800 (5.0s)
    r_peaks = np.array([360, 720, 936, 1440, 1800])
    classes = ["Normal", "Other", "PVC"]

    # Probs: beat #3 has high PVC probability
    beat_probs = np.array([
        [0.95, 0.02, 0.03],
        [0.92, 0.03, 0.05],
        [0.05, 0.05, 0.90],  # Beat #3 PVC
        [0.94, 0.02, 0.04],
        [0.96, 0.01, 0.03],
    ])

    evidences = analyze_beats_for_evidence(
        beats=beats,
        r_peaks=r_peaks,
        fs=fs,
        beat_probs=beat_probs,
        classes=classes,
    )

    assert len(evidences) == 5
    assert evidences[0].is_aberrant is False
    assert evidences[1].is_aberrant is False
    # Beat #3 (index 2)
    b3 = evidences[2]
    assert b3.beat_number == 3
    assert b3.is_aberrant is True
    assert b3.predicted_class == "PVC"
    assert b3.pvc_probability == 0.90
    assert b3.prematurity_index < 0.85
    assert b3.compensatory_ratio > 1.15
    assert len(b3.morphology_notes) > 0


def test_feature_attribution_for_beat():
    feature_names = ["local_rr_ratio", "pre_rr", "post_rr", "rms", "energy"]
    normal_features = np.array([
        [1.0, 0.8, 0.8, 0.5, 10.0],
        [1.02, 0.81, 0.79, 0.51, 10.2],
        [0.98, 0.79, 0.82, 0.49, 9.8],
    ])
    # Aberrant beat with high deviation on local_rr_ratio and pre_rr
    aberrant_beat = np.array([0.5, 0.4, 1.3, 0.5, 10.0])

    attrs = compute_feature_attribution_for_beat(
        beat_features=aberrant_beat,
        baseline_features=normal_features,
        feature_names=feature_names,
        top_k=2,
    )

    assert len(attrs) == 2
    top_feature_names = [a["feature_name"] for a in attrs]
    assert "local_rr_ratio" in top_feature_names or "pre_rr" in top_feature_names
    assert attrs[0]["z_score_deviation"] != 0.0
    assert "clinical_significance" in attrs[0]


def test_extract_waveform_snippet():
    fs = 360.0
    sig = np.sin(np.linspace(0, 10, 1000))
    snip = extract_waveform_snippet(
        signal=sig,
        peak_sample=500,
        fs=fs,
        beat_number=2,
        lead_name="II",
        window_before_sec=0.2,
        window_after_sec=0.3,
    )
    assert snip.beat_number == 2
    assert snip.start_sample == 500 - int(0.2 * fs)
    assert snip.end_sample == 500 + int(0.3 * fs)
    assert len(snip.signal_slice) == snip.end_sample - snip.start_sample


def test_generate_ai_evidence_full_report():
    fs = 360.0
    duration = 6.0
    t = np.arange(int(fs * duration)) / fs
    sig = 0.5 * np.sin(2 * np.pi * 1.0 * t)

    r_peaks = np.array([360, 720, 936, 1440, 1800])
    n_beats = len(r_peaks)
    beats = np.zeros((n_beats, 217))
    feature_names = [f"feat_{i}" for i in range(28)]
    features = np.ones((n_beats, 28))
    # Modify beat 3 features
    features[2] = features[2] * 3.0

    beat_probs = np.array([
        [0.95, 0.02, 0.03],
        [0.92, 0.03, 0.05],
        [0.05, 0.05, 0.90],
        [0.94, 0.02, 0.04],
        [0.96, 0.01, 0.03],
    ])
    classes = ["Normal", "Other", "PVC"]

    report = generate_ai_evidence(
        signal=sig,
        fs=fs,
        r_peaks=r_peaks,
        beats=beats,
        features=features,
        feature_names=feature_names,
        beat_probs=beat_probs,
        classes=classes,
        overall_classification="Premature Ventricular Contraction",
        analysis_id="ANL-TEST-EVD",
        lead_name="II",
    )

    assert report.total_beats_analyzed == 5
    assert report.aberrant_beats_count == 1
    assert report.aberrant_beat_numbers == [3]
    assert "Beat #3" in report.evidence_summary
    assert len(report.waveform_snippets) == 1
    assert 3 in report.feature_attributions
    assert len(report.clinician_verification_checklist) > 0
    d = report.to_dict()
    assert d["analysis_id"] == "ANL-TEST-EVD"
