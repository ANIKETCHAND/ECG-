"""
Tests for shadow-mode model comparison and the simulated streaming path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.models.registry import GLOBAL_MODEL_REGISTRY
from src.ml.shadow import (
    DEFAULT_CANDIDATE_MODEL_ID,
    PRODUCTION_MODEL_ID,
    run_shadow_comparison,
)
from src.streaming.simulated_stream import (
    run_streaming_demo,
    stream_beats,
    stream_window_indices,
    summarise_stream,
)

PROTECTED = {"label", "record_id", "symbol"}


def _feature_matrix(rows: int = 300) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv("data/processed/test_dataset.csv").head(rows)
    names = [c for c in frame.columns if c not in PROTECTED]
    return frame[names].to_numpy(float), names


# ------------------------------------------------------------------ shadow mode
def test_shadow_comparison_runs_and_reports_agreement():
    features, _ = _feature_matrix()
    report = run_shadow_comparison(features)

    assert report.status == "COMPLETED"
    assert report.production_model_id == PRODUCTION_MODEL_ID
    assert report.candidate_model_id == DEFAULT_CANDIDATE_MODEL_ID
    assert report.beats_compared == len(features)
    assert report.agreement_rate is not None
    assert 0.0 <= report.agreement_rate <= 1.0
    assert report.production_class_counts and report.candidate_class_counts
    assert report.recommendation


def test_shadow_refuses_a_placeholder_candidate():
    features, _ = _feature_matrix(50)
    report = run_shadow_comparison(features, candidate_model_id="ECG-AF-1.0.0-candidate")

    assert report.status == "BLOCKED_PLACEHOLDER"
    assert "fabricated" in report.note
    assert report.agreement_rate is None
    assert report.disagreements == []


def test_shadow_refuses_an_unregistered_model():
    features, _ = _feature_matrix(20)
    report = run_shadow_comparison(features, candidate_model_id="DOES-NOT-EXIST")
    assert report.status == "BLOCKED_MISSING_MODEL"
    assert report.agreement_rate is None


def test_shadow_with_no_features_fails_cleanly():
    report = run_shadow_comparison(np.zeros((0, 5)))
    assert report.status == "FAILED"
    assert report.note


def test_shadow_records_disagreement_details_when_present():
    features, _ = _feature_matrix(400)
    report = run_shadow_comparison(features, max_disagreements=5)
    for disagreement in report.disagreements:
        assert disagreement.production_prediction != disagreement.candidate_prediction
        assert disagreement.beat_index >= 0
    assert len(report.disagreements) <= 5


def test_shadow_report_is_serialisable():
    import json

    features, _ = _feature_matrix(100)
    payload = run_shadow_comparison(features).to_dict()
    assert json.loads(json.dumps(payload))["status"] == "COMPLETED"


def test_shadow_notes_that_candidate_output_is_not_clinical():
    features, _ = _feature_matrix(50)
    report = run_shadow_comparison(features)
    assert "never be returned to a clinician" in report.note


# -------------------------------------------------------------------- streaming
def test_window_indices_are_bounded_and_overlapping():
    windows = list(stream_window_indices(n_samples=3600, fs=360.0, window_sec=5.0, step_sec=1.0))
    assert len(windows) > 1
    for start, end in windows:
        assert 0 <= start < end <= 3600
    starts = [w[0] for w in windows]
    assert starts == sorted(starts)


def test_streaming_does_not_emit_duplicate_beats():
    result = run_streaming_demo(duration_s=20, heart_rate_bpm=75)
    beats = [e for e in result["events"] if e["event"] == "beat"]
    samples = [b["sample_index"] for b in beats]
    assert len(samples) == len(set(samples)), "overlapping windows re-emitted a beat"

    timestamps = [b["timestamp_s"] for b in beats]
    assert timestamps == sorted(timestamps)


@pytest.mark.parametrize("heart_rate", [60, 75, 100])
def test_streaming_estimates_heart_rate_close_to_ground_truth(heart_rate):
    result = run_streaming_demo(duration_s=30, heart_rate_bpm=heart_rate)
    estimated = result["summary"]["estimated_heart_rate_bpm"]
    assert estimated is not None
    assert abs(estimated - heart_rate) <= 5.0


def test_streaming_beat_count_matches_duration_and_rate():
    heart_rate, duration = 90, 30
    result = run_streaming_demo(duration_s=duration, heart_rate_bpm=heart_rate)
    expected = duration * heart_rate / 60.0
    assert abs(result["summary"]["beats_classified"] - expected) <= 3


def test_streaming_marks_signal_provenance():
    synthetic = run_streaming_demo(duration_s=10)
    assert synthetic["signal_provenance"] == "synthesised demonstration waveform"

    supplied = run_streaming_demo(signal=np.random.default_rng(0).normal(size=3600), fs=360.0)
    assert supplied["signal_provenance"] == "caller-supplied signal"


def test_streaming_reports_windows_and_quality_gate_decisions():
    events = list(stream_beats(np.random.default_rng(0).normal(0, 1.0, 7200), fs=360.0))
    windows = [e for e in events if e["event"] == "window"]
    assert windows
    assert all("quality_category" in w for w in windows)
    assert all(w["window_end_s"] >= w["window_start_s"] for w in windows)


def test_flatline_stream_is_gated_by_quality_check():
    events = list(stream_beats(np.zeros(7200), fs=360.0))
    summary = summarise_stream(events)
    assert summary["beats_classified"] == 0


def test_summary_handles_an_empty_stream():
    summary = summarise_stream([])
    assert summary["beats_classified"] == 0
    assert summary["estimated_heart_rate_bpm"] is None
    assert summary["windows_processed"] == 0


def test_summary_counts_predictions():
    result = run_streaming_demo(duration_s=15, heart_rate_bpm=70)
    summary = result["summary"]
    beats = [e for e in result["events"] if e["event"] == "beat"]
    assert sum(summary["prediction_counts"].values()) == len(beats)


def test_production_registry_models_were_not_disturbed_by_shadow_runs():
    """Shadow evaluation must be read-only with respect to model lifecycle."""
    production = GLOBAL_MODEL_REGISTRY.list_models()
    statuses = {m["model_id"]: m["status"] for m in production}
    assert statuses[PRODUCTION_MODEL_ID] == "PRODUCTION"
