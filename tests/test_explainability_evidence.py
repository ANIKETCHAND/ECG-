"""
Tests for per-beat attribution (SHAP when available, labelled fallback otherwise).

The tests assert the property that matters regardless of environment: the report
always states which method produced the numbers, and never presents
baseline-deviation evidence as a model attribution.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.evidence.shap_evidence import (
    METHOD_FALLBACK,
    METHOD_SHAP,
    attribute_beats_with_shap,
    shap_available,
    summarise_attributions,
)

FEATURE_NAMES = [f"f{i}" for i in range(6)]
CLASSES = ["Normal", "Other", "PVC"]


def _production_model():
    classifier = joblib.load("models/production/classifier.pkl")
    scaler = joblib.load("models/production/scaler.pkl")
    return classifier, scaler


def _feature_matrix(n: int = 40, seed: int = 0) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv("data/processed/test_dataset.csv").head(n)
    names = [c for c in frame.columns if c not in {"label", "record_id", "symbol"}]
    _, scaler = _production_model()
    return scaler.transform(frame[names].to_numpy(float))


def test_method_is_always_reported():
    classifier, _ = _production_model()
    X = _feature_matrix()
    report = attribute_beats_with_shap(
        classifier, X, FEATURE_NAMES * 5, list(classifier.classes_), beat_numbers=[1], max_beats=1
    )
    assert report["method"] in {METHOD_SHAP, METHOD_FALLBACK}
    if not shap_available():
        assert report["method"] == METHOD_FALLBACK
        assert report["enabled"] is False
        assert "shap" in report["reason"].lower()


def test_fallback_is_labelled_and_never_claims_model_attribution():
    classifier, _ = _production_model()
    X = _feature_matrix()
    report = attribute_beats_with_shap(
        classifier,
        X,
        FEATURE_NAMES * 5,
        list(classifier.classes_),
        beat_numbers=[1, 2],
        baseline_X=X[20:],
    )
    if report["method"] == METHOD_FALLBACK:
        joined = " ".join(report["notes"]).lower()
        assert "not a model attribution" in joined
        for beat in report["beats"]:
            for entry in beat["attributions"]:
                assert "z_score_deviation" in entry
                assert "shap_contribution" not in entry


def test_empty_feature_matrix_reports_a_reason():
    classifier, _ = _production_model()
    report = attribute_beats_with_shap(classifier, np.zeros((0, 5)), FEATURE_NAMES, CLASSES)
    assert report["beats"] == []
    assert report["reason"]


def test_non_tree_model_falls_back_with_explanation():
    class NotATree:
        classes_ = np.array(CLASSES)

        def predict(self, X):
            return np.array([CLASSES[0]] * len(X))

        def predict_proba(self, X):
            return np.tile([0.6, 0.2, 0.2], (len(X), 1))

    report = attribute_beats_with_shap(
        NotATree(), np.random.default_rng(0).normal(size=(5, 3)), ["a", "b", "c"], CLASSES, baseline_X=np.zeros((2, 3))
    )
    assert report["method"] == METHOD_FALLBACK
    assert report["reason"]
    # Whichever check short-circuits first, the reason must name a real cause.
    reason = report["reason"].lower()
    assert "shap is not installed" in reason or "no tree explainer" in reason


def test_fallback_without_baseline_explains_why_it_cannot_run():
    report = attribute_beats_with_shap(
        _production_model()[0], np.zeros((3, 4)), ["a", "b", "c", "d"], CLASSES, baseline_X=None
    )
    if report["method"] == METHOD_FALLBACK:
        assert report["beats"] == []
        assert any("baseline" in note.lower() for note in report["notes"])


def test_beat_numbers_are_respected_and_bounded():
    classifier, _ = _production_model()
    X = _feature_matrix(30)
    report = attribute_beats_with_shap(
        classifier, X, FEATURE_NAMES * 5, list(classifier.classes_), beat_numbers=[3], baseline_X=X[:10]
    )
    if report["beats"]:
        assert report["beats"][0]["beat_number"] == 3


def test_max_beats_caps_work():
    classifier, _ = _production_model()
    X = _feature_matrix(50)
    report = attribute_beats_with_shap(
        classifier, X, FEATURE_NAMES * 5, list(classifier.classes_), baseline_X=X, max_beats=3
    )
    assert len(report["beats"]) <= 3


def test_summary_lines_are_human_readable():
    classifier, _ = _production_model()
    X = _feature_matrix()
    report = attribute_beats_with_shap(
        classifier, X, FEATURE_NAMES * 5, list(classifier.classes_), beat_numbers=[1, 2], baseline_X=X
    )
    lines = summarise_attributions(report, top_n=2)
    assert lines[0].startswith("Attribution method:")
    assert len(lines) >= 2


def test_evidence_engine_accepts_model_and_exposes_attribution():
    """The evidence engine must surface the attribution block when given a model."""
    import pandas as pd

    from src.evidence.evidence_engine import generate_ai_evidence

    frame = pd.read_csv("data/processed/test_dataset.csv").head(60)
    names = [c for c in frame.columns if c not in {"label", "record_id", "symbol"}]
    classifier, scaler = _production_model()
    features = frame[names].to_numpy(float)
    scaled = scaler.transform(features)

    rng = np.random.default_rng(0)
    beats = rng.normal(size=(len(features), 217))
    r_peaks = np.arange(0, len(features) * 217, 217)[: len(features)]
    probabilities = classifier.predict_proba(scaled)

    report = generate_ai_evidence(
        signal=np.concatenate([b for b in beats]),
        fs=360.0,
        r_peaks=r_peaks,
        beats=beats,
        features=features,
        feature_names=names,
        beat_probs=probabilities,
        classes=list(classifier.classes_),
        overall_classification="Test",
        model=classifier,
        scaled_features=scaled,
    )
    payload = report.to_dict()
    assert "attribution_evidence" in payload
