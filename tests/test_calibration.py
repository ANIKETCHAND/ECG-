"""
Tests for probability calibration and Mondrian conformal abstention.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ml.calibration import (
    MondrianConformalClassifier,
    MulticlassCalibrator,
    apply_calibration,
    expected_calibration_error,
    load_calibration,
    maximum_calibration_error,
    multiclass_brier_score,
    reliability_bins,
    save_calibration,
)

CLASSES = ["Normal", "Other", "PVC"]


def _overconfident_data(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Probability rows that are systematically too confident.

    The true label is drawn from the model's own ordering, then the probabilities
    are sharpened, so the model is right about the ranking but wrong about the
    magnitude — exactly the Random Forest failure mode calibration exists for.
    """
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([0.7, 0.4, 1.2], size=n)
    labels = np.array([CLASSES[i] for i in np.argmax(raw, axis=1)])

    # Flip a known fraction so the model is imperfect but still informative.
    flip = rng.random(n) < 0.12
    for index in np.where(flip)[0]:
        alternatives = [c for c in CLASSES if c != labels[index]]
        labels[index] = alternatives[rng.integers(len(alternatives))]

    sharpened = raw**3
    sharpened = sharpened / sharpened.sum(axis=1, keepdims=True)
    return sharpened, labels


# --------------------------------------------------------------------- metrics
def test_ece_is_zero_for_perfectly_calibrated_certainty():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    probs = np.array([[0.95, 0.05]] * 10)
    labels = np.array([0] * 5 + [1] * 5)  # only 50% correct at 95% confidence
    ece = expected_calibration_error(probs, labels)
    assert ece == pytest.approx(0.45, abs=0.01)
    assert maximum_calibration_error(probs, labels) == pytest.approx(0.45, abs=0.01)


def test_brier_score_is_zero_for_perfect_predictions():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert multiclass_brier_score(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_reliability_bins_cover_the_unit_interval():
    probs, labels = _overconfident_data(200)
    index = np.array([CLASSES.index(label) for label in labels])
    bins = reliability_bins(probs, index, n_bins=5)
    assert len(bins) == 5
    assert bins[0]["lower"] == 0.0 and bins[-1]["upper"] == 1.0
    assert sum(b["count"] for b in bins) == pytest.approx(len(labels), abs=1e-6)


def test_metrics_handle_empty_input():
    empty = np.zeros((0, 3))
    assert expected_calibration_error(empty, np.array([], dtype=int)) == 0.0
    assert multiclass_brier_score(empty, np.array([], dtype=int)) == 0.0


# ----------------------------------------------------------------- calibrators
def test_calibration_reduces_ece_on_held_out_data():
    probs, labels = _overconfident_data(4000, seed=1)
    index = np.array([CLASSES.index(label) for label in labels])

    split = 2000
    calibrator = MulticlassCalibrator(method="isotonic", classes=CLASSES)
    calibrator.fit(probs[:split], labels[:split])

    before = expected_calibration_error(probs[split:], index[split:])
    after = expected_calibration_error(calibrator.transform(probs[split:]), index[split:])
    assert after <= before


def test_calibrated_probabilities_are_valid_distributions():
    probs, labels = _overconfident_data(1000, seed=2)
    calibrator = MulticlassCalibrator(method="isotonic", classes=CLASSES).fit(probs, labels)
    transformed = calibrator.transform(probs[:50])
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0)
    assert np.allclose(transformed.sum(axis=1), 1.0)


def test_platt_calibration_round_trips_through_json():
    probs, labels = _overconfident_data(1000, seed=3)
    original = MulticlassCalibrator(method="platt", classes=CLASSES).fit(probs, labels)
    restored = MulticlassCalibrator.from_dict(original.to_dict())

    assert restored.method == "platt"
    assert restored.classes == CLASSES
    assert np.allclose(original.transform(probs[:20]), restored.transform(probs[:20]))


def test_calibrator_rejects_mismatched_class_count():
    probs, labels = _overconfident_data(100, seed=4)
    with pytest.raises(ValueError, match="columns"):
        MulticlassCalibrator(classes=["A", "B"]).fit(probs, labels)


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError, match="isotonic"):
        MulticlassCalibrator(method="magic")


# ------------------------------------------------------------------- conformal
def test_conformal_predict_sets_are_never_empty():
    probs, labels = _overconfident_data(500, seed=5)
    conformal = MondrianConformalClassifier(alpha=0.1, classes=CLASSES).fit(probs, labels)
    sets = conformal.predict_set(probs[:20])
    assert all(len(s) >= 1 for s in sets)
    assert all(set(s).issubset(set(CLASSES)) for s in sets)


def test_conformal_empirical_coverage_meets_target_on_held_out_data():
    probs, labels = _overconfident_data(4000, seed=6)
    split = 2500
    conformal = MondrianConformalClassifier(alpha=0.10, classes=CLASSES).fit(
        probs[:split], labels[:split]
    )
    sets = conformal.predict_set(probs[split:])
    covered = np.mean([labels[split + i] in sets[i] for i in range(len(sets))])
    # Finite-sample correction makes coverage conservative; allow a small shortfall
    # for sampling noise.
    assert covered >= 0.85


def test_conformal_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="alpha"):
        MondrianConformalClassifier(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        MondrianConformalClassifier(alpha=1.5)


def test_sparse_classes_fall_back_to_pooled_quantile_and_say_so():
    probs, labels = _overconfident_data(600, seed=7)
    # Make one class extremely rare in the calibration set.
    labels = labels.copy()
    rare_index = CLASSES.index("Other")
    rare_positions = np.where(labels == CLASSES[rare_index])[0]
    for position in rare_positions[:-1]:
        labels[position] = "Normal"

    conformal = MondrianConformalClassifier(alpha=0.1, classes=CLASSES, min_class_samples=10).fit(probs, labels)
    assert "Other" in conformal.pooled_classes
    assert conformal.pooled_quantile is not None
    assert "coverage" in conformal.to_dict()["coverage_note"].lower()


def test_conformal_round_trips_through_json():
    probs, labels = _overconfident_data(800, seed=8)
    original = MondrianConformalClassifier(alpha=0.2, classes=CLASSES).fit(probs, labels)
    restored = MondrianConformalClassifier.from_dict(original.to_dict())
    assert restored.quantiles == original.quantiles
    assert restored.alpha == original.alpha


def test_abstain_flags_low_separation_rows():
    probs, labels = _overconfident_data(1000, seed=9)
    conformal = MondrianConformalClassifier(alpha=0.1, classes=CLASSES).fit(probs, labels)
    ambiguous = np.array([[0.34, 0.33, 0.33]])
    should_abstain, prediction_set = conformal.abstain(ambiguous)
    assert should_abstain is True
    assert len(prediction_set) > 1


# ---------------------------------------------------------------- persistence
def test_apply_calibration_without_artifact_reports_uncalibrated(tmp_path):
    probs = np.array([[0.6, 0.3, 0.1]])
    result = apply_calibration(probs, CLASSES, path=tmp_path / "nonexistent.json")
    assert result["calibration_applied"] is False
    assert result["conformal_applied"] is False
    assert result["calibrated_probabilities"] == probs.tolist()


def test_save_and_load_calibration_round_trip(tmp_path):
    probs, labels = _overconfident_data(1000, seed=10)
    calibrator = MulticlassCalibrator(method="isotonic", classes=CLASSES).fit(probs, labels)
    conformal = MondrianConformalClassifier(alpha=0.1, classes=CLASSES).fit(probs, labels)

    path = tmp_path / "calibration.json"
    save_calibration(calibrator, conformal, path=path)
    assert path.exists()

    loaded_calibrator, loaded_conformal, metadata = load_calibration(path)
    assert loaded_calibrator is not None and loaded_conformal is not None
    assert np.allclose(calibrator.transform(probs[:10]), loaded_calibrator.transform(probs[:10]))
    assert metadata  # saved_at present


def test_load_calibration_returns_none_when_file_is_corrupt(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")
    calibrator, conformal, metadata = load_calibration(path)
    assert calibrator is None and conformal is None and metadata == {}


def test_withheld_calibration_is_not_applied(tmp_path):
    """A calibration map that measured worse must not be used at inference."""
    probs, labels = _overconfident_data(500, seed=11)
    calibrator = MulticlassCalibrator(method="isotonic", classes=CLASSES).fit(probs, labels)
    path = tmp_path / "calibration.json"
    save_calibration(
        calibrator,
        None,
        path=path,
        extra={
            "deployment_recommended": False,
            "deployment_recommendation_reason": "ECE worsened on the evaluation partition",
        },
    )

    result = apply_calibration(np.array([[0.7, 0.2, 0.1]]), CLASSES, path=path)
    assert result["calibration_applied"] is False
    assert "worsened" in result["calibration_withheld_reason"]


def test_recommended_calibration_is_applied(tmp_path):
    probs, labels = _overconfident_data(1500, seed=12)
    calibrator = MulticlassCalibrator(method="isotonic", classes=CLASSES).fit(probs, labels)
    path = tmp_path / "calibration.json"
    save_calibration(calibrator, None, path=path, extra={"deployment_recommended": True})

    result = apply_calibration(np.array([[0.8, 0.15, 0.05]]), CLASSES, path=path)
    assert result["calibration_applied"] is True
    assert result["calibration_withheld_reason"] is None
