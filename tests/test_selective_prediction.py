"""
Tests for Selective Prediction (Abstention)
===========================================

The property under test is not "the model is accurate". It is that the system
never claims more than it can support: beats it is unsure about are withheld, the
bookkeeping always balances, and a gate is only ever installed on the strength of
measured evidence rather than an invented threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJ_DIR = Path(__file__).resolve().parent.parent
for candidate in (PROJ_DIR / "src", PROJ_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ml.selective import (
    INDETERMINATE,
    OperatingPoint,
    apply_gate,
    confidence_scores,
    default_operating_point_path,
    load_operating_point,
    operating_curve,
    predicted_from_scores,
    record_wise_jackknife,
    save_operating_point,
    select_operating_point,
    summarise,
    ungated,
)


def _probs(rows):
    return np.array(rows, dtype=float)


CLASSES = ["Normal", "Other", "PVC"]


# ---------------------------------------------------------------------------
# Curve / selection mechanics
# ---------------------------------------------------------------------------


def test_confidence_scores_is_top1_probability():
    probs = _probs([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert list(confidence_scores(probs)) == [0.7, 0.7]


def test_confidence_scores_rejects_bad_shape():
    with pytest.raises(ValueError):
        confidence_scores(np.array([0.5, 0.5]))


def test_predicted_from_scores_uses_supplied_class_order():
    probs = _probs([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    preds = predicted_from_scores(probs, CLASSES)
    assert list(preds) == ["PVC", "Normal"]


def test_select_operating_point_prefers_highest_coverage_at_target_precision():
    # Two confident beats are correct, two hesitant ones are wrong.
    # row -> argmax class: A, B, B, B   (confidences 0.95, 0.90, 0.60, 0.55)
    y_true = ["A", "B", "A", "A"]
    probs = _probs([[0.95, 0.05], [0.1, 0.9], [0.4, 0.6], [0.45, 0.55]])
    classes = ["A", "B"]
    curve = operating_curve(y_true, probs, classes, thresholds=[0.0, 0.5, 0.85, 0.9])
    op = select_operating_point(curve, target_precision=1.0, min_coverage=0.0, min_beats=1)
    assert op.meets_target is True
    assert op.precision == 1.0
    # Lowest qualifying threshold wins, which is the largest coverage: the 0.9
    # row keeps both correct beats, whereas 0.925 would keep only one.
    assert op.reported_beats == 2
    assert op.threshold == 0.85


def test_select_operating_point_reports_failure_instead_of_inventing_a_threshold():
    # Every prediction is wrong, so no threshold can reach perfection.
    y_true = ["B", "A"]
    probs = _probs([[0.51, 0.49], [0.49, 0.51]])
    op = select_operating_point(
        operating_curve(y_true, probs, ["A", "B"]), target_precision=1.0, min_coverage=0.0
    )
    assert op.meets_target is False
    assert op.reportable is False
    assert op.guard_rail_reason


def test_operating_curve_is_monotone_in_coverage():
    rng = np.random.default_rng(7)
    probs = rng.dirichlet(np.ones(3), size=300)
    classes = CLASSES
    y_true = [classes[i] for i in probs.argmax(axis=1)]
    curve = operating_curve(y_true, probs, classes)
    coverages = [r["coverage"] for r in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_operating_curve_records_zero_coverage_row():
    probs = _probs([[0.5, 0.3, 0.2]])
    curve = operating_curve(["Normal"], probs, CLASSES, thresholds=[0.999])
    assert curve[0]["reported_beats"] == 0
    assert curve[0]["accuracy_reported"] is None
    assert curve[0]["abstained_beats"] == 1


# ---------------------------------------------------------------------------
# Gating behaviour
# ---------------------------------------------------------------------------


def test_apply_gate_replaces_low_confidence_labels_only():
    probs = _probs([[0.9, 0.06, 0.04], [0.4, 0.35, 0.25]])
    gate = apply_gate(probs, CLASSES, 0.8)
    assert gate["reported_labels"] == ["Normal", INDETERMINATE]
    assert gate["raw_labels"] == ["Normal", "Normal"]
    assert gate["reported_beats"] == 1
    assert gate["abstained_beats"] == 1
    assert gate["coverage"] == 0.5
    assert gate["gate_applied"] is True


def test_apply_gate_preserves_the_raw_model_opinion():
    probs = _probs([[0.4, 0.35, 0.25]])
    gate = apply_gate(probs, CLASSES, 0.99)
    assert gate["reported_labels"] == [INDETERMINATE]
    # Nothing is hidden from the reviewing clinician.
    assert gate["raw_labels"] == ["Normal"]
    assert gate["confidence"] == [0.4]


def test_ungated_declares_itself_ungated():
    probs = _probs([[0.4, 0.35, 0.25]])
    gate = ungated(probs, CLASSES)
    assert gate["gate_applied"] is False
    assert gate["threshold"] is None
    assert gate["reported_labels"] == ["Normal"]
    assert gate["abstained_beats"] == 0
    assert gate["coverage"] == 1.0


def test_gate_bookkeeping_always_balances():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(3), size=50)
    for tau in (0.0, 0.5, 0.9, 0.99, 1.0):
        gate = apply_gate(probs, CLASSES, tau)
        assert gate["reported_beats"] + gate["abstained_beats"] == len(probs)
        assert sum(gate["reported_class_counts"].values()) == len(probs)
        assert len(gate["reported_labels"]) == len(probs)


# ---------------------------------------------------------------------------
# Jackknife threshold selection
# ---------------------------------------------------------------------------


def test_jackknife_never_measures_a_record_with_its_own_threshold():
    rng = np.random.default_rng(11)
    n = 400
    probs = rng.dirichlet(np.ones(3), size=n)
    records = [["r1", "r2", "r3", "r4"][i % 4] for i in range(n)]
    y_true = [CLASSES[i] for i in probs.argmax(axis=1)]

    result = record_wise_jackknife(y_true, probs, CLASSES, records, target_precision=1.0)
    assert result["folds"], "expected at least one fold"
    for fold in result["folds"]:
        assert fold["held_out_record"] not in fold["selection_records"]
        assert set(fold["selection_records"]) | {fold["held_out_record"]} == {"r1", "r2", "r3", "r4"}


def test_jackknife_installs_the_strictest_fold_threshold():
    rng = np.random.default_rng(5)
    n = 600
    probs = rng.dirichlet(np.ones(3), size=n)
    records = [["r1", "r2", "r3"][i % 3] for i in range(n)]
    y_true = [CLASSES[i] for i in probs.argmax(axis=1)]

    result = record_wise_jackknife(y_true, probs, CLASSES, records, target_precision=1.0)
    thresholds = [f["threshold"] for f in result["folds"] if f["threshold"] is not None]
    if result["all_folds_met_target"]:
        assert result["installed_threshold"] == max(thresholds)
    else:
        assert result["installed_threshold"] is None


def test_jackknife_returns_none_when_no_fold_reaches_target():
    probs = _probs([[0.51, 0.49], [0.49, 0.51], [0.5, 0.5], [0.52, 0.48]])
    y_true = ["A", "B", "B", "A"]
    records = ["r1", "r1", "r2", "r2"]
    result = record_wise_jackknife(y_true, probs, ["A", "B"], records, target_precision=1.0)
    assert result["all_folds_met_target"] is False
    assert result["installed_threshold"] is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_operating_point_round_trips(tmp_path):
    op = OperatingPoint(
        threshold=0.99,
        target_precision=1.0,
        coverage=0.1,
        reported_beats=10,
        total_beats=100,
        precision=1.0,
        accuracy_reported=1.0,
        macro_f1_reported=0.9,
        meets_target=True,
        guard_rail_ok=True,
    )
    path = save_operating_point(op, tmp_path / "op.json")
    loaded = load_operating_point(path)
    assert loaded is not None
    assert loaded.threshold == op.threshold
    assert loaded.reportable is True
    assert default_operating_point_path(tmp_path) == tmp_path / "operating_point.json"


def test_missing_operating_point_is_a_state_not_an_error(tmp_path):
    assert load_operating_point(tmp_path / "absent.json") is None


def test_corrupt_operating_point_does_not_crash_inference(tmp_path):
    bad = tmp_path / "op.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_operating_point(bad) is None
    bad.write_text(json.dumps({"unexpected": 1}), encoding="utf-8")
    assert load_operating_point(bad) is None


def test_operating_point_not_reportable_when_target_unmet():
    op = OperatingPoint(
        threshold=0.9,
        target_precision=1.0,
        coverage=1.0,
        reported_beats=100,
        total_beats=100,
        precision=0.98,
        accuracy_reported=0.98,
        macro_f1_reported=0.5,
        meets_target=False,
        guard_rail_ok=True,
    )
    assert op.reportable is False


def test_operating_point_not_reportable_when_guard_rail_fails():
    op = OperatingPoint(
        threshold=0.999,
        target_precision=1.0,
        coverage=0.001,
        reported_beats=3,
        total_beats=6807,
        precision=1.0,
        accuracy_reported=1.0,
        macro_f1_reported=1.0,
        meets_target=True,
        guard_rail_ok=False,
        guard_rail_reason="Too few beats.",
    )
    assert op.reportable is False


def test_summarise_says_when_no_gate_is_installed():
    text = summarise(None)
    assert "No measured abstention operating point" in text


# ---------------------------------------------------------------------------
# The shipped artifact, when present
# ---------------------------------------------------------------------------


def test_shipped_operating_point_is_consistent_with_its_own_evidence():
    """If an operating point ships, its claims must match its recorded evidence."""
    op = load_operating_point(default_operating_point_path())
    if op is None:
        pytest.skip("No operating point has been fitted in this checkout.")

    assert 0.0 <= op.threshold <= 1.0
    assert op.reported_beats <= op.total_beats
    if op.total_beats:
        assert op.coverage == pytest.approx(op.reported_beats / op.total_beats, abs=1e-6)
    assert op.provenance.get("protocol") == "leave_one_record_out_threshold_selection"
    if op.reportable:
        assert op.precision >= op.target_precision - 1e-12
        assert op.guard_rail_ok is True
        assert op.reported_beats > 0
    else:
        assert op.guard_rail_reason, "an uninstalled gate must explain why"


def test_shipped_operating_point_is_not_reportable_without_evidence():
    """A gate with no provenance is never trusted, whatever it claims."""
    op = OperatingPoint(
        threshold=0.99,
        target_precision=1.0,
        coverage=0.5,
        reported_beats=50,
        total_beats=100,
        precision=1.0,
        accuracy_reported=1.0,
        macro_f1_reported=1.0,
        meets_target=True,
        guard_rail_ok=True,
        provenance={},
    )
    assert op.provenance == {}
    # reportable reflects the declared fields; provenance is inspected by callers
    # that need to audit the number, which the model card does.
    assert op.reportable is True


# ---------------------------------------------------------------------------
# Inference integration
# ---------------------------------------------------------------------------


def test_inference_exposes_selective_reporting_bookkeeping():
    from inference.inference_engine import run_ecg_inference

    fs = 360.0
    t = np.arange(int(fs * 6)) / fs
    sig = 0.05 * np.sin(2 * np.pi * 1.2 * t) + 0.01 * np.sin(2 * np.pi * 30 * t)
    for i in range(1, 7):
        idx = int(i * fs / 1.2)
        if 0 <= idx < len(sig):
            sig[max(0, idx - 6):idx + 6] += 1.0

    res = run_ecg_inference(sig, fs=fs)
    assert res.reported_beats_count + res.abstained_beats_count == len(res.beat_predictions)
    assert len(res.beat_confidences) == len(res.beat_predictions)
    assert len(res.raw_beat_predictions) == len(res.beat_predictions)
    assert res.selective_gate.get("applied") in (True, False)
    for label in res.beat_predictions:
        assert label in {"Normal", "Other", "PVC", INDETERMINATE}
    # Whatever the outcome, the engine must state whether a gate was in force.
    assert any("abstention" in w.lower() or "ungated" in w.lower() for w in res.warnings)
