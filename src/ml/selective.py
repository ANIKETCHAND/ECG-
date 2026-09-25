"""
Selective Prediction (Abstention) for the Beat Arrhythmia Classifier
====================================================================

Why this module exists
----------------------
The shipped classifier is a three-class model (``Normal`` / ``Other`` / ``PVC``).
Its headline accuracy is already ~98.9% on record-disjoint data, but that number
is dominated by the easy majority class and hides the fact that the ``Other``
class has too little support in the source corpus to be learnable at all.

There is no honest way to make every beat prediction correct. There *is* an
honest way to make every **reported** beat prediction correct: let the model
decline to answer when it is not confident, and hand those beats to the
clinician instead. This is standard selective prediction (a.k.a. classification
with a reject option) and it composes with the project's core tenet:

    NO RELIABLE INPUT      -> NO AI RESULT
    NO CONFIDENT PREDICTION -> NO AI LABEL

Everything in here is *measured*, never asserted. The module only knows how to

* build the accuracy/coverage trade-off curve from real held-out data,
* pick a threshold that meets a target precision at the best achievable
  coverage,
* apply that threshold at inference time.

It deliberately does **not** invent a threshold. If no measured operating point
artifact exists, the callee must fall back to "report everything, flag it as
ungated" rather than silently pretending the predictions are vetted.

Research / educational use only. Not a diagnostic device.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: A beat the model declined to label. Never reported as a clinical finding.
INDETERMINATE = "INDETERMINATE"

#: Default search grid for the confidence threshold. Dense near 1.0 because the
#: interesting behaviour (100% precision) lives there.
DEFAULT_THRESHOLDS: Tuple[float, ...] = tuple(
    sorted({0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992,
            0.995, 0.997, 0.999})
)

#: Guard rail: below this coverage an operating point is statistically hollow no
#: matter how good the precision looks. 1.0 precision on 3 beats is noise.
MIN_USEFUL_COVERAGE = 0.05

#: Guard rail: a threshold must retain at least this many beats to be reported.
MIN_USEFUL_BEATS = 20


@dataclass
class OperatingPoint:
    """A measured accuracy/coverage choice, with the evidence that produced it."""

    threshold: float
    target_precision: float
    coverage: float
    reported_beats: int
    total_beats: int
    precision: float
    macro_f1_reported: float
    accuracy_reported: float
    classes_reported: List[str] = field(default_factory=list)
    meets_target: bool = False
    guard_rail_ok: bool = True
    guard_rail_reason: Optional[str] = None
    selection_bias: str = (
        "Threshold was selected on the same held-out partition it is reported on. "
        "Reported coverage is therefore an upper bound and must be re-fitted on a "
        "fresh dataset before clinical deployment."
    )
    source_dataset: str = ""
    evaluated_records: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def reportable(self) -> bool:
        """Whether this operating point may be used to gate predictions."""
        return bool(self.meets_target and self.guard_rail_ok)


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------


def confidence_scores(probabilities: np.ndarray) -> np.ndarray:
    """Confidence of the top-1 class = max class probability."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("probabilities must be a 2-D array of shape (n_samples, n_classes)")
    return probs.max(axis=1)


def predicted_from_scores(
    probabilities: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    """Argmax class label per row, using the supplied class ordering."""
    probs = np.asarray(probabilities, dtype=float)
    idx = probs.argmax(axis=1)
    cls = list(classes)
    return np.array([cls[i] for i in idx], dtype=object)


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = []
    for lab in labels:
        tp = int(np.sum((y_pred == lab) & (y_true == lab)))
        fp = int(np.sum((y_pred == lab) & (y_true != lab)))
        fn = int(np.sum((y_pred != lab) & (y_true == lab)))
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def operating_curve(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> List[Dict[str, Any]]:
    """Measure accuracy/coverage at each confidence threshold.

    Beats whose top-1 confidence falls below ``threshold`` are *withheld*
    (abstained) rather than predicted wrongly.
    """
    y = np.asarray(list(y_true), dtype=object)
    conf = confidence_scores(probabilities)
    pred = predicted_from_scores(probabilities, classes)
    n = len(y)

    rows: List[Dict[str, Any]] = []
    for tau in thresholds:
        keep = conf >= tau
        kept = int(np.sum(keep))
        if kept == 0:
            rows.append(
                {
                    "threshold": float(tau),
                    "reported_beats": 0,
                    "total_beats": n,
                    "coverage": 0.0,
                    "accuracy_reported": None,
                    "precision_reported": None,
                    "macro_f1_reported": None,
                    "errors_reported": None,
                    "abstained_beats": n,
                }
            )
            continue
        y_k = y[keep]
        p_k = pred[keep]
        rows.append(
            {
                "threshold": float(tau),
                "reported_beats": kept,
                "total_beats": n,
                "coverage": round(kept / n, 6) if n else 0.0,
                "accuracy_reported": round(_precision(y_k, p_k), 6),
                "precision_reported": round(_precision(y_k, p_k), 6),
                "macro_f1_reported": round(_macro_f1(y_k, p_k), 6),
                "errors_reported": int(np.sum(y_k != p_k)),
                "abstained_beats": n - kept,
            }
        )
    return rows


def select_operating_point(
    curve: Sequence[Dict[str, Any]],
    target_precision: float = 1.0,
    min_coverage: float = MIN_USEFUL_COVERAGE,
    min_beats: int = MIN_USEFUL_BEATS,
) -> OperatingPoint:
    """Pick the lowest threshold that still meets ``target_precision``.

    Lowest qualifying threshold == highest coverage at the requested precision,
    which is what a clinician actually wants: "only tell me things you are sure
    about, but tell me as many as you can."
    """
    qualifying = [
        row
        for row in curve
        if row["reported_beats"] > 0
        and row["precision_reported"] is not None
        and row["precision_reported"] >= target_precision - 1e-12
    ]
    if not qualifying:
        best = max(
            (r for r in curve if r["reported_beats"] > 0),
            key=lambda r: (r["precision_reported"] or 0.0, r["coverage"]),
        )
        return OperatingPoint(
            threshold=float(best["threshold"]),
            target_precision=float(target_precision),
            coverage=float(best["coverage"]),
            reported_beats=int(best["reported_beats"]),
            total_beats=int(best["total_beats"]),
            precision=float(best["precision_reported"] or 0.0),
            accuracy_reported=float(best["accuracy_reported"] or 0.0),
            macro_f1_reported=float(best["macro_f1_reported"] or 0.0),
            meets_target=False,
            guard_rail_ok=False,
            guard_rail_reason=(
                f"No confidence threshold reached the requested precision of "
                f"{target_precision:.4f} on the measured partition."
            ),
        )

    chosen = min(qualifying, key=lambda r: (r["threshold"], -r["coverage"]))
    guard_ok = True
    reason: Optional[str] = None
    if chosen["coverage"] < min_coverage:
        guard_ok = False
        reason = (
            f"Best achievable coverage at precision >= {target_precision:.4f} is "
            f"{chosen['coverage']:.4f}, below the {min_coverage:.2f} guard rail."
        )
    elif chosen["reported_beats"] < min_beats:
        guard_ok = False
        reason = (
            f"Only {chosen['reported_beats']} beats remain at this threshold "
            f"(minimum {min_beats})."
        )

    return OperatingPoint(
        threshold=float(chosen["threshold"]),
        target_precision=float(target_precision),
        coverage=float(chosen["coverage"]),
        reported_beats=int(chosen["reported_beats"]),
        total_beats=int(chosen["total_beats"]),
        precision=float(chosen["precision_reported"]),
        accuracy_reported=float(chosen["accuracy_reported"]),
        macro_f1_reported=float(chosen["macro_f1_reported"] or 0.0),
        meets_target=True,
        guard_rail_ok=guard_ok,
        guard_rail_reason=reason,
    )


def record_wise_jackknife(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    records: Sequence[str],
    target_precision: float = 1.0,
) -> Dict[str, Any]:
    """Leave-one-record-out threshold selection.

    A single train/test record split is far too coarse for threshold tuning: with
    only a couple of records per side, the split itself decides the answer. This
    routine instead repeats selection once per record, each time excluding that
    record entirely, so every record is measured with a threshold that never saw
    it. That is ordinary cross-validation applied to the reject option.

    The installed threshold is the **strictest** fold threshold
    (``max`` of the fold choices) rather than the median: being conservative costs
    coverage, never precision, so it is the safe direction to err.
    """
    y = np.asarray(list(y_true), dtype=object)
    recs = np.asarray([str(r) for r in records], dtype=object)
    probs = np.asarray(probabilities, dtype=float)
    unique = sorted(set(recs.tolist()))

    folds: List[Dict[str, Any]] = []
    for held_out in unique:
        others = recs != held_out
        if not others.any() or not (recs == held_out).any():
            continue
        curve = operating_curve(y[others].tolist(), probs[others], classes)
        pick = select_operating_point(
            curve, target_precision=target_precision, min_coverage=0.0, min_beats=1
        )
        folds.append(
            {
                "held_out_record": held_out,
                "selection_records": sorted(set(recs[others].tolist())),
                "threshold": pick.threshold if pick.meets_target else None,
                "selection_precision": pick.precision if pick.meets_target else None,
            }
        )

    fold_thresholds = [f["threshold"] for f in folds if f["threshold"] is not None]
    all_folds_met = len(fold_thresholds) == len(folds) and bool(folds)
    installed = max(fold_thresholds) if all_folds_met else None

    return {
        "folds": folds,
        "installed_threshold": installed,
        "all_folds_met_target": all_folds_met,
        "target_precision": float(target_precision),
    }


# ---------------------------------------------------------------------------
# Applying a gate
# ---------------------------------------------------------------------------


def apply_gate(
    probabilities: np.ndarray,
    classes: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    """Gate beat-level predictions, replacing low-confidence labels.

    Returns a dict with ``reported_labels`` (abstained beats become
    ``INDETERMINATE``), the raw ``raw_labels``, per-beat ``confidence`` and
    summary counts. The raw model opinion is preserved so nothing is hidden.
    """
    probs = np.asarray(probabilities, dtype=float)
    conf = confidence_scores(probs)
    raw = predicted_from_scores(probs, classes)
    keep = conf >= float(threshold)

    reported = np.array(
        [lab if k else INDETERMINATE for lab, k in zip(raw, keep)], dtype=object
    )
    reported_list = [str(x) for x in reported.tolist()]
    counts: Dict[str, int] = {}
    for lab in reported_list:
        counts[lab] = counts.get(lab, 0) + 1

    reported_n = int(np.sum(keep))
    return {
        "threshold": float(threshold),
        "raw_labels": [str(x) for x in raw.tolist()],
        "reported_labels": reported_list,
        "confidence": [round(float(c), 6) for c in conf.tolist()],
        "reported_beats": reported_n,
        "abstained_beats": int(len(conf) - reported_n),
        "coverage": round(reported_n / len(conf), 6) if len(conf) else 0.0,
        "reported_class_counts": counts,
        "gate_applied": True,
    }


def ungated(probabilities: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    """Explicitly *no* gate: every beat is reported, and we say so."""
    probs = np.asarray(probabilities, dtype=float)
    conf = confidence_scores(probs)
    raw = predicted_from_scores(probs, classes)
    reported_list = [str(x) for x in raw.tolist()]
    counts: Dict[str, int] = {}
    for lab in reported_list:
        counts[lab] = counts.get(lab, 0) + 1
    return {
        "threshold": None,
        "raw_labels": reported_list,
        "reported_labels": reported_list,
        "confidence": [round(float(c), 6) for c in conf.tolist()],
        "reported_beats": len(reported_list),
        "abstained_beats": 0,
        "coverage": 1.0 if reported_list else 0.0,
        "reported_class_counts": counts,
        "gate_applied": False,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def default_operating_point_path(models_dir: Optional[Path] = None) -> Path:
    base = Path(models_dir) if models_dir is not None else Path(__file__).resolve().parents[2] / "models"
    return Path(base) / "operating_point.json"


def save_operating_point(op: OperatingPoint, path: Optional[Path] = None) -> Path:
    target = Path(path) if path is not None else default_operating_point_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(op.to_dict(), fh, indent=2)
    return target


def load_operating_point(path: Optional[Path] = None) -> Optional[OperatingPoint]:
    """Load an operating point, or return ``None`` when none has been fitted.

    Returning ``None`` is a real state, not an error: it means the deployment has
    no measured justification for abstaining and must report predictions ungated.
    """
    target = Path(path) if path is not None else default_operating_point_path()
    if not target.exists():
        return None
    try:
        with open(target, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "threshold" not in payload:
        return None
    known = {k: v for k, v in payload.items() if k in OperatingPoint.__dataclass_fields__}
    try:
        return OperatingPoint(**known)
    except TypeError:  # pragma: no cover - defensive against hand-edited files
        return None


def summarise(op: Optional[OperatingPoint]) -> str:
    """One-line human summary used in reports and the API health block."""
    if op is None:
        return (
            "No measured abstention operating point is installed. All beat labels "
            "are reported ungated and must be treated as unvetted model opinions."
        )
    state = "active" if op.reportable else "installed but NOT reportable"
    return (
        f"Abstention gate {state}: report only when max-probability >= {op.threshold:.3f}; "
        f"measured precision {op.precision:.4f} on {op.reported_beats}/{op.total_beats} "
        f"held-out beats ({op.coverage * 100:.1f}% coverage)."
    )
