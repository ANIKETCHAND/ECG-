"""
Probability Calibration & Conformal Abstention
==============================================

The Random Forest's ``predict_proba`` output is a class vote frequency, not a
probability in any clinical sense. This module adds two things the safety
narrative needs:

1. **Calibration** — map raw scores onto probabilities that mean what they say,
   so "PVC 0.80" is right roughly four times in five. Per-class one-vs-rest
   Platt scaling (sigmoid) or isotonic regression, followed by renormalisation.

2. **Conformal abstention** — a Mondrian (class-conditional) split-conformal
   predictor that returns a *prediction set* with finite-sample coverage
   guarantee ``1 - alpha``. When the set contains more than one label the model
   has not separated the options, and the platform says so instead of emitting a
   confident single label. This is the "can I trust this prediction?" companion
   to the existing "can I trust this signal?" quality gate.

Everything here is deterministic numpy/sklearn, so it can be fitted offline in
``training/fit_calibration.py``, persisted to JSON, and replayed at inference
time without the original training data.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "calibration"
CALIBRATION_PATH = CALIBRATION_DIR / "calibration.json"


# --------------------------------------------------------------------- metrics
def expected_calibration_error(
    probabilities: np.ndarray, y_true_index: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label Expected Calibration Error."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_true_index = np.asarray(y_true_index, dtype=int)
    if probabilities.size == 0:
        return 0.0

    confidence = np.max(probabilities, axis=1)
    correct = (np.argmax(probabilities, axis=1) == y_true_index).astype(float)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        fraction = float(np.mean(in_bin))
        if fraction > 0:
            ece += abs(float(np.mean(confidence[in_bin])) - float(np.mean(correct[in_bin]))) * fraction
    return float(ece)


def maximum_calibration_error(
    probabilities: np.ndarray, y_true_index: np.ndarray, n_bins: int = 10
) -> float:
    """Worst-case bin gap between confidence and accuracy."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_true_index = np.asarray(y_true_index, dtype=int)
    if probabilities.size == 0:
        return 0.0

    confidence = np.max(probabilities, axis=1)
    correct = (np.argmax(probabilities, axis=1) == y_true_index).astype(float)

    worst = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if np.any(in_bin):
            worst = max(worst, abs(float(np.mean(confidence[in_bin])) - float(np.mean(correct[in_bin]))))
    return float(worst)


def multiclass_brier_score(probabilities: np.ndarray, y_true_index: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_true_index = np.asarray(y_true_index, dtype=int)
    if probabilities.size == 0:
        return 0.0
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true_index)), y_true_index] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability_bins(
    probabilities: np.ndarray, y_true_index: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Bin-wise confidence/accuracy pairs, for plotting a reliability diagram."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_true_index = np.asarray(y_true_index, dtype=int)
    confidence = np.max(probabilities, axis=1) if probabilities.size else np.array([])
    correct = (
        (np.argmax(probabilities, axis=1) == y_true_index).astype(float) if probabilities.size else np.array([])
    )

    bins: List[Dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": float(np.sum(in_bin)),
                "mean_confidence": float(np.mean(confidence[in_bin])) if np.any(in_bin) else None,
                "accuracy": float(np.mean(correct[in_bin])) if np.any(in_bin) else None,
            }
        )
    return bins


# ----------------------------------------------------------------- calibrators
class _IsotonicFunction:
    """Piecewise-linear isotonic map persisted as knots."""

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    @classmethod
    def fit(cls, scores: np.ndarray, targets: np.ndarray) -> "_IsotonicFunction":
        from sklearn.isotonic import IsotonicRegression

        scores = np.asarray(scores, dtype=float)
        targets = np.asarray(targets, dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores, targets)
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(scores, dtype=float), self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "isotonic", "x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "_IsotonicFunction":
        return cls(payload["x"], payload["y"])


class _PlattFunction:
    """Sigmoid (Platt) map persisted as slope/intercept."""

    def __init__(self, coef: float, intercept: float):
        self.coef = float(coef)
        self.intercept = float(intercept)

    @classmethod
    def fit(cls, scores: np.ndarray, targets: np.ndarray) -> "_PlattFunction":
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        targets = np.asarray(targets, dtype=int)
        model = LogisticRegression(max_iter=1000)
        model.fit(scores, targets)
        return cls(float(model.coef_[0][0]), float(model.intercept_[0]))

    def apply(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.coef * scores + self.intercept)))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "platt", "coef": self.coef, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "_PlattFunction":
        return cls(payload["coef"], payload["intercept"])


def _build_function(method: str, scores: np.ndarray, targets: np.ndarray):
    if method == "platt":
        return _PlattFunction.fit(scores, targets)
    return _IsotonicFunction.fit(scores, targets)


class MulticlassCalibrator:
    """Per-class one-vs-rest calibration with renormalisation.

    Example:
        >>> cal = MulticlassCalibrator(method="isotonic")
        >>> cal.fit(probs, ["Normal", "PVC", "PVC"])      # doctest: +SKIP
        >>> cal.transform(probs)                          # doctest: +SKIP
    """

    def __init__(self, method: str = "isotonic", classes: Optional[Sequence[str]] = None):
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self.classes: List[str] = list(classes) if classes else []
        self.functions: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, probabilities: np.ndarray, y_true: Sequence[str]) -> "MulticlassCalibrator":
        probabilities = np.asarray(probabilities, dtype=float)
        y_true = np.asarray(list(y_true), dtype=str)

        if not self.classes:
            self.classes = sorted(set(y_true.tolist()))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes):
            raise ValueError(
                f"Expected probabilities with {len(self.classes)} columns matching classes {self.classes}, "
                f"got shape {probabilities.shape}."
            )

        for idx, name in enumerate(self.classes):
            scores = probabilities[:, idx]
            targets = (y_true == name).astype(int)
            if len(np.unique(targets)) < 2:
                # A class absent from the calibration set cannot be calibrated;
                # fall back to the identity map rather than inventing a curve.
                self.functions[name] = None
                continue
            self.functions[name] = _build_function(self.method, scores, targets)

        self.is_fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated, renormalised class probabilities."""
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities[np.newaxis, :]

        calibrated = np.empty_like(probabilities)
        for idx, name in enumerate(self.classes):
            function = self.functions.get(name)
            calibrated[:, idx] = probabilities[:, idx] if function is None else function.apply(probabilities[:, idx])

        totals = calibrated.sum(axis=1, keepdims=True)
        totals[totals <= 0] = 1.0
        return calibrated / totals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "classes": self.classes,
            "functions": {
                name: (func.to_dict() if func is not None else None) for name, func in self.functions.items()
            },
            "is_fitted": self.is_fitted,
            "fitted_at": datetime.datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MulticlassCalibrator":
        calibrator = cls(method=payload.get("method", "isotonic"), classes=payload.get("classes"))
        for name, function_payload in (payload.get("functions") or {}).items():
            if function_payload is None:
                calibrator.functions[name] = None
            elif function_payload.get("type") == "platt":
                calibrator.functions[name] = _PlattFunction.from_dict(function_payload)
            else:
                calibrator.functions[name] = _IsotonicFunction.from_dict(function_payload)
        calibrator.is_fitted = bool(payload.get("is_fitted", False))
        return calibrator


# ------------------------------------------------------------------ conformal
class MondrianConformalClassifier:
    """Class-conditional split-conformal predictor over softmax outputs.

    Coverage guarantee: for a fresh exchangeable sample, the true label lies in
    ``predict_set`` with probability at least ``1 - alpha``, up to the finite
    calibration-set correction.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        classes: Optional[Sequence[str]] = None,
        min_class_samples: int = 10,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.alpha = float(alpha)
        self.classes: List[str] = list(classes) if classes else []
        self.quantiles: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        #: Classes too sparsely represented to estimate their own quantile.
        #: These fall back to a pooled quantile, which is recorded explicitly
        #: because it weakens the per-class coverage statement.
        self.min_class_samples = int(min_class_samples)
        self.pooled_classes: List[str] = []
        self.pooled_quantile: Optional[float] = None
        self.is_fitted = False

    def fit(self, probabilities: np.ndarray, y_true: Sequence[str]) -> "MondrianConformalClassifier":
        probabilities = np.asarray(probabilities, dtype=float)
        y_true = np.asarray(list(y_true), dtype=str)

        if not self.classes:
            self.classes = sorted(set(y_true.tolist()))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes):
            raise ValueError(
                f"Expected probabilities with {len(self.classes)} columns matching classes {self.classes}, "
                f"got shape {probabilities.shape}."
            )

        index_of = {name: i for i, name in enumerate(self.classes)}

        # Nonconformity of a calibration point = 1 - P(true class).
        all_scores = 1.0 - probabilities[np.arange(len(y_true)), [index_of[label] for label in y_true]]
        n_total = int(len(all_scores))
        pooled_level = min(1.0, np.ceil((n_total + 1) * (1.0 - self.alpha)) / n_total) if n_total else 1.0
        self.pooled_quantile = float(np.quantile(all_scores, pooled_level, method="higher")) if n_total else 1.0
        self.pooled_classes = []

        for name in self.classes:
            mask = y_true == name
            n = int(np.sum(mask))
            self.counts[name] = n

            if n < self.min_class_samples:
                # Too few examples to estimate this class's own quantile. Rather
                # than emitting a wild per-class threshold, fall back to the
                # pooled quantile and record that the per-class coverage for this
                # label is not individually guaranteed.
                self.quantiles[name] = self.pooled_quantile
                self.pooled_classes.append(name)
                continue

            nonconformity = 1.0 - probabilities[mask, index_of[name]]
            # Finite-sample corrected quantile.
            level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
            self.quantiles[name] = float(np.quantile(nonconformity, level, method="higher"))

        self.is_fitted = True
        return self

    def predict_set(self, probabilities: np.ndarray) -> List[str]:
        """Return the conformal prediction set for one or more probability rows."""
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities[np.newaxis, :]
        if not self.is_fitted:
            raise RuntimeError("Conformal classifier is not fitted.")

        sets: List[List[str]] = []
        for row in probabilities:
            members = [
                name
                for idx, name in enumerate(self.classes)
                if (1.0 - float(row[idx])) <= self.quantiles.get(name, 1.0)
            ]
            # A conformal set is never empty: an empty set would imply certainty.
            if not members:
                members = [self.classes[int(np.argmax(row))]]
            sets.append(members)
        return sets

    def abstain(self, probabilities: np.ndarray) -> Tuple[bool, List[str]]:
        """Return ``(should_abstain, prediction_set)``.

        Abstention means the calibrated evidence does not separate the candidate
        labels at the requested coverage level.
        """
        prediction_set = self.predict_set(probabilities)
        first = prediction_set[0] if prediction_set else []
        return (len(first) > 1), first

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "classes": self.classes,
            "quantiles": self.quantiles,
            "calibration_counts": self.counts,
            "min_class_samples": self.min_class_samples,
            "pooled_classes": self.pooled_classes,
            "pooled_quantile": self.pooled_quantile,
            "is_fitted": self.is_fitted,
            "target_coverage": 1.0 - self.alpha,
            "coverage_note": (
                "Per-class coverage is individually guaranteed for classes with at least "
                f"{self.min_class_samples} calibration samples. Classes {self.pooled_classes or []} use a "
                "pooled quantile and carry the marginal coverage statement only."
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MondrianConformalClassifier":
        conformal = cls(
            alpha=payload.get("alpha", 0.10),
            classes=payload.get("classes"),
            min_class_samples=payload.get("min_class_samples", 10),
        )
        conformal.quantiles = {k: float(v) for k, v in (payload.get("quantiles") or {}).items()}
        conformal.counts = {k: int(v) for k, v in (payload.get("calibration_counts") or {}).items()}
        conformal.pooled_classes = list(payload.get("pooled_classes") or [])
        pooled = payload.get("pooled_quantile")
        conformal.pooled_quantile = None if pooled is None else float(pooled)
        conformal.is_fitted = bool(payload.get("is_fitted", False))
        return conformal


# ---------------------------------------------------------------- persistence
def save_calibration(
    calibrator: MulticlassCalibrator,
    conformal: Optional[MondrianConformalClassifier] = None,
    *,
    path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist calibration artifacts to JSON."""
    target = Path(path) if path else CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibrator": calibrator.to_dict(),
        "conformal": conformal.to_dict() if conformal is not None else None,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def load_calibration(
    path: Optional[Path] = None,
) -> Tuple[Optional[MulticlassCalibrator], Optional[MondrianConformalClassifier], Dict[str, Any]]:
    """Load calibration artifacts, returning ``(calibrator, conformal, metadata)``.

    Returns ``(None, None, {})`` when nothing has been fitted yet, so that
    inference degrades to uncalibrated output rather than failing.
    """
    target = Path(path) if path else CALIBRATION_PATH
    if not target.exists():
        return None, None, {}

    try:
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, None, {}

    calibrator = MulticlassCalibrator.from_dict(payload["calibrator"]) if payload.get("calibrator") else None
    conformal = (
        MondrianConformalClassifier.from_dict(payload["conformal"]) if payload.get("conformal") else None
    )
    metadata = {k: v for k, v in payload.items() if k not in {"calibrator", "conformal"}}
    return calibrator, conformal, metadata


def apply_calibration(
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience wrapper used by the inference path.

    Returns a dict with ``raw_probabilities``, ``calibrated_probabilities``,
    ``prediction_set``, ``abstain`` and ``calibration_applied`` plus supporting
    metrics. When no calibration artifact exists, ``calibration_applied`` is
    False and the raw probabilities are echoed unchanged — an uncalibrated output
    is reported as uncalibrated rather than silently presented as calibrated.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities[np.newaxis, :]

    calibrator, conformal, metadata = load_calibration(path)
    class_list = list(classes)

    result: Dict[str, Any] = {
        "calibration_applied": False,
        "conformal_applied": False,
        "raw_probabilities": probabilities.tolist(),
        "calibrated_probabilities": probabilities.tolist(),
        "prediction_set": [],
        "abstain": False,
        "abstain_reason": None,
        "calibration_metadata": metadata,
        "calibration_withheld_reason": None,
    }

    # A calibration artifact only reaches production after it demonstrated an
    # improvement on a partition that did not fit it. Applying a map that measurably
    # made probabilities *worse* would be a step backwards dressed as rigour.
    if metadata and metadata.get("deployment_recommended") is False:
        result["calibration_withheld_reason"] = metadata.get("deployment_recommendation_reason") or (
            "Stored calibration artifact did not demonstrate improvement on its held-out evaluation "
            "partition, so raw probabilities are served uncalibrated."
        )
        return result

    if calibrator is not None and calibrator.is_fitted and calibrator.classes == class_list:
        result["calibrated_probabilities"] = calibrator.transform(probabilities).tolist()
        result["calibration_applied"] = True

    effective = np.asarray(result["calibrated_probabilities"], dtype=float)

    if conformal is not None and conformal.is_fitted and conformal.classes == class_list:
        should_abstain, prediction_set = conformal.abstain(effective)
        result["conformal_applied"] = True
        result["prediction_set"] = prediction_set
        result["abstain"] = bool(should_abstain)
        if should_abstain:
            result["abstain_reason"] = (
                f"Conformal prediction set contains {len(prediction_set)} labels "
                f"({', '.join(prediction_set)}) at {int((1 - conformal.alpha) * 100)}% target coverage. "
                "The model has not separated these classes for this recording."
            )

    return result
