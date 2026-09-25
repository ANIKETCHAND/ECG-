"""
SHAP-Based Beat Attribution
===========================

Quantitative "why did the model say that?" evidence for a single beat.

When ``shap`` is installed and the model is a supported tree ensemble, per-beat
Shapley attributions are computed from the model itself. When it is not, the
module **says so** and returns the project's existing baseline-deviation
attribution instead, labelled ``baseline_deviation_fallback``.

The distinction matters: a baseline deviation says "this beat differs from the
patient's own normal beats", whereas a Shapley value says "this feature moved the
model's output by this much". They are different claims, and the artifact records
which one was produced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # shap is optional; the pipeline must run without it.
    import shap  # type: ignore

    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - environment dependent
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(exc)


METHOD_SHAP = "shap_tree"
METHOD_FALLBACK = "baseline_deviation_fallback"


def shap_available() -> bool:
    """Return True when real Shapley attribution can be computed."""
    return SHAP_AVAILABLE


def _is_tree_model(model: Any) -> bool:
    return hasattr(model, "estimators_") or model.__class__.__name__ in {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "ExtraTreesClassifier",
        "DecisionTreeClassifier",
    }


def _shap_values_matrix(values: Any, class_index: int) -> Optional[np.ndarray]:
    """Normalise the several shapes ``shap`` returns across versions."""
    if isinstance(values, list):
        if not values:
            return None
        index = min(class_index, len(values) - 1)
        return np.asarray(values[index], dtype=float)

    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        # (n_samples, n_features, n_classes)
        index = min(class_index, array.shape[2] - 1)
        return array[:, :, index]
    return array


def attribute_beats_with_shap(
    model: Any,
    X: np.ndarray,
    feature_names: Sequence[str],
    classes: Sequence[str],
    *,
    beat_numbers: Optional[Sequence[int]] = None,
    top_k: int = 5,
    max_beats: int = 25,
    baseline_X: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Attribute model output to features for individual beats.

    Args:
        model: Fitted classifier exposing ``predict_proba``.
        X: Feature matrix (already scaled exactly as the model expects).
        feature_names: Ordered feature names matching ``X`` columns.
        classes: Class labels matching ``model.classes_``.
        beat_numbers: 1-based beat numbers to explain; defaults to all (capped).
        top_k: Number of features reported per beat.
        max_beats: Cap on explained beats, to bound runtime.
        baseline_X: Optional baseline feature matrix used only by the fallback.

    Returns:
        Dict with ``method``, ``enabled``, ``reason``, ``notes`` and ``beats``.
    """
    X = np.asarray(X, dtype=float)
    feature_names = list(feature_names)
    classes = [str(c) for c in classes]

    report: Dict[str, Any] = {
        "method": METHOD_FALLBACK,
        "enabled": False,
        "reason": None,
        "notes": [],
        "beats": [],
    }

    if X.size == 0:
        report["reason"] = "No feature rows were supplied."
        return report

    indices = list(range(min(len(X), max_beats)))
    if beat_numbers is not None:
        indices = [b - 1 for b in beat_numbers if 0 <= b - 1 < len(X)][:max_beats]

    explained_baseline = np.asarray(baseline_X, dtype=float) if baseline_X is not None else None

    if not SHAP_AVAILABLE:
        report["reason"] = (
            "shap is not installed; falling back to baseline-deviation attribution. "
            f"Install with: pip install shap  (import error: {SHAP_IMPORT_ERROR})"
        )
    elif not _is_tree_model(model):
        report["reason"] = (
            f"Model type '{model.__class__.__name__}' has no tree explainer configured; "
            "falling back to baseline-deviation attribution."
        )

    if report["reason"] is None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[indices])
            model_classes = [str(c) for c in getattr(model, "classes_", classes)]
            predictions = model.predict(X[indices])
            probabilities = model.predict_proba(X[indices])

            beats: List[Dict[str, Any]] = []
            for row, beat_idx in enumerate(indices):
                predicted = str(predictions[row])
                class_index = model_classes.index(predicted) if predicted in model_classes else 0
                values = _shap_values_matrix(shap_values, class_index)
                if values is None or row >= values.shape[0]:
                    continue

                contributions = values[row]
                order = np.argsort(np.abs(contributions))[::-1][:top_k]
                beats.append(
                    {
                        "beat_number": beat_idx + 1,
                        "predicted_class": predicted,
                        "prediction_probability": round(
                            float(probabilities[row][class_index]), 4
                        ),
                        "attributions": [
                            {
                                "feature_name": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                                "feature_value": round(float(X[beat_idx][i]), 4),
                                "shap_contribution": round(float(contributions[i]), 5),
                                "direction": "increases" if contributions[i] > 0 else "decreases",
                            }
                            for i in order
                        ],
                    }
                )

            report["method"] = METHOD_SHAP
            report["enabled"] = True
            report["notes"].append(
                "Shapley values are computed against this model's expected output. They explain model "
                "behaviour and are not causal claims about the patient."
            )
            report["beats"] = beats
            return report
        except Exception as exc:  # pragma: no cover - defensive
            report["reason"] = f"SHAP explainer failed ({exc}); falling back to baseline-deviation attribution."

    # Fallback: deviation of each beat from the baseline (normal) population.
    if explained_baseline is None or explained_baseline.size == 0:
        report["notes"].append(
            "No baseline feature matrix was supplied, so baseline-deviation attribution could not run."
        )
        return report

    mean_baseline = np.mean(explained_baseline, axis=0)
    std_baseline = np.std(explained_baseline, axis=0)
    std_baseline[std_baseline == 0] = 1e-6

    beats = []
    for beat_idx in indices:
        deviations = (X[beat_idx] - mean_baseline) / std_baseline
        order = np.argsort(np.abs(deviations))[::-1][:top_k]
        beats.append(
            {
                "beat_number": beat_idx + 1,
                "predicted_class": None,
                "attributions": [
                    {
                        "feature_name": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                        "feature_value": round(float(X[beat_idx][i]), 4),
                        "baseline_mean": round(float(mean_baseline[i]), 4),
                        "z_score_deviation": round(float(deviations[i]), 3),
                    }
                    for i in order
                ],
            }
        )

    report["beats"] = beats
    report["notes"].append(
        "Fallback attribution reports how a beat differs from the patient's other beats. It is NOT a "
        "model attribution and should not be read as one."
    )
    return report


def summarise_attributions(attribution_report: Dict[str, Any], top_n: int = 5) -> List[str]:
    """Human-readable summary lines for reports."""
    lines: List[str] = []
    method = attribution_report.get("method", METHOD_FALLBACK)
    lines.append(f"Attribution method: {method}")
    if attribution_report.get("reason"):
        lines.append(f"Attribution note: {attribution_report['reason']}")

    for beat in (attribution_report.get("beats") or [])[:top_n]:
        parts = []
        for entry in beat.get("attributions", [])[:3]:
            if "shap_contribution" in entry:
                parts.append(
                    f"{entry['feature_name']} ({entry['shap_contribution']:+.3f}, {entry.get('direction','')})"
                )
            else:
                parts.append(f"{entry['feature_name']} (z={entry.get('z_score_deviation', 0):+.2f})")
        label = beat.get("predicted_class") or "unclassified"
        lines.append(f"  Beat {beat['beat_number']} [{label}]: " + ", ".join(parts))
    return lines
