"""
External Validation & Domain Shift Assessment
=============================================

Compares in-domain performance against performance on an independent external
cohort, to characterise:

* demographic domain shift (age, sex, geography),
* technical variation (electrodes, filter cut-offs, sampling rates),
* performance degradation (sensitivity/specificity delta), and
* calibration drift (ECE increase).

Integrity rule
--------------
The report only describes a shift that was actually measured. If
``external_metrics`` is ``None`` — because the external cohort is not present, or
lacks the required columns — the report is returned with
``status = "NOT_EVALUATED"`` and every performance field is ``None``. It will not
estimate, extrapolate or nudge a number to look like a result.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def generate_external_validation_report(
    model_id: str,
    training_dataset: str,
    external_dataset: str,
    in_domain_metrics: Optional[Dict[str, Any]] = None,
    external_metrics: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an external-validation report.

    Args:
        model_id: Model under evaluation.
        training_dataset: Dataset identifier used for fitting.
        external_dataset: Independent dataset identifier.
        in_domain_metrics: Measured metrics on the held-out in-domain split.
        external_metrics: Measured metrics on the external cohort. ``None`` when
            no external cohort was available, which yields a NOT_EVALUATED report.
        note: Optional explanation appended to the report.

    Returns:
        Structured report. Contains no performance numbers unless the external
        cohort was genuinely evaluated.
    """
    if external_metrics is None:
        return {
            "status": "NOT_EVALUATED",
            "model_id": model_id,
            "training_cohort": training_dataset,
            "external_validation_cohort": external_dataset,
            "in_domain_accuracy": (in_domain_metrics or {}).get("accuracy"),
            "external_accuracy": None,
            "accuracy_delta": None,
            "f1_delta": None,
            "calibration_drift_ece": None,
            "domain_shift_findings": [
                "No external cohort was available, so no domain-shift statement can be made."
            ],
            "integrity_note": (
                "External validation was not performed. Cross-dataset generalisation for this model "
                "remains unmeasured; do not report a domain-shift figure for it."
            ),
            "note": note,
            "regulatory_note": "External validation pending per ISO 14971 / IEC 62304 Section 5.7.",
        }

    acc_drop = _delta(in_domain_metrics, external_metrics, "accuracy")
    f1_drop = _delta(in_domain_metrics, external_metrics, "weighted_f1")
    ece_shift = _delta(external_metrics, in_domain_metrics, "expected_calibration_error", fallback_key="ece")

    findings = []
    if acc_drop is not None and acc_drop > 0.03:
        findings.append(f"Accuracy degradation of {acc_drop * 100:.2f}% observed on the external cohort.")
    if ece_shift is not None and ece_shift > 0.02:
        findings.append(f"Calibration drift (ECE +{ece_shift:.4f}) on external recordings.")
    if not findings:
        findings.append("No material degradation detected across the measured external cohort.")

    return {
        "status": "EVALUATED",
        "model_id": model_id,
        "training_cohort": training_dataset,
        "external_validation_cohort": external_dataset,
        "in_domain_accuracy": _metric(in_domain_metrics, "accuracy"),
        "external_accuracy": _metric(external_metrics, "accuracy"),
        "accuracy_delta": None if acc_drop is None else round(acc_drop, 4),
        "f1_delta": None if f1_drop is None else round(f1_drop, 4),
        "calibration_drift_ece": None if ece_shift is None else round(ece_shift, 4),
        "in_domain_metrics": _subset(in_domain_metrics),
        "external_metrics": _subset(external_metrics),
        "domain_shift_findings": findings,
        "integrity_note": "All deltas computed from measured in-domain and external metrics.",
        "note": note,
        "regulatory_note": "External validation performed per ISO 14971 / IEC 62304 Section 5.7.",
    }


def _metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not metrics:
        return None
    value = metrics.get(key)
    return None if value is None else float(value)


def _delta(
    left: Optional[Dict[str, Any]],
    right: Optional[Dict[str, Any]],
    key: str,
    fallback_key: Optional[str] = None,
) -> Optional[float]:
    """Return ``left[key] - right[key]``, or None when either side is missing."""
    if not left or not right:
        return None
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None and fallback_key:
        left_value = left.get(fallback_key)
    if right_value is None and fallback_key:
        right_value = right.get(fallback_key)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _subset(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    keys = ("accuracy", "weighted_f1", "macro_f1", "auroc", "auprc", "ece", "expected_calibration_error")
    return {key: metrics.get(key) for key in keys if key in metrics}
