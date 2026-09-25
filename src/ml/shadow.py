"""
Shadow Mode Evaluation
======================

Runs a candidate model alongside the active production model on the same inputs,
records where they disagree, and writes an audit entry. Production output is
unaffected: the candidate's predictions are never returned to a caller.

This is what makes "promote the candidate" an evidence-based decision rather than
a leap of faith. Disagreements are the interesting signal — a candidate that
agrees 99.9% of the time is either strictly better or not worth the migration
risk, and you cannot tell which without looking at the cases where they differ.

Two guards are enforced:

* A candidate registered as ``SYNTHETIC_PLACEHOLDER`` is refused outright.
* Each model is evaluated with its **own** scaler, since a candidate fitted
  against differently scaled features would otherwise be silently mis-scaled and
  appear catastrophically wrong.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.ml.models.registry import GLOBAL_MODEL_REGISTRY, ModelLifecycleStatus

PRODUCTION_MODEL_ID = "ECG-RF-1.0.0"
DEFAULT_CANDIDATE_MODEL_ID = "ECG-RF-2.0.0-candidate"


@dataclass
class ShadowDisagreement:
    beat_index: int
    production_prediction: str
    candidate_prediction: str
    production_confidence: float
    candidate_confidence: float
    confidence_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowComparisonReport:
    status: str  # COMPLETED, BLOCKED_PLACEHOLDER, BLOCKED_MISSING_MODEL, FAILED
    production_model_id: Optional[str]
    candidate_model_id: Optional[str]
    beats_compared: int = 0
    agreement_rate: Optional[float] = None
    production_class_counts: Dict[str, int] = field(default_factory=dict)
    candidate_class_counts: Dict[str, int] = field(default_factory=dict)
    mean_confidence_delta: Optional[float] = None
    disagreements: List[ShadowDisagreement] = field(default_factory=list)
    recommendation: str = ""
    note: str = ""
    evaluated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    audit_logged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["disagreements"] = [d.to_dict() for d in self.disagreements]
        return data


def _predict(model: Any, scaler: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=float)
    if scaler is not None:
        matrix = scaler.transform(matrix)
    predictions = model.predict(matrix)
    try:
        probabilities = model.predict_proba(matrix)
        confidence = np.max(probabilities, axis=1)
    except Exception:
        confidence = np.ones(len(predictions), dtype=float)
    return np.asarray(predictions).astype(str), confidence


def run_shadow_comparison(
    features: np.ndarray,
    *,
    production_model_id: str = PRODUCTION_MODEL_ID,
    candidate_model_id: str = DEFAULT_CANDIDATE_MODEL_ID,
    max_disagreements: int = 50,
    audit_context: Optional[Dict[str, Any]] = None,
) -> ShadowComparisonReport:
    """Compare a candidate against production on the same feature matrix.

    Args:
        features: Feature matrix (unscaled) — one row per beat.
        production_model_id: Model currently serving patients.
        candidate_model_id: Model under evaluation.
        max_disagreements: Cap on retained disagreement rows.
        audit_context: Optional ``user_id``/``username``/``user_role``/
            ``patient_id`` values for the audit entry.

    Returns:
        A :class:`ShadowComparisonReport`. The candidate's predictions are
        recorded for evaluation only and must not be returned to a clinician.
    """
    # 1. Registration and provenance gates. Checked before anything is loaded so
    # that a fabricated or unknown candidate is refused with an accurate reason
    # rather than failing later as a generic load error.
    for model_id in (production_model_id, candidate_model_id):
        try:
            entry = GLOBAL_MODEL_REGISTRY.get_model_entry(model_id)
        except KeyError:
            return ShadowComparisonReport(
                status="BLOCKED_MISSING_MODEL",
                production_model_id=production_model_id,
                candidate_model_id=candidate_model_id,
                note=f"Model '{model_id}' is not registered in the model catalog.",
                recommendation="Register the model before shadow evaluation.",
            )

        if GLOBAL_MODEL_REGISTRY.is_placeholder(model_id):
            return ShadowComparisonReport(
                status="BLOCKED_PLACEHOLDER",
                production_model_id=production_model_id,
                candidate_model_id=candidate_model_id,
                note=(
                    f"Model '{model_id}' is a {ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value} "
                    f"trained on fabricated/demo data. Reason: "
                    f"{entry.get('provenance', {}).get('reason', 'unspecified')}. "
                    "Shadow comparison would produce meaningless disagreement statistics."
                ),
                recommendation="Train the model on real data before shadow evaluation.",
            )

    if features is None or len(features) == 0:
        return ShadowComparisonReport(
            status="FAILED",
            production_model_id=production_model_id,
            candidate_model_id=candidate_model_id,
            note="No feature rows were supplied.",
        )

    try:
        prod_model, prod_scaler, _ = GLOBAL_MODEL_REGISTRY.load_model(production_model_id)
        cand_model, cand_scaler, _ = GLOBAL_MODEL_REGISTRY.load_model(candidate_model_id)
    except Exception as exc:
        return ShadowComparisonReport(
            status="FAILED",
            production_model_id=production_model_id,
            candidate_model_id=candidate_model_id,
            note=f"Could not load model artifacts: {exc}",
        )

    prod_pred, prod_conf = _predict(prod_model, prod_scaler, features)
    cand_pred, cand_conf = _predict(cand_model, cand_scaler, features)

    if len(prod_pred) != len(cand_pred):
        return ShadowComparisonReport(
            status="FAILED",
            production_model_id=production_model_id,
            candidate_model_id=candidate_model_id,
            note="Models returned differing numbers of predictions.",
        )

    total = len(prod_pred)
    differing = np.where(prod_pred != cand_pred)[0]
    agreement_rate = float(1.0 - len(differing) / total) if total else None

    def counts(predictions: np.ndarray) -> Dict[str, int]:
        values, frequencies = np.unique(predictions, return_counts=True)
        return {str(v): int(f) for v, f in zip(values, frequencies)}

    disagreements = [
        ShadowDisagreement(
            beat_index=int(index),
            production_prediction=str(prod_pred[index]),
            candidate_prediction=str(cand_pred[index]),
            production_confidence=round(float(prod_conf[index]), 4),
            candidate_confidence=round(float(cand_conf[index]), 4),
            confidence_delta=round(float(cand_conf[index] - prod_conf[index]), 4),
        )
        for index in differing[:max_disagreements]
    ]

    mean_delta = float(np.mean(cand_conf - prod_conf)) if total else None

    # A candidate that never disagrees is not necessarily better; a candidate
    # that disagrees often needs clinician adjudication of those cases.
    disagreement_fraction = len(differing) / total if total else 0.0
    if disagreement_fraction == 0.0:
        recommendation = (
            "Candidate and production agree on every beat in this sample. Review the sample size "
            "before concluding equivalence."
        )
    elif disagreement_fraction < 0.05:
        recommendation = (
            f"Disagreement on {disagreement_fraction * 100:.2f}% of beats. Small enough for a clinician "
            "to adjudicate the listed cases directly."
        )
    else:
        recommendation = (
            f"Disagreement on {disagreement_fraction * 100:.2f}% of beats. Adjudicate the listed cases "
            "with a cardiologist before considering promotion; a divergence this large may reflect a "
            "genuine change in behaviour rather than noise."
        )

    report = ShadowComparisonReport(
        status="COMPLETED",
        production_model_id=production_model_id,
        candidate_model_id=candidate_model_id,
        beats_compared=total,
        agreement_rate=round(agreement_rate, 6) if agreement_rate is not None else None,
        production_class_counts=counts(prod_pred),
        candidate_class_counts=counts(cand_pred),
        mean_confidence_delta=round(mean_delta, 6) if mean_delta is not None else None,
        disagreements=disagreements,
        recommendation=recommendation,
        note=(
            "Candidate predictions are evaluation-only and must never be returned to a clinician. "
            "Each model was applied with its own scaler."
        ),
    )

    report.audit_logged = _log_to_audit(report, audit_context or {})
    return report


def _log_to_audit(report: ShadowComparisonReport, context: Dict[str, Any]) -> bool:
    """Record the shadow comparison in the tamper-evident audit trail."""
    try:
        from src.audit.audit_logger import AUDIT_LOGGER
    except ImportError:
        return False

    try:
        AUDIT_LOGGER.log_event(
            event_type="AI_MODEL_SHADOW_EVALUATED",
            user_id=str(context.get("user_id", "SYSTEM")),
            username=str(context.get("username", "shadow-evaluator")),
            user_role=str(context.get("user_role", "SYSTEM")),
            action="SHADOW_COMPARISON",
            details={
                "production_model_id": report.production_model_id,
                "candidate_model_id": report.candidate_model_id,
                "beats_compared": report.beats_compared,
                "agreement_rate": report.agreement_rate,
                "disagreement_count": len(report.disagreements),
            },
            status="SUCCESS" if report.status == "COMPLETED" else report.status,
            patient_id=context.get("patient_id"),
        )
        return True
    except Exception:
        # An audit-trail failure must not destroy the evaluation result, but it
        # must not be reported as logged either.
        return False
