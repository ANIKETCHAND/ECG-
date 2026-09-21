"""
AI vs. ECG Machine Comparison Engine
====================================
Cross-verifies automated ECG machine statements against AI inferences.
Flags disagreements, coverage gaps, and heart-rate measurement variations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from src.comparison.disagreement_detector import ComparisonResult, ComparisonStatus
from src.comparison.machine_interpretation import ParsedMachineInterpretation


def compare_ai_and_machine(
    ai_result: Dict[str, Any],
    machine_interp: ParsedMachineInterpretation,
) -> ComparisonResult:
    """Compare ECG Machine Automated Interpretation with AI Model Prediction.

    Args:
        ai_result: Dictionary returned by run_ecg_ml_inference.
        machine_interp: ParsedMachineInterpretation object.

    Returns:
        Structured ComparisonResult with status and clinical advisory.
    """
    ai_pred = str(ai_result.get("prediction", "NO_RESULT")).strip()
    ai_hr = ai_result.get("heart_rate")

    # 1. Missing or unreadable machine statement -> UNABLE_TO_COMPARE
    if machine_interp.normalized_category == "UNSPECIFIED":
        return ComparisonResult(
            status=ComparisonStatus.UNABLE_TO_COMPARE,
            summary="ECG machine interpretation was missing, unreadable, or unparsed.",
            machine_category="UNSPECIFIED",
            ai_prediction=ai_pred,
            discrepancy_details=["No machine rhythm statement provided for comparative verification."],
            clinical_advisory="Clinician must interpret the raw ECG tracing independently without machine baseline comparison.",
            coverage_limitation_warning=None,
        )

    # 2. AI Failed or Yielded NO_RESULT -> UNABLE_TO_COMPARE
    if "NO_RESULT" in ai_pred:
        return ComparisonResult(
            status=ComparisonStatus.UNABLE_TO_COMPARE,
            summary=f"AI model did not produce an inference ({ai_pred}).",
            machine_category=machine_interp.normalized_category,
            ai_prediction=ai_pred,
            discrepancy_details=[f"AI analysis halted: {ai_pred}"],
            clinical_advisory="Review machine statement and quality warnings. Manual clinician interpretation mandatory.",
        )

    # 3. Check Heart Rate Difference
    rate_diff: Optional[float] = None
    if ai_hr is not None and machine_interp.reported_heart_rate is not None:
        rate_diff = round(abs(float(ai_hr) - float(machine_interp.reported_heart_rate)), 1)

    # 4. Check for Machine Findings outside AI scope (STEMI, AFib, Block)
    if machine_interp.normalized_category in ["ISCHEMIA_INFARCT", "ATRIAL_FIBRILLATION", "CONDUCTION_BLOCK"]:
        pathology_desc = {
            "ISCHEMIA_INFARCT": "Acute Myocardial Infarction / Ischemia (STEMI/NSTEMI)",
            "ATRIAL_FIBRILLATION": "Atrial Fibrillation / Flutter",
            "CONDUCTION_BLOCK": "Atrioventricular or Bundle Branch Block",
        }.get(machine_interp.normalized_category, machine_interp.normalized_category)

        return ComparisonResult(
            status=ComparisonStatus.SIGNIFICANT_DISAGREEMENT,
            summary=f"CRITICAL DISCREPANCY: Machine reported {pathology_desc}, outside AI single-lead scope.",
            machine_category=machine_interp.normalized_category,
            ai_prediction=ai_pred,
            discrepancy_details=[
                f"Machine statement: '{machine_interp.raw_statement}'",
                f"AI rhythm prediction: '{ai_pred}'",
            ],
            clinical_advisory=(
                f"ALERT: The ECG machine algorithm flagged {pathology_desc}. "
                "The current AI model is ONLY trained for Lead II Normal vs. PVC screening and CANNOT rule out infarction or conduction blocks. "
                "Immediate 12-lead physician review is imperative."
            ),
            coverage_limitation_warning=(
                "Active AI model (ECG-RF-1.0.0) is not intended or validated for STEMI, AFib, or block detection."
            ),
            rate_difference_bpm=rate_diff,
        )

    # 5. Evaluate Normal vs. PVC
    ai_is_normal = "Normal" in ai_pred
    ai_is_pvc = "Premature Ventricular Contraction" in ai_pred or "PVC" in ai_pred

    mach_is_normal = machine_interp.normalized_category in ["NORMAL_SINUS", "BRADYCARDIA", "TACHYCARDIA"]
    mach_is_pvc = machine_interp.normalized_category == "VENTRICULAR_ECTOPY"

    # Both agree on Normal
    if ai_is_normal and mach_is_normal:
        if rate_diff is not None and rate_diff > 15.0:
            return ComparisonResult(
                status=ComparisonStatus.MINOR_DIFFERENCE,
                summary="Both agree on Normal Sinus Rhythm, but heart rate measurements diverge.",
                machine_category=machine_interp.normalized_category,
                ai_prediction=ai_pred,
                discrepancy_details=[
                    f"Machine HR: {machine_interp.reported_heart_rate} bpm vs. AI HR: {ai_hr} bpm (difference: {rate_diff} bpm)"
                ],
                clinical_advisory="Verify heart rate on rhythm strip to confirm whether machine or AI correctly detected all R-peaks.",
                rate_difference_bpm=rate_diff,
            )
        return ComparisonResult(
            status=ComparisonStatus.AGREE,
            summary="Machine and AI agree: Normal Sinus Rhythm.",
            machine_category=machine_interp.normalized_category,
            ai_prediction=ai_pred,
            discrepancy_details=[],
            clinical_advisory="Concordant normal findings. Physician review still required for final sign-off.",
            rate_difference_bpm=rate_diff,
        )

    # Both agree on PVC / Ventricular Ectopy
    if ai_is_pvc and mach_is_pvc:
        return ComparisonResult(
            status=ComparisonStatus.AGREE,
            summary="Machine and AI agree: Ventricular Ectopy / Premature Ventricular Contraction.",
            machine_category=machine_interp.normalized_category,
            ai_prediction=ai_pred,
            discrepancy_details=[],
            clinical_advisory="Concordant ventricular ectopy detected. Review coupling intervals and burden.",
            rate_difference_bpm=rate_diff,
        )

    # Disagreement: AI flags PVC but machine said Normal
    if ai_is_pvc and mach_is_normal:
        return ComparisonResult(
            status=ComparisonStatus.SIGNIFICANT_DISAGREEMENT,
            summary="DISAGREEMENT: AI detected Ventricular Ectopy (PVC), but machine reported Normal Sinus Rhythm.",
            machine_category=machine_interp.normalized_category,
            ai_prediction=ai_pred,
            discrepancy_details=[
                "AI flagged premature ventricular ectopy.",
                f"Machine statement reported '{machine_interp.raw_statement}'.",
            ],
            clinical_advisory=(
                "Physician attention required: Check AI Evidence Engine for highlighted aberrant beats. "
                "Determine if machine missed isolated ectopy or if AI encountered an artifact."
            ),
            rate_difference_bpm=rate_diff,
        )

    # Disagreement: Machine flags PVC but AI predicts Normal
    if mach_is_pvc and ai_is_normal:
        return ComparisonResult(
            status=ComparisonStatus.SIGNIFICANT_DISAGREEMENT,
            summary="DISAGREEMENT: Machine reported Ventricular Ectopy, but AI predicted Normal Sinus Rhythm.",
            machine_category=machine_interp.normalized_category,
            ai_prediction=ai_pred,
            discrepancy_details=[
                f"Machine reported: '{machine_interp.raw_statement}'.",
                "AI predicted normal rhythm.",
            ],
            clinical_advisory=(
                "Physician attention required: Inspect rhythm strip for transient ectopic beats that may have been averaged out or filtered."
            ),
            rate_difference_bpm=rate_diff,
        )

    # Fallback
    return ComparisonResult(
        status=ComparisonStatus.MINOR_DIFFERENCE,
        summary="Minor descriptor difference between machine interpretation and AI.",
        machine_category=machine_interp.normalized_category,
        ai_prediction=ai_pred,
        discrepancy_details=[
            f"Machine: '{machine_interp.raw_statement}'",
            f"AI: '{ai_pred}'",
        ],
        clinical_advisory="Review tracing to reconcile descriptive nuances.",
        rate_difference_bpm=rate_diff,
    )
