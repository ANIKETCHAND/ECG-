"""
ECG Quality Copilot
===================
Orchestrates multi-lead signal verification, lead-level quality mapping,
issue identification, and actionable clinical recommendations.

Output Format:
ECG QUALITY
Overall: GOOD
Lead II: GOOD | V1: GOOD | V2: POOR
Issue: Excessive artifact detected in V2.
Recommendation: Repeat ECG acquisition or check electrode placement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from src.ecg_core.models import ECGRecording
    from src.quality.lead_quality import evaluate_single_lead_quality
    from src.quality.quality_gate import QualityCategory, QualityGateDecision, evaluate_ecg_quality_gate
except ImportError:
    from ecg_core.models import ECGRecording
    from quality.lead_quality import evaluate_single_lead_quality
    from quality.quality_gate import QualityCategory, QualityGateDecision, evaluate_ecg_quality_gate


def assess_ecg_copilot_quality(
    ecg_or_signal: Union[ECGRecording, np.ndarray],
    fs: Optional[float] = None,
    lead_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Inspect all leads in an ECG recording and return overall quality assessment and display block."""
    if isinstance(ecg_or_signal, ECGRecording):
        signals = ecg_or_signal.signals
        sampling_rate = ecg_or_signal.sampling_rate
        leads = ecg_or_signal.lead_names
    else:
        signals = np.asarray(ecg_or_signal, dtype=float)
        if signals.ndim == 1:
            signals = np.expand_dims(signals, axis=0)
        sampling_rate = float(fs) if fs else 360.0
        leads = lead_names if lead_names and len(lead_names) == signals.shape[0] else [f"Lead_{i+1}" for i in range(signals.shape[0])]

    n_leads = signals.shape[0]
    lead_evaluations: List[Dict[str, Any]] = []
    all_issues: List[str] = []
    overall_scores: List[float] = []
    categories: List[str] = []

    for i in range(n_leads):
        lead_name = leads[i]
        lead_sig = signals[i]
        eval_res = evaluate_single_lead_quality(lead_sig, sampling_rate, lead_name=lead_name)
        lead_evaluations.append(eval_res)
        overall_scores.append(eval_res["quality_score"])
        categories.append(eval_res["status"])
        if eval_res["issues"]:
            all_issues.append(f"{lead_name}: {', '.join(eval_res['issues'])}")

    # Overall determination
    if "UNUSABLE" in categories:
        overall_status = "UNUSABLE"
        can_run_ai = False
        recommendation = "Repeat ECG acquisition. At least one critical lead is disconnected or corrupted."
    elif "POOR" in categories:
        overall_status = "POOR"
        can_run_ai = False
        recommendation = "Significant artifact present. Basic measurements permitted; AI rhythm classification suppressed."
    elif "ACCEPTABLE" in categories:
        overall_status = "ACCEPTABLE"
        can_run_ai = True
        recommendation = "Acceptable trace. Review signal baseline before clinical sign-off."
    else:
        overall_status = "GOOD"
        can_run_ai = True
        recommendation = "Optimal signal quality across all monitored leads. Proceed with clinical analysis."

    mean_score = round(float(np.mean(overall_scores)), 2) if overall_scores else 0.0

    # Format clinical UI text
    lead_summary_parts = [f"{eval_res['lead']}: {eval_res['status']}" for eval_res in lead_evaluations]
    lead_summary_str = " | ".join(lead_summary_parts)

    ui_text = (
        f"ECG QUALITY\n"
        f"Overall: {overall_status}\n"
        f"{lead_summary_str}\n"
        f"Issue: {'; '.join(all_issues) if all_issues else 'None detected.'}\n"
        f"Recommendation: {recommendation}"
    )

    return {
        "overall_status": overall_status,
        "can_run_ai": can_run_ai,
        "mean_quality_score": mean_score,
        "lead_evaluations": lead_evaluations,
        "issues": all_issues,
        "recommendation": recommendation,
        "ui_text": ui_text,
    }
