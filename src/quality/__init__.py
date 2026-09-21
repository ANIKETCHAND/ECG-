"""
ECG Quality Copilot Package
===========================
Lead-by-lead quality scoring, artifact/clipping/noise detection,
and pre-inference safety gatekeeper.
"""

from .artifact_detection import detect_motion_and_muscle_artifacts
from .baseline_wander import analyze_baseline_wander
from .clipping_detection import detect_clipping_and_flatline
from .lead_quality import evaluate_single_lead_quality
from .noise_detection import analyze_noise_and_powerline
from .quality_gate import QualityCategory, QualityGateDecision, evaluate_ecg_quality_gate
from .signal_quality import assess_ecg_copilot_quality

__all__ = [
    "QualityCategory",
    "QualityGateDecision",
    "evaluate_ecg_quality_gate",
    "assess_ecg_copilot_quality",
    "evaluate_single_lead_quality",
    "detect_clipping_and_flatline",
    "analyze_baseline_wander",
    "analyze_noise_and_powerline",
    "detect_motion_and_muscle_artifacts",
]
