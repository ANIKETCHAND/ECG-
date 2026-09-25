"""
ECG AI Evidence Engine
======================
Transparent, explainable evidence generation for clinical ECG analysis.
"""

from src.evidence.beat_evidence import BeatEvidence, analyze_beats_for_evidence
from src.evidence.evidence_engine import EvidenceReport, generate_ai_evidence
from src.evidence.feature_evidence import compute_feature_attribution_for_beat
from src.evidence.shap_evidence import (
    METHOD_FALLBACK,
    METHOD_SHAP,
    attribute_beats_with_shap,
    shap_available,
    summarise_attributions,
)
from src.evidence.waveform_evidence import WaveformSnippet, extract_waveform_snippet

__all__ = [
    "EvidenceReport",
    "generate_ai_evidence",
    "BeatEvidence",
    "analyze_beats_for_evidence",
    "compute_feature_attribution_for_beat",
    "WaveformSnippet",
    "extract_waveform_snippet",
    "attribute_beats_with_shap",
    "summarise_attributions",
    "shap_available",
    "METHOD_SHAP",
    "METHOD_FALLBACK",
]
