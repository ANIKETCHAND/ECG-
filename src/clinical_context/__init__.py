"""
Clinical Context Fusion Engine Package
"""

from src.clinical_context.context_engine import (
    GLOBAL_CONTEXT_FUSION_ENGINE,
    ClinicalContextFusionEngine,
    SourcedStatement,
)

__all__ = [
    "ClinicalContextFusionEngine",
    "GLOBAL_CONTEXT_FUSION_ENGINE",
    "SourcedStatement",
]
