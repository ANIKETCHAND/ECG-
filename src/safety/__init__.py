"""
Safety & Risk Mitigation Package
================================

Provides safety gatekeepers and risk mitigation barriers:
- evaluate_signal_quality_gate
- QualityCategory
- QualityGateResult
"""

try:
    from .signal_quality_gate import (
        QualityCategory,
        QualityGateResult,
        evaluate_signal_quality_gate,
    )
except ImportError:
    from safety.signal_quality_gate import (
        QualityCategory,
        QualityGateResult,
        evaluate_signal_quality_gate,
    )

__all__ = [
    "QualityCategory",
    "QualityGateResult",
    "evaluate_signal_quality_gate",
]
