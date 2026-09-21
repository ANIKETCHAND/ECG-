"""
ECG Machine vs. AI Verification Subsystem
=========================================
Automated concordance evaluation, discrepancy detection, and clinical alerts.
"""

from src.comparison.ai_machine_comparison import compare_ai_and_machine
from src.comparison.disagreement_detector import ComparisonResult, ComparisonStatus
from src.comparison.machine_interpretation import (
    ParsedMachineInterpretation,
    parse_machine_interpretation,
)

__all__ = [
    "ParsedMachineInterpretation",
    "parse_machine_interpretation",
    "ComparisonStatus",
    "ComparisonResult",
    "compare_ai_and_machine",
]
