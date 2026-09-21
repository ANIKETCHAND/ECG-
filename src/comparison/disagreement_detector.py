"""
Disagreement Detection Subsystem
================================
Detects diagnostic concordances and discrepancies between machine statements and AI predictions.
Mandates physician alerting when disagreement is discovered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ComparisonStatus(str, Enum):
    AGREE = "AGREE"
    MINOR_DIFFERENCE = "MINOR_DIFFERENCE"
    SIGNIFICANT_DISAGREEMENT = "SIGNIFICANT_DISAGREEMENT"
    UNABLE_TO_COMPARE = "UNABLE_TO_COMPARE"


@dataclass
class ComparisonResult:
    status: ComparisonStatus
    summary: str
    machine_category: str
    ai_prediction: str
    discrepancy_details: List[str] = field(default_factory=list)
    clinical_advisory: str = ""
    coverage_limitation_warning: Optional[str] = None
    rate_difference_bpm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data
