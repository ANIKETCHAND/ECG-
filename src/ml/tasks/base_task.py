"""
Base ML Task Definition Subsystem
=================================
Establishes standardized task boundaries, input constraints, and clinical metric evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    target_classes: List[str]
    is_multilabel: bool
    required_leads: List[str]
    recommended_sampling_rate: float
    compatible_datasets: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "target_classes": self.target_classes,
            "is_multilabel": self.is_multilabel,
            "required_leads": self.required_leads,
            "recommended_sampling_rate": self.recommended_sampling_rate,
            "compatible_datasets": self.compatible_datasets,
        }
