"""
Dataset Manifest Module
=======================
Maintains metadata, patient demographic descriptors, and data provenance
for training and clinical evaluation databases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PatientRecord:
    record_id: str
    sampling_rate: float
    duration_seconds: float
    leads: List[str]
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    known_diagnoses: List[str] = field(default_factory=list)
    database_source: str = "MIT-BIH Arrhythmia Database"


@dataclass
class DatasetManifest:
    manifest_id: str
    created_at: str
    database_name: str
    total_records: int
    train_records: List[str]
    test_records: List[str]
    val_records: List[str] = field(default_factory=list)
    records: Dict[str, PatientRecord] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
