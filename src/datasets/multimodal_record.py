"""
Universal Multimodal ECG Record Representation
===============================================

Phase 7:
Standardized in-memory and serialized record unifying:
- Raw ECG voltage traces & technical parameters
- Deterministic measurements & signal quality
- Clinical demographics, blood group, vital signs, labs
- Structured medications, allergies, symptoms, medical history
- Machine interpretations & diagnostic labels
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class MultimodalECGRecord:
    """Universal record schema fulfilling Phase 7 requirement."""
    patient_id: str
    ecg_id: str
    ecg_timestamp: Optional[str] = None
    ecg_waveform: Optional[Any] = None  # 1D or 2D np.ndarray (leads, samples)
    sampling_rate: float = 360.0
    lead_names: List[str] = field(default_factory=lambda: ["Lead II"])
    
    # Measurements & Quality
    ecg_measurements: Dict[str, Any] = field(default_factory=dict)
    ecg_quality: Dict[str, Any] = field(default_factory=dict)
    
    # Multimodal Clinical Context
    demographics: Dict[str, Any] = field(default_factory=dict)  # age, sex, blood_group, bmi, etc.
    symptoms: List[Dict[str, Any]] = field(default_factory=list)
    vitals: List[Dict[str, Any]] = field(default_factory=list)
    medical_history: List[Dict[str, Any]] = field(default_factory=list)
    medications: List[Dict[str, Any]] = field(default_factory=list)
    allergies: List[Dict[str, Any]] = field(default_factory=list)
    laboratory_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Machine Interpretation & Reference Labels
    machine_interpretation: Dict[str, Any] = field(default_factory=dict)
    clinical_labels: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_waveform: bool = False) -> Dict[str, Any]:
        d = {
            "patient_id": self.patient_id,
            "ecg_id": self.ecg_id,
            "ecg_timestamp": self.ecg_timestamp,
            "sampling_rate": self.sampling_rate,
            "lead_names": self.lead_names,
            "ecg_measurements": self.ecg_measurements,
            "ecg_quality": self.ecg_quality,
            "demographics": self.demographics,
            "symptoms": self.symptoms,
            "vitals": self.vitals,
            "medical_history": self.medical_history,
            "medications": self.medications,
            "allergies": self.allergies,
            "laboratory_results": self.laboratory_results,
            "machine_interpretation": self.machine_interpretation,
            "clinical_labels": self.clinical_labels,
        }
        if include_waveform and self.ecg_waveform is not None:
            if isinstance(self.ecg_waveform, np.ndarray):
                d["ecg_waveform"] = self.ecg_waveform.tolist()
            else:
                d["ecg_waveform"] = self.ecg_waveform
        return d
