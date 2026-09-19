"""
Unified ECG Data Models
=======================

Core domain entities for clinical electrocardiography:
- ECGRecording: Standardized internal object for any digital ECG recording
- ECGAnalysisResult: Structured, traceable analytical output from ML & signal engines
- ClinicianReview: Physician sign-off, clinical interpretation, and override record

References:
- Medical Devices Rules (MDR) 2017 (CDSCO)
- IEC 62304 Software Data Structures
- ISO 14971 Risk Management
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class ECGRecording:
    """Unified, standardized clinical ECG recording entity.

    Guarantees consistent representation across diverse acquisition devices,
    multi-lead geometries, and digital formats.
    """
    record_id: str
    sampling_rate: float
    duration: float
    lead_names: List[str]
    number_of_leads: int
    signals: np.ndarray  # Shape: (number_of_leads, n_samples)
    units: str = "mV"
    patient_id: Optional[str] = None
    acquisition_time: Optional[str] = None
    device: Optional[str] = None
    manufacturer: Optional[str] = None
    source_format: str = "UNKNOWN"
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[Dict[str, Any]] = None
    data_hash: str = ""

    def __post_init__(self):
        # Ensure signals is 2D numpy array: (leads, samples)
        if isinstance(self.signals, list):
            self.signals = np.array(self.signals, dtype=float)

        if self.signals.ndim == 1:
            self.signals = np.expand_dims(self.signals, axis=0)

        # Reconcile lead count
        actual_leads, actual_samples = self.signals.shape
        self.number_of_leads = actual_leads

        if not self.lead_names or len(self.lead_names) != actual_leads:
            if actual_leads == 1:
                self.lead_names = ["II"]
            elif actual_leads == 12:
                self.lead_names = list(STANDARD_12_LEADS)
            else:
                self.lead_names = [f"Lead_{i+1}" for i in range(actual_leads)]

        # Reconcile duration
        if self.sampling_rate > 0:
            self.duration = round(float(actual_samples) / float(self.sampling_rate), 4)

        # Compute SHA-256 data hash for tamper-evident traceability
        if not self.data_hash:
            self.data_hash = self._compute_data_hash()

    def _compute_data_hash(self) -> str:
        """Compute SHA-256 hash of the physiological signal matrix."""
        hasher = hashlib.sha256()
        hasher.update(self.signals.tobytes())
        hasher.update(f"{self.sampling_rate}".encode("utf-8"))
        hasher.update(",".join(self.lead_names).encode("utf-8"))
        return hasher.hexdigest()

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate clinical and technical integrity of the recording."""
        errors = []
        if self.sampling_rate <= 0:
            errors.append(f"Invalid sampling rate ({self.sampling_rate} Hz). Must be > 0.")
        if self.signals.size == 0:
            errors.append("Signal array is empty.")
        if self.number_of_leads != len(self.lead_names):
            errors.append(f"Lead count mismatch: {self.number_of_leads} leads vs {len(self.lead_names)} names.")
        if np.any(np.isinf(self.signals)):
            errors.append("Signal array contains infinite non-physiological values.")
        if self.duration < 0.5:
            errors.append(f"Recording duration ({self.duration:.2f}s) is too short for cardiac evaluation.")
        return len(errors) == 0, errors

    def get_lead(self, lead_name_or_index: Union[str, int]) -> Optional[np.ndarray]:
        """Safely retrieve signal series for a specific lead."""
        if isinstance(lead_name_or_index, int):
            if 0 <= lead_name_or_index < self.number_of_leads:
                return self.signals[lead_name_or_index]
            return None

        # Search by lead name (case-insensitive)
        target = lead_name_or_index.strip().upper()
        # Aliases mapping
        aliases = {"LEAD II": "II", "LEAD 2": "II", "MLII": "II", "LEAD I": "I", "LEAD 1": "I"}
        target = aliases.get(target, target)

        for idx, name in enumerate(self.lead_names):
            norm_name = aliases.get(name.strip().upper(), name.strip().upper())
            if norm_name == target:
                return self.signals[idx]
        return None

    def has_lead(self, lead_name: str) -> bool:
        """Check if recording contains a specific lead."""
        return self.get_lead(lead_name) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary (excluding large raw signal array)."""
        return {
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "sampling_rate": self.sampling_rate,
            "duration_sec": self.duration,
            "lead_names": self.lead_names,
            "number_of_leads": self.number_of_leads,
            "units": self.units,
            "acquisition_time": self.acquisition_time,
            "device": self.device,
            "manufacturer": self.manufacturer,
            "source_format": self.source_format,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
        }


@dataclass
class ECGAnalysisResult:
    """Standardized, traceable output from signal processing and AI inference."""
    analysis_id: str
    record_id: str
    model_id: str
    model_version: str
    preprocessing_version: str
    lead_analyzed: str
    signal_quality: str  # GOOD, ACCEPTABLE, POOR, UNUSABLE
    quality_score: float
    quality_indicators: Dict[str, Any]
    prediction: str
    model_probabilities: Dict[str, float]
    heart_rate_bpm: Optional[float]
    mean_rr_ms: Optional[float]
    detected_beats_count: int
    detected_r_peaks: List[int]
    rr_intervals_ms: List[float] = field(default_factory=list)
    beat_predictions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    patient_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    data_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis output to JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ClinicianReview:
    """Mandatory clinician review and diagnostic sign-off entity."""
    review_id: str
    analysis_id: str
    record_id: str
    clinician_id: str
    clinician_name: str
    clinician_role: str  # DOCTOR, CARDIOLOGIST, MEDICAL_OFFICER
    agreement_status: str  # CONFIRMED, MODIFIED, REJECTED
    clinician_interpretation: str
    clinical_notes: str = ""
    reviewed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    registration_number: Optional[str] = None  # Medical Council Registration No.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
