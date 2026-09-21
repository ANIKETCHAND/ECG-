"""
Universal Standardized ECG Record Representation
================================================
Defines the canonical in-memory and serialized data structure for any ECG waveform,
preserving multi-lead signals, expert annotations, multi-label diagnostic statements,
calibrations, and source provenance across all ingested datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class StandardizedECGRecord:
    record_id: str
    patient_id: str
    dataset_id: str
    sampling_rate: float
    duration: float
    lead_names: List[str]
    number_of_leads: int
    signals: np.ndarray  # Shape (n_leads, n_samples)
    units: str = "mV"
    acquisition_time: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)  # sample_indices, symbols, comments
    diagnostic_labels: List[str] = field(default_factory=list)  # canonical task labels
    patient_metadata: Dict[str, Any] = field(default_factory=dict)  # age, sex, clinical history
    device_metadata: Dict[str, Any] = field(default_factory=dict)  # manufacturer, model, filters
    quality_metadata: Dict[str, Any] = field(default_factory=dict)  # snr, clipping, baseline wander
    source_version: str = "1.0.0"
    checksum_sha256: str = ""

    def __post_init__(self):
        # Enforce 2D numpy array
        if not isinstance(self.signals, np.ndarray):
            self.signals = np.asarray(self.signals, dtype=float)
        if self.signals.ndim == 1:
            self.signals = self.signals.reshape(1, -1)

        self.number_of_leads = self.signals.shape[0]
        n_samples = self.signals.shape[1]
        if self.sampling_rate > 0 and n_samples > 0:
            self.duration = round(float(n_samples / self.sampling_rate), 4)

        if not self.checksum_sha256 and self.signals.size > 0:
            self.checksum_sha256 = hashlib.sha256(self.signals.tobytes()).hexdigest()

    def get_lead(self, lead_name: str) -> Optional[np.ndarray]:
        """Retrieve 1D waveform array for a specific lead."""
        upper_target = lead_name.upper().strip()
        for idx, name in enumerate(self.lead_names):
            if name.upper().strip() == upper_target:
                return self.signals[idx]
        return None

    def to_dict(self, include_signals: bool = False) -> Dict[str, Any]:
        """Serialize record metadata to dict."""
        return {
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "dataset_id": self.dataset_id,
            "sampling_rate": self.sampling_rate,
            "duration": self.duration,
            "lead_names": self.lead_names,
            "number_of_leads": self.number_of_leads,
            "units": self.units,
            "acquisition_time": self.acquisition_time,
            "annotations_count": len(self.annotations.get("sample_indices", [])),
            "diagnostic_labels": self.diagnostic_labels,
            "patient_metadata": self.patient_metadata,
            "device_metadata": self.device_metadata,
            "quality_metadata": self.quality_metadata,
            "source_version": self.source_version,
            "checksum_sha256": self.checksum_sha256,
            "signals": self.signals.tolist() if include_signals else None,
        }
