"""
Patient Longitudinal ECG History Store
======================================
Stores and retrieves historical ECG analyses for a patient to enable serial comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.longitudinal.ecg_comparison import (
    LongitudinalComparisonResult,
    compare_serial_ecgs,
)
from src.longitudinal.trend_analysis import (
    LongitudinalTrendReport,
    calculate_serial_trends,
)


class PatientECGHistoryStore:
    """In-memory and persistent manager for patient ECG time-series history."""

    def __init__(self) -> None:
        self._records_by_patient: Dict[str, List[Dict[str, Any]]] = {}

    def add_analysis(self, patient_id: str, analysis: Dict[str, Any]) -> None:
        """Add an analysis to the patient history in chronological sequence."""
        pid = str(patient_id).strip()
        if pid not in self._records_by_patient:
            self._records_by_patient[pid] = []
        self._records_by_patient[pid].append(analysis)

    def get_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieve all recorded analyses for a patient."""
        return self._records_by_patient.get(str(patient_id).strip(), [])

    def get_latest_comparison(self, patient_id: str) -> Optional[LongitudinalComparisonResult]:
        """Compare the most recent analysis with its immediately preceding baseline."""
        history = self.get_history(patient_id)
        if len(history) < 2:
            return None
        current = history[-1]
        prior = history[-2]
        return compare_serial_ecgs(current, prior, patient_id=patient_id)

    def get_trends(self, patient_id: str) -> LongitudinalTrendReport:
        """Calculate longitudinal trends across all recordings for a patient."""
        history = self.get_history(patient_id)
        return calculate_serial_trends(history, patient_id=patient_id)
