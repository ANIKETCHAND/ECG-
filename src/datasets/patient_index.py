"""
Patient Index & Identity Mapping Subsystem
==========================================
Maintains unambiguous mappings between individual recording IDs and true underlying
patient identities across all supported databases.
Strictly prevents multi-record leakage where multiple recordings from the same patient
are inadvertently partitioned into both training and testing sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
import pandas as pd


# Known multi-record patient groupings for MIT-BIH databases
MIT_BIH_PATIENT_GROUPS: Dict[str, str] = {
    # In MIT-BIH Arrhythmia Database, Records 201 and 202 originate from the same subject
    "201": "PT_MIT_201_202",
    "202": "PT_MIT_201_202",
}


def resolve_patient_id(
    dataset_id: str,
    record_id: str,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve the true patient identifier for a given recording.

    Args:
        dataset_id: Identifier of the source dataset.
        record_id: Identifier of the specific recording.
        metadata: Optional metadata dictionary (e.g., from PTB-XL CSV).

    Returns:
        Canonical string patient identifier.
    """
    rec_clean = str(record_id).strip()
    d_clean = dataset_id.lower().strip()

    if metadata and "patient_id" in metadata:
        return f"{d_clean}_pt_{metadata['patient_id']}"

    if d_clean == "mit_bih_arrhythmia" and rec_clean in MIT_BIH_PATIENT_GROUPS:
        return MIT_BIH_PATIENT_GROUPS[rec_clean]

    # In PTB Diagnostic, record paths follow patientXXX/sYYYY_re
    if d_clean == "ptbdb" and "patient" in rec_clean:
        parts = rec_clean.split("/")
        return parts[0]

    return f"{d_clean}_pt_{rec_clean}"


class PatientIndex:
    """Registry maintaining record-to-patient associations."""

    def __init__(self) -> None:
        self._record_to_patient: Dict[str, str] = {}
        self._patient_to_records: Dict[str, List[str]] = {}

    def register_record(self, dataset_id: str, record_id: str, metadata: Optional[Dict[str, str]] = None) -> str:
        key = f"{dataset_id}:{record_id}"
        patient_id = resolve_patient_id(dataset_id, record_id, metadata)
        self._record_to_patient[key] = patient_id

        if patient_id not in self._patient_to_records:
            self._patient_to_records[patient_id] = []
        if key not in self._patient_to_records[patient_id]:
            self._patient_to_records[patient_id].append(key)

        return patient_id

    def get_patient_id(self, dataset_id: str, record_id: str) -> str:
        key = f"{dataset_id}:{record_id}"
        return self._record_to_patient.get(key, resolve_patient_id(dataset_id, record_id))

    def get_records_for_patient(self, patient_id: str) -> List[str]:
        return self._patient_to_records.get(patient_id, [])

    def verify_no_patient_cross_contamination(
        self,
        train_record_keys: List[str],
        test_record_keys: List[str],
    ) -> bool:
        """Verify zero patient overlap between train and test record keys."""
        train_pts = {self._record_to_patient.get(k) for k in train_record_keys if k in self._record_to_patient}
        test_pts = {self._record_to_patient.get(k) for k in test_record_keys if k in self._record_to_patient}

        overlap = train_pts.intersection(test_pts)
        if overlap and None not in overlap:
            raise ValueError(f"Cross-contamination detected! Patients {overlap} present in both train and test splits.")
        return True


GLOBAL_PATIENT_INDEX = PatientIndex()
