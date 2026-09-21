"""
Patient-Level Splitting Subsystem
=================================
Prevents patient data leakage across training, validation, and testing partitions.
"""

from training.splitting.patient_splitter import (
    split_records_by_patient,
    validate_dataset_leakage,
    verify_patient_isolation,
)

__all__ = [
    "verify_patient_isolation",
    "split_records_by_patient",
    "validate_dataset_leakage",
]
