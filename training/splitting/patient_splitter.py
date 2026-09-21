"""
Patient-Level Splitting Module
==============================
Strictly prevents cardiac data leakage between training, tuning, and evaluation sets.
In medical machine learning, beats from the same patient must NEVER be randomly partitioned
into both train and test splits, as this causes catastrophic overestimation of performance.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Set, Tuple
import pandas as pd


class DataLeakageError(ValueError):
    """Raised when patient records or identities overlap between dataset splits."""
    pass


def verify_patient_isolation(
    train_records: Sequence[str],
    test_records: Sequence[str],
    val_records: Sequence[str] | None = None,
) -> bool:
    """Verify zero overlap between patient/record identifiers across partitions.

    Args:
        train_records: List of record/patient identifiers in training split.
        test_records: List of record/patient identifiers in testing split.
        val_records: Optional list of record/patient identifiers in validation split.

    Returns:
        True if all partitions are completely disjoint.

    Raises:
        DataLeakageError: If any identifier is shared across partitions.
    """
    s_train: Set[str] = set(str(r).strip() for r in train_records)
    s_test: Set[str] = set(str(r).strip() for r in test_records)
    s_val: Set[str] = set(str(r).strip() for r in (val_records or []))

    train_test_overlap = s_train.intersection(s_test)
    if train_test_overlap:
        raise DataLeakageError(
            f"FATAL LEAKAGE DETECTED: Patients {sorted(list(train_test_overlap))} present in both train and test sets!"
        )

    if s_val:
        train_val_overlap = s_train.intersection(s_val)
        if train_val_overlap:
            raise DataLeakageError(
                f"FATAL LEAKAGE DETECTED: Patients {sorted(list(train_val_overlap))} present in both train and val sets!"
            )
        val_test_overlap = s_val.intersection(s_test)
        if val_test_overlap:
            raise DataLeakageError(
                f"FATAL LEAKAGE DETECTED: Patients {sorted(list(val_test_overlap))} present in both val and test sets!"
            )

    return True


def split_records_by_patient(
    records: Sequence[str],
    test_ratio: float = 0.3,
    val_ratio: float = 0.0,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Partition records at the patient/recording level.

    Args:
        records: List of unique record identifiers.
        test_ratio: Fraction of records to allocate to test set.
        val_ratio: Fraction of records to allocate to validation set.
        random_seed: Random state for deterministic reproducibility.

    Returns:
        Tuple of (train_records, val_records, test_records)
    """
    unique_records = sorted(list(set(str(r).strip() for r in records)))
    if not unique_records:
        raise ValueError("Cannot split empty record list.")

    rng = random.Random(random_seed)
    shuffled = list(unique_records)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n_total * val_ratio))) if val_ratio > 0 else 0

    if n_test + n_val >= n_total:
        raise ValueError(
            f"Requested split ratios (test={test_ratio}, val={val_ratio}) leave 0 records for training."
        )

    test_recs = sorted(shuffled[:n_test])
    val_recs = sorted(shuffled[n_test : n_test + n_val])
    train_recs = sorted(shuffled[n_test + n_val :])

    verify_patient_isolation(train_recs, test_recs, val_recs)
    return train_recs, val_recs, test_recs


def validate_dataset_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_col: str = "record_id",
) -> Dict[str, Any]:
    """Validate that two DataFrames share zero patients.

    Args:
        train_df: Training DataFrame containing beat-level samples.
        test_df: Testing DataFrame containing beat-level samples.
        patient_col: Column name identifying the patient or recording ID.

    Returns:
        Dictionary summarizing isolation verification.
    """
    if patient_col not in train_df.columns:
        raise KeyError(f"Column '{patient_col}' not found in training DataFrame.")
    if patient_col not in test_df.columns:
        raise KeyError(f"Column '{patient_col}' not found in testing DataFrame.")

    train_pts = set(train_df[patient_col].dropna().astype(str).unique())
    test_pts = set(test_df[patient_col].dropna().astype(str).unique())

    overlap = train_pts.intersection(test_pts)
    if overlap:
        raise DataLeakageError(
            f"Patient leakage detected between train and test datasets: {sorted(list(overlap))}"
        )

    return {
        "status": "PASS",
        "train_patients": sorted(list(train_pts)),
        "test_patients": sorted(list(test_pts)),
        "train_beats": len(train_df),
        "test_beats": len(test_df),
        "overlap_count": 0,
    }
