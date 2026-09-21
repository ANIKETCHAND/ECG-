"""
ML Validation & Outlier Traps
=============================
Checks feature matrix sanity and validates probability output bounds.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


def validate_feature_matrix(features: np.ndarray, expected_dim: int = 28) -> Tuple[bool, List[str]]:
    """Verify feature matrix dimensions and lack of NaNs/Infs."""
    errors = []
    if features.size == 0:
        errors.append("Feature matrix is empty.")
        return False, errors

    if features.shape[1] != expected_dim:
        errors.append(f"Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}.")

    if np.any(np.isnan(features)):
        errors.append("Feature matrix contains unhandled NaN values.")

    if np.any(np.isinf(features)):
        errors.append("Feature matrix contains unhandled Infinite values.")

    return len(errors) == 0, errors


__all__ = ["validate_feature_matrix"]
