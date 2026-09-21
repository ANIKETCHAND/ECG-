"""
Clinical Evaluation Subsystem
=============================
Computes medical device validation metrics: Sensitivity, Specificity, PPV, NPV,
AUROC, AUPRC, Expected Calibration Error (ECE), and Confusion Matrices.
"""

from training.evaluation.clinical_metrics import (
    calculate_binary_clinical_metrics,
    calculate_multiclass_clinical_metrics,
    compute_expected_calibration_error,
    evaluate_diagnostic_performance,
)

__all__ = [
    "calculate_binary_clinical_metrics",
    "calculate_multiclass_clinical_metrics",
    "compute_expected_calibration_error",
    "evaluate_diagnostic_performance",
]
