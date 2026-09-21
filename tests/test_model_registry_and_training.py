"""
Unit Tests for Phase 5: Model Registry & Training Infrastructure
================================================================
Validates:
1. Patient-level splitting strictly rejects data leakage.
2. Clinical metrics: Sensitivity, Specificity, PPV, NPV, F1, ECE.
3. Model registry artifacts and MODEL_CARD metadata consistency.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.splitting.patient_splitter import (
    DataLeakageError,
    split_records_by_patient,
    validate_dataset_leakage,
    verify_patient_isolation,
)
from training.evaluation.clinical_metrics import (
    calculate_binary_clinical_metrics,
    calculate_multiclass_clinical_metrics,
    compute_expected_calibration_error,
)
from src.ml.models import load_model_artifacts


def test_patient_isolation_valid():
    train_recs = ["100", "106", "200", "213"]
    test_recs = ["101", "119", "208"]
    assert verify_patient_isolation(train_recs, test_recs) is True


def test_patient_isolation_detects_leakage():
    train_recs = ["100", "106", "200"]
    test_recs = ["101", "200", "208"]  # '200' is in both!
    with pytest.raises(DataLeakageError, match="FATAL LEAKAGE DETECTED"):
        verify_patient_isolation(train_recs, test_recs)


def test_split_records_by_patient():
    recs = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109"]
    train_r, val_r, test_r = split_records_by_patient(recs, test_ratio=0.3, val_ratio=0.1, random_seed=42)
    assert len(test_r) == 3
    assert len(val_r) == 1
    assert len(train_r) == 6
    assert set(train_r).isdisjoint(set(test_r))
    assert set(train_r).isdisjoint(set(val_r))
    assert set(val_r).isdisjoint(set(test_r))


def test_validate_dataset_leakage():
    train_df = pd.DataFrame({"record_id": ["100", "100", "106", "200"], "feature": [1, 2, 3, 4]})
    test_df = pd.DataFrame({"record_id": ["101", "119", "208"], "feature": [5, 6, 7]})
    res = validate_dataset_leakage(train_df, test_df)
    assert res["status"] == "PASS"
    assert res["overlap_count"] == 0

    leaky_df = pd.DataFrame({"record_id": ["100", "119"], "feature": [8, 9]})
    with pytest.raises(DataLeakageError, match="Patient leakage detected"):
        validate_dataset_leakage(train_df, leaky_df)


def test_binary_clinical_metrics():
    # True: 10 positives, 10 negatives
    y_true = [1] * 10 + [0] * 10
    # Pred: 8 TP, 2 FN, 1 FP, 9 TN
    y_pred = [1] * 8 + [0] * 2 + [1] * 1 + [0] * 9
    metrics = calculate_binary_clinical_metrics(y_true, y_pred)

    assert metrics["true_positives"] == 8
    assert metrics["false_negatives"] == 2
    assert metrics["true_negatives"] == 9
    assert metrics["false_positives"] == 1
    assert metrics["sensitivity"] == 0.8
    assert metrics["specificity"] == 0.9
    assert metrics["positive_predictive_value"] == round(8 / 9, 4)
    assert metrics["negative_predictive_value"] == round(9 / 11, 4)


def test_expected_calibration_error():
    y_true = [1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.85, 0.8, 0.1, 0.2, 0.15]
    ece_res = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert "expected_calibration_error" in ece_res
    assert ece_res["expected_calibration_error"] >= 0.0


def test_multiclass_clinical_metrics():
    classes = ["Normal", "PVC", "Other"]
    y_true = ["Normal", "Normal", "PVC", "PVC", "Other"]
    y_pred = ["Normal", "PVC", "PVC", "Normal", "Other"]
    res = calculate_multiclass_clinical_metrics(y_true, y_pred, classes=classes)
    assert res["overall_accuracy"] == 0.6
    assert "per_class_metrics" in res
    assert "Normal" in res["per_class_metrics"]
    assert "PVC" in res["per_class_metrics"]


def test_model_registry_artifacts():
    clf, scaler, meta = load_model_artifacts()
    assert clf is not None
    assert scaler is not None
    assert meta["model_name"] == "RandomForestClassifier"
    assert len(meta["feature_names"]) == 28
    assert meta["classes"] == ["Normal", "Other", "PVC"]
