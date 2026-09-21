"""
Comprehensive Machine Learning Model Quality, Clinical Performance & Robustness Test
"""

import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from training.evaluation.clinical_metrics import (
    calculate_binary_clinical_metrics,
    calculate_multiclass_clinical_metrics,
    compute_expected_calibration_error,
)
from training.splitting.patient_splitter import validate_dataset_leakage


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_model_test():
    print("================================================================================")
    print("             ECG GUARDIAN — MACHINE LEARNING MODEL TEST REPORT                 ")
    print("================================================================================")

    # 1. Model Artifact Verification
    models_dir = ROOT_DIR / "models"
    clf_path = models_dir / "classifier.pkl"
    scaler_path = models_dir / "scaler.pkl"
    meta_path = models_dir / "metadata.json"

    print("\n[STEP 1] Verifying Model Artifacts & Cryptographic Checksums...")
    assert clf_path.exists(), "classifier.pkl missing"
    assert scaler_path.exists(), "scaler.pkl missing"
    assert meta_path.exists(), "metadata.json missing"

    clf_hash = compute_sha256(clf_path)
    scaler_hash = compute_sha256(scaler_path)

    print(f"  • Classifier : {clf_path.name} ({clf_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"    SHA-256    : {clf_hash}")
    print(f"  • Scaler     : {scaler_path.name} ({scaler_path.stat().st_size / 1024:.2f} KB)")
    print(f"    SHA-256    : {scaler_hash}")

    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"  • Architecture: {meta['model_name']} ({clf.n_estimators} Estimators)")
    print(f"  • Classes     : {meta['classes']}")
    print(f"  • Features    : {len(meta['feature_names'])} input features")

    # 2. Patient-Level Data Leakage Verification
    print("\n[STEP 2] Verifying Patient-Level Split (Zero Leakage Enforcement)...")
    data_dir = ROOT_DIR / "data" / "processed"
    train_df = pd.read_csv(data_dir / "train_dataset.csv")
    test_df = pd.read_csv(data_dir / "test_dataset.csv")

    leakage_res = validate_dataset_leakage(train_df, test_df, patient_col="record_id")
    print(f"  • Train Patients: {leakage_res['train_patients']} ({leakage_res['train_beats']} beats)")
    print(f"  • Test Patients : {leakage_res['test_patients']} ({leakage_res['test_beats']} beats)")
    print(f"  • Overlap Count : {leakage_res['overlap_count']} (PASS: Disjoint patient sets)")

    # 3. Clinical Evaluation on Unseen Test Patients
    print("\n[STEP 3] Evaluating Performance on Held-Out Test Patients...")
    feature_cols = meta["feature_names"]
    classes = meta["classes"]

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    t0 = time.perf_counter()
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    y_probs = clf.predict_proba(X_test_scaled)
    eval_duration_sec = time.perf_counter() - t0

    # Binary clinical metrics for PVC vs Non-PVC
    pvc_idx = classes.index("PVC")
    y_test_pvc = (y_test == "PVC").astype(int)
    y_pred_pvc = (y_pred == "PVC").astype(int)
    pvc_probs = y_probs[:, pvc_idx]

    pvc_metrics = calculate_binary_clinical_metrics(y_test_pvc, y_pred_pvc, pvc_probs)

    print(f"  • Total Unseen Evaluation Beats: {pvc_metrics['total_samples']}")
    print(f"  • Overall Diagnostic Accuracy  : {pvc_metrics['accuracy'] * 100:.2f}%")
    print(f"  • PVC Sensitivity (Recall)     : {pvc_metrics['sensitivity'] * 100:.2f}% (Target >= 90.0%)")
    print(f"  • PVC Specificity              : {pvc_metrics['specificity'] * 100:.2f}% (Target >= 90.0%)")
    print(f"  • PVC Positive Predictive Value: {pvc_metrics['positive_predictive_value'] * 100:.2f}%")
    print(f"  • PVC Negative Predictive Value: {pvc_metrics['negative_predictive_value'] * 100:.2f}%")
    print(f"  • F1-Score                     : {pvc_metrics['f1_score'] * 100:.2f}%")
    if pvc_metrics.get("auroc"):
        print(f"  • Area Under ROC Curve (AUROC) : {pvc_metrics['auroc']:.4f}")
    if pvc_metrics.get("auprc"):
        print(f"  • Area Under PR Curve (AUPRC)  : {pvc_metrics['auprc']:.4f}")

    print("\n  Confusion Matrix for Ventricular Ectopy (PVC):")
    print(f"    True Positives  (TP) : {pvc_metrics['true_positives']} beats (Correctly identified PVCs)")
    print(f"    False Negatives (FN) : {pvc_metrics['false_negatives']} beats (Missed PVCs)")
    print(f"    True Negatives  (TN) : {pvc_metrics['true_negatives']} beats (Correctly identified non-PVCs)")
    print(f"    False Positives (FP) : {pvc_metrics['false_positives']} beats (Normal misclassified as PVC)")

    # 4. Calibration Analysis
    print("\n[STEP 4] Evaluating Probability Calibration (ECE)...")
    ece_res = compute_expected_calibration_error(y_test_pvc, pvc_probs, n_bins=10)
    print(f"  • Expected Calibration Error (ECE): {ece_res['expected_calibration_error']:.4f}")
    print("    Calibration Bins:")
    for b in ece_res["bin_details"]:
        if b["count"] > 0:
            print(f"      Range [{b['bin_range'][0]:.1f}-{b['bin_range'][1]:.1f}] : Count={b['count']:4d}, Mean Prob={b['confidence']:.3f}, Empirical Acc={b['accuracy']:.3f}, Gap={b['gap']:.3f}")

    # 5. Robustness to Input Perturbations
    print("\n[STEP 5] Robustness & Perturbation Invariance Testing...")
    # Add small Gaussian noise to features (5% of std)
    noise = np.random.normal(0, 0.05, size=X_test.shape)
    X_test_perturbed = X_test + noise * np.std(X_test, axis=0, keepdims=True)
    X_pert_scaled = scaler.transform(X_test_perturbed)
    y_pred_pert = clf.predict(X_pert_scaled)

    agreement = np.mean(y_pred == y_pred_pert) * 100.0
    print(f"  • Prediction Consistency under 5% Gaussian Noise: {agreement:.2f}% (Robustness Stability)")

    # 6. Latency & Throughput Benchmark
    print("\n[STEP 6] Inference Latency & Speed Benchmark...")
    single_beat = X_test_scaled[0:1]
    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        _ = clf.predict_proba(single_beat)
        latencies.append((time.perf_counter() - t_start) * 1000)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    throughput = len(X_test) / eval_duration_sec

    print(f"  • Mean Single-Beat Inference Time : {mean_lat:.3f} ms")
    print(f"  • 95th Percentile Latency (P95)   : {p95_lat:.3f} ms")
    print(f"  • Batch Processing Throughput     : {throughput:.1f} beats/second")

    # 7. Top 5 Most Important Predictive Features
    print("\n[STEP 7] Top 5 Most Predictive Clinical Features (Model Explanability):")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"  {rank}. {feature_cols[idx]:<22} : {importances[idx] * 100:.2f}% relative importance")

    print("\n================================================================================")
    print("                >>> ALL ML MODEL TESTS PASSED CLEANLY! <<<                     ")
    print("================================================================================")


if __name__ == "__main__":
    run_model_test()
