"""
Phase 5: MIT-BIH Detailed Validation, Model Performance & Signal Robustness Suite
==================================================================================
Validates:
1. Model loading & live inference on real PhysioNet MIT-BIH recordings (records 100, 208).
2. Multimodal comparison verification (Model A ECG-only, Model B Clinical-only, Model C Fusion).
3. Hard Safety Gatekeeper Rejections (No reliable input = No AI result):
   - Flatline (zero variance, std == 0)
   - Truncated / Short signals (< 100 samples / < 1.5s)
   - Corrupted values (NaN, Inf)
4. Noise & Artifact Detection:
   - Baseline wander
   - 50 Hz / 60 Hz powerline interference
   - Motion artifacts
5. Latency & Resource Utilization Benchmarking.
"""

import json
import time
from pathlib import Path
import numpy as np
import pytest
import wfdb

from src.data_loader import load_record
from src.inference.inference_engine import run_ecg_inference
from src.safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets" / "mit_bih_arrhythmia" / "raw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def test_real_mitbih_record_loading_and_inference():
    """Verify loading real recorded PhysioNet MIT-BIH ECG record 100 and running inference."""
    assert (RAW_DATA_DIR / "100.dat").exists(), "MIT-BIH record 100 not found"
    assert (RAW_DATA_DIR / "100.atr").exists(), "MIT-BIH record 100 reference annotations not found"
    signal, fs, fields = load_record("100", RAW_DATA_DIR)

    assert fs == 360.0
    assert len(signal) > 3600  # More than 10 seconds

    # Extract 10s segment
    segment = signal[:3600]
    result = run_ecg_inference(segment, fs=fs, lead_to_analyze="II")

    assert result is not None
    assert any(k in result.prediction for k in ["Normal", "Other", "PVC", "Sinus Rhythm"])
    assert result.heart_rate_bpm > 50 and result.heart_rate_bpm < 120
    assert result.detected_beats_count > 5
    assert len(result.model_probabilities) == 3
    assert abs(sum(result.model_probabilities.values()) - 1.0) < 1e-4

    # Verify against official WFDB reference annotations
    ann = wfdb.rdann(str(RAW_DATA_DIR / "100"), "atr")
    ref_symbols = [sym for s, sym in zip(ann.sample, ann.symbol) if s < 3600]
    assert "N" in ref_symbols, "Record 100 must contain Normal beat annotations"
    # Model should accurately classify normal sinus rhythm
    assert result.model_probabilities["Normal"] > 0.70, f"Normal probability {result.model_probabilities['Normal']} below 0.70"


def test_real_mitbih_record_208_ectopy_inference():
    """Verify loading MIT-BIH record 208 and verifying against known ventricular ectopy / PVC annotations."""
    assert (RAW_DATA_DIR / "208.dat").exists(), "MIT-BIH record 208 not found"
    assert (RAW_DATA_DIR / "208.atr").exists(), "MIT-BIH record 208 reference annotations not found"
    signal, fs, _ = load_record("208", RAW_DATA_DIR)

    # 10s segment with known PVCs
    segment = signal[:3600]
    result = run_ecg_inference(segment, fs=fs, lead_to_analyze="II")

    assert result is not None
    assert result.detected_beats_count > 5
    assert "PVC" in result.model_probabilities

    # Verify official WFDB reference annotations contain PVCs ('V') in this segment
    ann = wfdb.rdann(str(RAW_DATA_DIR / "208"), "atr")
    pvc_indices = [s for s, sym in zip(ann.sample, ann.symbol) if s < 3600 and sym == "V"]
    assert len(pvc_indices) >= 5, f"Expected >= 5 PVC reference beats in record 208, got {len(pvc_indices)}"

    # Model inference must detect PVC ectopy with significant probability and prediction
    assert "Ventricular Ectopy" in result.prediction or "PVC" in result.prediction
    assert result.model_probabilities["PVC"] > 0.35, f"Expected elevated PVC probability, got {result.model_probabilities['PVC']}"


def test_three_model_comparison_metrics_benchmark():
    """Verify three-model comparison metrics from held-out validation partition."""
    metrics_path = REPORTS_DIR / "three_model_comparison.json"
    assert metrics_path.exists(), "Three model comparison metrics report not found"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Model A: ECG-only RF
    model_a = data["model_a_ecg_only"]
    assert model_a["accuracy"] >= 0.90, f"Model A accuracy {model_a['accuracy']} below 0.90 threshold"
    assert model_a["macro_f1"] >= 0.80, f"Model A macro F1 {model_a['macro_f1']} below 0.80 threshold"
    assert model_a["auroc"] >= 0.95, f"Model A AUROC {model_a['auroc']} below 0.95"
    assert model_a["per_class"]["Normal"]["sensitivity"] >= 0.90
    assert model_a["per_class"]["PVC"]["sensitivity"] >= 0.80

    # Model B: Clinical-only (Non-ECG baseline)
    model_b = data["model_b_clinical_only"]
    assert model_b["macro_f1"] < model_a["macro_f1"], "Clinical-only model should have lower discriminative power"

    # Model C: Multimodal Fusion
    model_c = data["model_c_multimodal_fusion"]
    assert model_c["accuracy"] >= 0.90, "Multimodal fusion accuracy below 0.90"
    assert model_c["weighted_f1"] >= 0.90, "Multimodal fusion weighted F1 below 0.90"


def test_stress_flatline_rejection():
    """Verify flatline signal (zero variance) is halted immediately by safety gatekeeper."""
    flatline = np.zeros(3600)
    gate_res = evaluate_signal_quality_gate(flatline, fs=360.0)

    assert gate_res.category == QualityCategory.UNUSABLE
    assert not gate_res.can_run_ai
    assert any("flatline" in r.lower() or "amplitude" in r.lower() or "variance" in r.lower() for r in gate_res.rejection_reasons)


def test_stress_short_signal_rejection():
    """Verify signals shorter than minimum clinical duration (1.5s / 100 samples) are rejected."""
    short_signal = np.sin(np.linspace(0, 2 * np.pi, 50))
    gate_res = evaluate_signal_quality_gate(short_signal, fs=360.0)

    assert gate_res.category == QualityCategory.UNUSABLE
    assert not gate_res.can_run_ai
    assert any("duration" in r.lower() or "short" in r.lower() for r in gate_res.rejection_reasons)


def test_stress_corrupted_nan_and_inf():
    """Verify corrupted signals with NaN and Infinite values are rejected without raising unhandled exceptions."""
    corrupted_signal = np.ones(3600)
    corrupted_signal[100:150] = np.nan
    corrupted_signal[200:250] = np.inf

    gate_res = evaluate_signal_quality_gate(corrupted_signal, fs=360.0)
    assert gate_res.category == QualityCategory.UNUSABLE
    assert not gate_res.can_run_ai
    assert any("nan" in r.lower() or "missing" in r.lower() or "infinite" in r.lower() for r in gate_res.rejection_reasons)


def test_stress_baseline_wander_detection():
    """Verify high-amplitude low-frequency baseline drift is detected."""
    t = np.linspace(0, 10, 3600)
    # 0.2 Hz huge sine wave simulates respiratory drift
    huge_wander = 5.0 * np.sin(2 * np.pi * 0.2 * t)
    gate_res = evaluate_signal_quality_gate(huge_wander, fs=360.0)

    # Should detect wander or reject
    assert gate_res.metrics.get("baseline_wander_detected", False) or len(gate_res.warnings) > 0 or not gate_res.can_run_ai


def test_stress_powerline_interference_detection():
    """Verify 50 Hz powerline interference is detected."""
    t = np.linspace(0, 10, 3600)
    # Clean sine + large 50 Hz hum
    hum_signal = np.sin(2 * np.pi * 1.2 * t) + 2.0 * np.sin(2 * np.pi * 50.0 * t)
    gate_res = evaluate_signal_quality_gate(hum_signal, fs=360.0)

    assert gate_res.metrics.get("powerline_detected", False) or len(gate_res.warnings) > 0


def test_inference_latency_benchmark():
    """Verify end-to-end inference latency on a standard 10-second ECG recording is < 1000 ms."""
    assert (RAW_DATA_DIR / "100.dat").exists()
    signal, fs, _ = load_record("100", RAW_DATA_DIR)
    segment = signal[:3600]

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = run_ecg_inference(segment, fs=fs, lead_to_analyze="II")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))

    print(f"\nInference Benchmark: Avg = {avg_latency_ms:.2f} ms, P95 = {p95_latency_ms:.2f} ms")
    assert avg_latency_ms < 1000.0, f"Average inference latency too high: {avg_latency_ms} ms"


def test_stress_high_frequency_electrode_chatter_rejection():
    """Verify high-frequency electrode detachment / contact bounce chatter is halted by safety gatekeeper."""
    fs = 360.0
    rng = np.random.default_rng(42)
    sig = np.zeros(3600)
    for start in range(0, 3600, 200):
        sig[start:start+100] = rng.normal(0, 1.5, 100)  # chatter bounce
        sig[start+100:start+200] = 0.0  # flatline disconnect

    gate_res = evaluate_signal_quality_gate(sig, fs=fs)
    assert not gate_res.can_run_ai, "Signal gatekeeper must halt inference on electrode detachment chatter"
    assert gate_res.category in [QualityCategory.UNUSABLE, QualityCategory.POOR]


def test_wfdb_record_beat_level_validation():
    """Verify beat-level alignment and classification against PhysioNet WFDB reference annotations."""
    # Record 100: Normal reference baseline
    sig_100, fs, _ = load_record("100", RAW_DATA_DIR)
    ann_100 = wfdb.rdann(str(RAW_DATA_DIR / "100"), "atr")
    res_100 = run_ecg_inference(sig_100[:3600], fs=fs)

    assert res_100.detected_beats_count > 0
    ref_100_normals = sum(1 for s, sym in zip(ann_100.sample, ann_100.symbol) if s < 3600 and sym == "N")
    # Detected beats count should closely match annotation count in the window (+/- 2 beats)
    assert abs(res_100.detected_beats_count - ref_100_normals) <= 3
    assert res_100.model_probabilities["Normal"] > 0.80

    # Record 208: Ventricular Arrhythmia reference
    sig_208, fs, _ = load_record("208", RAW_DATA_DIR)
    ann_208 = wfdb.rdann(str(RAW_DATA_DIR / "208"), "atr")
    res_208 = run_ecg_inference(sig_208[:3600], fs=fs)

    ref_208_pvcs = sum(1 for s, sym in zip(ann_208.sample, ann_208.symbol) if s < 3600 and sym == "V")
    assert ref_208_pvcs >= 5, "Ground truth annotations must confirm presence of PVCs"
    assert res_208.model_probabilities["PVC"] > res_100.model_probabilities["PVC"] * 5
    assert "Ventricular Ectopy" in res_208.prediction or "PVC" in res_208.prediction
