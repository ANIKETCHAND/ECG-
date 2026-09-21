"""
Comprehensive Unit and Integration Tests for Autonomous Multi-Dataset ML Pipeline.
Covers:
- Dataset registry and patient resolver
- Preprocessing pipeline with zero-leakage scaler
- Model registry lifecycle and safety gates
- Unified multi-task inference API
- Migration and benchmark report integrity
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.datasets.patient_index import PatientIndex
from src.datasets.registry import APPROVED_DATASETS
from src.ml.inference import analyze_ecg
from src.ml.models.registry import GLOBAL_MODEL_REGISTRY, ModelLifecycleStatus
from src.ml.preprocessing import ECGPreprocessingPipeline
from src.ml.tasks import TASK_BEAT_ARRHYTHMIA, TASK_QUALITY_GATE, TASK_AF_DETECTION

PROJ_DIR = Path(__file__).resolve().parent.parent


def test_dataset_catalog_completeness():
    assert "mit_bih_arrhythmia" in APPROVED_DATASETS
    assert "ptb_xl" in APPROVED_DATASETS
    assert "mit_bih_afdb" in APPROVED_DATASETS
    for dset_id, entry in APPROVED_DATASETS.items():
        assert entry.dataset_id == dset_id
        assert len(entry.sampling_rates) > 0
        assert len(entry.lead_names) > 0


def test_patient_index_cross_contamination_prevention():
    idx = PatientIndex()
    p1 = idx.get_patient_id("mit_bih_arrhythmia", "100")
    p2 = idx.get_patient_id("mit_bih_arrhythmia", "101")
    assert p1 != p2
    assert p1 == "mit_bih_arrhythmia_pt_100"


def test_preprocessing_pipeline_resampling_and_filtering():
    cfg_path = PROJ_DIR / "configs" / "preprocessing" / "beat_arrhythmia_v1.json"
    pipe = ECGPreprocessingPipeline(cfg_path)
    
    # 500 Hz input to 360 Hz output
    raw = np.sin(np.linspace(0, 10, 1000))
    transformed, fs = pipe.transform_array(raw, 500.0)
    assert fs == 360.0
    assert len(transformed) == 720
    assert not np.isnan(transformed).any()


def test_model_registry_lifecycle_and_production_lock():
    models = GLOBAL_MODEL_REGISTRY.list_models()
    assert len(models) >= 3

    # Production model must be ECG-RF-1.0.0
    prod_entry = GLOBAL_MODEL_REGISTRY.get_model_entry("ECG-RF-1.0.0")
    assert prod_entry["status"] == ModelLifecycleStatus.PRODUCTION.value

    # Candidate models must not be PRODUCTION without explicit migration
    cand_entry = GLOBAL_MODEL_REGISTRY.get_model_entry("ECG-RF-2.0.0-candidate")
    assert cand_entry["status"] == ModelLifecycleStatus.VALIDATED.value

    clf, scaler, meta = GLOBAL_MODEL_REGISTRY.load_model()
    assert meta["model_id"] == "ECG-RF-1.0.0"
    assert clf is not None


def test_unified_api_quality_gate_dispatch():
    noise = np.random.randn(1000) * 100.0
    res = analyze_ecg(noise, fs=360.0, task=TASK_QUALITY_GATE)
    assert res["task"] == "quality_gate"
    assert "prediction" in res
    assert "quality_score" in res


def test_unified_api_research_task_safety_barrier():
    signal = np.sin(np.linspace(0, 10, 1000))
    res = analyze_ecg(signal, fs=360.0, task=TASK_AF_DETECTION)
    assert res["task"] == "af_detection"
    assert res["prediction"] == "RESEARCH_TASK_EVALUATION_ONLY"
    assert len(res["warnings"]) > 0
    assert len(res["limitations"]) > 0


def test_migration_and_benchmark_reports_exist():
    migration_report = PROJ_DIR / "reports" / "migration" / "current_vs_new.md"
    assert migration_report.exists()
    content = migration_report.read_text(encoding="utf-8")
    assert "ECG-RF-1.0.0" in content
    assert "ECG-RF-2.0.0" in content
    assert "AUROC" in content

    benchmark_json = PROJ_DIR / "reports" / "experiments" / "multi_model_benchmark.json"
    assert benchmark_json.exists()
    data = json.loads(benchmark_json.read_text(encoding="utf-8"))
    assert "production" in data
    assert "candidate_rf" in data

