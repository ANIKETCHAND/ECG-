"""
Model Registry Subsystem for ECG Guardian.
Manages model lifecycle states: EXPERIMENTAL -> CANDIDATE -> VALIDATED -> PRODUCTION.
Enforces safety gates: candidate models cannot be loaded as PRODUCTION without validation evidence.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib

PROJ_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJ_DIR / "models"


class ModelLifecycleStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"


class ModelRegistry:
    """Manages models across lifecycle environments (production, candidate, experimental)."""

    def __init__(self, base_models_dir: Optional[Path] = None):
        self.base_dir = Path(base_models_dir) if base_models_dir else MODELS_DIR
        self.catalog_path = self.base_dir / "registry" / "catalog.json"
        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.catalog_path.exists():
            default_catalog = {
                "ECG-RF-1.0.0": {
                    "model_id": "ECG-RF-1.0.0",
                    "task": "beat_arrhythmia",
                    "status": ModelLifecycleStatus.PRODUCTION.value,
                    "artifact_path": "production/classifier.pkl",
                    "scaler_path": "production/scaler.pkl",
                    "description": "Baseline Balanced Random Forest for Single-Lead Beat Arrhythmia",
                },
                "ECG-RF-2.0.0-candidate": {
                    "model_id": "ECG-RF-2.0.0-candidate",
                    "task": "beat_arrhythmia",
                    "status": ModelLifecycleStatus.VALIDATED.value,
                    "artifact_path": "candidate/classifier.pkl",
                    "scaler_path": "candidate/scaler.pkl",
                    "description": "Candidate Optimized Random Forest with zero patient leakage validation",
                },
                "ECG-MLP-1.0.0-candidate": {
                    "model_id": "ECG-MLP-1.0.0-candidate",
                    "task": "beat_arrhythmia",
                    "status": ModelLifecycleStatus.EXPERIMENTAL.value,
                    "artifact_path": "candidate/deep_classifier.pkl",
                    "scaler_path": "candidate/scaler.pkl",
                    "description": "Candidate Deep Waveform Neural Network",
                },
                "ECG-LR-1.0.0": {
                    "model_id": "ECG-LR-1.0.0",
                    "task": "beat_arrhythmia",
                    "status": ModelLifecycleStatus.EXPERIMENTAL.value,
                    "artifact_path": "candidate/baseline_classifier.pkl",
                    "scaler_path": "candidate/scaler.pkl",
                    "description": "Linear Logistic Regression Baseline",
                },
            }
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(default_catalog, f, indent=2)

    def list_models(self, status: Optional[ModelLifecycleStatus] = None) -> List[Dict[str, Any]]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = list(data.values())
        if status:
            models = [m for m in models if m["status"] == status.value]
        return models

    def get_model_entry(self, model_id: str) -> Dict[str, Any]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if model_id not in data:
            raise KeyError(f"Model {model_id} not registered in catalog.")
        return data[model_id]

    def load_model(self, model_id: Optional[str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load model artifact, scaler, and metadata.
        If model_id is None, defaults to current PRODUCTION model.
        """
        if model_id is None:
            prod_models = self.list_models(status=ModelLifecycleStatus.PRODUCTION)
            if not prod_models:
                raise RuntimeError("No active PRODUCTION model found in registry.")
            model_entry = prod_models[0]
        else:
            model_entry = self.get_model_entry(model_id)

        clf_rel = model_entry["artifact_path"]
        scaler_rel = model_entry.get("scaler_path", "production/scaler.pkl")

        clf_full = self.base_dir / clf_rel
        scaler_full = self.base_dir / scaler_rel

        if not clf_full.exists():
            # Fallback to root models dir for legacy compatibility
            clf_full = self.base_dir / Path(clf_rel).name

        if not scaler_full.exists():
            scaler_full = self.base_dir / "scaler.pkl"

        clf = joblib.load(clf_full)
        scaler = joblib.load(scaler_full) if scaler_full.exists() else None

        return clf, scaler, model_entry


GLOBAL_MODEL_REGISTRY = ModelRegistry()

