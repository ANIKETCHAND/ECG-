"""
Model Registry Subsystem for ECG Guardian.
Manages model lifecycle states: EXPERIMENTAL -> CANDIDATE -> VALIDATED -> PRODUCTION.
Enforces safety gates: candidate models cannot be loaded as PRODUCTION without validation evidence.

Additionally enforces *data provenance* gates:

* Every entry carries a ``provenance`` block recording where its training data
  came from (see ``training/guards.py``).
* Artifacts fitted to fabricated samples are stamped
  ``ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER`` and can never be loaded as the
  active production model, nor returned by the default ``load_model()`` path.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib

PROJ_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJ_DIR / "models"

#: Provenance statuses that make an artifact ineligible for production use.
NON_PRODUCTION_DATA_STATUSES = {"SYNTHETIC_PLACEHOLDER", "DEMO_FIXTURE"}


class ModelLifecycleStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    #: Trained on fabricated or demo data. Structurally barred from production.
    SYNTHETIC_PLACEHOLDER = "SYNTHETIC_PLACEHOLDER"


#: Models known to have been fitted to fabricated samples by the legacy
#: ``numpy.random`` training fallbacks, before guard enforcement was added.
_LEGACY_SYNTHETIC_MODEL_IDS = {
    "ECG-AF-1.0.0-candidate",
    "ECG-PTBXL-1.0.0-candidate",
    "ECG-ST-1.0.0-candidate",
    "ECG-QUALITY-1.0.0",
}


def _default_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        "ECG-RF-1.0.0": {
            "model_id": "ECG-RF-1.0.0",
            "task": "beat_arrhythmia",
            "status": ModelLifecycleStatus.PRODUCTION.value,
            "artifact_path": "production/classifier.pkl",
            "scaler_path": "production/scaler.pkl",
            "description": "Baseline Balanced Random Forest for Single-Lead Beat Arrhythmia",
            "provenance": {
                "data_status": "REAL_DATASET",
                "dataset_id": "mit_bih_arrhythmia",
                "task": "beat_arrhythmia",
                "clinical_claim": "Research prototype - not a certified diagnostic device",
                "may_be_promoted_to_production": True,
            },
        },
        "ECG-RF-2.0.0-candidate": {
            "model_id": "ECG-RF-2.0.0-candidate",
            "task": "beat_arrhythmia",
            "status": ModelLifecycleStatus.VALIDATED.value,
            "artifact_path": "candidate/classifier.pkl",
            "scaler_path": "candidate/scaler.pkl",
            "description": "Candidate Optimized Random Forest with zero patient leakage validation",
            "provenance": {
                "data_status": "REAL_DATASET",
                "dataset_id": "mit_bih_arrhythmia",
                "task": "beat_arrhythmia",
                "patient_level_split_enforced": True,
                "clinical_claim": "Research prototype - not a certified diagnostic device",
                "may_be_promoted_to_production": True,
            },
        },
        "ECG-MLP-1.0.0-candidate": {
            "model_id": "ECG-MLP-1.0.0-candidate",
            "task": "beat_arrhythmia",
            "status": ModelLifecycleStatus.EXPERIMENTAL.value,
            "artifact_path": "candidate/deep_classifier.pkl",
            "scaler_path": "candidate/scaler.pkl",
            "description": "Candidate neural waveform model (scikit-learn MLP on extracted features)",
            "provenance": {
                "data_status": "REAL_DATASET",
                "dataset_id": "mit_bih_arrhythmia",
                "task": "beat_arrhythmia",
                "clinical_claim": "Research prototype - not a certified diagnostic device",
                "may_be_promoted_to_production": True,
            },
        },
        "ECG-LR-1.0.0": {
            "model_id": "ECG-LR-1.0.0",
            "task": "beat_arrhythmia",
            "status": ModelLifecycleStatus.EXPERIMENTAL.value,
            "artifact_path": "candidate/baseline_classifier.pkl",
            "scaler_path": "candidate/scaler.pkl",
            "description": "Linear Logistic Regression Baseline",
            "provenance": {
                "data_status": "REAL_DATASET",
                "dataset_id": "mit_bih_arrhythmia",
                "task": "beat_arrhythmia",
                "clinical_claim": "Research prototype - not a certified diagnostic device",
                "may_be_promoted_to_production": True,
            },
        },
    }


def _unverified_provenance(task: str) -> Dict[str, Any]:
    return {
        "data_status": "UNVERIFIED_LEGACY",
        "task": task,
        "clinical_claim": "NONE - training data provenance not recorded for this legacy artifact",
        "may_be_promoted_to_production": False,
    }


def _synthetic_provenance(task: str, reason: str) -> Dict[str, Any]:
    return {
        "data_status": "SYNTHETIC_PLACEHOLDER",
        "task": task,
        "generator": "numpy.random",
        "reason": reason,
        "clinical_claim": "NONE - placeholder exists solely to exercise application plumbing",
        "may_be_promoted_to_production": False,
    }


class ModelRegistry:
    """Manages models across lifecycle environments (production, candidate, experimental)."""

    def __init__(self, base_models_dir: Optional[Path] = None):
        self.base_dir = Path(base_models_dir) if base_models_dir else MODELS_DIR
        self.catalog_path = self.base_dir / "registry" / "catalog.json"
        self._ensure_catalog()

    # ------------------------------------------------------------------ catalog
    def _ensure_catalog(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.catalog_path.exists():
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(_default_catalog(), f, indent=2)
            return
        self._migrate_catalog()

    def _read_catalog(self) -> Dict[str, Dict[str, Any]]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_catalog(self, data: Dict[str, Dict[str, Any]]) -> None:
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _migrate_catalog(self) -> None:
        """Bring an existing catalog up to the provenance-aware schema.

        Two things happen:

        1. Every entry without a ``provenance`` block gets an honest
           ``UNVERIFIED_LEGACY`` stamp.
        2. Artifacts from the known ``numpy.random`` training fallbacks are
           reclassified as ``SYNTHETIC_PLACEHOLDER``.
        """
        data = self._read_catalog()
        changed = False

        for model_id, entry in data.items():
            task = entry.get("task", "unclassified")

            if "provenance" not in entry:
                entry["provenance"] = _unverified_provenance(task)
                changed = True

            if model_id in _LEGACY_SYNTHETIC_MODEL_IDS:
                entry["provenance"] = _synthetic_provenance(
                    task,
                    "Fitted to fabricated numpy.random samples by the pre-guard training fallback.",
                )
                if entry.get("status") != ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value:
                    entry["status"] = ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value
                changed = True

        if changed:
            self._write_catalog(data)

    # ---------------------------------------------------------------- queries
    def list_models(
        self, status: Optional[ModelLifecycleStatus] = None
    ) -> List[Dict[str, Any]]:
        data = self._read_catalog()
        models = list(data.values())
        if status:
            models = [m for m in models if m["status"] == status.value]
        return models

    def get_model_entry(self, model_id: str) -> Dict[str, Any]:
        data = self._read_catalog()
        if model_id not in data:
            raise KeyError(f"Model {model_id} not registered in catalog.")
        return data[model_id]

    def is_placeholder(self, model_id: str) -> bool:
        """Return True when ``model_id`` may not be used for clinical output."""
        try:
            entry = self.get_model_entry(model_id)
        except KeyError:
            return False
        provenance = entry.get("provenance", {})
        return (
            entry.get("status") == ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value
            or provenance.get("data_status") in NON_PRODUCTION_DATA_STATUSES
        )

    # ---------------------------------------------------------------- mutation
    def register_model(
        self,
        *,
        model_id: str,
        status: str,
        task: str,
        provenance: Optional[Dict[str, Any]] = None,
        artifact_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a catalog entry, enforcing provenance safety rules.

        A model whose provenance is synthetic/fixture can never hold a
        production-grade status: the status is coerced to
        ``SYNTHETIC_PLACEHOLDER``.
        """
        data = self._read_catalog()
        entry = data.get(model_id, {"model_id": model_id})
        provenance = provenance or {}

        if provenance.get("data_status") in NON_PRODUCTION_DATA_STATUSES:
            status = ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value

        entry.update(
            {
                "model_id": model_id,
                "task": task,
                "status": status,
                "provenance": provenance,
            }
        )
        for key, value in (
            ("artifact_path", artifact_path),
            ("scaler_path", scaler_path),
            ("description", description),
        ):
            if value is not None:
                entry[key] = value

        data[model_id] = entry
        self._write_catalog(data)
        return entry

    # ------------------------------------------------------------------- load
    def load_model(
        self,
        model_id: Optional[str] = None,
        allow_placeholder: bool = False,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load model artifact, scaler, and metadata.

        Args:
            model_id: Explicit model to load. When ``None``, the active
                PRODUCTION model is used.
            allow_placeholder: When False (default), a ``SYNTHETIC_PLACEHOLDER``
                artifact is refused rather than silently served.

        Raises:
            RuntimeError: if no production model exists, or if the resolved
                entry is a placeholder and ``allow_placeholder`` is False.
        """
        if model_id is None:
            prod_models = self.list_models(status=ModelLifecycleStatus.PRODUCTION)
            if not prod_models:
                raise RuntimeError("No active PRODUCTION model found in registry.")
            model_entry = prod_models[0]
        else:
            model_entry = self.get_model_entry(model_id)

        resolved_id = model_entry.get("model_id", "<unknown>")
        provenance = model_entry.get("provenance", {})
        is_placeholder = (
            model_entry.get("status") == ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value
            or provenance.get("data_status") in NON_PRODUCTION_DATA_STATUSES
        )
        if is_placeholder and not allow_placeholder:
            raise RuntimeError(
                f"Model '{resolved_id}' is a {ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value} "
                f"trained on fabricated/demo data and cannot be served. "
                f"Reason: {provenance.get('reason', 'unspecified')}. "
                f"Train this task on a real dataset before use."
            )

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
