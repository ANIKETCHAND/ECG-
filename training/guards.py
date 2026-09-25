"""
Training Data Integrity Guards
==============================

ECG Guardian's core clinical tenet is:

    "NO RELIABLE INPUT = NO AI RESULT"

That tenet must apply to *training* as well as to inference. Previously, every
training entrypoint silently fell back to `numpy.random` samples whenever the
real PhysioNet dataset was absent from disk, then saved the resulting artifact
under a clinical-sounding model identifier (for example
``ECG-PTBXL-1.0.0-candidate``). Weights fitted to random numbers carry no
biological signal whatsoever, yet they were registered with lifecycle statuses
such as ``VALIDATED`` and could therefore be surfaced to clinicians.

This module makes that impossible by default:

* Missing dataset  ->  :class:`PlaceholderTrainingNotAllowed` (training aborts).
* Explicit opt-in via ``ECG_ALLOW_PLACEHOLDER_TRAINING=1`` (or
  ``--allow-placeholder``)  ->  training proceeds, but the artifact is
  permanently stamped ``SYNTHETIC_PLACEHOLDER`` in the model registry and can
  never be loaded as the active production model.

Every training script must route its "dataset is missing" branch through
:func:`require_real_dataset` and every provenance blob must be produced by
:func:`real_provenance` or :func:`synthetic_provenance`.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

PLACEHOLDER_ENV_VAR = "ECG_ALLOW_PLACEHOLDER_TRAINING"

#: Provenance status meaning "this artifact was fitted to real recorded data".
DATA_STATUS_REAL = "REAL_DATASET"
#: Provenance status meaning "this artifact was fitted to fabricated samples".
DATA_STATUS_SYNTHETIC = "SYNTHETIC_PLACEHOLDER"
#: Provenance status meaning "this artifact was fitted to bundled demo fixtures".
DATA_STATUS_FIXTURE = "DEMO_FIXTURE"


class MissingDatasetError(RuntimeError):
    """Base class for missing/incomplete training dataset conditions."""


class PlaceholderTrainingNotAllowed(MissingDatasetError):
    """Raised when a real dataset is required but absent.

    Raised instead of silently generating synthetic samples. Callers may opt in
    to placeholder generation explicitly (see :func:`placeholder_training_allowed`).
    """


def placeholder_training_allowed(explicit: Optional[bool] = None) -> bool:
    """Return True when synthetic placeholder artifacts may be generated.

    Args:
        explicit: Caller-supplied override (e.g. from ``--allow-placeholder``).
            When provided it wins over the environment variable.
    """
    if explicit is not None:
        return bool(explicit)
    value = os.environ.get(PLACEHOLDER_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def require_real_dataset(
    *,
    dataset_id: str,
    task: str,
    records: Sequence[Any],
    location: Any,
    allow_placeholder: Optional[bool] = None,
    download_hint: Optional[str] = None,
) -> bool:
    """Enforce the presence of real training data.

    Args:
        dataset_id: Registry identifier of the expected dataset.
        task: Human-readable task name, used only in the error message.
        records: Discovered record files. Non-empty means real data is present.
        location: Filesystem path searched, used in the error message.
        allow_placeholder: Explicit opt-in override.
        download_hint: Optional shell command shown in the error message.

    Returns:
        ``False`` when real records exist (proceed with real training).
        ``True`` when no records exist *and* placeholder generation was
        explicitly authorised.

    Raises:
        PlaceholderTrainingNotAllowed: when records are missing and placeholder
            generation was not authorised.
    """
    if records:
        return False

    if placeholder_training_allowed(allow_placeholder):
        return True

    hint = download_hint or f"python training/download_datasets.py --dataset {dataset_id}"
    raise PlaceholderTrainingNotAllowed(
        f"\n"
        f"{'=' * 78}\n"
        f"TRAINING ABORTED - NO RELIABLE INPUT = NO AI RESULT\n"
        f"{'=' * 78}\n"
        f"Task            : {task}\n"
        f"Dataset         : {dataset_id}\n"
        f"Expected at     : {location}\n"
        f"\n"
        f"Refusing to fabricate a training set. An artifact fitted to random\n"
        f"samples is clinically meaningless and must never be registered as a\n"
        f"model for this task.\n"
        f"\n"
        f"To train for real:\n"
        f"    {hint}\n"
        f"\n"
        f"To build a clearly-labelled non-clinical placeholder for UI plumbing\n"
        f"only (it can never be promoted to PRODUCTION):\n"
        f"    {PLACEHOLDER_ENV_VAR}=1 <same command>\n"
        f"    or pass --allow-placeholder\n"
        f"{'=' * 78}\n"
    )


def _timestamp() -> str:
    return datetime.datetime.now().isoformat()


def synthetic_provenance(
    *,
    dataset_id: str,
    task: str,
    reason: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Provenance blob for an artifact fitted to fabricated samples."""
    return {
        "data_status": DATA_STATUS_SYNTHETIC,
        "dataset_id": dataset_id,
        "task": task,
        "n_samples": n_samples,
        "generator": f"numpy.random (deterministic seed={seed})",
        "reason": reason,
        "clinical_claim": "NONE - placeholder exists solely to exercise application plumbing",
        "may_be_promoted_to_production": False,
        "trained_at": _timestamp(),
    }


def fixture_provenance(
    *,
    dataset_id: str,
    task: str,
    source_files: Iterable[str],
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Provenance blob for an artifact fitted to bundled demo fixtures.

    Demo fixtures are genuinely recorded signals (though not a clinical
    training corpus), so they are distinguished from fabricated samples.
    """
    return {
        "data_status": DATA_STATUS_FIXTURE,
        "dataset_id": dataset_id,
        "task": task,
        "n_samples": n_samples,
        "source_files": sorted(str(s) for s in source_files),
        "reason": "Fitted to repository demo fixtures, not a validated training corpus.",
        "clinical_claim": "NONE - insufficient sample size and label coverage for clinical use",
        "may_be_promoted_to_production": False,
        "trained_at": _timestamp(),
    }


def real_provenance(
    *,
    dataset_id: str,
    task: str,
    records: Sequence[Any],
    train_records: Optional[Sequence[str]] = None,
    test_records: Optional[Sequence[str]] = None,
    validation_records: Optional[Sequence[str]] = None,
    n_train_samples: Optional[int] = None,
    n_test_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Provenance blob for an artifact fitted to real recorded data."""
    return {
        "data_status": DATA_STATUS_REAL,
        "dataset_id": dataset_id,
        "task": task,
        "n_records": len(records),
        "train_records": list(train_records) if train_records else None,
        "validation_records": list(validation_records) if validation_records else None,
        "test_records": list(test_records) if test_records else None,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "patient_level_split_enforced": bool(train_records and test_records),
        "clinical_claim": "Research prototype - not a certified diagnostic device",
        "may_be_promoted_to_production": True,
        "trained_at": _timestamp(),
    }


def record_provenance(
    *,
    model_id: str,
    provenance: Dict[str, Any],
    status: str,
    artifact_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist provenance for ``model_id`` into the model registry catalog.

    Placeholder provenance is automatically paired with the
    ``SYNTHETIC_PLACEHOLDER`` lifecycle status so it can never be picked up as
    the active production model.

    Returns the updated catalog entry.
    """
    # Imported lazily so training scripts can import guards without pulling the
    # whole registry (and its artifact I/O) into scope at module import time.
    from src.ml.models.registry import (
        GLOBAL_MODEL_REGISTRY,
        ModelLifecycleStatus,
    )

    data_status = provenance.get("data_status")
    if data_status in {DATA_STATUS_SYNTHETIC, DATA_STATUS_FIXTURE}:
        # Structural safety rule: non-real provenance can never be production-grade.
        status = ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value

    return GLOBAL_MODEL_REGISTRY.register_model(
        model_id=model_id,
        status=status,
        task=provenance.get("task", "unclassified"),
        provenance=provenance,
        artifact_path=artifact_path,
        scaler_path=scaler_path,
        description=description,
    )


def assert_metric_is_measured(metric_name: str, value: Any, source: Any) -> None:
    """Guard against invented evaluation numbers.

    Any reported metric must be traceable to an actual computation over real
    held-out data. This helper exists so that "we ran out of time so we nudged
    the number" cannot quietly re-enter the codebase.
    """
    if value is None:
        raise ValueError(
            f"Metric '{metric_name}' has no measured value. "
            f"Refusing to emit a placeholder performance number."
        )
    if source is None:
        raise ValueError(
            f"Metric '{metric_name}' has no measured data source. "
            f"Refusing to report a number that was not computed from real data."
        )
