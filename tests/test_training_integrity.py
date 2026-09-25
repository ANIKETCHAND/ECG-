"""
Tests for training data-integrity guards and provenance safety gates.

These tests exist to prevent a regression of the behaviour they were written to
remove: training on fabricated data and registering the result as a clinical
model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.ml.models.registry import (
    GLOBAL_MODEL_REGISTRY,
    ModelLifecycleStatus,
    ModelRegistry,
)
from training.external_validation import generate_external_validation_report
from training.guards import (
    DATA_STATUS_FIXTURE,
    DATA_STATUS_REAL,
    DATA_STATUS_SYNTHETIC,
    PLACEHOLDER_ENV_VAR,
    PlaceholderTrainingNotAllowed,
    fixture_provenance,
    placeholder_training_allowed,
    real_provenance,
    record_provenance,
    require_real_dataset,
    synthetic_provenance,
)


# ------------------------------------------------------------------------ guards
def test_require_real_dataset_returns_false_when_records_exist():
    assert require_real_dataset(
        dataset_id="demo", task="demo", records=["a.hea", "b.hea"], location="/tmp"
    ) is False


def test_require_real_dataset_aborts_when_data_missing(monkeypatch):
    monkeypatch.delenv(PLACEHOLDER_ENV_VAR, raising=False)
    with pytest.raises(PlaceholderTrainingNotAllowed) as exc:
        require_real_dataset(
            dataset_id="mit_bih_afdb", task="af_detection", records=[], location="/missing"
        )
    message = str(exc.value)
    assert "NO RELIABLE INPUT = NO AI RESULT" in message
    assert "mit_bih_afdb" in message
    # The error must tell the operator how to fix it.
    assert "download_datasets.py" in message


def test_require_real_dataset_allows_placeholder_only_with_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(PLACEHOLDER_ENV_VAR, raising=False)

    # Explicit opt-in authorises the placeholder path.
    assert require_real_dataset(
        dataset_id="d", task="t", records=[], location="/x", allow_placeholder=True
    ) is True

    # Explicit opt-out must raise even with no environment variable present.
    with pytest.raises(PlaceholderTrainingNotAllowed):
        require_real_dataset(
            dataset_id="d", task="t", records=[], location="/x", allow_placeholder=False
        )


def test_require_real_dataset_env_var_opts_in(monkeypatch):
    monkeypatch.setenv(PLACEHOLDER_ENV_VAR, "1")
    assert placeholder_training_allowed() is True
    assert require_real_dataset(
        dataset_id="d", task="t", records=[], location="/x"
    ) is True


def test_require_real_dataset_explicit_false_overrides_env(monkeypatch):
    monkeypatch.setenv(PLACEHOLDER_ENV_VAR, "1")
    assert placeholder_training_allowed(False) is False
    with pytest.raises(PlaceholderTrainingNotAllowed):
        require_real_dataset(
            dataset_id="d", task="t", records=[], location="/x", allow_placeholder=False
        )


def test_placeholder_training_allowed_parses_common_values(monkeypatch):
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(PLACEHOLDER_ENV_VAR, value)
        assert placeholder_training_allowed() is True
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv(PLACEHOLDER_ENV_VAR, value)
        assert placeholder_training_allowed() is False


# -------------------------------------------------------------------- provenance
def test_synthetic_provenance_denies_clinical_claims():
    provenance = synthetic_provenance(dataset_id="d", task="t", reason="absent", n_samples=10)
    assert provenance["data_status"] == DATA_STATUS_SYNTHETIC
    assert provenance["clinical_claim"] == "NONE - placeholder exists solely to exercise application plumbing"
    assert provenance["may_be_promoted_to_production"] is False


def test_real_provenance_records_split_and_permits_promotion():
    provenance = real_provenance(
        dataset_id="mit_bih_arrhythmia",
        task="beat_arrhythmia",
        records=["100", "101"],
        train_records=["100"],
        test_records=["101"],
        n_train_samples=100,
        n_test_samples=50,
    )
    assert provenance["data_status"] == DATA_STATUS_REAL
    assert provenance["patient_level_split_enforced"] is True
    assert provenance["may_be_promoted_to_production"] is True


def test_fixture_provenance_denies_promotion():
    provenance = fixture_provenance(dataset_id="d", task="t", source_files=["a.csv"], n_samples=5)
    assert provenance["data_status"] == DATA_STATUS_FIXTURE
    assert provenance["may_be_promoted_to_production"] is False


def test_record_provenance_forces_placeholder_status(tmp_path, monkeypatch):
    from src.ml import models as models_pkg
    from src.ml.models import registry as registry_module

    scoped = ModelRegistry(tmp_path)
    monkeypatch.setattr(registry_module, "GLOBAL_MODEL_REGISTRY", scoped)

    entry = record_provenance(
        model_id="TEST-SYNTH-1.0.0",
        provenance=synthetic_provenance(dataset_id="d", task="t", reason="absent"),
        status="VALIDATED",  # must be overridden
        artifact_path="candidate/thing.pkl",
    )
    assert entry["status"] == ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value
    assert scoped.is_placeholder("TEST-SYNTH-1.0.0") is True


# ---------------------------------------------------------------------- registry
def test_default_registry_carries_provenance_for_every_entry():
    for entry in GLOBAL_MODEL_REGISTRY.list_models():
        assert "provenance" in entry, entry["model_id"]
        assert "data_status" in entry["provenance"], entry["model_id"]


def test_legacy_synthetic_models_are_reclassified_or_flagged():
    # The known numpy.random fallbacks must never be production-grade.
    for model_id in (
        "ECG-AF-1.0.0-candidate",
        "ECG-PTBXL-1.0.0-candidate",
        "ECG-ST-1.0.0-candidate",
        "ECG-QUALITY-1.0.0",
    ):
        try:
            entry = GLOBAL_MODEL_REGISTRY.get_model_entry(model_id)
        except KeyError:
            continue
        assert entry["status"] == ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value, model_id
        assert GLOBAL_MODEL_REGISTRY.is_placeholder(model_id) is True


def test_no_placeholder_model_is_registered_as_production():
    production = GLOBAL_MODEL_REGISTRY.list_models(status=ModelLifecycleStatus.PRODUCTION)
    assert production, "Expected at least one production model"
    for entry in production:
        assert not GLOBAL_MODEL_REGISTRY.is_placeholder(entry["model_id"]), entry["model_id"]


def test_loading_a_placeholder_model_is_refused(tmp_path):
    scoped = ModelRegistry(tmp_path)
    scoped.register_model(
        model_id="TEST-PLACEHOLDER",
        status="VALIDATED",
        task="demo",
        provenance=synthetic_provenance(dataset_id="d", task="demo", reason="absent"),
        artifact_path="candidate/x.pkl",
    )
    assert scoped.is_placeholder("TEST-PLACEHOLDER") is True
    with pytest.raises(RuntimeError, match="cannot be served"):
        scoped.load_model("TEST-PLACEHOLDER")


def test_register_model_cannot_promote_synthetic_to_production(tmp_path):
    scoped = ModelRegistry(tmp_path)
    entry = scoped.register_model(
        model_id="TEST-PROMOTE",
        status=ModelLifecycleStatus.PRODUCTION.value,
        task="demo",
        provenance=fixture_provenance(dataset_id="d", task="demo", source_files=["a.csv"]),
    )
    assert entry["status"] == ModelLifecycleStatus.SYNTHETIC_PLACEHOLDER.value


def test_new_registry_in_a_fresh_directory_writes_a_catalog(tmp_path):
    scoped = ModelRegistry(tmp_path / "models")
    assert scoped.catalog_path.exists()
    data = json.loads(scoped.catalog_path.read_text(encoding="utf-8"))
    assert "ECG-RF-1.0.0" in data


# --------------------------------------------------------- external validation
def test_external_validation_returns_not_evaluated_without_numbers():
    report = generate_external_validation_report(
        model_id="ECG-RF-2.0.0-candidate",
        training_dataset="mit_bih_arrhythmia",
        external_dataset="incart_12lead_arrhythmia",
        in_domain_metrics={"accuracy": 0.98, "weighted_f1": 0.98},
        external_metrics=None,
    )
    assert report["status"] == "NOT_EVALUATED"
    assert report["external_accuracy"] is None
    assert report["accuracy_delta"] is None
    assert report["calibration_drift_ece"] is None
    assert "unmeasured" in report["integrity_note"]


def test_external_validation_computes_deltas_from_measured_metrics():
    report = generate_external_validation_report(
        model_id="m",
        training_dataset="a",
        external_dataset="b",
        in_domain_metrics={"accuracy": 0.98, "weighted_f1": 0.97, "ece": 0.02},
        external_metrics={"accuracy": 0.91, "weighted_f1": 0.90, "ece": 0.06},
    )
    assert report["status"] == "EVALUATED"
    assert report["accuracy_delta"] == pytest.approx(0.07, abs=1e-6)
    assert report["calibration_drift_ece"] == pytest.approx(0.04, abs=1e-6)
    assert any("degradation" in finding for finding in report["domain_shift_findings"])


def test_external_validation_never_invents_a_delta_when_one_side_is_missing():
    report = generate_external_validation_report(
        model_id="m",
        training_dataset="a",
        external_dataset="b",
        in_domain_metrics={"accuracy": 0.98},
        external_metrics={"accuracy": 0.91},  # no f1 on either side
    )
    assert report["accuracy_delta"] == pytest.approx(0.07, abs=1e-6)
    assert report["f1_delta"] is None


#: Sampling calls that would fabricate data if they appeared in a training path.
_FABRICATION_ATTRS = {
    "normal",
    "randn",
    "uniform",
    "rand",
    "random",
    "choice",
    "integers",
    "standard_normal",
    "poisson",
}


def test_training_scripts_do_not_contain_random_fallbacks():
    """Static guard: no training entrypoint may fabricate a dataset silently.

    Sampling calls are allowed only inside the explicitly gated placeholder
    builder. Legitimate uses of randomness for *splitting* existing real data
    (``shuffle``, ``permutation``) are not sampling and are not flagged.
    """
    import ast

    training_dir = Path(__file__).resolve().parent.parent / "training"
    guarded = ["train_af.py", "train_ptbxl.py", "train_st.py", "train_quality.py"]

    for filename in guarded:
        source = (training_dir / filename).read_text(encoding="utf-8")
        # Dataset-backed tasks call require_real_dataset; corpus-backed tasks
        # (e.g. quality) gate on placeholder_training_allowed instead.
        assert (
            "require_real_dataset" in source or "placeholder_training_allowed" in source
        ), f"{filename} must gate its training data"

        tree = ast.parse(source)
        placeholder_functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_build_placeholder"
        ]

        def in_placeholder(lineno: int) -> bool:
            return any(
                node.lineno <= lineno <= (node.end_lineno or node.lineno)
                for node in placeholder_functions
            )

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _FABRICATION_ATTRS:
                continue
            # Only flag numpy / Generator sampling, not unrelated objects.
            target = func.value
            is_numpy = isinstance(target, ast.Name) and target.id in {"np", "rng", "random", "numpy"}
            is_numpy_attr = isinstance(target, ast.Attribute) and target.attr == "random"
            if (is_numpy or is_numpy_attr) and not in_placeholder(node.lineno):
                offenders.append((node.lineno, func.attr))

        assert not offenders, (
            f"{filename} fabricates samples outside the gated placeholder builder at "
            f"{offenders}. Training must consume real data or refuse."
        )


def test_train_all_has_no_fabricated_external_metrics():
    source = (Path(__file__).resolve().parent.parent / "training" / "train_all.py").read_text(encoding="utf-8")
    # The previous implementation subtracted constants from in-domain scores.
    assert "accuracy\"] - 0.021" not in source
    assert "- 0.025" not in source
