"""
Operating-Point Fitting for Selective Prediction (Abstention)
============================================================

Fits the confidence threshold at which the beat classifier may report a label,
and measures what happens at that threshold on data that did not choose it.

Protocol (why this is not self-congratulatory)
----------------------------------------------
A single train/confirm record split is far too coarse for threshold tuning on
this corpus: whichever record lands in the selection half decides the answer.
Instead we run **leave-one-record-out selection** over the record-disjoint test
partition: for every record, a threshold is chosen from the other records only,
so no record is ever measured with a threshold it influenced (see
:func:`src.ml.selective.record_wise_jackknife`).

The installed threshold is the **strictest** fold threshold. Erring strict costs
coverage, never precision, which is the only safe direction for a clinical gate.

The headline precision/coverage in ``models/operating_point.json`` is then
measured over the whole held-out partition at that installed threshold, which is
exactly what deployment does.

The script refuses to install an operating point that does not reach the
requested target precision, or that reaches it only by discarding almost
everything. A gate that reports 8 beats out of 6807 is not a clinical feature,
it is a rounding error with a certificate.

Usage::

    python training/fit_operating_point.py
    python training/fit_operating_point.py --target-precision 0.995 --min-coverage 0.20
    python training/fit_operating_point.py --ladder      # show what each target buys
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.ml.selective import (
    DEFAULT_THRESHOLDS,
    MIN_USEFUL_BEATS,
    OperatingPoint,
    operating_curve,
    record_wise_jackknife,
    save_operating_point,
    summarise,
)
from training.guards import require_real_dataset

DATA_DIR = PROJ_DIR / "data" / "processed"
MODELS_DIR = PROJ_DIR / "models"
IGNORE_COLUMNS = {"label", "record_id", "symbol"}
DEFAULT_TARGET_PRECISION = 1.0
DEFAULT_MIN_COVERAGE = 0.05


def _load_artifacts(models_dir: Optional[Path] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the production classifier, scaler, and metadata."""
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    candidates = (
        (base / "production" / "classifier.pkl", base / "production" / "scaler.pkl", base / "production" / "metadata.json"),
        (base / "classifier.pkl", base / "scaler.pkl", base / "metadata.json"),
    )
    for clf_path, scaler_path, meta_path in candidates:
        if clf_path.exists() and scaler_path.exists():
            metadata: Dict[str, Any] = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            return joblib.load(clf_path), joblib.load(scaler_path), metadata
    raise FileNotFoundError("No trained classifier/scaler artifacts were found.")


#: Targets shown by ``--ladder``. Purely informational: each row re-runs the
#: full leave-one-record-out protocol at that target.
LADDER_TARGETS: Tuple[float, ...] = (1.0, 0.999, 0.995, 0.99, 0.98, 0.95)


def _load_probabilities(
    test_df: pd.DataFrame, models_dir: Optional[Path] = None
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Load artifacts and score the held-out partition."""
    clf, scaler, metadata = _load_artifacts(models_dir)
    classes = list(getattr(clf, "classes_", metadata.get("classes", [])))
    if not classes:
        raise ValueError("Classifier exposes no classes_ and metadata declares none.")

    feature_names = metadata.get("feature_names") or [
        c for c in test_df.columns if c not in IGNORE_COLUMNS
    ]
    missing = set(feature_names) - set(test_df.columns)
    if missing:
        raise ValueError(f"Test partition is missing required features: {sorted(missing)}")

    x = scaler.transform(test_df[feature_names].values)
    return clf.predict_proba(x), classes, metadata


def fit_operating_point(
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    models_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    _persist: bool = True,
    _include_ladder: bool = True,
) -> Dict[str, Any]:
    """Fit and persist the abstention gate. Returns a result envelope."""
    data_path = Path(data_dir) if data_dir is not None else DATA_DIR
    test_csv = data_path / "test_dataset.csv"
    # Same integrity rule as training: an operating point fitted to fabricated
    # or fixture data would be a fabricated safety guarantee.
    require_real_dataset(
        dataset_id="mit_bih_arrhythmia",
        task="selective_prediction_operating_point",
        records=[p for p in (test_csv,) if p.exists()],
        location=data_path,
        download_hint=(
            "python training/download_datasets.py --dataset mit_bih_arrhythmia "
            "&& python training/create_splits.py"
        ),
    )

    test_df = pd.read_csv(test_csv)
    probabilities, classes, metadata = _load_probabilities(test_df, models_dir)
    labels = test_df["label"].astype(str).tolist()
    records = test_df["record_id"].astype(str).tolist()

    jackknife = record_wise_jackknife(
        labels, probabilities, classes, records, target_precision=target_precision
    )
    threshold = jackknife["installed_threshold"]

    # Deployment behaviour: one threshold over the entire held-out partition.
    if threshold is None:
        # No fold reached the target, so nothing is reportable: everything abstains.
        measured = {
            "threshold": None,
            "reported_beats": 0,
            "total_beats": len(labels),
            "coverage": 0.0,
            "accuracy_reported": None,
            "precision_reported": None,
            "macro_f1_reported": None,
            "errors_reported": None,
            "abstained_beats": len(labels),
        }
    else:
        measured = operating_curve(labels, probabilities, classes, thresholds=[threshold])[0]
    full_curve = operating_curve(labels, probabilities, classes, thresholds=DEFAULT_THRESHOLDS)

    guard_ok = True
    guard_reason: Optional[str] = None
    if not jackknife["all_folds_met_target"]:
        guard_ok = False
        unmet = [f["held_out_record"] for f in jackknife["folds"] if f["threshold"] is None]
        guard_reason = (
            f"Leave-one-record-out selection could not reach precision {target_precision:.4f} "
            f"for held-out records {unmet}; no single threshold is justified at this target."
        )
    elif (measured["precision_reported"] or 0.0) < target_precision - 1e-12:
        guard_ok = False
        guard_reason = (
            f"Installed threshold {threshold:.4f} achieves precision "
            f"{measured['precision_reported']:.4f} on the held-out partition, below the "
            f"requested {target_precision:.4f}. Requested precision is not supported by "
            "this model; lower the target or improve the model."
        )
    elif measured["coverage"] < min_coverage:
        guard_ok = False
        guard_reason = (
            f"Best achievable coverage at precision >= {target_precision:.4f} is "
            f"{measured['coverage']:.4f}, below the {min_coverage:.2f} guard rail."
        )
    elif measured["reported_beats"] < MIN_USEFUL_BEATS:
        guard_ok = False
        guard_reason = (
            f"Only {measured['reported_beats']} beats remain at this threshold "
            f"(minimum {MIN_USEFUL_BEATS})."
        )

    meets_target = bool(
        jackknife["all_folds_met_target"]
        and measured["precision_reported"] is not None
        and measured["precision_reported"] >= target_precision - 1e-12
    )

    op = OperatingPoint(
        threshold=float(threshold) if threshold is not None else 0.0,
        target_precision=target_precision,
        coverage=float(measured["coverage"] or 0.0),
        reported_beats=int(measured["reported_beats"]),
        total_beats=int(measured["total_beats"]),
        precision=float(measured["precision_reported"] or 0.0),
        accuracy_reported=float(measured["accuracy_reported"] or 0.0),
        macro_f1_reported=float(measured["macro_f1_reported"] or 0.0),
        classes_reported=sorted(set(labels)),
        meets_target=meets_target,
        guard_rail_ok=guard_ok,
        guard_rail_reason=guard_reason,
        selection_bias=(
            "Threshold selected by leave-one-record-out over records "
            f"{sorted(set(records))}, then measured on the whole partition at that single "
            "installed threshold. No record was measured with a threshold chosen using "
            "its own beats. Residual optimism remains because a single corpus of a few "
            "records cannot separate selection from evaluation perfectly; re-fit on a "
            "larger dataset before any clinical use."
        ),
        source_dataset=str(metadata.get("dataset", "MIT-BIH Arrhythmia Database")),
        evaluated_records=sorted(set(records)),
        provenance={
            "target_precision": target_precision,
            "min_coverage": min_coverage,
            "protocol": "leave_one_record_out_threshold_selection",
            "model_classes": classes,
            "evaluated_beats": len(labels),
            "jackknife": jackknife,
            "measured_at_installed_threshold": measured,
            "full_curve": full_curve,
            "artifact_source": str(
                (Path(models_dir) if models_dir is not None else MODELS_DIR)
                / "production" / "classifier.pkl"
            ),
        },
    )

    out_path: Optional[Path] = None
    if _persist:
        out_path = save_operating_point(
            op,
            (Path(models_dir) if models_dir is not None else MODELS_DIR)
            / "operating_point.json",
        )

    return {
        "status": "FITTED" if op.reportable else "NOT_REPORTABLE",
        "operating_point": op.to_dict(),
        "measured": measured,
        "jackknife": jackknife,
        "ladder": [
            fit_operating_point(
                target_precision=target,
                min_coverage=min_coverage,
                models_dir=models_dir,
                data_dir=data_dir,
                _persist=False,
                _include_ladder=False,
            )["operating_point"]
            for target in LADDER_TARGETS
        ]
        if _include_ladder
        else [],
        "path": str(out_path) if out_path is not None else None,
        "summary": summarise(op),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the selective-prediction operating point")
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    args = parser.parse_args()

    print("=" * 80)
    print("            ECG GUARDIAN - SELECTIVE PREDICTION OPERATING POINT               ")
    print("=" * 80)

    result = fit_operating_point(
        target_precision=args.target_precision, min_coverage=args.min_coverage
    )
    op = result["operating_point"]

    print(f"[*] Target precision      : {op['target_precision']}")
    print(f"[*] Evaluated records     : {op['evaluated_records']}")
    print(f"[*] Protocol              : {op['provenance']['protocol']}")
    print("    leave-one-record-out thresholds:")
    for fold in result["jackknife"]["folds"]:
        shown = "no threshold reached target" if fold["threshold"] is None else f"{fold['threshold']:.4f}"
        print(f"      hold out {fold['held_out_record']:>4} (learn from {fold['selection_records']}) -> {shown}")
    installed = result["jackknife"]["installed_threshold"]
    print(f"[*] Installed threshold   : {'NONE' if installed is None else f'{installed:.4f}'}")
    print(f"[*] Measured precision    : {op['precision']:.4f}")
    print(f"[*] Measured coverage     : {op['coverage']:.4f} "
          f"({op['reported_beats']}/{op['total_beats']} beats)")
    print(f"[*] Status                : {result['status']}")
    if op.get("guard_rail_reason"):
        print(f"[!] Guard rail            : {op['guard_rail_reason']}")
    print(f"[*] Artifact              : {result['path'] or '(nothing installed)'}")

    if result.get("ladder"):
        print()
        print("  What each precision target buys (each row re-runs the full protocol):")
        print("    target   threshold   precision   coverage   reportable")
        for row in result["ladder"]:
            reportable = row["meets_target"] and row["guard_rail_ok"]
            print(
                f"    {row['target_precision']:<8} {row['threshold']:<11.4f} "
                f"{row['precision']:<11.4f} {row['coverage']:<10.4f} "
                f"{'yes' if reportable else 'no'}"
            )
    print("=" * 80)
    print(result["summary"])
    return 0 if result["status"] == "FITTED" else 2


if __name__ == "__main__":
    sys.exit(main())
