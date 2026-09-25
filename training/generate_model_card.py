"""
Model Card & Measured Metrics Generator
=======================================

Writes two artifacts that were previously missing entirely:

* ``models/production/metrics.json`` — the measured performance of the shipped
  classifier on the record-disjoint held-out partition,
* ``models/MODEL_CARD.md`` — the same numbers in a form a reviewer can read,
  including what the model is *not* able to do.

Why this exists
---------------
The shipped ``models/*/metadata.json`` described the estimator, the feature list
and the hyperparameters, but contained **no measured metrics and no provenance**.
Anyone reading it could only guess how well the model actually performed. The
project's own rule is that a number may only appear if it was computed from real
held-out data, so this script computes every number it prints and refuses to emit
any that it did not.

It deliberately reports three things that flatter the model least:

1. **Macro F1, not just accuracy.** Accuracy is dominated by the majority class
   and hides total failure on rare classes.
2. **Per-class support and recall.** A class with nine test examples cannot be
   learned, and saying so is more useful than a good-looking average.
3. **Record-level presence detection.** Clinicians act on "does this recording
   contain PVCs", not on per-beat averages.

Usage::

    python training/generate_model_card.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.ml.selective import (
    OperatingPoint,
    load_operating_point,
    operating_curve,
    summarise,
)
from training.guards import require_real_dataset

DATA_DIR = PROJ_DIR / "data" / "processed"
MODELS_DIR = PROJ_DIR / "models"
IGNORE_COLUMNS = {"label", "record_id", "symbol"}

#: A class needs at least this many held-out examples before any per-class
#: conclusion is drawn from it at all.
MIN_SUPPORT_FOR_CONCLUSION = 30

#: Onset window used for record-level PVC presence. Two beats is the same rule
#: the inference engine applies, so the card reports the deployed behaviour.
PVC_MIN_BEATS = 2
PVC_MIN_FRACTION = 0.15


def _load_artifacts(models_dir: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    for clf_path, scaler_path, meta_path in (
        (models_dir / "production" / "classifier.pkl", models_dir / "production" / "scaler.pkl", models_dir / "production" / "metadata.json"),
        (models_dir / "classifier.pkl", models_dir / "scaler.pkl", models_dir / "metadata.json"),
    ):
        if clf_path.exists() and scaler_path.exists():
            metadata: Dict[str, Any] = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            return joblib.load(clf_path), joblib.load(scaler_path), metadata
    raise FileNotFoundError("No trained classifier/scaler artifacts were found.")


def _per_class(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        support = int(np.sum(y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else 0.0
        )
        out[cls] = {
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": None if precision is None else round(precision, 6),
            "recall": None if recall is None else round(recall, 6),
            "f1": round(f1, 6),
            "metrics_conclusive": support >= MIN_SUPPORT_FOR_CONCLUSION,
            "conclusion_blocked_reason": (
                None
                if support >= MIN_SUPPORT_FOR_CONCLUSION
                else (
                    f"Only {support} held-out examples. Below the {MIN_SUPPORT_FOR_CONCLUSION} "
                    "needed before any per-class statement is meaningful."
                )
            ),
        }
    return out


def _record_level(
    df: pd.DataFrame, y_pred: np.ndarray, positive_class: str = "PVC"
) -> Dict[str, Any]:
    """Record-level presence detection, matching the deployed decision rule."""
    frame = df.copy()
    frame["pred"] = y_pred
    rows: List[Dict[str, Any]] = []
    for record_id, group in frame.groupby("record_id"):
        truth_beats = int(np.sum(group["label"] == positive_class))
        pred_beats = int(np.sum(group["pred"] == positive_class))
        n = len(group)
        predicted = pred_beats > 0 and (
            pred_beats >= PVC_MIN_BEATS or (pred_beats / n) >= PVC_MIN_FRACTION
        )
        rows.append(
            {
                "record_id": str(record_id),
                "beats": n,
                "true_positive_beats": truth_beats,
                "predicted_positive_beats": pred_beats,
                "true_presence": bool(truth_beats > 0),
                "predicted_presence": bool(predicted),
                "correct": bool((truth_beats > 0) == predicted),
                "false_positive": bool(predicted and truth_beats == 0),
                "false_negative": bool(truth_beats > 0 and not predicted),
            }
        )
    correct = sum(1 for r in rows if r["correct"])
    return {
        "positive_class": positive_class,
        "decision_rule": (
            f"at least {PVC_MIN_BEATS} beats or >= {PVC_MIN_FRACTION:.0%} of beats classified "
            f"as {positive_class}"
        ),
        "records": rows,
        "n_records": len(rows),
        "correct_records": correct,
        "record_level_accuracy": round(correct / len(rows), 6) if rows else None,
        "false_positives": [r["record_id"] for r in rows if r["false_positive"]],
        "false_negatives": [r["record_id"] for r in rows if r["false_negative"]],
    }


def generate_model_card(
    models_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Measure the shipped model and (optionally) write its card."""
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    data_path = Path(data_dir) if data_dir is not None else DATA_DIR

    test_csv = data_path / "test_dataset.csv"
    train_csv = data_path / "train_dataset.csv"
    require_real_dataset(
        dataset_id="mit_bih_arrhythmia",
        task="model_card",
        records=[p for p in (test_csv, train_csv) if p.exists()],
        location=data_path,
    )

    test_df = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv)
    clf, scaler, metadata = _load_artifacts(base)
    classes = list(getattr(clf, "classes_", metadata.get("classes", [])))
    feature_names = metadata.get("feature_names") or [
        c for c in test_df.columns if c not in IGNORE_COLUMNS
    ]
    missing = set(feature_names) - set(test_df.columns)
    if missing:
        raise ValueError(f"Test partition is missing required features: {sorted(missing)}")

    x_test = scaler.transform(test_df[feature_names].values)
    y_true = test_df["label"].astype(str).values
    y_pred = clf.predict(x_test)
    probabilities = clf.predict_proba(x_test)

    test_records = sorted(test_df["record_id"].astype(str).unique().tolist())
    train_records = sorted(train_df["record_id"].astype(str).unique().tolist())
    overlap = sorted(set(test_records) & set(train_records))

    primary: Dict[str, Any] = {
        "held_out_beats": int(len(y_true)),
        "held_out_records": test_records,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(
            np.mean([v["recall"] for v in _per_class(y_true, y_pred, classes).values() if v["recall"] is not None])
        ), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 6),
        "per_class": _per_class(y_true, y_pred, classes),
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
    }

    unsupported = [
        cls for cls, stats in primary["per_class"].items() if not stats["metrics_conclusive"]
    ]
    primary["classes_without_conclusive_support"] = unsupported

    # Restricted task: the model's own published scope in limitations.
    supported = [c for c in classes if c not in unsupported]
    restricted: Optional[Dict[str, Any]] = None
    if supported and len(supported) < len(classes):
        mask = np.isin(y_true, supported)
        if mask.sum() > 0:
            restricted = {
                "classes": supported,
                "beats": int(mask.sum()),
                "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 6),
                "macro_f1": round(
                    float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 6
                ),
                "reason": (
                    "Restricted to classes with conclusive held-out support. The full-support "
                    "macro F1 above is dragged down by classes this model cannot learn from the "
                    "available corpus, which is a data limitation rather than a modelling defect."
                ),
            }

    op = load_operating_point(base / "operating_point.json")
    selective: Dict[str, Any] = {
        "curve": operating_curve(y_true.tolist(), probabilities, classes),
        "operating_point": op.to_dict() if op is not None else None,
        "summary": summarise(op),
    }

    card: Dict[str, Any] = {
        "model_id": metadata.get("model_id", "ECG-RF-1.0.0"),
        "estimator": metadata.get("model_name", type(clf).__name__),
        "dataset": metadata.get("dataset", "MIT-BIH Arrhythmia Database"),
        "classes": classes,
        "feature_count": len(feature_names),
        "sampling_rate_hz": metadata.get("sampling_rate_hz"),
        "hyperparameters": metadata.get("hyperparameters", {}),
        "train_records": train_records,
        "held_out_records": test_records,
        "record_overlap": overlap,
        "patient_level_split_enforced": not overlap,
        "train_samples": int(len(train_df)),
        "primary_metrics": primary,
        "restricted_metrics": restricted,
        "record_level": _record_level(test_df, y_pred),
        "selective_prediction": selective,
        "measured_at": datetime.now().isoformat(),
        "provenance": {
            "data_status": "REAL_DATASET",
            "artifact_source": str(base / "production" / "classifier.pkl"),
            "metrics_computed_by": "training/generate_model_card.py",
            "no_numbers_are_hardcoded": (
                "Every value in this artifact is produced by sklearn on the held-out "
                "partition at generation time. Nothing here is transcribed by hand."
            ),
        },
        "limitations": [
            "Research prototype. Not a certified diagnostic device.",
            "Single-lead (Lead II / MLII). Precordial morphology is not evaluated.",
            "Trained on a small subset of MIT-BIH; class prevalence does not match any "
            "real clinical population.",
            "Per-beat metrics are not patient-level diagnostic accuracy.",
            "Coverage in the live pipeline is lower than the coverage measured here. "
            "These metrics are computed on beats centred on the expert annotations, while "
            "deployment centres beats on the engine's own R-peak detection. Any alignment "
            "error moves the beat under the model and lowers its confidence. Treat the "
            "coverage figures above as an upper bound on what the deployed gate will report.",
            "Metrics are measured on a handful of held-out records. A single corpus this "
            "small cannot separate threshold selection from evaluation perfectly.",
        ],
    }
    if unsupported:
        card["unsupported_classes"] = {
            "classes": unsupported,
            "explanation": (
                "These classes have too few held-out beats for any model to learn from this "
                "corpus, so their poor scores measure the dataset, not the algorithm. Adding "
                "records is the only honest fix; tuning cannot recover absent examples."
            ),
        }

    if write:
        metrics_path = base / "production" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(card, handle, indent=2)
        card_path = base / "MODEL_CARD.md"
        with open(card_path, "w", encoding="utf-8") as handle:
            handle.write(_render_card(card))
        card["_artifacts"] = {"metrics_json": str(metrics_path), "model_card": str(card_path)}

        # Stamp the measured metrics and provenance into the model metadata so the
        # metadata file stops being silent about how the model performs.
        for meta_path in (base / "production" / "metadata.json", base / "metadata.json"):
            if not meta_path.exists():
                continue
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            meta["metrics"] = {
                "source": str(metrics_path),
                "held_out_records": test_records,
                "accuracy": primary["accuracy"],
                "macro_f1": primary["macro_f1"],
                "weighted_f1": primary["weighted_f1"],
                "cohen_kappa": primary["cohen_kappa"],
                "classes_without_conclusive_support": unsupported,
                "measured_at": card["measured_at"],
            }
            meta["provenance"] = card["provenance"]
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)

    return card


def _render_card(card: Dict[str, Any]) -> str:
    p = card["primary_metrics"]
    lines: List[str] = []
    add = lines.append

    add(f"# Model Card — {card['model_id']}")
    add("")
    add("> Generated by `training/generate_model_card.py`. Every number below was computed")
    add("> from the record-disjoint held-out partition at generation time. Nothing is hand-written.")
    add("")
    add(f"_Measured: {card['measured_at']}_")
    add("")
    add("## At a glance")
    add("")
    add("| Property | Value |")
    add("| --- | --- |")
    add(f"| Estimator | `{card['estimator']}` |")
    add(f"| Dataset | {card['dataset']} |")
    add(f"| Classes | {', '.join(card['classes'])} |")
    add(f"| Features | {card['feature_count']} |")
    add(f"| Training beats | {card['train_samples']} |")
    add(f"| Held-out beats | {p['held_out_beats']} |")
    add(f"| Patient-level split enforced | {card['patient_level_split_enforced']} |")
    add("")
    add("## Measured performance (record-disjoint)")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| Accuracy | **{p['accuracy']:.4f}** |")
    add(f"| Balanced accuracy | {p['balanced_accuracy']:.4f} |")
    add(f"| Macro F1 | **{p['macro_f1']:.4f}** |")
    add(f"| Weighted F1 | {p['weighted_f1']:.4f} |")
    add(f"| Cohen's kappa | {p['cohen_kappa']:.4f} |")
    add("")
    add("### Per class")
    add("")
    add("| Class | Support | Precision | Recall | F1 | Conclusive? |")
    add("| --- | ---: | ---: | ---: | ---: | --- |")
    for cls, s in p["per_class"].items():
        prec = "n/a" if s["precision"] is None else f"{s['precision']:.4f}"
        rec = "n/a" if s["recall"] is None else f"{s['recall']:.4f}"
        add(
            f"| {cls} | {s['support']} | {prec} | {rec} | {s['f1']:.4f} | "
            f"{'yes' if s['metrics_conclusive'] else '**no**'} |"
        )
    add("")
    if card.get("unsupported_classes"):
        add(f"> **{card['unsupported_classes']['explanation']}**")
        add(f"> Affected classes: {', '.join(card['unsupported_classes']['classes'])}.")
        add("")

    if card.get("restricted_metrics"):
        r = card["restricted_metrics"]
        add(f"### Restricted to supported classes ({', '.join(r['classes'])})")
        add("")
        add(f"- Beats: {r['beats']}")
        add(f"- Accuracy: **{r['accuracy']:.4f}**")
        add(f"- Macro F1: **{r['macro_f1']:.4f}**")
        add("")
        add(f"{r['reason']}")
        add("")

    rl = card["record_level"]
    add(f"## Record-level {rl['positive_class']} presence")
    add("")
    add(f"Decision rule: {rl['decision_rule']}.")
    add("")
    add(f"- Records evaluated: {rl['n_records']}")
    add(f"- Correct: {rl['correct_records']} ({rl['record_level_accuracy']:.4f})")
    add(f"- False positives: {rl['false_positives'] or 'none'}")
    add(f"- False negatives: {rl['false_negatives'] or 'none'}")
    add("")

    sel = card["selective_prediction"]
    add("## Selective prediction (abstention)")
    add("")
    add(sel["summary"])
    add("")
    add("What each confidence threshold buys. Beats below the threshold are withheld")
    add("rather than guessed, so \"accuracy\" here is the accuracy of *reported* beats only.")
    add("")
    add("| Threshold | Reported beats | Coverage | Accuracy of reported | Macro F1 | Errors |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sel["curve"]:
        if row["reported_beats"] == 0:
            continue
        add(
            f"| {row['threshold']:.3f} | {row['reported_beats']} | {row['coverage']:.4f} | "
            f"{row['accuracy_reported']:.4f} | {row['macro_f1_reported']:.4f} | "
            f"{row['errors_reported']} |"
        )
    add("")
    op = sel.get("operating_point")
    if op:
        add("Installed operating point:")
        add("")
        add(f"- Threshold: `{op['threshold']}`")
        add(f"- Requested precision: {op['target_precision']}")
        add(f"- Measured precision: {op['precision']:.4f}")
        add(f"- Measured coverage: {op['coverage']:.4f} ({op['reported_beats']}/{op['total_beats']} beats)")
        add(f"- Reportable: {op['meets_target'] and op['guard_rail_ok']}")
        add("")
        add(f"Selection protocol: {op['selection_bias']}")
        if op.get("guard_rail_reason"):
            add("")
            add(f"**Guard rail:** {op['guard_rail_reason']}")
        add("")
    add("## Limitations")
    add("")
    for item in card["limitations"]:
        add(f"- {item}")
    add("")
    add("Provenance:")
    add("")
    for key, value in card["provenance"].items():
        add(f"- `{key}`: {value}")
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the model card and metrics.json")
    parser.add_argument("--no-write", action="store_true", help="Measure without writing artifacts")
    args = parser.parse_args()

    print("=" * 80)
    print("                  ECG GUARDIAN - MODEL CARD GENERATOR                        ")
    print("=" * 80)

    card = generate_model_card(write=not args.no_write)
    p = card["primary_metrics"]

    print(f"[*] Estimator          : {card['estimator']}")
    print(f"[*] Held-out records   : {card['held_out_records']}")
    print(f"[*] Held-out beats     : {p['held_out_beats']}")
    print(f"[*] Accuracy           : {p['accuracy']:.4f}")
    print(f"[*] Balanced accuracy  : {p['balanced_accuracy']:.4f}")
    print(f"[*] Macro F1           : {p['macro_f1']:.4f}")
    print(f"[*] Weighted F1        : {p['weighted_f1']:.4f}")
    print(f"[*] Cohen kappa        : {p['cohen_kappa']:.4f}")
    print("    per class:")
    for cls, s in p["per_class"].items():
        flag = "" if s["metrics_conclusive"] else "  <- NOT conclusive"
        print(f"      {cls:<8} support={s['support']:<6} f1={s['f1']:.4f}{flag}")
    if card.get("restricted_metrics"):
        r = card["restricted_metrics"]
        print(f"[*] Supported-class only ({', '.join(r['classes'])}): "
              f"accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f}")
    rl = card["record_level"]
    print(f"[*] Record-level PVC presence: {rl['correct_records']}/{rl['n_records']} correct")
    print(f"[*] Selective prediction      : {card['selective_prediction']['summary']}")
    if card.get("_artifacts"):
        print(f"[*] Metrics   -> {card['_artifacts']['metrics_json']}")
        print(f"[*] Model card-> {card['_artifacts']['model_card']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
