"""
Command-Line Utility for Patient-Level Dataset Splitting
========================================================
Partitions records strictly by patient ID, guaranteeing zero data leakage.
Saves split manifests to data/splits/<task>/<version>.json.

Usage:
    python training/create_splits.py --task beat_arrhythmia --version v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets.patient_index import resolve_patient_id
from datasets.registry import DATASET_REGISTRY
from training.splitting.patient_splitter import split_records_by_patient, verify_patient_isolation


def create_task_split(
    task: str = "beat_arrhythmia",
    version: str = "v1",
    test_ratio: float = 0.3,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> Path:
    splits_dir = ROOT_DIR / "data" / "splits" / task
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_file = splits_dir / f"{version}.json"

    # Default to MIT-BIH Arrhythmia available records
    raw_dir = DATASET_REGISTRY.get_local_path("mit_bih_arrhythmia") / "raw"
    if not list(raw_dir.glob("*.hea")):
        raw_dir = ROOT_DIR / "data" / "raw"

    hea_files = sorted([f.stem for f in raw_dir.glob("*.hea")])
    if not hea_files:
        raise FileNotFoundError(f"No records found in {raw_dir} to split.")

    # Group records by patient
    patient_to_records = {}
    for rec in hea_files:
        pid = resolve_patient_id("mit_bih_arrhythmia", rec)
        if pid not in patient_to_records:
            patient_to_records[pid] = []
        patient_to_records[pid].append(rec)

    unique_patients = sorted(list(patient_to_records.keys()))
    train_pts, val_pts, test_pts = split_records_by_patient(
        unique_patients,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    verify_patient_isolation(train_pts, test_pts, val_pts)

    train_recs = [r for pid in train_pts for r in patient_to_records[pid]]
    val_recs = [r for pid in val_pts for r in patient_to_records[pid]]
    test_recs = [r for pid in test_pts for r in patient_to_records[pid]]

    manifest = {
        "task": task,
        "split_version": version,
        "random_seed": random_seed,
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "train": {
            "patients": train_pts,
            "records": train_recs,
        },
        "validation": {
            "patients": val_pts,
            "records": val_recs,
        },
        "test": {
            "patients": test_pts,
            "records": test_recs,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Create Patient-Level Split Manifests")
    parser.add_argument("--task", type=str, default="beat_arrhythmia")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("================================================================================")
    print("                     ECG GUARDIAN — PATIENT SPLITTER                           ")
    print("================================================================================")

    out_path = create_task_split(
        task=args.task,
        version=args.version,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )

    with open(out_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    print(f"[*] Task           : {m['task']}")
    print(f"[*] Split Version  : {m['split_version']}")
    print(f"[*] Train Patients : {len(m['train']['patients'])} (Records: {m['train']['records']})")
    print(f"[*] Val Patients   : {len(m['validation']['patients'])} (Records: {m['validation']['records']})")
    print(f"[*] Test Patients  : {len(m['test']['patients'])} (Records: {m['test']['records']})")
    print(f"[*] Saved Manifest : {out_path}")
    print("================================================================================")


if __name__ == "__main__":
    main()
