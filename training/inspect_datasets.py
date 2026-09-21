"""
Command-Line Utility for Automated Dataset Inspection
=====================================================
Usage:
    python training/inspect_datasets.py --dataset mit_bih_arrhythmia
    python training/inspect_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets.inspector import DatasetInspector
from datasets.registry import APPROVED_DATASETS, DATASET_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Automated ECG Dataset Inspector")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mit_bih_arrhythmia",
        help="Dataset identifier to inspect, or 'all'.",
    )
    args = parser.parse_args()

    inspector = DatasetInspector()

    if args.dataset.lower() == "all":
        target_ids = list(APPROVED_DATASETS.keys())
    else:
        target_ids = [args.dataset.lower().strip()]

    print("================================================================================")
    print("                     ECG GUARDIAN — DATASET INSPECTOR                          ")
    print("================================================================================")

    for d_id in target_ids:
        entry = DATASET_REGISTRY.get_dataset(d_id)
        if not entry:
            print(f"[-] Unknown dataset: '{d_id}'.")
            continue

        raw_dir = DATASET_REGISTRY.get_local_path(d_id) / "raw"
        # Fallback to data/raw for mit_bih_arrhythmia if not yet copied to data/datasets/
        if d_id == "mit_bih_arrhythmia" and not list(raw_dir.glob("*.hea")):
            legacy_raw = ROOT_DIR / "data" / "raw"
            if list(legacy_raw.glob("*.hea")):
                raw_dir = legacy_raw

        hea_files = list(raw_dir.glob("*.hea"))
        if not hea_files:
            print(f"\n[-] Dataset '{d_id}' has no .hea files at {raw_dir}. Run download first.")
            continue

        print(f"\n[*] Inspecting: {entry.name} [{d_id}] ({len(hea_files)} records)...")
        report = inspector.inspect_dataset(d_id, raw_dir)

        print(f"    Records Inspected   : {report.records_inspected}")
        print(f"    Patients Identified : {report.patients_count}")
        print(f"    Sampling Rates (Hz) : {report.sampling_rates}")
        print(f"    Leads Present       : {report.lead_names}")
        print(f"    Signal Duration     : {report.total_duration_hours} hours")
        print(f"    Annotations Count   : {sum(report.annotations_summary.values()):,} beats")
        print(f"    Reports Written     : reports/datasets/{d_id}_inspection.json / .md")

    print("\n================================================================================")
    print("                        INSPECTION COMPLETE                                    ")
    print("================================================================================")


if __name__ == "__main__":
    main()
