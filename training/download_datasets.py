"""
Command-Line Utility for Autonomous Dataset Download
=====================================================
Usage:
    python training/download_datasets.py --dataset all
    python training/download_datasets.py --dataset mit_bih_arrhythmia
    python training/download_datasets.py --dataset mit_bih_svdb
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets.downloader import DatasetDownloader
from datasets.registry import APPROVED_DATASETS, DATASET_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Autonomous ECG Dataset Downloader")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mit_bih_arrhythmia",
        help="Dataset identifier to download, or 'all' for all approved datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset files already exist.",
    )
    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.dataset.lower() == "all":
        target_ids = list(APPROVED_DATASETS.keys())
    else:
        target_ids = [args.dataset.lower().strip()]

    print("================================================================================")
    print("                     ECG GUARDIAN — DATASET DOWNLOADER                         ")
    print("================================================================================")

    for d_id in target_ids:
        entry = DATASET_REGISTRY.get_dataset(d_id)
        if not entry:
            print(f"[-] Unknown dataset: '{d_id}'. Run --help or check src/datasets/registry.py.")
            continue

        print(f"\n[*] Processing: {entry.name} [{d_id}]")
        res = downloader.download_dataset(d_id, force_redownload=args.force)

        print(f"    Status : {res['status']}")
        print(f"    Message: {res['message']}")
        if "path" in res:
            print(f"    Path   : {res['path']}")

    print("\n================================================================================")
    print("                        DOWNLOAD SEQUENCE COMPLETE                             ")
    print("================================================================================")


if __name__ == "__main__":
    main()
