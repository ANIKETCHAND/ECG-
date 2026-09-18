"""
Train/Test Split Module
=======================

Implements patient/record-level train/test split to strictly prevent data leakage.
Extracts heartbeat features aligned with expert annotations and generates
separate training and evaluation sets.

Research/educational use only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    get_available_records,
    load_annotations,
    load_record,
)
from feature_extraction import extract_all_features
from label_mapping import is_beat_annotation, map_symbol_to_class
from preprocessing import preprocess_pipeline

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def process_record_annotated(
    record_id: str,
    raw_dir: Path,
    pre_window: float = 0.2,
    post_window: float = 0.4,
) -> Tuple[pd.DataFrame, Dict]:
    """Process one record using expert beat annotations with local peak refinement.

    Args:
        record_id: Record identifier string (e.g. '100')
        raw_dir: Path to raw data folder
        pre_window: Seconds before R-peak (0.2s = 72 samples at 360Hz)
        post_window: Seconds after R-peak (0.4s = 144 samples at 360Hz)

    Returns:
        Tuple of (features_df, metadata_dict)
    """
    signal, fs, sample_count = load_record(record_id, raw_dir)
    annotations = load_annotations(record_id, raw_dir)

    # Preprocess ECG signal (baseline removal, bandpass filter, z-score normalization)
    processed = preprocess_pipeline(signal, fs)

    # Filter for cardiac beat annotations (discard non-beat comments '+', '~', etc.)
    beat_mask = [is_beat_annotation(sym) for sym in annotations["symbol"]]
    beat_ann = annotations[beat_mask].reset_index(drop=True)

    if len(beat_ann) == 0:
        return pd.DataFrame(), {}

    samples = beat_ann["sample_index"].values.astype(int)
    symbols = beat_ann["symbol"].values

    pre_samples = int(pre_window * fs)
    post_samples = int(post_window * fs)
    win_len = pre_samples + post_samples + 1

    # Pre-calculate RR intervals (in seconds)
    rr_intervals = np.diff(samples) / fs
    mean_rr = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.8

    rows = []
    class_counts: Dict[str, int] = {}

    for i in range(len(samples)):
        center = samples[i]
        sym = symbols[i]
        label = map_symbol_to_class(sym, mode="3class")

        if label == "Unknown":
            continue

        # Local peak refinement within ±10 samples (~28 ms)
        search_start = max(0, center - 10)
        search_end = min(len(processed), center + 11)
        refined_center = search_start + int(np.argmax(np.abs(processed[search_start:search_end])))

        start = refined_center - pre_samples
        end = refined_center + post_samples

        # Boundary padding if near beginning or end of recording
        if start < 0:
            beat = np.zeros(win_len)
            beat[-start:] = processed[:end + 1]
        elif end >= len(processed):
            beat = np.zeros(win_len)
            valid_len = len(processed[start:])
            beat[:valid_len] = processed[start:]
        else:
            beat = processed[start:end + 1]

        # Calculate rhythm features
        pre_rr = float(rr_intervals[i - 1]) if i > 0 else mean_rr
        post_rr = float(rr_intervals[i]) if i < len(rr_intervals) else mean_rr
        ratio = pre_rr / mean_rr if mean_rr > 0 else 1.0

        # Extract features
        feat = extract_all_features(
            beat,
            fs,
            pre_rr=pre_rr,
            post_rr=post_rr,
            local_rr_ratio=ratio,
        )
        feat["label"] = label
        feat["record_id"] = record_id
        feat["symbol"] = sym
        rows.append(feat)

        class_counts[label] = class_counts.get(label, 0) + 1

    df = pd.DataFrame(rows)
    metadata = {
        "record_id": record_id,
        "total_samples": sample_count,
        "fs": fs,
        "total_beats": len(df),
        "class_counts": class_counts,
    }
    return df, metadata


def create_split(
    train_records: List[str],
    test_records: List[str],
    output_train: Path = PROCESSED_DIR / "train_dataset.csv",
    output_test: Path = PROCESSED_DIR / "test_dataset.csv",
) -> Dict[str, Dict]:
    """Create and save train/test datasets with record-level splitting."""
    print("Processing Training Records...")
    train_dfs = []
    train_meta = {}
    for rid in train_records:
        print(f"  Train Record: {rid}")
        df, meta = process_record_annotated(rid, RAW_DIR)
        if len(df) > 0:
            train_dfs.append(df)
            train_meta[rid] = meta

    print("Processing Testing Records...")
    test_dfs = []
    test_meta = {}
    for rid in test_records:
        print(f"  Test Record: {rid}")
        df, meta = process_record_annotated(rid, RAW_DIR)
        if len(df) > 0:
            test_dfs.append(df)
            test_meta[rid] = meta

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    # Drop any NaNs
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    info = {
        "train": {
            "records": train_records,
            "samples": len(train_df),
            "class_distribution": train_df["label"].value_counts().to_dict(),
            "metadata": train_meta,
        },
        "test": {
            "records": test_records,
            "samples": len(test_df),
            "class_distribution": test_df["label"].value_counts().to_dict(),
            "metadata": test_meta,
        },
    }

    with open(PROCESSED_DIR / "split_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    # Also save overall dataset
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df.to_csv(PROCESSED_DIR / "ecg_dataset.csv", index=False)

    feature_cols = [c for c in train_df.columns if c not in ("label", "record_id", "symbol")]
    with open(PROCESSED_DIR / "dataset_info.json", "w") as f:
        json.dump(
            {
                "records_processed": train_records + test_records,
                "total_beats": len(all_df),
                "sampling_rate_hz": 360,
                "feature_columns": feature_cols,
                "class_distribution": all_df["label"].value_counts().to_dict(),
            },
            f,
            indent=2,
        )

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test split for ECG dataset")
    parser.add_argument("--train-records", nargs="+", default=["100", "106", "200", "213"])
    parser.add_argument("--test-records", nargs="+", default=["101", "119", "208"])
    args = parser.parse_args()

    available = get_available_records(RAW_DIR)
    train_recs = [r for r in args.train_records if r in available]
    test_recs = [r for r in args.test_records if r in available]

    print(f"Available records: {available}")
    print(f"Train records ({len(train_recs)}): {train_recs}")
    print(f"Test records ({len(test_recs)}): {test_recs}")

    info = create_split(train_recs, test_recs)

    print("\n" + "=" * 50)
    print("TRAIN SET (Record-level split):")
    print(f"  Records: {info['train']['records']}")
    print(f"  Total beats: {info['train']['samples']}")
    for k, v in info['train']['class_distribution'].items():
        print(f"    {k}: {v}")

    print("\nTEST SET (Completely unseen patient records):")
    print(f"  Records: {info['test']['records']}")
    print(f"  Total beats: {info['test']['samples']}")
    for k, v in info['test']['class_distribution'].items():
        print(f"    {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()