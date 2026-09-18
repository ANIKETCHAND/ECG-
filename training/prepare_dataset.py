"""
Dataset Preparation for ECG Classification
=============================================

Phase 2: Setup MIT-BIH Arrhythmia Database and prepare ML-ready dataset.

Downloads and processes ECG data from MIT-BIH, extracts annotations,
performs preprocessing, detects R-peaks, segments heartbeats, extracts
features, assigns labels, and saves processed data for ML training.

Research/educational use only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    RECORDS,
    SAMPLING_RATE_HZ,
    load_record,
    load_annotations,
    get_available_records,
    download_record,
)
from preprocessing import preprocess_pipeline
from peak_detection import detect_r_peaks
from segmentation import extract_beats, estimate_heart_rate
from feature_extraction import extract_all_features

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Local AAMI-style beat groups used for labeling
NORMAL_SYMBOLS = {"N", "L", "R", "e"}
PVC_SYMBOLS = {"V", "W"}

WINDOW_PRE_SEC = 0.2   # seconds before R-peak
WINDOW_POST_SEC = 0.4  # seconds after R-peak


def setup_mitdb(download_missing: bool = False) -> list[str]:
    """Ensure MIT-BIH records are present locally."""
    available = get_available_records(RAW_DIR)
    print(f"Already downloaded: {len(available)}/{len(RECORDS)} records")

    if download_missing and len(available) < len(RECORDS):
        missing = [r for r in RECORDS if r not in available]
        print(f"Downloading {len(missing)} missing records...")
        downloaded, failed = download_records(missing)
        print(f"Downloaded: {len(downloaded)}, Failed: {len(failed)}")
        if failed:
            print(f"Failed records: {failed}")

    available = get_available_records(RAW_DIR)
    if not available:
        raise ValueError("No MIT-BIH records found in data/raw")
    return available


def download_records(record_ids: list[str]) -> tuple[list[str], list[str]]:
    """Download records, returning (downloaded, failed)."""
    downloaded: list[str] = []
    failed: list[str] = []
    for record_id in record_ids:
        try:
            ok = download_record(record_id, RAW_DIR)
            (downloaded if ok else failed).append(record_id)
        except Exception as exc:
            print(f"  Error downloading {record_id}: {exc}")
            failed.append(record_id)
    return downloaded, failed


def label_symbol(symbol: str) -> str:
    """Map annotation symbol to a simple label."""
    if symbol in NORMAL_SYMBOLS:
        return "Normal"
    if symbol in PVC_SYMBOLS:
        return "PVC"
    return "Other"


def process_record(record_id: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load, preprocess, detect R-peaks, segment, and label beats for one record."""
    print(f"  Processing {record_id}...", end=" ")

    signal, fs, _ = load_record(record_id, RAW_DIR)
    annotations = load_annotations(record_id, RAW_DIR)

    # Preprocess
    processed = preprocess_pipeline(signal, fs)

    # Detect R-peaks (use annotation samples as ground-truth peaks)
    r_peaks = annotations["sample_index"].values.astype(int)

    # Segment beats around R-peaks
    beats, valid_indices = extract_beats(
        processed, r_peaks, fs,
        pre_window=WINDOW_PRE_SEC,
        post_window=WINDOW_POST_SEC,
    )

    # Build labels DataFrame using annotations for valid beats
    valid_annots = annotations.iloc[valid_indices]
    beat_df = pd.DataFrame({
        "record_id": record_id,
        "sample_index": valid_annots["sample_index"].values,
        "symbol": valid_annots["symbol"].values,
        "label": [label_symbol(s) for s in valid_annots["symbol"]],
    })

    print(f"extracted {len(beats)} beats")
    return beats, beat_df


def prepare_dataset(
    record_ids: list[str] | None = None,
    download_missing: bool = False,
) -> dict:
    """End-to-end dataset preparation for a list of records.

    Returns a dict with 'features', 'labels', 'metadata', and 'class_distribution'.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    available = setup_mitdb(download_missing=download_missing)
    if record_ids is not None:
        available = [r for r in available if r in record_ids]
    if not available:
        raise ValueError("No records available to process")

    print(f"\nProcessing {len(available)} record(s): {available}")

    all_beats: list[np.ndarray] = []
    all_labels = pd.DataFrame()

    for record_id in available:
        beats, beat_df = process_record(record_id)
        all_beats.append(beats)
        all_labels = pd.concat([all_labels, beat_df], ignore_index=True)

    beats_array = np.vstack(all_beats)

    # Feature extraction
    print(f"Extracting features from {len(beats_array)} beats...")
    feature_dict = extract_features_batch(beats_array, SAMPLING_RATE_HZ)
    features_df = pd.DataFrame(feature_dict)

    # Assemble final dataset
    dataset = pd.concat([features_df, all_labels], axis=1)

    # Save
    output_csv = PROCESSED_DIR / "ecg_dataset.csv"
    dataset.to_csv(output_csv, index=False)
    print(f"Saved dataset to {output_csv}  ({len(dataset)} rows, {len(dataset.columns)} cols)")

    # Metadata
    metadata = {
        "records_processed": available,
        "total_beats": len(dataset),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_pre_sec": WINDOW_PRE_SEC,
        "window_post_sec": WINDOW_POST_SEC,
        "feature_columns": features_df.columns.tolist(),
        "class_distribution": dataset["label"].value_counts().to_dict(),
    }
    with open(PROCESSED_DIR / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\nClass distribution:")
    for label, count in metadata["class_distribution"].items():
        print(f"  {label}: {count}")

    return metadata


def extract_features_batch(beats: np.ndarray, fs: float) -> pd.DataFrame:
    """Extract features for every beat in the batch."""
    rows = []
    for i in range(len(beats)):
        rows.append(extract_all_features(beats[i], fs))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ECG dataset from MIT-BIH")
    parser.add_argument("--download", action="store_true", help="Download missing records")
    parser.add_argument("--records", nargs="*", default=None, help="Specific record IDs")
    args = parser.parse_args()

    try:
        metadata = prepare_dataset(record_ids=args.records, download_missing=args.download)
        print("\nDataset preparation complete.")
        print(f"Records: {metadata['records_processed']}")
        print(f"Beats: {metadata['total_beats']}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()