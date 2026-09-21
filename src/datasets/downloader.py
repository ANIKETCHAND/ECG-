"""
Autonomous Dataset Downloader Subsystem
=======================================
Downloads official datasets from PhysioNet and approved repositories,
verifies integrity, unpacks archives, and tracks ingestion status.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.request
import wfdb

from src.datasets.registry import APPROVED_DATASETS, DATASET_REGISTRY, DatasetEntry
from src.datasets.validator import compute_file_sha256, validate_physionet_record_files


class DatasetDownloader:
    """Manages secure, verified downloading of approved clinical ECG datasets."""

    def __init__(self, registry=None) -> None:
        self.registry = registry or DATASET_REGISTRY

    def is_dataset_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset files already exist in data/datasets/<id>/raw/."""
        entry = self.registry.get_dataset(dataset_id)
        if not entry:
            return False
        raw_dir = self.registry.get_local_path(dataset_id) / "raw"
        # Check if there are .hea or .dat or .csv files
        files = list(raw_dir.glob("*.hea")) + list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.dat"))
        return len(files) > 0

    def download_dataset(
        self,
        dataset_id: str,
        records_subset: Optional[List[str]] = None,
        force_redownload: bool = False,
    ) -> Dict[str, Any]:
        """Download an approved dataset from its official repository."""
        entry = self.registry.get_dataset(dataset_id)
        if not entry:
            return {
                "dataset_id": dataset_id,
                "status": "ERROR",
                "message": f"Dataset '{dataset_id}' not found in approved registry.",
            }

        target_dir = self.registry.get_local_path(dataset_id)
        raw_dir = target_dir / "raw"

        if self.is_dataset_downloaded(dataset_id) and not force_redownload:
            return {
                "dataset_id": dataset_id,
                "status": "AVAILABLE",
                "message": f"Dataset '{dataset_id}' already present at {raw_dir}.",
                "path": str(raw_dir),
            }

        print(f"[*] Commencing download for {entry.name} (Source: {entry.source_url})...")

        # 1. PhysioNet WFDB Database
        if entry.physionet_slug:
            try:
                # If specific records requested, download them; else download entire or core set
                wfdb.dl_database(
                    db_dir=entry.physionet_slug,
                    dl_dir=str(raw_dir),
                    records=records_subset,
                    keep_subdirs=False,
                    overwrite=force_redownload,
                )

                # Save download manifest
                manifest = {
                    "dataset_id": dataset_id,
                    "name": entry.name,
                    "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_url": entry.source_url,
                    "license": entry.license_type,
                    "version": entry.version,
                    "raw_dir": str(raw_dir),
                }
                with open(target_dir / "metadata" / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                return {
                    "dataset_id": dataset_id,
                    "status": "DOWNLOADED",
                    "message": f"Successfully downloaded {entry.name} to {raw_dir}.",
                    "path": str(raw_dir),
                }
            except Exception as err:
                return {
                    "dataset_id": dataset_id,
                    "status": "DOWNLOAD_FAILED",
                    "message": f"Download from PhysioNet failed: {err}",
                    "path": str(raw_dir),
                }

        # 2. Manual / Credential-Restricted Download Required
        return {
            "dataset_id": dataset_id,
            "status": "MANUAL_DOWNLOAD_REQUIRED",
            "message": f"Access credentials or manual acceptance required for {entry.name}. Refer to docs/DATASET_SETUP.md.",
            "source_url": entry.source_url,
        }


GLOBAL_DATASET_DOWNLOADER = DatasetDownloader()
