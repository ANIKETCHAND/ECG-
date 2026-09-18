"""MIT-BIH Arrhythmia Database loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

MIT_BIH_URL = "https://physionet.org/content/mitdb/1.0.0/"
SAMPLING_RATE_HZ = 360
RECORDS: Tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
    "213", "214", "215", "217", "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234",
)

# AAMI-style groups are implemented in Phase 7. These aliases keep the loader useful.
ANNOTATION_DESCRIPTIONS: Dict[str, str] = {
    "N": "Normal beat",
    "L": "Left bundle branch block beat",
    "R": "Right bundle branch block beat",
    "e": "Junctional escape beat",
    "j": "Nodal (junctional) premature beat",
    "A": "Atrial premature beat",
    "a": "Aberrant atrial premature beat",
    "J": "Nodal (junctional) premature beat",
    "S": "Supraventricular premature or ectopic beat",
    "V": "Premature ventricular contraction",
    "W": "Ventricular escape beat",
    "F": "Fusion of ventricular and normal beat",
    "Q": "Unclassifiable beat",
    "P": "Paced beat",
    "p": "Paced beat that failed to capture",
    "T": "Fusion of paced and normal beat",
    "/": "Pacemaker pulse annotation",
    "!": "Ventricular flutter wave",
    "[": "Start of ventricular flutter/fibrillation",
    "]": "End of ventricular flutter/fibrillation",
    "x": "Non-conducted P-wave (blocked APB)",
    "f": "Atrial fibrillation",
    "|": "Isolated QRS-like artifact",
}


def _record_path(data_dir: str | Path, record_id: str) -> Path:
    return Path(data_dir) / record_id


def record_is_available(data_dir: str | Path, record_id: str) -> bool:
    """Return whether the signal, header, and rhythm annotations exist."""
    base = _record_path(data_dir, record_id)
    return all(base.with_suffix(extension).is_file() for extension in (".hea", ".dat", ".atr"))


def get_available_records(data_dir: str | Path) -> List[str]:
    """Return known MIT-BIH records that are complete on disk."""
    return [record_id for record_id in RECORDS if record_is_available(data_dir, record_id)]


def download_record(
    record_id: str,
    data_dir: str | Path,
    *,
    overwrite: bool = False,
) -> bool:
    """Download one MIT-BIH record and its ``.atr`` annotations.

    Files are placed directly in ``data_dir`` so the project layout stays simple.
    """
    if record_id not in RECORDS:
        raise ValueError(f"Unknown MIT-BIH record: {record_id}")

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if not overwrite and record_is_available(destination, record_id):
        return True

    try:
        wfdb.dl_database(
            db_dir="mitdb",
            dl_dir=str(destination),
            records=[record_id],
            annotators=["atr"],
            keep_subdirs=False,
            overwrite=overwrite,
        )
    except Exception as exc:
        print(f"Could not download record {record_id}: {exc}")
        return False

    return record_is_available(destination, record_id)


def download_records(
    record_ids: Iterable[str],
    data_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Tuple[List[str], List[str]]:
    """Download records and return ``(downloaded, failed)`` lists."""
    requested = list(record_ids)
    downloaded: List[str] = []
    failed: List[str] = []

    for record_id in requested:
        if download_record(record_id, data_dir, overwrite=overwrite):
            downloaded.append(record_id)
        else:
            failed.append(record_id)

    return downloaded, failed


def load_record(
    record_id: str,
    data_dir: str | Path,
    *,
    channel: int = 0,
) -> Tuple[np.ndarray, float, int]:
    """Load one ECG channel from a local MIT-BIH record.

    Returns:
        A tuple of ``(signal, sampling_rate_hz, sample_count)``.
    """
    record = wfdb.rdrecord(str(_record_path(data_dir, record_id)), channels=[channel])
    signal = np.asarray(record.p_signal, dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError("The selected channel must contain one signal")
    return signal, float(record.fs), int(signal.size)


def load_annotations(record_id: str, data_dir: str | Path) -> pd.DataFrame:
    """Load beat annotations into a normalized DataFrame."""
    annotation = wfdb.rdann(
        str(_record_path(data_dir, record_id)),
        "atr",
        return_label_elements=["symbol", "description"],
    )

    return pd.DataFrame(
        {
            "record_id": record_id,
            "sample_index": np.asarray(annotation.sample, dtype=np.int64),
            "symbol": np.asarray(annotation.symbol),
            "description": np.asarray(annotation.description),
        }
    )


def inspect_record(record_id: str, data_dir: str | Path) -> Dict[str, object]:
    """Return concise signal and annotation metadata for one record."""
    signal, fs, sample_count = load_record(record_id, data_dir)
    annotations = load_annotations(record_id, data_dir)
    return {
        "record_id": record_id,
        "sampling_rate_hz": fs,
        "sample_count": sample_count,
        "duration_seconds": sample_count / fs,
        "annotation_count": len(annotations),
        "annotation_symbols": sorted(annotations["symbol"].unique().tolist()),
    }


def inspect_records(record_ids: Iterable[str], data_dir: str | Path) -> pd.DataFrame:
    """Inspect multiple local records and return a metadata table."""
    return pd.DataFrame([inspect_record(record_id, data_dir) for record_id in record_ids])


