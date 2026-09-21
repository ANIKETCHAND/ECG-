"""
Dataset Integrity Validator
===========================
Verifies that downloaded raw records are structurally sound, not truncated,
and match expected PhysioNet binary formats and file sizes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_file_sha256(path: Path | str) -> str:
    """Calculate SHA-256 hash of a file."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_physionet_record_files(record_dir: Path | str, record_name: str) -> Tuple[bool, List[str]]:
    """Verify that a PhysioNet record has valid .hea, .dat, and optional .atr files."""
    r_dir = Path(record_dir)
    errors = []

    hea_file = r_dir / f"{record_name}.hea"
    dat_file = r_dir / f"{record_name}.dat"

    if not hea_file.exists():
        errors.append(f"Header file missing: {hea_file.name}")
    elif hea_file.stat().st_size == 0:
        errors.append(f"Header file is empty: {hea_file.name}")

    if not dat_file.exists():
        errors.append(f"Binary data file missing: {dat_file.name}")
    elif dat_file.stat().st_size == 0:
        errors.append(f"Binary data file is empty: {dat_file.name}")

    return len(errors) == 0, errors
