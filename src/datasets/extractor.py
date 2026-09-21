"""
Dataset Extractor Subsystem
===========================
Safely unpacks tar, tar.gz, tar.bz2, and zip archives while strictly
protecting against zip-slip and path-traversal vulnerabilities.
"""

from __future__ import annotations

import os
from pathlib import Path
import tarfile
import zipfile


class ArchiveExtractionError(Exception):
    pass


def extract_archive(archive_path: Path | str, target_dir: Path | str) -> Path:
    """Extract an archive file safely into target_dir."""
    arc = Path(archive_path)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if not arc.exists():
        raise FileNotFoundError(f"Archive file not found: {arc}")

    target_resolved = target.resolve()

    if zipfile.is_zipfile(arc):
        with zipfile.ZipFile(arc, "r") as zf:
            for member in zf.infolist():
                dest = (target / member.filename).resolve()
                if not str(dest).startswith(str(target_resolved)):
                    raise ArchiveExtractionError(f"Security Alert: Path traversal attempt in archive: {member.filename}")
            zf.extractall(target)
        return target

    if tarfile.is_tarfile(arc):
        with tarfile.open(arc, "r:*") as tf:
            for member in tf.getmembers():
                dest = (target / member.name).resolve()
                if not str(dest).startswith(str(target_resolved)):
                    raise ArchiveExtractionError(f"Security Alert: Path traversal attempt in tarball: {member.name}")
            tf.extractall(target)
        return target

    raise ArchiveExtractionError(f"Unsupported archive format: {arc.name}")
