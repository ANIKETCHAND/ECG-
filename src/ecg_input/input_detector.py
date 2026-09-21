"""
ECG Input Detector Module
=========================

Identifies the format and modality of uploaded ECG files:
- DIGITAL_SIGNAL: CSV, TXT, NPY, EDF, XML, JSON, DICOM, WFDB (.dat)
- REPORT_IMAGE: JPG, JPEG, PNG, BMP, TIFF
- REPORT_PDF: PDF
- UNSUPPORTED: Non-ECG or unknown file types
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union


class InputModality(str, Enum):
    DIGITAL_SIGNAL = "DIGITAL_SIGNAL"
    REPORT_IMAGE = "REPORT_IMAGE"
    REPORT_PDF = "REPORT_PDF"
    UNSUPPORTED = "UNSUPPORTED"


# Supported file extension mappings
EXT_MAPPING = {
    ".csv": (InputModality.DIGITAL_SIGNAL, "csv"),
    ".txt": (InputModality.DIGITAL_SIGNAL, "txt"),
    ".npy": (InputModality.DIGITAL_SIGNAL, "npy"),
    ".dat": (InputModality.DIGITAL_SIGNAL, "dat"),
    ".hea": (InputModality.DIGITAL_SIGNAL, "hea"),
    ".edf": (InputModality.DIGITAL_SIGNAL, "edf"),
    ".rec": (InputModality.DIGITAL_SIGNAL, "edf"),
    ".xml": (InputModality.DIGITAL_SIGNAL, "xml"),
    ".hl7": (InputModality.DIGITAL_SIGNAL, "xml"),
    ".json": (InputModality.DIGITAL_SIGNAL, "json"),
    ".dcm": (InputModality.DIGITAL_SIGNAL, "dcm"),
    ".dicom": (InputModality.DIGITAL_SIGNAL, "dcm"),
    ".jpg": (InputModality.REPORT_IMAGE, "jpg"),
    ".jpeg": (InputModality.REPORT_IMAGE, "jpeg"),
    ".png": (InputModality.REPORT_IMAGE, "png"),
    ".bmp": (InputModality.REPORT_IMAGE, "bmp"),
    ".tiff": (InputModality.REPORT_IMAGE, "tiff"),
    ".tif": (InputModality.REPORT_IMAGE, "tif"),
    ".pdf": (InputModality.REPORT_PDF, "pdf"),
}


def detect_input_modality(
    filename_or_path: Union[str, Path],
    file_bytes: Optional[bytes] = None,
) -> Tuple[InputModality, str]:
    """Detect file modality and normalized extension with magic byte verification."""
    path_obj = Path(str(filename_or_path))
    ext = path_obj.suffix.lower()

    # Magic byte checking if bytes provided
    if file_bytes is not None and len(file_bytes) >= 4:
        # PDF magic number: %PDF
        if file_bytes.startswith(b"%PDF"):
            return InputModality.REPORT_PDF, "pdf"
        # PNG magic number: \x89PNG
        if file_bytes.startswith(b"\x89PNG"):
            return InputModality.REPORT_IMAGE, "png"
        # JPEG magic number: \xff\xd8\xff
        if file_bytes.startswith(b"\xff\xd8\xff"):
            return InputModality.REPORT_IMAGE, "jpeg"
        # NPY magic number: \x93NUMPY
        if file_bytes.startswith(b"\x93NUMPY"):
            return InputModality.DIGITAL_SIGNAL, "npy"
        # DICOM magic number at offset 128: DICM
        if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
            return InputModality.DIGITAL_SIGNAL, "dcm"
        # XML magic starts with <?xml or <
        stripped = file_bytes.strip()
        if stripped.startswith(b"<?xml") or (stripped.startswith(b"<") and b">" in stripped[:100]):
            return InputModality.DIGITAL_SIGNAL, "xml"
        # JSON magic starts with { or [
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            return InputModality.DIGITAL_SIGNAL, "json"

    if ext in EXT_MAPPING:
        return EXT_MAPPING[ext]

    return InputModality.UNSUPPORTED, ext.lstrip(".")
