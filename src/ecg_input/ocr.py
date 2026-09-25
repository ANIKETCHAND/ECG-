"""
Scanned-Report OCR
==================

Recovers printed text from *scanned* ECG reports — JPG/PNG/TIFF photographs and
image-only PDFs that carry no selectable text layer.

Why this exists
---------------
The PDF path relies on ``pypdf`` text extraction, which returns nothing for a
scanned document. The image path could digitise the waveform but could not read
the printed header, so machine-printed measurements (HR, PR, QRS, QT/QTc,
auto-interpretation) were unavailable for exactly the reports most often handed
to a clinician as a scan.

OCR is optional. When the ``pytesseract`` binding or the Tesseract binary is
absent, :func:`is_ocr_available` returns False and callers receive an explicit
``status: OCR_UNAVAILABLE`` instead of silently empty text. Empty text and
"could not read the image" are different facts and must not be conflated.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:  # optional dependency
    import pytesseract  # type: ignore

    PYTESSERACT_AVAILABLE = True
    PYTESSERACT_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - environment dependent
    pytesseract = None  # type: ignore[assignment]
    PYTESSERACT_AVAILABLE = False
    PYTESSERACT_IMPORT_ERROR = str(exc)


#: Minimum characters for OCR output to be treated as a usable header.
MIN_USEFUL_CHARACTERS = 12


def _tesseract_binary_reachable() -> bool:
    """Check that the underlying Tesseract binary answers, not just the binding."""
    if not PYTESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def is_ocr_available() -> bool:
    """Return True when OCR can actually run (binding present and binary reachable)."""
    return _tesseract_binary_reachable()


def ocr_unavailable_reason() -> Optional[str]:
    """Explain why OCR cannot run, or None when it can."""
    if not PYTESSERACT_AVAILABLE:
        return (
            "pytesseract is not installed, so scanned reports cannot be read. "
            f"Install with: pip install pytesseract (import error: {PYTESSERACT_IMPORT_ERROR})"
        )
    try:
        pytesseract.get_tesseract_version()
        return None
    except Exception as exc:
        return (
            "The Tesseract OCR engine binary is not reachable, so scanned reports cannot be read. "
            f"Install the Tesseract engine and ensure it is on PATH. (error: {exc})"
        )


def _prepare_for_ocr(image: Union[np.ndarray, Image.Image, bytes], upscale: float = 2.0) -> Optional[Image.Image]:
    """Convert input to a grayscale, contrast-stretched PIL image for OCR."""
    try:
        if isinstance(image, bytes):
            pil = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            pil = image
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                pil = Image.fromarray(array.astype("uint8"), mode="L")
            elif array.shape[2] == 4:
                pil = Image.fromarray(array, mode="RGBA").convert("RGB")
            else:
                pil = Image.fromarray(array[:, :, ::-1] if array.shape[2] == 3 else array)  # BGR -> RGB
    except Exception:
        return None

    pil = pil.convert("L")

    if upscale and upscale > 1.0:
        pil = pil.resize((int(pil.width * upscale), int(pil.height * upscale)), Image.LANCZOS)

    # Autocontrast helps with pale photocopies, the common failure case.
    try:
        from PIL import ImageOps

        pil = ImageOps.autocontrast(pil)
    except Exception:
        pass

    return pil


def extract_text_from_image(
    image: Union[np.ndarray, Image.Image, bytes, str, Path],
    *,
    psm: int = 6,
    upscale: float = 2.0,
    min_characters: int = MIN_USEFUL_CHARACTERS,
) -> Dict[str, Any]:
    """Run OCR over an ECG report image.

    Args:
        image: BGR numpy array, PIL image, encoded bytes, or a path.
        psm: Tesseract page-segmentation mode (6 = uniform block of text).
        upscale: Pre-processing upscale factor, which materially improves results
            on typical ECG screenshots.
        min_characters: Below this, the result is reported as too sparse to use.

    Returns:
        Dict with ``status`` (one of ``SUCCESS``, ``OCR_UNAVAILABLE``,
        ``EMPTY_TEXT``, ``LOW_CONFIDENCE``, ``PROCESSING_FAILED``), ``text``,
        ``characters``, ``reason`` and ``duration_ms``.
    """
    import time

    started = time.perf_counter()
    result: Dict[str, Any] = {
        "status": "PROCESSING_FAILED",
        "text": "",
        "characters": 0,
        "reason": None,
        "duration_ms": 0.0,
    }

    reason = ocr_unavailable_reason()
    if reason:
        result["status"] = "OCR_UNAVAILABLE"
        result["reason"] = reason
        result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return result

    if isinstance(image, (str, Path)):
        try:
            image = Image.open(str(image))
        except Exception as exc:
            result["reason"] = f"Could not open image for OCR: {exc}"
            result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            return result

    prepared = _prepare_for_ocr(image, upscale=upscale)
    if prepared is None:
        result["reason"] = "Could not convert the supplied image into an OCR-able buffer."
        result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return result

    try:
        text = pytesseract.image_to_string(prepared, config=f"--psm {psm}")
    except Exception as exc:
        result["reason"] = f"Tesseract execution failed: {exc}"
        result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return result

    text = (text or "").strip()
    result["text"] = text
    result["characters"] = len(text)

    if not text:
        result["status"] = "EMPTY_TEXT"
        result["reason"] = "OCR ran but found no text in the image."
    elif len(text) < min_characters:
        result["status"] = "LOW_CONFIDENCE"
        result["reason"] = (
            f"OCR returned only {len(text)} characters, below the {min_characters}-character "
            "threshold for a trustworthy header."
        )
    else:
        result["status"] = "SUCCESS"

    result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return result


def ocr_report_measurements(
    image: Union[np.ndarray, Image.Image, bytes, str, Path],
    *,
    psm: int = 6,
) -> Dict[str, Any]:
    """OCR an ECG report image and parse printed clinical measurements.

    Combines OCR with the existing regex measurement extractor so a scan yields
    the same structured measurements as a text-layer PDF.

    Returns:
        The OCR result dict, plus ``measurements`` when text was recovered.
    """
    result = extract_text_from_image(image, psm=psm)

    if result["status"] != "SUCCESS":
        result["measurements"] = {}
        return result

    try:
        from src.ecg_input.measurement_extractor import extract_report_measurements
    except ImportError:
        from .measurement_extractor import extract_report_measurements

    result["measurements"] = extract_report_measurements(result["text"])
    return result


def rasterise_pdf_pages(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    *,
    dpi: int = 200,
    max_pages: int = 3,
) -> Tuple[List[np.ndarray], Optional[str]]:
    """Rasterise PDF pages to images for OCR.

    Requires ``pypdfium2`` or ``pdf2image``/poppler. Returns ``([], reason)`` when
    no renderer is installed, so the caller can report that honestly rather than
    treating a scanned PDF as empty.
    """
    try:
        import pypdfium2  # type: ignore

        if isinstance(file_or_path, (str, Path)):
            pdf = pypdfium2.PdfDocument(str(file_or_path))
        else:
            data = file_or_path if isinstance(file_or_path, bytes) else file_or_path.read()
            pdf = pypdfium2.PdfDocument(io.BytesIO(data))

        pages: List[np.ndarray] = []
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            bitmap = page.render(scale=dpi / 72.0)
            image = bitmap.to_pil().convert("RGB")
            pages.append(np.asarray(image))
        return pages, None
    except ImportError:
        pass
    except Exception as exc:
        return [], f"PDF rasterisation failed: {exc}"

    return [], (
        "No PDF rasteriser is installed, so scanned PDF pages cannot be converted to images for OCR. "
        "Install with: pip install pypdfium2"
    )


def ocr_pdf_report(
    file_or_path: Union[str, Path, bytes, io.BytesIO],
    *,
    dpi: int = 200,
) -> Dict[str, Any]:
    """OCR a scanned (image-only) PDF report."""
    reason = ocr_unavailable_reason()
    if reason:
        return {"status": "OCR_UNAVAILABLE", "text": "", "characters": 0, "reason": reason, "measurements": {}}

    pages, raster_reason = rasterise_pdf_pages(file_or_path, dpi=dpi)
    if raster_reason:
        return {
            "status": "RASTERISER_UNAVAILABLE",
            "text": "",
            "characters": 0,
            "reason": raster_reason,
            "measurements": {},
        }

    texts: List[str] = []
    for page in pages:
        page_result = extract_text_from_image(page)
        if page_result["text"]:
            texts.append(page_result["text"])

    combined = "\n".join(texts).strip()
    status = "SUCCESS" if len(combined) >= MIN_USEFUL_CHARACTERS else ("EMPTY_TEXT" if not combined else "LOW_CONFIDENCE")

    measurements: Dict[str, Any] = {}
    if status == "SUCCESS":
        try:
            from src.ecg_input.measurement_extractor import extract_report_measurements
        except ImportError:
            from .measurement_extractor import extract_report_measurements

        measurements = extract_report_measurements(combined)

    return {
        "status": status,
        "text": combined,
        "characters": len(combined),
        "pages_processed": len(pages),
        "reason": None if status == "SUCCESS" else "OCR did not recover usable text from the scanned PDF.",
        "measurements": measurements,
    }
