"""
ECG PDF Report Processor
========================

Extracts selectable text, metadata, and embedded images from clinical ECG PDFs using pypdf:
- Determines whether the PDF contains selectable text vs scanned raster images
- Verifies whether the document is a genuine ECG report
- Extracts patient metadata, printed clinical measurements, and waveform images

Research/educational use only.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union
from PIL import Image
import pypdf

try:
    from src.ecg_input.measurement_extractor import extract_report_measurements
except ImportError:
    from .measurement_extractor import extract_report_measurements

__all__ = ["process_pdf_report", "ECG_REPORT_KEYWORDS"]



ECG_REPORT_KEYWORDS = [
    "ecg", "ekg", "electrocardiogram", "electrocardiograph",
    "vent. rate", "heart rate", "pr interval", "qrs duration",
    "qt/qtc", "sinus rhythm", "lead i", "lead ii", "v1", "v2", "v3",
    "v4", "v5", "v6", "bpm", "25 mm/s", "10 mm/mv"
]


def process_pdf_report(
    file_or_path: Union[str, Path, BinaryIO],
) -> Dict[str, Any]:
    """Process an uploaded ECG PDF document.

    Args:
        file_or_path: File path or binary stream of the PDF

    Returns:
        Structured dictionary containing:
        - 'is_ecg': bool
        - 'text': str
        - 'measurements': dict
        - 'images': List[Image.Image]
        - 'has_embedded_images': bool
        - 'page_count': int
        - 'status_message': str
    """
    try:
        reader = pypdf.PdfReader(file_or_path)
    except Exception as exc:
        return {
            "is_ecg": False,
            "text": "",
            "measurements": {},
            "images": [],
            "has_embedded_images": False,
            "page_count": 0,
            "status_message": f"Unable to read PDF file: {exc}",
        }

    page_count = len(reader.pages)
    if page_count == 0:
        return {
            "is_ecg": False,
            "text": "",
            "measurements": {},
            "images": [],
            "has_embedded_images": False,
            "page_count": 0,
            "status_message": "PDF contains no pages.",
        }

    # Extract text from all pages
    all_text = []
    extracted_images = []

    for page_idx, page in enumerate(reader.pages):
        # Text extraction
        try:
            p_text = page.extract_text() or ""
            if p_text.strip():
                all_text.append(p_text)
        except Exception:
            pass

        # Image extraction
        try:
            for img_file in page.images:
                try:
                    pil_img = Image.open(io.BytesIO(img_file.data))
                    extracted_images.append(pil_img)
                except Exception:
                    continue
        except Exception:
            pass

    combined_text = "\n".join(all_text)
    ocr_status = "NOT_ATTEMPTED"
    ocr_reason: Optional[str] = None

    # Scanned PDFs carry no selectable text. Fall back to OCR rather than
    # reporting an empty document, which would otherwise look like "no ECG here".
    if not combined_text.strip():
        try:
            from src.ecg_input.ocr import ocr_pdf_report
        except ImportError:
            from .ocr import ocr_pdf_report

        ocr_result = ocr_pdf_report(file_or_path)
        ocr_status = ocr_result.get("status", "PROCESSING_FAILED")
        ocr_reason = ocr_result.get("reason")
        if ocr_result.get("text"):
            combined_text = ocr_result["text"]

    # Check if document appears to be an ECG report
    text_lower = combined_text.lower()
    keyword_matches = sum(1 for kw in ECG_REPORT_KEYWORDS if kw in text_lower)
    is_ecg = keyword_matches >= 2 or len(extracted_images) > 0

    # Extract clinical measurements from selectable text (or recovered OCR text)
    measurements = extract_report_measurements(combined_text)

    if not is_ecg:
        status_msg = "This PDF does not appear to contain an ECG report."
    elif measurements["has_extracted_data"]:
        status_msg = "Successfully extracted text measurements from ECG PDF."
    elif len(extracted_images) > 0:
        status_msg = f"Found {len(extracted_images)} embedded image(s) in ECG PDF."
    else:
        status_msg = "PDF detected as ECG, but text extraction was limited."

    if ocr_status not in ("NOT_ATTEMPTED", "SUCCESS") and not combined_text.strip():
        status_msg = (
            "PDF has no selectable text layer. "
            f"Scanned-report OCR could not be used: {ocr_reason or ocr_status}"
        )

    return {
        "is_ecg": is_ecg,
        "text": combined_text,
        "measurements": measurements,
        "images": extracted_images,
        "has_embedded_images": len(extracted_images) > 0,
        "page_count": page_count,
        "status_message": status_msg,
        "ocr_status": ocr_status,
        "ocr_reason": ocr_reason,
        "text_source": "ocr" if ocr_status == "SUCCESS" and combined_text.strip() else "pdf_text_layer",
    }
