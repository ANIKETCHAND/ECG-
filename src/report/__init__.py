"""
ECG Report Generation Package
=============================

Exports structured report models and multi-format exporters (PDF, JSON, TXT).
"""

try:
    from report.report_generator import (
        generate_structured_report,
        export_report_to_text,
        export_report_to_json,
    )
    from report.pdf_generator import (
        generate_pdf_report,
        generate_doctor_report,
        generate_patient_report,
    )
except ImportError:
    from src.report.report_generator import (
        generate_structured_report,
        export_report_to_text,
        export_report_to_json,
    )
    from src.report.pdf_generator import (
        generate_pdf_report,
        generate_doctor_report,
        generate_patient_report,
    )

__all__ = [
    "generate_structured_report",
    "export_report_to_text",
    "export_report_to_json",
    "generate_pdf_report",
    "generate_doctor_report",
    "generate_patient_report",
]
