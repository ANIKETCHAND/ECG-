"""
Dataset Inspection Engine
=========================
Introspects raw and processed datasets, extracting ground-truth sampling rates,
leads, durations, patient counts, diagnostic labels, and data quality indicators.
Generates comprehensive JSON and Markdown inspection reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import wfdb

from src.datasets.patient_index import resolve_patient_id
from src.datasets.registry import DATASET_REGISTRY


@dataclass
class InspectionReport:
    dataset_id: str
    dataset_name: str
    records_inspected: int
    patients_count: int
    sampling_rates: List[float]
    lead_names: List[str]
    total_duration_hours: float
    signal_units: List[str]
    annotations_summary: Dict[str, int]
    missing_records: List[str] = field(default_factory=list)
    corrupted_records: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetInspector:
    """Introspects local ECG databases and outputs audit documentation."""

    def __init__(self, reports_dir: Optional[Path | str] = None) -> None:
        self.reports_dir = (
            Path(reports_dir)
            if reports_dir
            else Path(__file__).resolve().parent.parent.parent / "reports" / "datasets"
        )
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def inspect_dataset(self, dataset_id: str, raw_dir: Path | str) -> InspectionReport:
        r_dir = Path(raw_dir)
        hea_files = sorted(list(r_dir.glob("*.hea")))

        sampling_rates: set = set()
        leads: set = set()
        units: set = set()
        total_duration_sec = 0.0
        patient_ids: set = set()
        annotations_counts: Dict[str, int] = {}
        corrupted = []

        for hea in hea_files:
            rec_name = hea.stem
            try:
                header = wfdb.rdheader(str(r_dir / rec_name))
                sampling_rates.add(float(header.fs))
                for s_name in header.sig_name:
                    leads.add(str(s_name).strip())
                for u in header.units:
                    units.add(str(u).strip())

                if header.fs > 0:
                    dur = header.sig_len / header.fs
                    total_duration_sec += dur

                pid = resolve_patient_id(dataset_id, rec_name)
                patient_ids.add(pid)

                # Check if annotation exists
                atr_file = r_dir / f"{rec_name}.atr"
                if atr_file.exists():
                    ann = wfdb.rdann(str(r_dir / rec_name), "atr")
                    for sym in ann.symbol:
                        annotations_counts[sym] = annotations_counts.get(sym, 0) + 1

            except Exception as err:
                corrupted.append(f"{rec_name}: {str(err)}")

        report = InspectionReport(
            dataset_id=dataset_id,
            dataset_name=DATASET_REGISTRY.get_dataset(dataset_id).name if DATASET_REGISTRY.get_dataset(dataset_id) else dataset_id,
            records_inspected=len(hea_files),
            patients_count=len(patient_ids),
            sampling_rates=sorted(list(sampling_rates)),
            lead_names=sorted(list(leads)),
            total_duration_hours=round(total_duration_sec / 3600.0, 2),
            signal_units=sorted(list(units)),
            annotations_summary=annotations_counts,
            corrupted_records=corrupted,
        )

        # Save JSON
        json_path = self.reports_dir / f"{dataset_id}_inspection.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save Markdown
        md_path = self.reports_dir / f"{dataset_id}_inspection.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._render_markdown_report(report))

        return report

    @staticmethod
    def _render_markdown_report(report: InspectionReport) -> str:
        lines = [
            f"# Dataset Inspection Report: {report.dataset_name}",
            f"**Dataset ID:** `{report.dataset_id}`  ",
            f"**Records Inspected:** {report.records_inspected}  ",
            f"**Distinct Patients:** {report.patients_count}  ",
            f"**Total ECG Signal Duration:** {report.total_duration_hours} hours  ",
            "",
            "## 1. Acquisition & Technical Parameters",
            f"- **Sampling Rates (Hz):** {report.sampling_rates}",
            f"- **Lead Names:** {report.lead_names}",
            f"- **Signal Units:** {report.signal_units}",
            "",
            "## 2. Annotations & Diagnostic Symbols",
        ]
        if report.annotations_summary:
            lines.append("| Symbol | Frequency | Description / Clinical Meaning |")
            lines.append("| :--- | :--- | :--- |")
            for sym, cnt in sorted(report.annotations_summary.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| `{sym}` | {cnt:,} | Annotated cardiac beat / event |")
        else:
            lines.append("No beat annotations recorded in header inspection.")

        if report.corrupted_records:
            lines.extend(["", "## 3. Data Integrity Issues", "- " + "\n- ".join(report.corrupted_records)])
        else:
            lines.extend(["", "## 3. Data Integrity Issues", "Zero corrupted or unreadable records detected."])

        return "\n".join(lines)


GLOBAL_DATASET_INSPECTOR = DatasetInspector()
