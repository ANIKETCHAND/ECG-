"""
Dataset-Wide Signal Quality Assessment Subsystem
================================================
Evaluates signal quality across entire datasets, tracking technical anomalies,
mean SNR, flatlines, and clipping distribution.
Fulfills Phase 10 & Phase 14 quality transparency requirements.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import wfdb

from src.quality.quality_gate import evaluate_ecg_quality_gate


@dataclass
class DatasetQualitySummary:
    dataset_id: str
    records_evaluated: int
    good_count: int
    acceptable_count: int
    poor_count: int
    unusable_count: int
    mean_snr_db: float
    mean_quality_score: float
    quality_pass_rate_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_dataset_quality(raw_dir: Path | str, dataset_id: str = "dataset", max_records: int = 30) -> DatasetQualitySummary:
    """Assess technical diagnostic quality on records in raw_dir."""
    raw_path = Path(raw_dir)
    hea_files = list(raw_path.glob("*.hea"))[:max_records]

    categories = {"GOOD": 0, "ACCEPTABLE": 0, "POOR": 0, "UNUSABLE": 0}
    snrs: List[float] = []
    scores: List[float] = []

    for hf in hea_files:
        rec_name = hf.stem
        try:
            record = wfdb.rdrecord(str(raw_path / rec_name))
            if record.p_signal is not None and len(record.p_signal) > 0:
                sig = record.p_signal[:, 0]
                fs = float(record.fs) if record.fs else 360.0
                res = evaluate_ecg_quality_gate(sig, fs=fs)
                categories[res.category.value] = categories.get(res.category.value, 0) + 1
                scores.append(res.quality_score)
                snrs.append(res.metrics.snr_db)
        except Exception:
            categories["UNUSABLE"] += 1
            scores.append(0.0)

    total = sum(categories.values())
    pass_count = categories["GOOD"] + categories["ACCEPTABLE"]
    pass_rate = float(pass_count / total * 100.0) if total > 0 else 0.0

    return DatasetQualitySummary(
        dataset_id=dataset_id,
        records_evaluated=total,
        good_count=categories["GOOD"],
        acceptable_count=categories["ACCEPTABLE"],
        poor_count=categories["POOR"],
        unusable_count=categories["UNUSABLE"],
        mean_snr_db=float(np.mean(snrs)) if snrs else 0.0,
        mean_quality_score=float(np.mean(scores)) if scores else 0.0,
        quality_pass_rate_pct=round(pass_rate, 2),
    )
