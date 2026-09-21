"""
Experiment Tracker Module
=========================
Maintains an immutable ledger of training runs, hyperparameters, patient splits,
evaluation metrics, and artifact checksums for regulatory auditability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRun:
    run_id: str
    model_name: str
    model_version: str
    created_at: str
    train_patients: List[str]
    test_patients: List[str]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    artifact_checksums: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    def __init__(self, ledger_path: Optional[Path | str] = None) -> None:
        self.ledger_path = (
            Path(ledger_path)
            if ledger_path
            else Path(__file__).resolve().parent.parent.parent / "reports" / "experiments.jsonl"
        )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        p = Path(file_path)
        if not p.exists():
            return "FILE_NOT_FOUND"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def log_run(self, run: ExperimentRun) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run.to_dict()) + "\n")

    def list_runs(self) -> List[Dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        runs = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        return runs
