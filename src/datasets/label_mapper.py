"""
Deterministic Label Mapping Engine
==================================
Maps heterogeneous dataset-specific diagnostic codes, PhysioNet beat annotations,
and SCP-ECG statements into standardized, task-specific canonical classes.
Strictly returns UNMAPPED for ambiguous or unverified clinical statements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LabelMapper:
    """Loads and manages task-specific deterministic label translation tables."""

    def __init__(self, configs_dir: Optional[Path | str] = None) -> None:
        self.configs_dir = (
            Path(configs_dir)
            if configs_dir
            else Path(__file__).resolve().parent.parent.parent / "configs" / "label_mappings"
        )
        self._tables: Dict[str, Dict[str, Any]] = {}
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        if not self.configs_dir.exists():
            return
        for file in self.configs_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    d_id = data.get("dataset_id")
                    if d_id:
                        self._tables[d_id] = data
            except Exception as err:
                print(f"Warning: Failed to load mapping config {file.name}: {err}")

    def map_label(
        self,
        dataset_id: str,
        source_symbol: str,
    ) -> Tuple[str, str]:
        """Map a source annotation/statement to a canonical label.

        Returns:
            (canonical_label, reason)
        """
        table = self._tables.get(dataset_id.lower().strip())
        if not table:
            return "UNMAPPED", f"No mapping table registered for dataset '{dataset_id}'"

        mappings = table.get("mappings", {})
        sym_clean = str(source_symbol).strip()

        if sym_clean in mappings:
            entry = mappings[sym_clean]
            return entry.get("canonical", "UNMAPPED"), entry.get("reason", "Mapped")

        excluded = table.get("unmapped_excluded", [])
        if sym_clean in excluded:
            return "EXCLUDED", f"Symbol '{sym_clean}' designated as non-diagnostic / artifact"

        return "UNMAPPED", f"Unrecognized clinical statement/symbol '{sym_clean}'"

    def get_canonical_classes(self, dataset_id: str) -> List[str]:
        """Retrieve list of canonical target classes for a dataset."""
        table = self._tables.get(dataset_id.lower().strip())
        if table:
            return table.get("canonical_classes", [])
        return []


GLOBAL_LABEL_MAPPER = LabelMapper()
