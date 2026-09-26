"""
DailyMed Structured Knowledge Extraction Pipeline
=================================================

Phases 3 & 7:
Ingests and indexes structured product labeling (SPL) from DailyMed / FDA:
- Drug classes, generic names, and brand names
- Approved cardiac indications (AHA/ACC aligned)
- Black-box warnings and contraindications
- Multi-drug interaction matrices
- Organ impairment precautions (Renal / Hepatic / Beers criteria)
- Evidence level tags and regulatory labeling versions

Functions strictly as an EVIDENCE & SAFETY LAYER, never as an unconstrained ML label source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.medications.medication_database import CARDIOVASCULAR_MEDICATION_REPOSITORY


def export_dailymed_knowledge_base(output_json: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Exports normalized, structured DailyMed cardiovascular knowledge layer.
    """
    kb: Dict[str, Any] = {
        "metadata": {
            "source": "DailyMed / US National Library of Medicine & FDA Prescribing Information",
            "version": "2026.04",
            "extracted_at": "2026-09-26",
            "governance": "Clinician-facing reference evidence; zero autonomous prescribing",
        },
        "medications": {},
    }

    for key, entry in CARDIOVASCULAR_MEDICATION_REPOSITORY.items():
        kb["medications"][key] = {
            "generic_name": entry.generic_name,
            "brand_names": entry.brand_names,
            "drug_class": entry.drug_class,
            "mechanism_of_action": entry.mechanism_of_action,
            "indications": entry.cardiac_indications,
            "contraindications": entry.contraindications,
            "black_box_warnings": entry.black_box_warnings,
            "qt_prolongation_risk": entry.qt_prolongation_risk,
            "renal_considerations": entry.renal_considerations,
            "hepatic_considerations": entry.hepatic_considerations,
            "pregnancy_category": entry.pregnancy_category,
            "monitoring_parameters": entry.monitoring_parameters,
            "interactions": [
                {
                    "interacting_drug": i.interacting_drug,
                    "severity": i.severity,
                    "clinical_effect": i.clinical_effect,
                    "recommendation": i.management_recommendation,
                    "source": i.source,
                    "evidence_level": i.evidence_level,
                }
                for i in entry.known_interactions
            ],
            "authoritative_source": entry.authoritative_source,
            "source_version": entry.source_version,
            "last_verified": entry.last_verified,
        }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2)
        print(f"[DailyMed] Exported {len(kb['medications'])} structured drug profiles to {out_p}")

    return kb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DailyMed knowledge base")
    parser.add_argument("--output", type=str, default="src/medications/dailymed_knowledge_base.json")
    args = parser.parse_args()
    export_dailymed_knowledge_base(args.output)
