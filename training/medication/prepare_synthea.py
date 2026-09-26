"""
Synthea Synthetic Patient Generator & Edge-Case Safety Suite
============================================================

Phases 3 & 6:
Generates synthetic patient profiles specifically targeting:
- Known drug allergy edge cases (e.g. Amiodarone with Iodine allergy, Metoprolol with Bronchospasm)
- Severe organ dysfunction (CrCl < 15 mL/min, Severe hepatic cirrhosis)
- Critical electrolyte derangements (K+ = 2.4 mEq/L, K+ = 6.2 mEq/L)
- Extreme vitals (SBP = 70 mmHg, HR = 36 bpm)
- Missing clinical context stress tests

Strict Governance (Part 3 & 6):
1. Marked explicitly as SYNTHETIC.
2. Used exclusively for safety-gate stress testing, missing-data verification, and unit tests.
3. NEVER used to claim real-world clinical accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


SYNTHETIC_SAFETY_EDGE_CASES = [
    {
        "scenario_id": "SYN-ALLERGY-01",
        "title": "Known Severe Beta-Blocker Allergy",
        "patient_name": "Synthea Allergy Patient A",
        "age": 54,
        "sex": "Female",
        "known_allergies": ["Metoprolol", "Carvedilol"],
        "existing_conditions": ["Hypertension"],
        "ecg_finding": "PVC",
        "heart_rate": 84,
        "expected_safety_outcome": "BLOCKED_ALLERGY",
        "expected_blocked_drug": "metoprolol",
    },
    {
        "scenario_id": "SYN-HEPATIC-02",
        "title": "Known Amiodarone / Iodine Anaphylaxis",
        "patient_name": "Synthea Iodine Allergy Patient",
        "age": 68,
        "sex": "Male",
        "known_allergies": ["Iodine", "Amiodarone"],
        "existing_conditions": ["Coronary Artery Disease"],
        "ecg_finding": "PVC",
        "heart_rate": 92,
        "expected_safety_outcome": "BLOCKED_ALLERGY",
        "expected_blocked_drug": "amiodarone",
    },
    {
        "scenario_id": "SYN-ELECTROLYTE-03",
        "title": "Severe Hyperkalemia (K+ = 6.2 mEq/L) with ACE-Inhibitor Candidate",
        "patient_name": "Synthea Renal Hyperkalemia",
        "age": 72,
        "sex": "Male",
        "known_allergies": [],
        "existing_conditions": ["Chronic Kidney Disease Stage 4", "Heart Failure"],
        "laboratory_results": {"potassium": 6.2, "creatinine": 2.8, "egfr": 22},
        "vital_signs": {"systolic_bp": 138, "diastolic_bp": 82, "heart_rate": 66},
        "ecg_finding": "Normal",
        "expected_safety_outcome": "BLOCKED_LAB_CONTRAINDICATION",
        "expected_blocked_drug": "spironolactone",
    },
    {
        "scenario_id": "SYN-BRADYCARDIA-04",
        "title": "Marked Sinus Bradycardia (HR = 38 bpm)",
        "patient_name": "Synthea Profound Bradycardia",
        "age": 81,
        "sex": "Female",
        "known_allergies": [],
        "existing_conditions": ["Sick Sinus Syndrome"],
        "vital_signs": {"systolic_bp": 98, "diastolic_bp": 54, "heart_rate": 38},
        "ecg_finding": "PVC",
        "expected_safety_outcome": "BLOCKED_VITALS_CONTRAINDICATION",
        "expected_blocked_drug": "metoprolol",
    },
    {
        "scenario_id": "SYN-MISSING-05",
        "title": "Total Absence of Clinical Context (ECG-Only)",
        "patient_name": "Anonymous Zero Context",
        "age": None,
        "sex": None,
        "known_allergies": [],
        "existing_conditions": [],
        "vital_signs": {},
        "laboratory_results": {},
        "ecg_finding": "PVC",
        "expected_safety_outcome": "SAFETY_VERIFICATION_INCOMPLETE",
        "expected_blocked_drug": "ALL",
    },
]


def generate_synthea_scenarios(output_json: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Generate structured synthetic edge-case dataset."""
    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(SYNTHETIC_SAFETY_EDGE_CASES, f, indent=2)
        print(f"[Synthea] Generated {len(SYNTHETIC_SAFETY_EDGE_CASES)} safety test scenarios -> {p}")

    return SYNTHETIC_SAFETY_EDGE_CASES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthea synthetic edge cases")
    parser.add_argument("--output", type=str, default="training/medication/synthea_safety_scenarios.json")
    args = parser.parse_args()
    generate_synthea_scenarios(args.output)
