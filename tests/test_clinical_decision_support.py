"""
Unit tests for Clinical Decision Support (CDS) Engine.
Verifies:
- Recommendation generation for PVC, Atrial Fibrillation, Normal Sinus Rhythm.
- Triage urgency classification based on heart rate and rhythm findings.
- Evidence-based guideline citation presence (ACC, AHA, ESC).
- Absolute non-autonomous prescribing boundary: no automated prescriptions or dosage orders.
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical.recommendation_engine import (
    ClinicalDecisionSupportEngine,
    ClinicalRecommendation,
    GLOBAL_CDS_ENGINE,
)
from database.db_manager import PatientRecord


def test_cds_evaluates_pvc_finding():
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        ecg_finding="Premature Ventricular Contractions (PVC)",
        heart_rate=72.0,
    )

    assert isinstance(rec, ClinicalRecommendation)
    assert "Premature Ventricular" in rec.finding
    assert rec.urgency in ["ROUTINE REVIEW", "PROMPT REVIEW"]
    assert len(rec.clinical_considerations) > 0
    assert any("electrolyte" in c.lower() for c in rec.clinical_considerations)
    assert any("CAST" in c or "structural heart disease" in c for c in rec.contraindication_warnings)
    assert len(rec.relevant_guidelines) > 0
    assert "AHA" in rec.relevant_guidelines[0] or "ESC" in rec.relevant_guidelines[0]
    assert rec.clinician_action_required != ""


def test_cds_evaluates_pvc_with_extreme_tachycardia_prompts_prompt_review():
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        ecg_finding="PVC",
        heart_rate=135.0,  # Tachycardia
    )
    assert rec.urgency == "PROMPT REVIEW"


def test_cds_evaluates_afib_finding():
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        ecg_finding="Atrial Fibrillation (AF)",
        heart_rate=110.0,
    )
    assert "Atrial Fibrillation" in rec.finding
    assert rec.urgency == "PROMPT REVIEW"
    assert any("CHA2DS2-VASc" in c for c in rec.clinical_considerations)
    assert any("HAS-BLED" in c for c in rec.clinical_considerations)
    assert any("mechanical heart valves" in c for c in rec.contraindication_warnings)


def test_cds_evaluates_afib_with_critical_rate_triggers_urgent_review():
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        ecg_finding="Atrial Fibrillation (AF)",
        heart_rate=155.0,  # Critical rapid ventricular response
    )
    assert rec.urgency == "URGENT CLINICIAN REVIEW"


def test_cds_evaluates_normal_sinus_rhythm():
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        ecg_finding="Normal Sinus Rhythm",
        heart_rate=70.0,
    )
    assert rec.finding == "Normal Sinus Rhythm"
    assert rec.urgency == "ROUTINE REVIEW"
    assert len(rec.clinical_considerations) > 0
    assert len(rec.contraindication_warnings) > 0


def test_cds_evaluates_uncategorized_rhythm():
    cds = ClinicalDecisionSupportEngine()
    rec = cds.evaluate_finding(
        ecg_finding="Supraventricular Ectopy / Atypical morphology",
        heart_rate=80.0,
    )
    assert rec.urgency == "ROUTINE REVIEW"
    assert any("empiric antiarrhythmic" in c for c in rec.contraindication_warnings)


def test_cds_never_contains_dosage_or_prescription():
    """Safety validation: CDS must never issue direct drug orders or doses."""
    cds = ClinicalDecisionSupportEngine()
    for finding in ["PVC", "AF", "Normal Sinus Rhythm", "Unknown"]:
        rec = cds.evaluate_finding(finding, heart_rate=80.0)
        # Verify no direct prescriptive phrasing
        for c in rec.clinical_considerations + rec.suggested_assessments:
            assert "take " not in c.lower()
            assert "mg daily" not in c.lower()
            assert "prescribe " not in c.lower()
