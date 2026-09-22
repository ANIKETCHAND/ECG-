"""
Clinical Decision Support (CDS) Engine.
Phases 12 & 18:
- Provides traceable clinical considerations, suggested next assessments, and triage urgency.
- Strictly adheres to Rule 1: No autonomous prescriptions.
- Cites authoritative medical guidelines (AHA, ACC, ESC).
- Clinician retains mandatory final review and therapeutic authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database.db_manager import PatientRecord


@dataclass
class ClinicalRecommendation:
    finding: str
    urgency: str  # ROUTINE REVIEW, PROMPT REVIEW, URGENT CLINICIAN REVIEW
    clinical_considerations: List[str]
    suggested_assessments: List[str]
    contraindication_warnings: List[str]
    relevant_guidelines: List[str]
    guideline_source: str
    last_verified: str
    clinician_action_required: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalDecisionSupportEngine:
    """Generates evidence-backed clinical decision support considerations."""

    def evaluate_finding(
        self,
        ecg_finding: str,
        heart_rate: Optional[float] = None,
        patient: Optional[PatientRecord] = None,
        machine_finding: Optional[str] = None,
    ) -> ClinicalRecommendation:
        finding_upper = ecg_finding.upper()

        # 1. Ventricular Ectopy / PVC Pattern
        if "PVC" in finding_upper or ("VENTRICULAR" in finding_upper and "SUPRAVENTRICULAR" not in finding_upper):
            considerations = [
                "Premature Ventricular Contraction pattern detected on rhythm strip.",
                "Frequent or multifocal PVCs may warrant quantification of 24-hour ectopy burden via ambulatory Holter monitoring.",
                "Evaluate for reversible secondary causes: electrolyte imbalance (hypokalemia, hypomagnesemia), caffeine/sympathomimetic excess, or myocardial ischemia.",
            ]
            assessments = [
                "Check serum electrolytes (Potassium, Magnesium).",
                "Assess for clinical symptoms: palpitations, presyncope, or chest tightness.",
                "Review baseline echocardiogram if ectopy burden is high (> 10-15%).",
                "Review current medication profile for proarrhythmic or QT-prolonging agents.",
            ]
            contraindications = [
                "Class IC antiarrhythmics (flecainide, propafenone) are contraindicated in patients with structural heart disease or prior myocardial infarction (CAST trial).",
            ]
            guidelines = [
                "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure",
                "2019 ESC Guidelines for the Management of Patients with Ventricular Arrhythmias",
            ]
            urgency = "PROMPT REVIEW" if (heart_rate and (heart_rate > 120 or heart_rate < 45)) else "ROUTINE REVIEW"
            action = "Attending physician review required to correlate with clinical history and order electrolyte panel if indicated."

            return ClinicalRecommendation(
                finding="Premature Ventricular Contractions (PVC)",
                urgency=urgency,
                clinical_considerations=considerations,
                suggested_assessments=assessments,
                contraindication_warnings=contraindications,
                relevant_guidelines=guidelines,
                guideline_source="AHA/ACC/HRS Guideline on Ventricular Arrhythmias",
                last_verified="2026-04-12",
                clinician_action_required=action,
            )

        # 2. Atrial Fibrillation Pattern
        elif "AF" in finding_upper or "FIBRILLATION" in finding_upper:
            considerations = [
                "Irregularly irregular ventricular response suggestive of Atrial Fibrillation.",
                "Systemic thromboembolism risk assessment mandatory (calculate CHA2DS2-VASc score).",
                "Evaluate bleeding risk score (HAS-BLED) before considering anticoagulation therapy.",
                "Assess hemodynamic stability and identify precipitating causes (thyrotoxicosis, infection, alcohol, acute pulmonary event).",
            ]
            assessments = [
                "Assess vital signs (orthostatic blood pressure, pulse deficit).",
                "Calculate CHA2DS2-VASc and HAS-BLED clinical risk scores.",
                "Order complete metabolic panel, thyroid stimulating hormone (TSH), and transthoracic echocardiogram.",
                "Obtain previous ECG to determine if rhythm shift is paroxysmal, persistent, or long-standing.",
            ]
            contraindications = [
                "Direct Oral Anticoagulants (DOACs) are contraindicated in patients with mechanical heart valves or moderate-to-severe mitral stenosis.",
            ]
            guidelines = [
                "2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation",
                "2020 ESC Guidelines for the Diagnosis and Management of Atrial Fibrillation",
            ]
            urgency = "URGENT CLINICIAN REVIEW" if (heart_rate and (heart_rate > 140 or heart_rate < 40)) else "PROMPT REVIEW"
            action = "Urgent clinician evaluation required. Determine rhythm vs. rate control strategy and thromboembolic risk profile."

            return ClinicalRecommendation(
                finding="Atrial Fibrillation Pattern",
                urgency=urgency,
                clinical_considerations=considerations,
                suggested_assessments=assessments,
                contraindication_warnings=contraindications,
                relevant_guidelines=guidelines,
                guideline_source="ACC/AHA/HRS Guideline on Atrial Fibrillation",
                last_verified="2026-05-18",
                clinician_action_required=action,
            )

        # 3. Normal Sinus Rhythm
        elif "NORMAL" in finding_upper:
            considerations = [
                "Normal Sinus Rhythm with regular P waves and physiological PR intervals.",
                "Normal rhythm on a resting strip does not exclude paroxysmal dysrhythmias or transient ischemic episodes if patient is symptomatic.",
            ]
            assessments = [
                "Correlate with current patient symptoms (chest pain, dyspnea, syncope).",
                "Compare with prior ECGs for subtle repolarization, axis, or voltage changes.",
            ]
            contraindications = ["None based on resting ECG alone."]
            guidelines = ["AHA/ACC Standard Guidelines for Electrocardiography Interpretation"]
            urgency = "ROUTINE REVIEW"
            action = "Routine clinician confirmation and archival."

            return ClinicalRecommendation(
                finding="Normal Sinus Rhythm",
                urgency=urgency,
                clinical_considerations=considerations,
                suggested_assessments=assessments,
                contraindication_warnings=contraindications,
                relevant_guidelines=guidelines,
                guideline_source="AHA/ACC Recommendations for Standardization and Interpretation of the ECG",
                last_verified="2026-03-01",
                clinician_action_required=action,
            )

        # 4. Non-diagnostic / Uncategorized finding
        else:
            return ClinicalRecommendation(
                finding=ecg_finding,
                urgency="ROUTINE REVIEW",
                clinical_considerations=[
                    "Atypical morphology or unclassified rhythm finding.",
                    "Review full 12-lead context, patient clinical presentation, and prior tracings.",
                ],
                suggested_assessments=[
                    "Clinical correlation with symptoms and cardiovascular physical exam.",
                    "Repeat ECG acquisition if recording contains baseline wander or high-frequency tremor.",
                ],
                contraindication_warnings=["Avoid empiric antiarrhythmic therapy without verified diagnostic etiology."],
                relevant_guidelines=["Clinical Electrocardiography Core Curriculum"],
                guideline_source="AHA/ACC Standards",
                last_verified="2026-01-15",
                clinician_action_required="Attending physician review and diagnostic classification.",
            )


GLOBAL_CDS_ENGINE = ClinicalDecisionSupportEngine()

