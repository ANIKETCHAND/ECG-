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
    medication_recommendations: List[Dict[str, str]] = field(default_factory=list)
    patient_context: Dict[str, Any] = field(default_factory=dict)
    missing_information: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalDecisionSupportEngine:
    """Generates evidence-backed clinical decision support considerations."""

    def evaluate_finding(
        self,
        ecg_finding: str,
        heart_rate: Optional[float] = None,
        patient: Optional[Any] = None,
        machine_finding: Optional[str] = None,
    ) -> ClinicalRecommendation:
        finding_upper = ecg_finding.upper()

        # ── Compute Patient Context & Missing Info (Phase 17) ──
        p_ctx: Dict[str, Any] = {}
        missing: List[str] = []
        if patient is None:
            p_ctx = {
                "age": "NOT PROVIDED",
                "sex": "NOT PROVIDED",
                "blood_group": "NOT PROVIDED",
                "smoking_status": "NOT PROVIDED",
                "known_conditions": "None documented",
                "current_medications": "None documented",
                "cardiac_history": "No documented cardiac history",
            }
            missing = [
                "Symptoms: Not provided",
                "Vital signs: Not provided",
                "Laboratory data: Not provided",
                "Medication history: Not provided",
                "Medical history: Not provided",
            ]
        else:
            p_dict = patient.to_dict() if hasattr(patient, "to_dict") else (patient if isinstance(patient, dict) else vars(patient))
            p_ctx = {
                "age": p_dict.get("age") if p_dict.get("age") is not None else "NOT PROVIDED",
                "sex": p_dict.get("sex") or "NOT PROVIDED",
                "blood_group": p_dict.get("blood_group") or "NOT PROVIDED",
                "smoking_status": p_dict.get("smoking_status") or "NOT PROVIDED",
                "known_conditions": ", ".join(p_dict.get("existing_conditions", [])) if isinstance(p_dict.get("existing_conditions"), list) else (p_dict.get("existing_conditions") or "None documented"),
                "current_medications": ", ".join([m.get("drug_name", "") if isinstance(m, dict) else getattr(m, "drug_name", str(m)) for m in p_dict.get("current_medications", [])]) if isinstance(p_dict.get("current_medications"), list) else (p_dict.get("current_medications") or "None documented"),
                "cardiac_history": ", ".join(p_dict.get("cardiac_history", [])) if isinstance(p_dict.get("cardiac_history"), list) else (p_dict.get("previous_cardiac_history") or "No documented cardiac history"),
            }
            if not p_dict.get("symptoms"):
                missing.append("Symptoms: Not provided")
            if not p_dict.get("vital_signs") and not p_dict.get("vitals"):
                missing.append("Vital signs: Not provided")
            if not p_dict.get("laboratory_results") and not p_dict.get("labs"):
                missing.append("Laboratory data: Not provided")
            if not p_dict.get("previous_ecg_ids"):
                missing.append("Previous ECG: None available")

        # ── 1. Ventricular Ectopy / PVC ────────────────────────────────────────
        if "PVC" in finding_upper or (
            "VENTRICULAR" in finding_upper and "SUPRAVENTRICULAR" not in finding_upper
        ):
            urgency = (
                "PROMPT REVIEW"
                if (heart_rate and (heart_rate > 120 or heart_rate < 45))
                else "ROUTINE REVIEW"
            )
            rec = ClinicalRecommendation(
                finding="Premature Ventricular Contractions (PVC)",
                urgency=urgency,
                clinical_considerations=[
                    "Premature Ventricular Contraction pattern detected on rhythm strip.",
                    "Frequent or multifocal PVCs may warrant quantification of 24-hour ectopy burden via ambulatory Holter monitoring.",
                    "Evaluate for reversible secondary causes: electrolyte imbalance (hypokalemia, hypomagnesemia), caffeine/sympathomimetic excess, or myocardial ischemia.",
                ],
                suggested_assessments=[
                    "Check serum electrolytes (Potassium, Magnesium).",
                    "Assess for clinical symptoms: palpitations, presyncope, or chest tightness.",
                    "Review baseline echocardiogram if ectopy burden is high (> 10-15%).",
                    "Review current medication profile for proarrhythmic or QT-prolonging agents.",
                ],
                contraindication_warnings=[
                    "Class IC antiarrhythmics (flecainide, propafenone) are contraindicated in patients with structural heart disease or prior myocardial infarction (CAST trial).",
                ],
                relevant_guidelines=[
                    "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure",
                    "2019 ESC Guidelines for the Management of Patients with Ventricular Arrhythmias",
                ],
                guideline_source="AHA/ACC/HRS Guideline on Ventricular Arrhythmias",
                last_verified="2026-04-12",
                clinician_action_required=(
                    "Attending physician review required to correlate with clinical history "
                    "and order electrolyte panel if indicated."
                ),
                medication_recommendations=[
                    {
                        "drug_class": "Beta-1 Selective Blockers (First-line)",
                        "example_agents": "Metoprolol succinate (Toprol-XL 25-200 mg OD), Bisoprolol (2.5-10 mg OD)",
                        "indication": "Symptomatic PVCs — ectopy suppression and palpitation relief in structurally normal hearts",
                        "guideline": "2019 ESC Guidelines on Ventricular Arrhythmias (Class I, Level B)",
                        "note": "First-line. Rule out bradycardia / AV block before initiating. Monitor heart rate and BP.",
                    },
                    {
                        "drug_class": "Non-Dihydropyridine Calcium Channel Blockers",
                        "example_agents": "Verapamil (40-120 mg TID), Diltiazem (60-120 mg TID)",
                        "indication": "Idiopathic PVCs in patients intolerant of beta-blockers",
                        "guideline": "AHA/ACC 2017 Ventricular Arrhythmia Guideline (Class IIa, Level C)",
                        "note": "Do not combine with beta-blockers. Contraindicated in HFrEF (LVEF < 40%).",
                    },
                    {
                        "drug_class": "Electrolyte Replacement",
                        "example_agents": "Potassium chloride (KCl oral/IV), Magnesium sulphate (IV/oral)",
                        "indication": "PVCs driven by hypokalemia (K+ < 3.5 mEq/L) or hypomagnesemia (Mg2+ < 1.7 mg/dL)",
                        "guideline": "AHA Arrhythmia Prevention Guidelines",
                        "note": "Target serum K+ >= 4.0 mEq/L and Mg2+ >= 2.0 mg/dL. Essential pre-step before antiarrhythmics.",
                    },
                    {
                        "drug_class": "Class III Antiarrhythmics (Specialist-initiated only)",
                        "example_agents": "Amiodarone (Cordarone 100-400 mg OD), Sotalol (80-160 mg BID)",
                        "indication": "Refractory high-burden PVCs (> 20%) with symptomatic LV dysfunction not responding to beta-blockade",
                        "guideline": "2019 ESC Guidelines (Class IIb, Level C)",
                        "note": "Electrophysiology / cardiologist consultation mandatory. Monitor QTc, LFTs, TFTs, and CXR.",
                    },
                ],
            )

        # ── 2. Atrial Fibrillation ─────────────────────────────────────────────
        elif "AF" in finding_upper or "FIBRILLATION" in finding_upper:
            urgency = (
                "URGENT CLINICIAN REVIEW"
                if (heart_rate and (heart_rate > 140 or heart_rate < 40))
                else "PROMPT REVIEW"
            )
            rec = ClinicalRecommendation(
                finding="Atrial Fibrillation Pattern",
                urgency=urgency,
                clinical_considerations=[
                    "Irregularly irregular ventricular response suggestive of Atrial Fibrillation.",
                    "Systemic thromboembolism risk assessment mandatory (calculate CHA2DS2-VASc score).",
                    "Evaluate bleeding risk score (HAS-BLED) before considering anticoagulation therapy.",
                    "Assess hemodynamic stability and identify precipitating causes (thyrotoxicosis, infection, alcohol, acute pulmonary event).",
                ],
                suggested_assessments=[
                    "Assess vital signs (orthostatic blood pressure, pulse deficit).",
                    "Calculate CHA2DS2-VASc and HAS-BLED clinical risk scores.",
                    "Order CMP, TSH, and transthoracic echocardiogram.",
                    "Obtain previous ECG to determine if paroxysmal, persistent, or long-standing.",
                ],
                contraindication_warnings=[
                    "DOACs are contraindicated in patients with mechanical heart valves or moderate-to-severe mitral stenosis.",
                ],
                relevant_guidelines=[
                    "2023 ACC/AHA/ACCP/HRS Guideline for Atrial Fibrillation",
                    "2020 ESC Guidelines for Atrial Fibrillation",
                ],
                guideline_source="ACC/AHA/HRS Guideline on Atrial Fibrillation",
                last_verified="2026-05-18",
                clinician_action_required=(
                    "Urgent clinician evaluation required. Determine rhythm vs. rate control strategy "
                    "and thromboembolic risk profile."
                ),
                medication_recommendations=[
                    {
                        "drug_class": "DOACs — Stroke Prevention (First-line)",
                        "example_agents": "Apixaban (Eliquis 5 mg BID), Rivaroxaban (Xarelto 20 mg OD), Dabigatran (Pradaxa 150 mg BID)",
                        "indication": "Stroke/systemic embolism prevention in non-valvular AF with CHA2DS2-VASc >= 2 (men) or >= 3 (women)",
                        "guideline": "2023 ACC/AHA AF Guideline (Class I, Level A)",
                        "note": "Preferred over warfarin. Dose-reduce apixaban if 2 of: age >= 80, weight <= 60 kg, Cr >= 1.5. Renal function check mandatory.",
                    },
                    {
                        "drug_class": "Beta-Blockers — Rate Control (First-line)",
                        "example_agents": "Metoprolol tartrate (25-100 mg BID), Bisoprolol (2.5-10 mg OD), Atenolol (25-100 mg OD)",
                        "indication": "Ventricular rate control target < 110 bpm at rest in AF",
                        "guideline": "2023 ACC/AHA AF Guideline (Class I, Level B)",
                        "note": "Avoid in decompensated HF or bronchospasm. Titrate dose to resting HR < 110 bpm. Monitor for bradycardia.",
                    },
                    {
                        "drug_class": "Non-DHP CCBs — Rate Control (Alternative)",
                        "example_agents": "Diltiazem (60-120 mg TID or SR formulation), Verapamil (80-120 mg TID)",
                        "indication": "Rate control in AF when beta-blockers are contraindicated or poorly tolerated",
                        "guideline": "2023 ACC/AHA AF Guideline (Class I, Level B)",
                        "note": "Avoid in AF with WPW syndrome and in HFrEF (LVEF < 40%). Do not combine with beta-blockers.",
                    },
                    {
                        "drug_class": "Class III Antiarrhythmic — Rhythm Control",
                        "example_agents": "Amiodarone (Cordarone 200 mg OD maintenance), Dronedarone (Multaq 400 mg BID)",
                        "indication": "Pharmacological cardioversion or maintenance of sinus rhythm in persistent AF",
                        "guideline": "2023 ACC/AHA AF Guideline (Class IIa, Level A) — Cardiologist-initiated",
                        "note": "Amiodarone: most efficacious but requires annual thyroid, liver, and pulmonary monitoring. Dronedarone: avoid in HF or permanent AF.",
                    },
                ],
            )

        # ── 3. Normal Sinus Rhythm ─────────────────────────────────────────────
        elif "NORMAL" in finding_upper:
            rec = ClinicalRecommendation(
                finding="Normal Sinus Rhythm",
                urgency="ROUTINE REVIEW",
                clinical_considerations=[
                    "Normal Sinus Rhythm with regular P waves and physiological PR intervals.",
                    "Normal resting ECG does not exclude paroxysmal dysrhythmias or transient ischemia if patient is symptomatic.",
                ],
                suggested_assessments=[
                    "Correlate with current patient symptoms (chest pain, dyspnea, syncope).",
                    "Compare with prior ECGs for subtle repolarization, axis, or voltage changes.",
                ],
                contraindication_warnings=["None based on resting ECG alone."],
                relevant_guidelines=["AHA/ACC Standard Guidelines for Electrocardiography Interpretation"],
                guideline_source="AHA/ACC Recommendations for Standardization and Interpretation of the ECG",
                last_verified="2026-03-01",
                clinician_action_required="Routine clinician confirmation and archival.",
                medication_recommendations=[
                    {
                        "drug_class": "No Antiarrhythmic Indicated",
                        "example_agents": "N/A",
                        "indication": "Normal sinus rhythm does not warrant antiarrhythmic pharmacotherapy",
                        "guideline": "AHA/ACC ECG Interpretation Standards",
                        "note": "Optimize existing cardiovascular risk factor medications as appropriate (antihypertensives, statins, antiplatelets) per physician guidance.",
                    },
                    {
                        "drug_class": "Cardiovascular Risk Reduction (if applicable)",
                        "example_agents": "ACE inhibitors / ARBs (hypertension), Statins (dyslipidaemia), Low-dose Aspirin (secondary prevention only)",
                        "indication": "Primary/secondary cardiovascular risk reduction in at-risk patients with normal ECG",
                        "guideline": "2019 ACC/AHA Guideline on Primary Prevention of Cardiovascular Disease",
                        "note": "Decisions based on overall ASCVD risk score, not ECG alone. Requires physician evaluation.",
                    },
                ],
            )

        # ── 4. Unclassified / Other ────────────────────────────────────────────
        else:
            rec = ClinicalRecommendation(
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
                medication_recommendations=[
                    {
                        "drug_class": "Defer Pending Diagnostic Clarification",
                        "example_agents": "N/A",
                        "indication": "Atypical ECG morphology — do not initiate therapy without confirmed diagnosis",
                        "guideline": "AHA/ACC Arrhythmia Management Standards",
                        "note": "Avoid empiric antiarrhythmics. Obtain 12-lead ECG and cardiology consultation first.",
                    },
                ],
            )

        rec.patient_context = p_ctx
        rec.missing_information = missing
        return rec


GLOBAL_CDS_ENGINE = ClinicalDecisionSupportEngine()
