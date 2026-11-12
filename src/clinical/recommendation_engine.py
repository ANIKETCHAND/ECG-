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
from typing import Any, Dict, List, Optional, Union

from src.database.db_manager import PatientRecord


@dataclass
class FindingReliability:
    """Whether the ECG finding being acted on is trustworthy enough to act on.

    A therapy consideration attached to an unreliable reading is worse than no
    consideration at all: it launders a bad measurement into a clinical decision.
    This object carries the measured facts that decide the question, so the
    decision is auditable rather than a judgement call buried in code.
    """

    reliable: bool = False
    reasons: List[str] = field(default_factory=list)
    signal_quality: str = "UNKNOWN"
    selective_gate_applied: bool = False
    reported_coverage: Optional[float] = None
    abstained: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_analysis_result(
        cls, analysis: Any, min_reported_coverage: float = 0.50
    ) -> "FindingReliability":
        """Derive reliability from an :class:`ECGAnalysisResult`.

        ``analysis`` may be an object or an already-serialised dict.
        """
        get = (
            (lambda key, default=None: analysis.get(key, default))
            if isinstance(analysis, dict)
            else (lambda key, default=None: getattr(analysis, key, default))
        )
        quality = str(get("signal_quality", "UNKNOWN") or "UNKNOWN").upper()
        abstained = bool(get("abstain", False))
        gate_applied = bool(get("selective_gate_applied", False))
        coverage_raw = get("reported_coverage", None)
        coverage = None if coverage_raw is None else float(coverage_raw)
        prediction = str(get("prediction", "") or "")

        reasons: List[str] = []
        if quality not in {"GOOD", "ACCEPTABLE"}:
            reasons.append(
                f"Signal quality is {quality}; the waveform itself is not reliable enough "
                "to justify a therapy consideration."
            )
        if abstained:
            reasons.append(
                "The analysis declined to commit to a rhythm statement "
                f"({get('abstain_reason') or 'abstention raised by the inference engine'})."
            )
        if prediction.startswith(("NO_RESULT", "AI_SUPPRESSED", "INSUFFICIENT_CONFIDENCE", "PIPELINE_EXECUTION_FAILURE", "UNSUPPORTED_LEAD")):
            reasons.append(f"No usable AI finding was produced (prediction={prediction!r}).")
        if not gate_applied:
            reasons.append(
                "No measured abstention operating point is installed, so the model's "
                "confidence in this finding is unquantified."
            )
        elif coverage is not None and coverage < min_reported_coverage:
            reasons.append(
                f"Only {coverage:.0%} of detected beats could be reported with confidence "
                f"(minimum {min_reported_coverage:.0%}). The rhythm conclusion rests on a "
                "minority of the recording."
            )

        return cls(
            reliable=not reasons,
            reasons=reasons,
            signal_quality=quality,
            selective_gate_applied=gate_applied,
            reported_coverage=coverage,
            abstained=abstained,
            evidence={
                "prediction": prediction,
                "min_reported_coverage": min_reported_coverage,
                "abstain_reason": get("abstain_reason", None),
                "quality_score": get("quality_score", None),
                "selective_gate": get("selective_gate", None),
            },
        )


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
    # Evidence gate: when the reading behind a recommendation is not reliable, the
    # therapy suggestions are withheld rather than presented. They are retained in
    # ``withheld_medication_recommendations`` for audit, never deleted silently.
    medication_recommendations_withheld: bool = False
    withheld_medication_recommendations: List[Dict[str, str]] = field(default_factory=list)
    withheld_reason: Optional[str] = None
    reliability: Optional[Dict[str, Any]] = None
    relevant_conditions: List[Dict[str, str]] = field(default_factory=list)
    relevant_medications: List[Dict[str, str]] = field(default_factory=list)
    relevant_allergies: List[Dict[str, str]] = field(default_factory=list)
    relevant_laboratory_results: List[Dict[str, str]] = field(default_factory=list)
    potential_contraindications: List[str] = field(default_factory=list)
    medication_related_ecg_concerns: List[Dict[str, str]] = field(default_factory=list)
    relevance_explanations: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalDecisionSupportEngine:
    """Generates evidence-backed clinical decision support considerations."""

    def explain_clinical_context_relevance(
        self,
        ecg_finding: str,
        heart_rate: Optional[float] = None,
        patient: Optional[Any] = None,
        vital_signs: Optional[Dict[str, Any]] = None,
        laboratory_results: Optional[Dict[str, Any]] = None,
        ecg_measurements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Explain WHY patient factors are relevant to the active ECG interpretation.

        Uses clinician-reviewed, evidence-based guidelines (ACC/AHA, ESC, KDIGO, ADA).
        Avoids unsupported causal claims and provides traceable citations.
        """
        p_dict = {}
        if patient is not None:
            p_dict = patient.to_dict() if hasattr(patient, "to_dict") else (patient if isinstance(patient, dict) else vars(patient))

        v_dict = vital_signs or (p_dict.get("vital_signs") if isinstance(p_dict.get("vital_signs"), dict) else {}) or (p_dict.get("vitals") if isinstance(p_dict.get("vitals"), dict) else {}) or {}
        l_dict = laboratory_results or (p_dict.get("laboratory_results") if isinstance(p_dict.get("laboratory_results"), dict) else {}) or (p_dict.get("labs") if isinstance(p_dict.get("labs"), dict) else {}) or {}
        m_dict = ecg_measurements or {}

        finding_u = str(ecg_finding or "").upper()
        cond_parts = [
            str(p_dict.get("conditions", "")),
            str(p_dict.get("existing_conditions", "")),
            str(p_dict.get("cardiac_history", "")),
            str(p_dict.get("previous_cardiac_history", "")),
            str(p_dict.get("past_medical_history", "")),
        ]
        for c_field, c_label in [
            ("diabetes", "diabetes mellitus"),
            ("hypertension", "hypertension"),
            ("coronary_artery_disease", "coronary artery disease"),
            ("previous_heart_attack", "myocardial infarction"),
            ("heart_failure", "heart failure"),
            ("arrhythmia_history", "arrhythmia history"),
            ("kidney_disease", "chronic kidney disease ckd renal"),
            ("liver_disease", "liver disease cirrhosis hepatic"),
            ("thyroid_disorders", "thyroid disorder"),
            ("other_diagnosed_diseases", "other diagnosed diseases"),
        ]:
            val = p_dict.get(c_field)
            if val and str(val).lower() not in ["none", "no", "0", "false", ""]:
                cond_parts.append(c_label)
        cond_str = " ".join(cond_parts).lower()

        allergies_parts = [
            str(p_dict.get("allergies") or ""),
            str(p_dict.get("known_allergies") or ""),
            str(p_dict.get("food_allergies") or ""),
            str(p_dict.get("other_allergies") or ""),
        ]
        allergies_str = " ".join(allergies_parts).lower().strip()

        meds_parts = [
            str(p_dict.get("current_medications") or ""),
            str(p_dict.get("past_medications") or ""),
            str(p_dict.get("discontinued_medications") or ""),
        ]
        curr_meds_str = " ".join(meds_parts).lower().strip()

        relevant_conditions: List[Dict[str, str]] = []
        relevant_medications: List[Dict[str, str]] = []
        relevant_allergies: List[Dict[str, str]] = []
        relevant_labs: List[Dict[str, str]] = []
        potential_contraindications: List[str] = []
        med_ecg_concerns: List[Dict[str, str]] = []
        relevance_explanations: List[Dict[str, str]] = []
        missing_info: List[str] = []

        # 1. Condition Relevance Explainability
        if any(k in cond_str for k in ["hypertension", "htn", "high blood pressure"]):
            exp = {
                "factor": "Hypertension",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Chronic left ventricular afterload promotes left ventricular hypertrophy (LVH), repolarization strain, and atrial dilation predisposing to supraventricular ectopy / atrial fibrillation.",
                "clinical_guideline": "2017 ACC/AHA / 2024 ESC Hypertension Guidelines",
            }
            relevant_conditions.append({"condition": "Hypertension", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["diabetes", "dm", "t2dm", "t1dm"]):
            exp = {
                "factor": "Diabetes Mellitus",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Diabetic autonomic neuropathy blunts heart rate variability, causes resting sinus tachycardia, and elevates risk of asymptomatic / silent myocardial ischemia.",
                "clinical_guideline": "2024 ADA Standards of Care / 2023 ESC CVD in Diabetes",
            }
            relevant_conditions.append({"condition": "Diabetes Mellitus", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["cad", "coronary", "myocardial infarction", "prior mi", "infarction", "stent", "cabg"]):
            exp = {
                "factor": "Coronary Artery Disease / Prior MI",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Ischemic or scarred myocardium creates localized conduction deceleration, fragmented QRS complexes, pathologic Q waves, and re-entrant ventricular ectopy substrate.",
                "clinical_guideline": "2017 AHA/ACC/HRS Ventricular Arrhythmia Guidelines",
            }
            relevant_conditions.append({"condition": "Coronary Artery Disease / Prior MI", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["heart failure", "hf", "chf", "hfref", "hfpef", "cardiomyopathy"]):
            exp = {
                "factor": "Heart Failure",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Elevated ventricular diastolic pressures and neurohormonal remodeling cause intraventricular conduction delays (LBBB), secondary ST-T changes, and increased ectopy burden.",
                "clinical_guideline": "2022 AHA/ACC/HFSA Heart Failure Guideline",
            }
            relevant_conditions.append({"condition": "Heart Failure", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["ckd", "kidney", "renal", "nephropathy"]):
            exp = {
                "factor": "Chronic Kidney Disease",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Impaired electrolyte filtration predisposes to hyperkalemia (peaked T waves, QRS widening) and reduced renal excretion of antiarrhythmic and cardiorenal drugs.",
                "clinical_guideline": "2024 KDIGO Clinical Practice Guideline",
            }
            relevant_conditions.append({"condition": "Chronic Kidney Disease", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["liver", "hepatic", "cirrhosis", "hepatitis"]):
            exp = {
                "factor": "Liver Disease / Hepatic Impairment",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Cirrhotic cardiomyopathy presents with baseline electrophysiological QTc prolongation and blunted chronotropic response; reduced hepatic CYP drug clearance.",
                "clinical_guideline": "2023 EASL Practice Guidelines",
            }
            relevant_conditions.append({"condition": "Liver Disease", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["thyroid", "hypothyroid", "hyperthyroid", "graves", "hashimoto"]):
            exp = {
                "factor": "Thyroid Disorder",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Thyroid status directly regulates cardiac beta-receptors; hyperthyroidism promotes sinus tachycardia and AFib, while hypothyroidism causes bradycardia, low voltage, and QTc prolongation.",
                "clinical_guideline": "2022 American Thyroid Association Guidelines",
            }
            relevant_conditions.append({"condition": "Thyroid Disorder", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        if any(k in cond_str for k in ["arrhythmia", "pvc", "afib", "a-fib", "flutter", "svt"]):
            exp = {
                "factor": "Documented Arrhythmia History",
                "category": "EXISTING_CONDITION",
                "ecg_relationship": "Historical dysrhythmia indicates pre-existing electrophysiological substrate; active finding must be compared against baseline ECG recordings.",
                "clinical_guideline": "2019/2020 ESC Arrhythmia Guidelines",
            }
            relevant_conditions.append({"condition": "Arrhythmia History", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        # 2. Medication ECG Concerns & Relevance
        hr_val = float(heart_rate or m_dict.get("heart_rate_bpm") or v_dict.get("heart_rate") or 72.0)
        qtc_val = float(m_dict.get("qtc_bazett_ms") or m_dict.get("qtc_ms") or 420.0)

        if any(k in curr_meds_str for k in ["amiodarone", "sotalol", "dronedarone", "flecainide"]):
            exp = {
                "factor": "Class I/III Antiarrhythmics",
                "category": "MEDICATION_REGIMEN",
                "ecg_relationship": "Blocks cardiac potassium or sodium channels; prolongs repolarization (QTc interval) or conduction (QRS width). Strict surveillance required to avoid Torsades de Pointes.",
                "clinical_guideline": "AHA/ACC/HRS Antiarrhythmic Safety Standards",
            }
            relevant_medications.append({"medication": "Class I/III Antiarrhythmic", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)
            if qtc_val >= 470.0:
                med_ecg_concerns.append({
                    "concern": "Antiarrhythmic QTc Prolongation Risk",
                    "mechanism": f"Current antiarrhythmic therapy combined with measured QTc of {qtc_val:.0f} ms exceeds safe threshold. Risk of ventricular tachyarrhythmia.",
                    "action": "Urgent ECG over-read and serum potassium/magnesium check mandatory.",
                })

        if any(k in curr_meds_str for k in ["metoprolol", "bisoprolol", "carvedilol", "atenolol", "diltiazem", "verapamil"]):
            exp = {
                "factor": "AV Nodal Blocking Agents (Beta-Blocker / Non-DHP CCB)",
                "category": "MEDICATION_REGIMEN",
                "ecg_relationship": "Suppresses sinoatrial node firing and prolongs AV conduction. Can induce sinus bradycardia, sinus pause, or higher-degree AV block.",
                "clinical_guideline": "2018 ACC/AHA/HRS Bradycardia Guideline",
            }
            relevant_medications.append({"medication": "AV Nodal Blocker", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)
            if hr_val < 55.0:
                med_ecg_concerns.append({
                    "concern": "Medication-Induced Bradycardia",
                    "mechanism": f"Heart rate {hr_val:.0f} bpm under AV nodal blockade. Patient at risk of symptomatic chronotropic incompetence or syncope.",
                    "action": "Consider dose titration or temporary withholding under physician direction.",
                })

        if any(k in curr_meds_str for k in ["digoxin", "lanoxin"]):
            exp = {
                "factor": "Digoxin",
                "category": "MEDICATION_REGIMEN",
                "ecg_relationship": "Inhibits Na+/K+-ATPase; characteristic 'sagging' ST-T depression ('digitalis effect'); toxicity triggers PVCs, bidirectional VT, and AV junctional rhythms.",
                "clinical_guideline": "ACC/AHA HF Guidelines",
            }
            relevant_medications.append({"medication": "Digoxin", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
            relevance_explanations.append(exp)

        # 3. Laboratory Relevance
        k_val = l_dict.get("potassium_mmol_l") or l_dict.get("potassium_meq_l") or l_dict.get("serum_potassium") or l_dict.get("potassium")
        if k_val is not None:
            k_f = float(k_val)
            if k_f < 3.5:
                exp = {
                    "factor": f"Hypokalemia (K+ {k_f:.2f} mmol/L)",
                    "category": "LABORATORY_PANEL",
                    "ecg_relationship": "Low serum potassium hyperpolarizes myocardial resting potential, promotes early afterdepolarizations, prolongs QTc, and directly triggers ventricular ectopy / PVCs.",
                    "clinical_guideline": "AHA Arrhythmia Prevention Standards",
                }
                relevant_labs.append({"lab": f"Potassium: {k_f:.2f} mmol/L (LOW)", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                potential_contraindications.append(f"Hypokalemia ({k_f:.2f} mmol/L) potentiates fatal arrhythmias with Class III antiarrhythmics or Digoxin. Repletion required.")
            elif k_f >= 5.2:
                exp = {
                    "factor": f"Hyperkalemia (K+ {k_f:.2f} mmol/L)",
                    "category": "LABORATORY_PANEL",
                    "ecg_relationship": "Elevated serum potassium decreases resting membrane potential, leading to peaked T waves, flattened P waves, PR prolongation, and terminal sinoventricular conduction arrest.",
                    "clinical_guideline": "ERC / AHA Resuscitation Guidelines",
                }
                relevant_labs.append({"lab": f"Potassium: {k_f:.2f} mmol/L (HIGH)", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                potential_contraindications.append(f"Hyperkalemia ({k_f:.2f} mmol/L) contraindicates potassium repletion, potassium-sparing diuretics, and ACEi/ARBs.")

        cr_val = l_dict.get("serum_creatinine_mg_dl") or l_dict.get("creatinine_mg_dl") or l_dict.get("serum_creatinine") or l_dict.get("creatinine")
        egfr_val = l_dict.get("egfr_ml_min") or l_dict.get("egfr")
        if cr_val is not None or egfr_val is not None:
            cr_f = float(cr_val) if cr_val is not None else 1.0
            egfr_f = float(egfr_val) if egfr_val is not None else max(15.0, 110.0 - (float(p_dict.get("age") or 55.0) * 0.6) - (cr_f * 22.0))
            if cr_f > 1.5 or egfr_f < 60.0:
                exp = {
                    "factor": f"Renal Dysfunction (Cr {cr_f:.2f} mg/dL, eGFR {egfr_f:.0f} mL/min)",
                    "category": "LABORATORY_PANEL",
                    "ecg_relationship": "Decreased glomerular filtration impairs renal clearance of renally eliminated cardiovascular agents (sotalol, flecainide, digoxin, DOACs, metformin).",
                    "clinical_guideline": "KDIGO 2024 Clinical Practice Guideline",
                }
                relevant_labs.append({"lab": f"Creatinine: {cr_f:.2f} mg/dL, eGFR: {egfr_f:.0f} mL/min", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                if egfr_f < 30.0:
                    potential_contraindications.append(f"Severe CKD (eGFR {egfr_f:.0f} mL/min < 30) strictly contraindicates Metformin (lactic acidosis) and requires DOAC dose reduction / avoidance.")

        # LFTs
        alt_val = l_dict.get("alt_u_l") or l_dict.get("alt")
        ast_val = l_dict.get("ast_u_l") or l_dict.get("ast")
        bili_val = l_dict.get("bilirubin_mg_dl") or l_dict.get("bilirubin")
        if alt_val is not None or ast_val is not None or bili_val is not None:
            alt_f = float(alt_val or 0.0)
            ast_f = float(ast_val or 0.0)
            bili_f = float(bili_val or 0.0)
            if alt_f >= 120.0 or ast_f >= 120.0 or bili_f >= 2.0:
                exp = {
                    "factor": f"Hepatic Transaminitis (ALT {alt_f:.0f} U/L, AST {ast_f:.0f} U/L, Bilirubin {bili_f:.1f} mg/dL)",
                    "category": "LABORATORY_PANEL",
                    "ecg_relationship": "Severe hepatocellular injury impairs hepatic CYP-mediated drug clearance and alters systemic volume dynamics; high risk of hepatotoxic drug accumulation.",
                    "clinical_guideline": "FDA Drug Labeling / EASL Guidelines",
                }
                relevant_labs.append({"lab": f"LFTs elevated (ALT {alt_f:.0f}, AST {ast_f:.0f}, Bili {bili_f:.1f})", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                potential_contraindications.append(f"Active hepatic failure / transaminitis contraindicates statin initiation and warrants extreme caution with Amiodarone.")

        # 4. Allergy Relevance
        if allergies_str and allergies_str not in ("none", "no known allergies", "none documented"):
            if "beta" in allergies_str:
                exp = {
                    "factor": "Documented Beta-Blocker Allergy",
                    "category": "ALLERGIC_HYPERSENSITIVITY",
                    "ecg_relationship": "Anaphylaxis or severe bronchospasm risk; first-line rate control and ectopy suppression must be achieved via Non-DHP CCB or other non-beta agent.",
                    "clinical_guideline": "DailyMed Product Labeling / ACC Guidelines",
                }
                relevant_allergies.append({"allergy": "Beta-Blockers", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                potential_contraindications.append("Beta-blocker class contraindicated due to documented hypersensitivity.")

            if any(k in allergies_str for k in ["sulfa", "sulfonamide"]):
                exp = {
                    "factor": "Sulfa Hypersensitivity",
                    "category": "ALLERGIC_HYPERSENSITIVITY",
                    "ecg_relationship": "Cross-reactivity risk with sulfonamide-derived diuretics (furosemide, bumetanide, torsemide) and certain antiarrhythmics (sotalol).",
                    "clinical_guideline": "FDA SPL / DailyMed",
                }
                relevant_allergies.append({"allergy": "Sulfa Drugs", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)

            if any(k in allergies_str for k in ["aspirin", "nsaid", "salicylate"]):
                exp = {
                    "factor": "Aspirin / NSAID Hypersensitivity",
                    "category": "ALLERGIC_HYPERSENSITIVITY",
                    "ecg_relationship": "Contraindicates aspirin for secondary ischemic cardioprotection; alternative antiplatelet (clopidogrel) indicated.",
                    "clinical_guideline": "2023 ESC ACS Guidelines",
                }
                relevant_allergies.append({"allergy": "Aspirin / NSAIDs", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                potential_contraindications.append("Aspirin contraindicated due to documented hypersensitivity. Use Clopidogrel.")

            if any(k in allergies_str for k in ["ace", "lisinopril", "ramipril", "angioedema"]):
                exp = {
                    "factor": "ACE Inhibitor Allergy / Angioedema",
                    "category": "ALLERGIC_HYPERSENSITIVITY",
                    "ecg_relationship": "Risk of life-threatening angioedema; ACE inhibitors strictly contraindicated; ARB or ARNI substituted under close observation.",
                    "clinical_guideline": "2022 AHA/ACC HF Guidelines",
                }
                relevant_allergies.append({"allergy": "ACE Inhibitors", "relevance": exp["ecg_relationship"], "guideline": exp["clinical_guideline"]})
                relevance_explanations.append(exp)
                potential_contraindications.append("ACE inhibitors strictly contraindicated due to history of hypersensitivity/angioedema.")

        # 5. Missing Clinical Information Audit
        has_bp = bool(v_dict.get("systolic_bp") or v_dict.get("systolic_bp_mmhg") or v_dict.get("blood_pressure") or v_dict.get("sbp"))
        if not has_bp:
            missing_info.append("Blood pressure not recorded: hemodynamic stability during dysrhythmia cannot be verified.")
        if k_val is None:
            missing_info.append("Serum potassium not recorded: electrolyte trigger for arrhythmia cannot be excluded.")
        if cr_val is None and egfr_val is None:
            missing_info.append("Renal function (Creatinine/eGFR) not recorded: antiarrhythmic and cardiorenal clearance unverified.")
        if not p_dict.get("current_medications") or str(p_dict.get("current_medications")).lower() in ["none recorded", "none documented", ""]:
            missing_info.append("Complete medication history unverified: drug-induced arrhythmia or interactions cannot be ruled out.")
        if not allergies_str or allergies_str in ["none", "none documented", ""]:
            missing_info.append("Allergy status unverified: patient hypersensitivities require bedside confirmation.")

        return {
            "relevant_conditions": relevant_conditions,
            "relevant_medications": relevant_medications,
            "relevant_allergies": relevant_allergies,
            "relevant_laboratory_results": relevant_labs,
            "potential_contraindications": potential_contraindications,
            "medication_related_ecg_concerns": med_ecg_concerns,
            "relevance_explanations": relevance_explanations,
            "missing_information": missing_info,
        }

    def evaluate_finding(
        self,
        ecg_finding: str,
        heart_rate: Optional[float] = None,
        patient: Optional[Any] = None,
        machine_finding: Optional[str] = None,
        reliability: Optional[Union["FindingReliability", Dict[str, Any], Any]] = None,
        vital_signs: Optional[Dict[str, Any]] = None,
        laboratory_results: Optional[Dict[str, Any]] = None,
        ecg_measurements: Optional[Dict[str, Any]] = None,
    ) -> ClinicalRecommendation:
        """Build traceable clinical considerations for an ECG finding.

        Args:
            reliability: The reliability of the finding being acted on. Pass a
                :class:`FindingReliability`, a raw ``ECGAnalysisResult``, or a
                dict of its fields. When the finding is unreliable, therapy
                considerations are withheld instead of presented. ``None`` means
                "not supplied" and is treated as unverified.
        """
        finding_upper = ecg_finding.upper()

        resolved_reliability: Optional[FindingReliability] = None
        if reliability is not None:
            if isinstance(reliability, FindingReliability):
                resolved_reliability = reliability
            elif isinstance(reliability, dict) and "signal_quality" in reliability:
                resolved_reliability = FindingReliability(**{
                    k: v for k, v in reliability.items()
                    if k in FindingReliability.__dataclass_fields__
                })
            else:
                resolved_reliability = FindingReliability.from_analysis_result(reliability)

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
                "Medication history: Not provided",
                "Medical history: Not provided",
            ]
            if not vital_signs:
                missing.append("Vital signs: Not provided")
            if not laboratory_results:
                missing.append("Laboratory data: Not provided")
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
            if not vital_signs and not p_dict.get("vital_signs") and not p_dict.get("vitals"):
                missing.append("Vital signs: Not provided")
            if not laboratory_results and not p_dict.get("laboratory_results") and not p_dict.get("labs"):
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

        # ── Evidence gate: no reliable finding -> no therapy suggestion ────────
        # A drug consideration derived from an unreliable read is worse than no
        # consideration, because it carries the authority of an AI finding. The
        # suggestions are moved aside, never deleted, so an auditor can see both
        # what was proposed and why it was held back.
        if resolved_reliability is None:
            rec.reliability = FindingReliability(
                reliable=False,
                reasons=[
                    "No reliability information was supplied with this finding, so it "
                    "cannot be verified as trustworthy enough to act on."
                ],
            ).to_dict()
            _withhold_medications(
                rec,
                "Reliability of the underlying ECG finding was not supplied. Therapy "
                "considerations are withheld until the source analysis is attached.",
            )
        else:
            rec.reliability = resolved_reliability.to_dict()
            if not resolved_reliability.reliable:
                _withhold_medications(rec, " ".join(resolved_reliability.reasons))

        rec.patient_context = p_ctx
        rec.missing_information = missing

        # Integrate contextual explainability & Section 3 evidence
        relevance_data = self.explain_clinical_context_relevance(
            ecg_finding=ecg_finding,
            heart_rate=heart_rate,
            patient=patient,
            vital_signs=vital_signs,
            laboratory_results=laboratory_results,
            ecg_measurements=ecg_measurements,
        )
        rec.relevant_conditions = relevance_data["relevant_conditions"]
        rec.relevant_medications = relevance_data["relevant_medications"]
        rec.relevant_allergies = relevance_data["relevant_allergies"]
        rec.relevant_laboratory_results = relevance_data["relevant_laboratory_results"]
        rec.potential_contraindications = list(set(rec.contraindication_warnings + relevance_data["potential_contraindications"]))
        rec.medication_related_ecg_concerns = relevance_data["medication_related_ecg_concerns"]
        rec.relevance_explanations = relevance_data["relevance_explanations"]
        for m_item in relevance_data["missing_information"]:
            if m_item not in rec.missing_information:
                rec.missing_information.append(m_item)

        return rec


def _withhold_medications(rec: ClinicalRecommendation, reason: str) -> None:
    """Move therapy suggestions aside when the underlying finding is unreliable."""
    if not rec.medication_recommendations and not rec.withheld_medication_recommendations:
        return
    if rec.medication_recommendations:
        rec.withheld_medication_recommendations.extend(rec.medication_recommendations)
        rec.medication_recommendations = []
    rec.medication_recommendations_withheld = True
    rec.withheld_reason = reason
    rec.urgency = _escalate_urgency(rec.urgency)
    rec.clinical_considerations.append(
        "Medication considerations were withheld: the ECG finding supporting them was "
        f"not reliable enough to act on. {reason}"
    )
    rec.clinician_action_required = (
        "Repeat acquisition or manual over-read required before any therapy decision is "
        "considered. " + rec.clinician_action_required
    )


def _escalate_urgency(urgency: str) -> str:
    """Raise the review urgency to at least PROMPT REVIEW."""
    order = ["ROUTINE REVIEW", "PROMPT REVIEW", "URGENT CLINICIAN REVIEW"]
    try:
        idx = order.index(urgency)
    except ValueError:
        return urgency
    return order[max(idx, 1)]


GLOBAL_CDS_ENGINE = ClinicalDecisionSupportEngine()
