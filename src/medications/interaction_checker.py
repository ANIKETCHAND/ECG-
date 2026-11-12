"""
Medication Interaction & Safety Verification Engine.
Checks:
- Drug-drug interactions across patient's current medications.
- Contraindications against patient's known clinical conditions.
- Known allergy conflicts.
- Missing context safety halts ("INSUFFICIENT CLINICAL CONTEXT").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from database.db_manager import PatientRecord
    from medications.medication_database import GLOBAL_MEDICATION_DB, MedicationEntry
except (ImportError, ValueError):
    from src.database.db_manager import PatientRecord
    from src.medications.medication_database import GLOBAL_MEDICATION_DB, MedicationEntry


@dataclass
class SafetyAlert:
    category: str  # INTERACTION, CONTRAINDICATION, ALLERGY, CONTEXT_WARNING
    severity: str  # CRITICAL, MAJOR, MODERATE, INFO
    title: str
    description: str
    clinical_recommendation: str
    source: str
    evidence_level: str


@dataclass
class MedicationSafetyReport:
    patient_id: str
    analyzed_medications: List[str]
    has_critical_alerts: bool
    context_completeness: str  # COMPLETE, PARTIAL, INSUFFICIENT
    missing_context_fields: List[str]
    alerts: List[SafetyAlert]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "analyzed_medications": self.analyzed_medications,
            "has_critical_alerts": self.has_critical_alerts,
            "context_completeness": self.context_completeness,
            "missing_context_fields": self.missing_context_fields,
            "alerts": [asdict(a) for a in self.alerts],
        }


def check_medication_safety(
    medication_names: Optional[List[str]] = None,
    patient: Optional[Any] = None,
    ecg_finding: Optional[str] = None,
    ecg_measurements: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
    medications: Optional[List[str]] = None,
) -> MedicationSafetyReport:
    """Evaluate drug interactions, contraindications, allergy safety, and multimodal physiological conflicts."""
    med_names = medication_names or medications or []
    alerts: List[SafetyAlert] = []
    missing_fields: List[str] = []

    # 1. Evaluate Patient Clinical Context Completeness
    missing_fields = []
    p_dict = {}
    if patient is not None:
        p_dict = patient.to_dict() if hasattr(patient, "to_dict") else (patient if isinstance(patient, dict) else vars(patient))

    if not p_dict:
        missing_fields = ["Patient Record", "Age", "Allergies", "Known Conditions", "Renal/Hepatic Status"]
        context_status = "INSUFFICIENT"
        alerts.append(
            SafetyAlert(
                category="CONTEXT_WARNING",
                severity="CRITICAL",
                title="INSUFFICIENT CLINICAL CONTEXT",
                description="No verified patient clinical record linked to this safety evaluation.",
                clinical_recommendation="Mandatory clinician review: verify complete patient history and allergies before administering therapy.",
                source="AAMI TIR57 / Clinical Safety Guidelines",
                evidence_level="Level A",
            )
        )
    else:
        known_allergies_val = str(p_dict.get("known_allergies") or p_dict.get("allergies") or "").strip()
        existing_cond_val = str(p_dict.get("existing_conditions") or p_dict.get("conditions") or "").strip()
        age_val = p_dict.get("age")

        if not known_allergies_val or known_allergies_val.strip() == "" or known_allergies_val.lower() in ["unknown", "unverified"]:
            missing_fields.append("Allergy Record")
        if not existing_cond_val or existing_cond_val.strip() == "" or existing_cond_val.lower() in ["unknown", "unverified"]:
            # Check specific condition flags before declaring conditions missing
            has_specific = any(
                p_dict.get(k) and str(p_dict.get(k)).lower() not in ["none", "no", "0", "false", ""]
                for k in ["diabetes", "hypertension", "coronary_artery_disease", "kidney_disease", "liver_disease", "heart_failure"]
            )
            if not has_specific:
                missing_fields.append("Existing Conditions")
        if age_val is None:
            missing_fields.append("Age")

        if len(missing_fields) >= 2:
            context_status = "INSUFFICIENT"
            alerts.append(
                SafetyAlert(
                    category="CONTEXT_WARNING",
                    severity="CRITICAL",
                    title="INSUFFICIENT CLINICAL CONTEXT",
                    description=f"Missing critical clinical context: {', '.join(missing_fields)}.",
                    clinical_recommendation="Do not assume absence of allergies or organ impairment. Verify with clinician.",
                    source="Hospital Patient Safety Guidelines",
                    evidence_level="Level A",
                )
            )
        elif len(missing_fields) > 0:
            context_status = "PARTIAL"
        else:
            context_status = "COMPLETE"

    # Normalize medications
    cleaned_meds = [m.strip().lower() for m in med_names if m and str(m).strip()]
    med_entries: Dict[str, MedicationEntry] = {}
    for m in cleaned_meds:
        entry = GLOBAL_MEDICATION_DB.get_by_name(m)
        if entry:
            med_entries[entry.generic_name.lower()] = entry

    # 2. Check Drug-Drug Interactions
    entry_list = list(med_entries.values())
    checked_pairs = set()
    for i in range(len(entry_list)):
        for j in range(i + 1, len(entry_list)):
            m1 = entry_list[i]
            m2 = entry_list[j]
            for source_med, target_med in [(m1, m2), (m2, m1)]:
                for inter in source_med.known_interactions:
                    target_name_lower = target_med.generic_name.lower()
                    inter_drug_lower = inter.interacting_drug.lower()
                    if (
                        inter_drug_lower in target_name_lower
                        or any(inter_drug_lower in b.lower() for b in target_med.brand_names)
                        or any(part.strip() in target_name_lower for part in inter_drug_lower.split("/") if len(part.strip()) > 3)
                    ):
                        pair_key = (tuple(sorted([source_med.generic_name, target_med.generic_name])), inter.clinical_effect[:30])
                        if pair_key not in checked_pairs:
                            checked_pairs.add(pair_key)
                            alerts.append(
                                SafetyAlert(
                                    category="INTERACTION",
                                    severity=inter.severity,
                                    title=f"Potential Interaction: {source_med.generic_name} + {target_med.generic_name}",
                                    description=inter.clinical_effect,
                                    clinical_recommendation=inter.management_recommendation,
                                    source=inter.source,
                                    evidence_level=inter.evidence_level,
                                )
                            )

    # 3. Check Contraindications against patient's existing conditions
    cond_text = ""
    if p_dict:
        cond_parts = [
            str(p_dict.get("existing_conditions") or ""),
            str(p_dict.get("conditions") or ""),
            str(p_dict.get("cardiac_history") or ""),
            str(p_dict.get("past_medical_history") or ""),
        ]
        for c_field, c_label in [
            ("diabetes", "diabetes mellitus"),
            ("hypertension", "hypertension"),
            ("coronary_artery_disease", "coronary artery disease"),
            ("previous_heart_attack", "myocardial infarction"),
            ("heart_failure", "heart failure"),
            ("arrhythmia_history", "arrhythmia"),
            ("kidney_disease", "kidney disease ckd"),
            ("liver_disease", "liver disease cirrhosis"),
            ("thyroid_disorders", "thyroid disorder"),
        ]:
            val = p_dict.get(c_field)
            if val and str(val).lower() not in ["none", "no", "0", "false", ""]:
                cond_parts.append(c_label)
        cond_text = " ".join(cond_parts).lower()

    if cond_text.strip():
        for med in entry_list:
            for contra in med.contraindications:
                # Check keyword overlap (e.g., bradycardia, block, asthma)
                words = [w for w in contra.lower().split() if len(w) > 4]
                if any(w in cond_text for w in words):
                    alerts.append(
                        SafetyAlert(
                            category="CONTRAINDICATION",
                            severity="CRITICAL",
                            title=f"Potential Contraindication: {med.generic_name}",
                            description=f"Patient condition '{cond_text[:80]}...' may conflict with contraindication: '{contra}'.",
                            clinical_recommendation="Clinician must evaluate risk/benefit and review alternative therapeutic strategies.",
                            source=med.authoritative_source,
                            evidence_level="Level A",
                        )
                    )

    # 4. Check Known Allergies
    if p_dict:
        allergy_parts = [
            str(p_dict.get("known_allergies") or ""),
            str(p_dict.get("food_allergies") or ""),
            str(p_dict.get("other_allergies") or ""),
        ]
        raw_allergies = p_dict.get("allergies")
        if isinstance(raw_allergies, list):
            for a in raw_allergies:
                a_name = a.allergen if hasattr(a, "allergen") else (a.get("allergen", "") if isinstance(a, dict) else str(a))
                allergy_parts.append(str(a_name))
        elif raw_allergies:
            allergy_parts.append(str(raw_allergies))

        allergy_text = " ".join(allergy_parts).lower()

        if allergy_text.strip() and allergy_text.strip() not in ["none", "none documented", "no known allergies"]:
            for med in entry_list:
                matches_name = med.generic_name.lower() in allergy_text or any(b.lower() in allergy_text for b in med.brand_names)

                # Drug class & cross-reactivity matching
                class_match = False
                med_class_lower = med.drug_class.lower()
                gen_name_lower = med.generic_name.lower()
                if "sulfa" in allergy_text or "sulfonamide" in allergy_text:
                    if "sulfa" in med_class_lower or "sulfonamide" in med_class_lower or gen_name_lower in ["furosemide"]:
                        class_match = True
                if "iodine" in allergy_text and gen_name_lower in ["amiodarone"]:
                    class_match = True
                if ("aspirin" in allergy_text or "nsaid" in allergy_text or "salicylate" in allergy_text) and gen_name_lower in ["aspirin"]:
                    class_match = True
                if ("ace inhibitor" in allergy_text or "acei" in allergy_text or "angioedema" in allergy_text) and "angiotensin-converting enzyme" in med_class_lower:
                    class_match = True
                if ("beta blocker" in allergy_text or "beta-blocker" in allergy_text) and "beta" in med_class_lower:
                    class_match = True
                if "statin" in allergy_text and "statin" in med_class_lower:
                    class_match = True

                matches_contra_hypersensitivity = False
                for contra in med.contraindications:
                    if "hypersensitivity" in contra.lower() or "allergy" in contra.lower():
                        words = [w for w in contra.lower().split() if len(w) > 4 and w not in ["known", "hypersensitivity", "allergy"]]
                        if any(w in allergy_text for w in words):
                            matches_contra_hypersensitivity = True
                            break

                if matches_name or class_match or matches_contra_hypersensitivity:
                    alerts.append(
                        SafetyAlert(
                            category="ALLERGY",
                            severity="CRITICAL",
                            title=f"Allergy Conflict Detected: {med.generic_name}",
                            description=f"Patient has documented allergy/hypersensitivity record matching '{allergy_text.strip()}'.",
                            clinical_recommendation="Do not administer without explicit allergy clearance by attending physician.",
                            source="Hospital Pharmacy Safety Standards / FDA SPL",
                            evidence_level="Level A",
                        )
                    )

    # 5. Multimodal Cross-Checks: ECG Findings & Waveform Measurements
    qtc_val = None
    ecg_hr = None
    pr_val = None
    if ecg_measurements:
        qtc_val = ecg_measurements.get("qtc_ms") or ecg_measurements.get("qtc")
        ecg_hr = ecg_measurements.get("heart_rate") or ecg_measurements.get("hr")
        pr_val = ecg_measurements.get("pr_interval_ms") or ecg_measurements.get("pr_ms")

    # Resolve effective multimodal physiological observations
    eff_vitals = dict(p_dict)
    if vital_signs:
        eff_vitals.update(vital_signs)

    eff_labs = dict(p_dict)
    if laboratory_results:
        eff_labs.update(laboratory_results)

    for med in entry_list:
        # 5a. QT Prolonging Agents + Prolonged QTc
        if med.qt_prolongation_risk in ["HIGH", "MODERATE"]:
            p_sex_str = str(p_dict.get("sex") or getattr(patient, "sex", "") or "").upper()
            is_female = p_sex_str in ["F", "FEMALE"]
            threshold = 460 if is_female else 450
            if qtc_val is not None and qtc_val > threshold:
                alerts.append(
                    SafetyAlert(
                        category="PHYSIOLOGICAL_CONFLICT",
                        severity="CRITICAL",
                        title=f"Critical QTc Alert: {med.generic_name} with Prolonged QTc ({qtc_val} ms)",
                        description=(
                            f"{med.generic_name} carries {med.qt_prolongation_risk} QT-prolongation risk. "
                            f"Current measured QTc is {qtc_val:.0f} ms (normal upper limit: {threshold} ms), "
                            "substantially elevating risk of Torsades de Pointes and ventricular fibrillation."
                        ),
                        clinical_recommendation="Immediate clinician evaluation: hold or dose-reduce QT-prolonging agent, order serum K+/Mg2+, and maintain continuous telemetry.",
                        source="AHA/ACC Scientific Statement on Prevention of Torsades de Pointes",
                        evidence_level="Level A",
                    )
                )

        # 5a-bis. High-Risk QTc Prolongation Safety Gate (>= 480 ms)
        if qtc_val is not None and qtc_val >= 480:
            alerts.append(
                SafetyAlert(
                    category="PHYSIOLOGICAL_CONFLICT",
                    severity="CRITICAL",
                    title=f"High-Risk QTc Safety Gate: QTc Prolongation ({qtc_val:.0f} ms >= 480 ms)",
                    description=(
                        f"Patient QTc interval is severely prolonged at {qtc_val:.0f} ms (established safety gate threshold: 480 ms). "
                        f"Administering {med.generic_name} requires urgent electrophysiology consultation and continuous cardiac monitoring."
                    ),
                    clinical_recommendation="Critical action required: Review all concurrent medications, correct electrolyte imbalances (hypokalemia, hypomagnesemia), and avoid QT-prolonging drugs.",
                    source="AHA/HRS/ACC Clinical Safety Standards for Drug-Induced Proarrhythmia",
                    evidence_level="Level A",
                )
            )

        # 5b. Bradycardia / AV Nodal Conduction Slowing Agents
        is_nodal_agent = any(k in med.drug_class.lower() for k in ["beta", "calcium channel", "glycoside"])
        effective_hr = ecg_hr
        v_hr = eff_vitals.get("heart_rate") if eff_vitals.get("heart_rate") is not None else (eff_vitals.get("heart_rate_bpm") or eff_vitals.get("hr"))
        if v_hr is not None:
            effective_hr = float(v_hr)

        if is_nodal_agent and effective_hr is not None and effective_hr < 50:
            alerts.append(
                SafetyAlert(
                    category="PHYSIOLOGICAL_CONFLICT",
                    severity="CRITICAL",
                    title=f"Severe Bradycardia Alert: {med.generic_name} (HR {effective_hr:.0f} bpm)",
                    description=(
                        f"{med.generic_name} impairs AV nodal conduction and slows sinus node rate. "
                        f"Patient is currently bradycardic (Heart Rate: {effective_hr:.0f} bpm)."
                    ),
                    clinical_recommendation="Urgent clinician review: evaluate for hemodynamically unstable bradycardia, hold nodal-blocking medications.",
                    source="2023 ACC/AHA Bradycardia Management Guidelines",
                    evidence_level="Level A",
                )
            )

        if is_nodal_agent and pr_val is not None and pr_val > 220:
            alerts.append(
                SafetyAlert(
                    category="PHYSIOLOGICAL_CONFLICT",
                    severity="MAJOR",
                    title=f"First-Degree AV Block Warning: {med.generic_name} (PR {pr_val:.0f} ms)",
                    description=f"{med.generic_name} can further lengthen PR interval ({pr_val:.0f} ms) towards high-degree AV block.",
                    clinical_recommendation="Review baseline PR interval; monitor serial ECGs if continuation is clinically necessary.",
                    source="AHA/ACC Guidelines",
                    evidence_level="Level B",
                )
            )

    # 6. Multimodal Cross-Checks: Vital Signs (Hypotension)
    if eff_vitals:
        sbp = eff_vitals.get("systolic_bp") or eff_vitals.get("systolic_bp_mmhg") or eff_vitals.get("sbp")
        if sbp is not None and float(sbp) < 90:
            for med in entry_list:
                if any(k in med.drug_class.lower() for k in ["antagonist", "inhibitor", "blocker"]):
                    alerts.append(
                        SafetyAlert(
                            category="PHYSIOLOGICAL_CONFLICT",
                            severity="CRITICAL",
                            title=f"Severe Hypotension Alert: {med.generic_name} (SBP {float(sbp):.0f} mmHg)",
                            description=f"Patient systolic BP is {float(sbp):.0f} mmHg. Antihypertensive agent {med.generic_name} may precipitate shock.",
                            clinical_recommendation="Urgent clinician bed-side evaluation: hold antihypertensive therapies and evaluate perfusion.",
                            source="Hospital Critical Care Protocol",
                            evidence_level="Level A",
                        )
                    )

    # 7. Multimodal Cross-Checks: Laboratory Results (Electrolytes & Renal Function)
    if eff_labs:
        k_val = (
            eff_labs.get("potassium_mmol_l")
            or eff_labs.get("potassium_meq_l")
            or eff_labs.get("potassium")
            or eff_labs.get("k")
            or eff_labs.get("serum_potassium")
        )
        cr_val = (
            eff_labs.get("serum_creatinine_mg_dl")
            or eff_labs.get("creatinine_mg_dl")
            or eff_labs.get("creatinine")
            or eff_labs.get("serum_creatinine")
        )
        egfr_val = eff_labs.get("egfr_ml_min") or eff_labs.get("egfr")
        mg_val = (
            eff_labs.get("magnesium_mg_dl")
            or eff_labs.get("magnesium")
            or eff_labs.get("mg")
        )

        # Hypokalemia / Hypomagnesemia + Digoxin / QT agents
        if k_val is not None and k_val < 3.5:
            for med in entry_list:
                if "digoxin" in med.generic_name.lower():
                    alerts.append(
                        SafetyAlert(
                            category="LABORATORY_CONFLICT",
                            severity="CRITICAL",
                            title="Potassium-Digoxin Toxicity Alert (K+ < 3.5 mEq/L)",
                            description=f"Serum potassium is {k_val:.1f} mEq/L. Hypokalemia markedly sensitizes myocardium to lethal digoxin toxicity and dysrhythmias.",
                            clinical_recommendation="Check serum digoxin level, correct potassium urgently (target >= 4.0 mEq/L), and withhold digoxin.",
                            source="AHA Scientific Statement on Digoxin Toxicity",
                            evidence_level="Level A",
                        )
                    )
                elif med.qt_prolongation_risk in ["HIGH", "MODERATE"]:
                    alerts.append(
                        SafetyAlert(
                            category="LABORATORY_CONFLICT",
                            severity="MAJOR",
                            title=f"Electrolyte Risk with QT Agent: {med.generic_name} (K+ {k_val:.1f} mEq/L)",
                            description="Hypokalemia compounds QT dispersion and heightens vulnerability to Torsades de Pointes.",
                            clinical_recommendation="Promptly re-pleat potassium to >= 4.0 mEq/L under clinician supervision.",
                            source="ACC/AHA Guidelines",
                            evidence_level="Level A",
                        )
                    )

        # Hyperkalemia + K-sparing diuretics / ACEi
        if k_val is not None and k_val > 5.0:
            for med in entry_list:
                is_k_retaining = (
                    "spironolactone" in med.generic_name.lower()
                    or any(k in med.drug_class.lower() for k in ["potassium-sparing", "aldosterone", "mineralocorticoid", "angiotensin", "ace"])
                )
                if is_k_retaining:
                    alerts.append(
                        SafetyAlert(
                            category="LABORATORY_CONFLICT",
                            severity="CRITICAL",
                            title=f"Hyperkalemia Hazard: {med.generic_name} (K+ {k_val:.1f} mEq/L)",
                            description=f"{med.generic_name} impairs renal potassium excretion. Current K+ is elevated at {k_val:.1f} mEq/L.",
                            clinical_recommendation="Hold potassium-retaining agents immediately; check urgent stat ECG for peaked T waves / sine waves.",
                            source="AHA/ACC Heart Failure Guidelines",
                            evidence_level="Level A",
                        )
                    )

        # Renal Impairment (Creatinine / eGFR) + DOACs / Digoxin / Metformin / Sotalol
        has_renal_impairment = (cr_val is not None and cr_val >= 1.5) or (egfr_val is not None and egfr_val < 50)
        if has_renal_impairment or (egfr_val is not None and egfr_val < 60):
            for med in entry_list:
                gen_name_lower = med.generic_name.lower()
                if "apixaban" in gen_name_lower:
                    alerts.append(
                        SafetyAlert(
                            category="LABORATORY_CONFLICT",
                            severity="MAJOR",
                            title="Apixaban Renal Dosing Advisory",
                            description=f"Patient exhibits impaired renal function (Cr: {cr_val or 'N/A'}, eGFR: {egfr_val or 'N/A'}). DOAC accumulation increases bleeding risk.",
                            clinical_recommendation="Clinician must evaluate apixaban dose-reduction criteria (reduce to 2.5 mg BID if 2 of: Age >= 80, Wt <= 60 kg, Cr >= 1.5 mg/dL).",
                            source="FDA Prescribing Information (Eliquis) / CHEST Guidelines",
                            evidence_level="Level A",
                        )
                    )
                elif "digoxin" in gen_name_lower:
                    alerts.append(
                        SafetyAlert(
                            category="LABORATORY_CONFLICT",
                            severity="CRITICAL",
                            title="Digoxin Renal Accumulation Hazard",
                            description=f"Digoxin is cleared by glomerular filtration. Renal insufficiency (Cr: {cr_val or 'N/A'}, eGFR: {egfr_val or 'N/A'}) predisposes to fatal toxicity.",
                            clinical_recommendation="Mandatory dosage adjustment and therapeutic drug monitoring of serum digoxin level.",
                            source="FDA Prescribing Information (Lanoxin)",
                            evidence_level="Level A",
                        )
                    )
                elif "metformin" in gen_name_lower and egfr_val is not None and egfr_val < 30:
                    alerts.append(
                        SafetyAlert(
                            category="CONTRAINDICATION",
                            severity="CRITICAL",
                            title="Metformin Contraindicated: Severe Renal Impairment (eGFR < 30 mL/min)",
                            description=f"Patient eGFR is {egfr_val:.0f} mL/min (< 30 mL/min threshold). Metformin accumulation carries high risk of fatal lactic acidosis.",
                            clinical_recommendation="Discontinue metformin immediately; switch to renally safe antihyperglycemic therapy.",
                            source="FDA Drug Safety Communication / ADA Standards of Care",
                            evidence_level="Level A",
                        )
                    )
                elif "sotalol" in gen_name_lower and egfr_val is not None and egfr_val < 40:
                    alerts.append(
                        SafetyAlert(
                            category="CONTRAINDICATION",
                            severity="CRITICAL",
                            title="Sotalol Contraindicated: Renal Impairment (eGFR < 40 mL/min)",
                            description=f"Patient eGFR is {egfr_val:.0f} mL/min (< 40 mL/min cutoff). Renal excretion failure leads to lethal Torsades de Pointes.",
                            clinical_recommendation="Sotalol is contraindicated; select alternative rhythm or rate control agent.",
                            source="FDA Prescribing Information (Betapace) / AHA Guidelines",
                            evidence_level="Level A",
                        )
                    )
                elif "spironolactone" in gen_name_lower and ((egfr_val is not None and egfr_val < 30) or (cr_val is not None and cr_val > 2.5)):
                    alerts.append(
                        SafetyAlert(
                            category="CONTRAINDICATION",
                            severity="CRITICAL",
                            title="Spironolactone Contraindicated: Severe Renal Impairment",
                            description=f"Severe renal insufficiency (eGFR {egfr_val or 'N/A'}, Cr {cr_val or 'N/A'}) severely impedes potassium clearance, risking fatal hyperkalemia.",
                            clinical_recommendation="Withhold aldosterone antagonist; monitor serum potassium stat.",
                            source="AHA/ACC Heart Failure Guidelines",
                            evidence_level="Level A",
                        )
                    )
                elif med.egfr_cutoff_ml_min is not None and egfr_val is not None and float(egfr_val) < med.egfr_cutoff_ml_min:
                    alerts.append(
                        SafetyAlert(
                            category="CONTRAINDICATION" if float(egfr_val) < 30 else "LABORATORY_CONFLICT",
                            severity="CRITICAL" if float(egfr_val) < 30 else "MAJOR",
                            title=f"{med.generic_name} Renal Threshold Advisory (eGFR < {med.egfr_cutoff_ml_min:.0f} mL/min)",
                            description=f"Patient eGFR is {float(egfr_val):.0f} mL/min (below recommended clinical cutoff of {med.egfr_cutoff_ml_min:.0f} mL/min). {med.renal_considerations}",
                            clinical_recommendation=f"Review renal dosing for {med.generic_name}: {med.renal_considerations}",
                            source=med.authoritative_source,
                            evidence_level="Level A",
                        )
                    )

    # 8. Hepatic Function Safety Checks
    alt_val = eff_labs.get("alt_u_l") or eff_labs.get("alt")
    ast_val = eff_labs.get("ast_u_l") or eff_labs.get("ast")
    bili_val = eff_labs.get("bilirubin_mg_dl") or eff_labs.get("bilirubin")
    lft_summary = str(eff_labs.get("lft_summary") or "").lower()

    cond_text_lower = cond_text.lower()
    has_liver_impairment = (
        (alt_val is not None and float(alt_val) > 120.0)
        or (ast_val is not None and float(ast_val) > 105.0)
        or (bili_val is not None and float(bili_val) > 2.0)
        or "impairment" in lft_summary or "cirrhosis" in lft_summary or "elevated" in lft_summary
        or any(k in cond_text_lower for k in ["cirrhosis", "hepatitis", "liver disease", "hepatic", "fatty liver", "jaundice"])
    )

    if has_liver_impairment:
        for med in entry_list:
            gen_lower = med.generic_name.lower()
            if gen_lower in ["atorvastatin", "rosuvastatin"]:
                alerts.append(
                    SafetyAlert(
                        category="HEPATIC_SAFETY",
                        severity="CRITICAL" if ((alt_val and float(alt_val) > 150) or "cirrhosis" in cond_text_lower) else "MAJOR",
                        title=f"Hepatic Safety Advisory: {med.generic_name}",
                        description=f"Patient exhibits liver impairment or transaminitis (ALT: {alt_val or 'N/A'}, AST: {ast_val or 'N/A'}, Bilirubin: {bili_val or 'N/A'}). Statin metabolism is impaired with elevated risk of hepatotoxicity.",
                        clinical_recommendation="Evaluate baseline LFTs, consider dose reduction or temporary suspension; monitor transaminases in 4-6 weeks.",
                        source="FDA DailyMed / SIDER / DrugCentral Hepatic Guidelines",
                        evidence_level="Level A",
                    )
                )
            elif gen_lower in ["amiodarone"]:
                alerts.append(
                    SafetyAlert(
                        category="HEPATIC_SAFETY",
                        severity="CRITICAL",
                        title="Amiodarone Hepatotoxicity Hazard",
                        description=f"Amiodarone carries a black-box warning for severe hepatotoxicity. Patient has documented hepatic impairment (ALT: {alt_val or 'N/A'}, AST: {ast_val or 'N/A'}).",
                        clinical_recommendation="Serial monitoring of hepatic transaminases mandatory; discontinue or seek alternative rhythm agent if AST/ALT > 3x baseline.",
                        source="FDA Boxed Warning (DailyMed) / SIDER Side Effect Resource",
                        evidence_level="Level A",
                    )
                )
            elif gen_lower in ["carvedilol", "metoprolol"]:
                alerts.append(
                    SafetyAlert(
                        category="HEPATIC_SAFETY",
                        severity="MAJOR",
                        title=f"Hepatic Clearance Caution: {med.generic_name}",
                        description=f"{med.generic_name} is extensively cleared by hepatic first-pass metabolism. Plasma levels may be markedly elevated in liver impairment.",
                        clinical_recommendation="Initiate at lowest starting dose and titrate cautiously under frequent hemodynamic monitoring.",
                        source="FDA DailyMed SPL / DrugCentral",
                        evidence_level="Level B",
                    )
                )

    # 9. Age-Related Beers Criteria
    eff_age_val = p_dict.get("age") if p_dict else (getattr(patient, "age", None) if patient else None)
    try:
        eff_age_int = int(eff_age_val) if eff_age_val is not None else None
    except (ValueError, TypeError):
        eff_age_int = None

    if eff_age_int is not None and eff_age_int >= 75:
        for med in entry_list:
            if "digoxin" in med.generic_name.lower():
                alerts.append(
                    SafetyAlert(
                        category="AGE_PRECAUTION",
                        severity="MODERATE",
                        title=f"AGS Beers Criteria Advisory: Digoxin in Geriatric Patient (Age {eff_age_int})",
                        description="Digoxin in patients aged >= 65 is associated with increased toxicity risk due to age-related decline in renal clearance.",
                        clinical_recommendation="Consider alternative rate-control therapies; if required, keep dose <= 0.125 mg daily.",
                        source="American Geriatrics Society Beers Criteria (2023 Update)",
                        evidence_level="Level B",
                    )
                )

    has_critical = any(a.severity == "CRITICAL" for a in alerts)

    resolved_pat_id = (
        p_dict.get("patient_id")
        or (getattr(patient, "patient_id", None) if patient else None)
        or "UNKNOWN"
    )

    return MedicationSafetyReport(
        patient_id=resolved_pat_id,
        analyzed_medications=cleaned_meds,
        has_critical_alerts=has_critical,
        context_completeness=context_status,
        missing_context_fields=missing_fields,
        alerts=alerts,
    )


@dataclass
class MedicationReconciliationReport:
    patient_id: str
    reconciled_at: str
    active_regimen: List[str]
    discontinued_regimen: List[str]
    therapeutic_duplications: List[str]
    adverse_reaction_resumptions: List[str]
    new_interactions: List[SafetyAlert]
    reconciliation_summary: str
    dosage_changes: List[str] = field(default_factory=list)
    discontinued: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "reconciled_at": self.reconciled_at,
            "active_regimen": self.active_regimen,
            "discontinued_regimen": self.discontinued_regimen,
            "discontinued": self.discontinued,
            "dosage_changes": self.dosage_changes,
            "therapeutic_duplications": self.therapeutic_duplications,
            "adverse_reaction_resumptions": self.adverse_reaction_resumptions,
            "new_interactions": [asdict(a) for a in self.new_interactions],
            "reconciliation_summary": self.reconciliation_summary,
        }


def _normalize_med_item(item: Any) -> str:
    """Safely extracts a clean drug representation string from str, dict, or Medication object."""
    if not item:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        name = (
            item.get("medication_name")
            or item.get("medication")
            or item.get("drug_name")
            or item.get("generic_name")
            or item.get("brand_name")
            or item.get("drug")
            or item.get("name")
            or ""
        )
        dose = item.get("dosage") or item.get("dose") or ""
        freq = item.get("frequency") or ""
        parts = [str(p).strip() for p in [name, dose, freq] if p and str(p).strip()]
        return " ".join(parts).strip()
    if hasattr(item, "medication_name") or hasattr(item, "drug_name") or hasattr(item, "generic_name"):
        name = getattr(item, "medication_name", None) or getattr(item, "drug_name", None) or getattr(item, "generic_name", None) or getattr(item, "name", "") or ""
        dose = getattr(item, "dosage", "") or getattr(item, "dose", "") or ""
        freq = getattr(item, "frequency", "") or ""
        parts = [str(p).strip() for p in [name, dose, freq] if p and str(p).strip()]
        return " ".join(parts).strip()
    return str(item).strip()


def reconcile_medications(
    current_medications: List[Union[str, Dict[str, Any], Any]],
    previous_medications: Optional[List[Union[str, Dict[str, Any], Any]]] = None,
    proposed_medications: Optional[List[Union[str, Dict[str, Any], Any]]] = None,
    discontinued_medications: Optional[List[Union[str, Dict[str, Any], Any]]] = None,
    discontinued_with_reactions: Optional[Dict[str, str]] = None,
    patient: Optional[Any] = None,
    patient_id: Optional[str] = None,
) -> MedicationReconciliationReport:
    """Reconciles current, previous, and newly proposed medications for clinical safety."""
    from datetime import datetime as dt
    now_ts = dt.now().isoformat()
    prev_list = previous_medications or []
    prop_list = proposed_medications or []
    disc_list = discontinued_medications or []
    disc_reactions = discontinued_with_reactions or {}

    active_clean = [_normalize_med_item(m) for m in current_medications if _normalize_med_item(m)]
    prev_clean = [_normalize_med_item(m) for m in prev_list if _normalize_med_item(m)]
    prop_clean = [_normalize_med_item(m) for m in prop_list if _normalize_med_item(m)]
    disc_clean = [_normalize_med_item(m) for m in disc_list if _normalize_med_item(m)]

    duplications: List[str] = []
    adverse_resumptions: List[str] = []
    dosage_changes: List[str] = []

    # Detect dosage changes between previous and current medications
    for prev_med in prev_clean:
        p_entry = GLOBAL_MEDICATION_DB.get_by_name(prev_med)
        p_base = p_entry.generic_name.lower() if p_entry else prev_med.lower().split()[0]
        prev_norm = re.sub(r'\s+', '', prev_med.lower())
        for cur_med in active_clean:
            c_entry = GLOBAL_MEDICATION_DB.get_by_name(cur_med)
            c_base = c_entry.generic_name.lower() if c_entry else cur_med.lower().split()[0]
            cur_norm = re.sub(r'\s+', '', cur_med.lower())
            if p_base == c_base and prev_norm != cur_norm:
                dosage_changes.append(f"{prev_med} -> {cur_med}")

    # Determine discontinued list if not explicitly provided
    if not disc_clean and prev_clean:
        for prev_med in prev_clean:
            p_entry = GLOBAL_MEDICATION_DB.get_by_name(prev_med)
            p_base = p_entry.generic_name.lower() if p_entry else prev_med.lower().split()[0]
            if not any(p_base in cur_med.lower() for cur_med in active_clean):
                disc_clean.append(prev_med)

    # Detect active ingredient and therapeutic class duplications across the combined regimen
    # A single combination drug (e.g. Entresto / Sacubitril/Valsartan alone) MUST NEVER flag duplication against itself.
    # Concurrently prescribed distinct drugs with identical active ingredient (e.g. Prinivil + Zestril) MUST be flagged.
    combined = list(dict.fromkeys(active_clean + prop_clean))

    items_constituents: Dict[int, List[Dict[str, str]]] = {}
    for idx, med_str in enumerate(combined):
        entry = GLOBAL_MEDICATION_DB.get_by_name(med_str)
        constituents: List[Dict[str, str]] = []
        if entry and "/" in entry.generic_name:
            main_cls = entry.drug_class
            parts = [p.strip() for p in entry.generic_name.split("/") if p.strip()]
            for p in parts:
                sub_entry = GLOBAL_MEDICATION_DB.get_by_name(p)
                sub_cls = sub_entry.drug_class if sub_entry else main_cls
                sub_gen = sub_entry.generic_name if sub_entry else p
                constituents.append({
                    "generic": sub_gen,
                    "class": sub_cls,
                    "label": f"{med_str} ({sub_gen})",
                })
            if ("ARNI" in main_cls or "Neprilysin" in main_cls) and not any("ARB" in c["class"] or "Angiotensin II" in c["class"] for c in constituents):
                constituents.append({
                    "generic": "Valsartan",
                    "class": "Angiotensin II Receptor Blocker (ARB)",
                    "label": f"{med_str} (ARB component)",
                })
        elif "/" in med_str:
            name_only = re.sub(r'[\d/\.\+\-]+|\b(mg|mcg|g|ml|tab|tablet|capsule)\b', ' ', med_str, flags=re.I).strip()
            parts = [p.strip() for p in re.split(r'[/+]', med_str.split()[0] if med_str else "") if p.strip()]
            if len(parts) >= 2:
                for p in parts:
                    sub_entry = GLOBAL_MEDICATION_DB.get_by_name(p)
                    sub_cls = sub_entry.drug_class if sub_entry else "Combination Component"
                    sub_gen = sub_entry.generic_name if sub_entry else p
                    constituents.append({
                        "generic": sub_gen,
                        "class": sub_cls,
                        "label": f"{med_str} ({sub_gen})",
                    })
            elif entry:
                constituents.append({
                    "generic": entry.generic_name,
                    "class": entry.drug_class,
                    "label": entry.generic_name,
                })
            else:
                constituents.append({
                    "generic": name_only or med_str,
                    "class": "Unknown",
                    "label": med_str,
                })
        elif entry:
            cls = entry.drug_class
            constituents.append({
                "generic": entry.generic_name,
                "class": cls,
                "label": entry.generic_name,
            })
            if "ARNI" in cls or "Neprilysin" in cls:
                constituents.append({
                    "generic": "Valsartan",
                    "class": "Angiotensin II Receptor Blocker (ARB)",
                    "label": f"{entry.generic_name} (ARB component)",
                })
        else:
            clean_name = re.sub(r'[\d/\.\+\-]+|\b(mg|mcg|g|ml|tab|tablet|capsule|daily|bid|tid|qid|od|prn|po|iv)\b', ' ', med_str, flags=re.I).strip()
            constituents.append({
                "generic": clean_name or med_str,
                "class": "Unknown",
                "label": med_str,
            })
        items_constituents[idx] = constituents

    # 1. Direct Duplicate Active Ingredient Check across DISTINCT items
    ingredient_map: Dict[str, Dict[int, str]] = {}
    for idx, consts in items_constituents.items():
        for c in consts:
            g_key = c["generic"].lower()
            ingredient_map.setdefault(g_key, {})[idx] = c["label"]

    for g_key, item_labels in ingredient_map.items():
        if len(item_labels) > 1:
            duplications.append(
                f"Duplicate active ingredient '{g_key.title()}': {', '.join(item_labels.values())}"
            )

    # 2. Therapeutic Class Duplications across DISTINCT items
    class_map: Dict[str, Dict[int, List[str]]] = {}
    for idx, consts in items_constituents.items():
        for c in consts:
            cls = c["class"]
            if cls and cls != "Unknown":
                class_map.setdefault(cls, {}).setdefault(idx, []).append(c["label"])

    for cls, item_labels in class_map.items():
        if len(item_labels) > 1:
            distinct_agents = []
            for idx, labels in item_labels.items():
                distinct_agents.append(labels[0])
            duplications.append(f"Duplicate {cls}: {', '.join(distinct_agents)}")

    # Check resumption of discontinued medications with adverse reactions
    for prop in prop_clean:
        p_entry = GLOBAL_MEDICATION_DB.get_by_name(prop)
        p_name = p_entry.generic_name.lower() if p_entry else prop.lower()
        for disc_med, reason in disc_reactions.items():
            d_entry = GLOBAL_MEDICATION_DB.get_by_name(disc_med)
            d_name = d_entry.generic_name.lower() if d_entry else disc_med.lower()
            if p_name == d_name or prop.lower() in disc_med.lower():
                adverse_resumptions.append(
                    f"Resumption of previously discontinued medication '{prop}' (Prior documented reaction: {reason})"
                )

    # Run safety checks on proposed combined regimen
    safety_rep = check_medication_safety(combined, patient=patient)

    summary_parts = []
    if duplications:
        summary_parts.append(f"Identified {len(duplications)} therapeutic duplication(s).")
    if adverse_resumptions:
        summary_parts.append(f"Flagged {len(adverse_resumptions)} resumption(s) of discontinued therapy.")
    if dosage_changes:
        summary_parts.append(f"Detected {len(dosage_changes)} dosage modification(s).")
    if safety_rep.has_critical_alerts:
        summary_parts.append("Critical drug-drug, physiological, or allergy conflicts detected.")
    if not summary_parts:
        summary_parts.append("Medication regimen reconciled: no duplications or adverse resumptions detected.")

    resolved_patient_id = patient_id or (getattr(patient, "patient_id", None) if patient else "UNKNOWN") or "UNKNOWN"

    return MedicationReconciliationReport(
        patient_id=resolved_patient_id,
        reconciled_at=now_ts,
        active_regimen=active_clean,
        discontinued_regimen=disc_clean,
        discontinued=disc_clean,
        dosage_changes=dosage_changes,
        therapeutic_duplications=duplications,
        adverse_reaction_resumptions=adverse_resumptions,
        new_interactions=safety_rep.alerts,
        reconciliation_summary=" ".join(summary_parts),
    )

