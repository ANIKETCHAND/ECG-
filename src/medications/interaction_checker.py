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
from typing import Any, Dict, List, Optional

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
    medication_names: List[str],
    patient: Optional[PatientRecord] = None,
    ecg_finding: Optional[str] = None,
    ecg_measurements: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
) -> MedicationSafetyReport:
    """Evaluate drug interactions, contraindications, allergy safety, and multimodal physiological conflicts."""
    alerts: List[SafetyAlert] = []
    missing_fields: List[str] = []

    # 1. Evaluate Patient Clinical Context Completeness
    if patient is None:
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
        if not patient.known_allergies or patient.known_allergies.strip() == "":
            missing_fields.append("Allergy Record")
        if not patient.existing_conditions or patient.existing_conditions.strip() == "":
            missing_fields.append("Existing Conditions")
        if patient.age is None:
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
    cleaned_meds = [m.strip().lower() for m in medication_names if m.strip()]
    med_entries: Dict[str, MedicationEntry] = {}
    for m in cleaned_meds:
        entry = GLOBAL_MEDICATION_DB.get_by_name(m)
        if entry:
            med_entries[entry.generic_name.lower()] = entry

    # 2. Check Drug-Drug Interactions
    entry_list = list(med_entries.values())
    for i in range(len(entry_list)):
        for j in range(i + 1, len(entry_list)):
            m1 = entry_list[i]
            m2 = entry_list[j]
            for inter in m1.known_interactions:
                if inter.interacting_drug.lower() in m2.generic_name.lower() or any(
                    inter.interacting_drug.lower() in b.lower() for b in m2.brand_names
                ):
                    alerts.append(
                        SafetyAlert(
                            category="INTERACTION",
                            severity=inter.severity,
                            title=f"Potential Interaction: {m1.generic_name} + {m2.generic_name}",
                            description=inter.clinical_effect,
                            clinical_recommendation=inter.management_recommendation,
                            source=inter.source,
                            evidence_level=inter.evidence_level,
                        )
                    )

    # 3. Check Contraindications against patient's existing conditions
    if patient and patient.existing_conditions:
        cond_text = patient.existing_conditions.lower()
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
                            description=f"Patient condition '{patient.existing_conditions}' may conflict with contraindication: '{contra}'.",
                            clinical_recommendation="Clinician must evaluate risk/benefit and review alternative therapeutic strategies.",
                            source=med.authoritative_source,
                            evidence_level="Level A",
                        )
                    )

    # 4. Check Known Allergies
    if patient and patient.known_allergies:
        allergy_text = patient.known_allergies.lower()
        for med in entry_list:
            matches_name = med.generic_name.lower() in allergy_text or any(b.lower() in allergy_text for b in med.brand_names)
            matches_contra_hypersensitivity = False
            for contra in med.contraindications:
                if "hypersensitivity" in contra.lower() or "allergy" in contra.lower():
                    words = [w for w in contra.lower().split() if len(w) > 4 and w not in ["known", "hypersensitivity", "allergy"]]
                    if any(w in allergy_text for w in words):
                        matches_contra_hypersensitivity = True
                        break

            if matches_name or matches_contra_hypersensitivity:
                alerts.append(
                    SafetyAlert(
                        category="ALLERGY",
                        severity="CRITICAL",
                        title=f"Allergy Conflict Detected: {med.generic_name}",
                        description=f"Patient has documented allergy matching '{patient.known_allergies}'.",
                        clinical_recommendation="Do not administer without explicit allergy clearance by attending physician.",
                        source="Hospital Pharmacy Safety Standards",
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

    for med in entry_list:
        # 5a. QT Prolonging Agents + Prolonged QTc
        if med.qt_prolongation_risk in ["HIGH", "MODERATE"]:
            threshold = 460 if (patient and patient.sex == "F") else 450
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

        # 5b. Bradycardia / AV Nodal Conduction Slowing Agents
        is_nodal_agent = any(k in med.drug_class.lower() for k in ["beta", "calcium channel", "glycoside"])
        effective_hr = ecg_hr
        if vital_signs and vital_signs.get("heart_rate") is not None:
            effective_hr = vital_signs.get("heart_rate")

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
    if vital_signs:
        sbp = vital_signs.get("systolic_bp") or vital_signs.get("sbp")
        if sbp is not None and sbp < 90:
            for med in entry_list:
                if any(k in med.drug_class.lower() for k in ["antagonist", "inhibitor", "blocker"]):
                    alerts.append(
                        SafetyAlert(
                            category="PHYSIOLOGICAL_CONFLICT",
                            severity="CRITICAL",
                            title=f"Severe Hypotension Alert: {med.generic_name} (SBP {sbp:.0f} mmHg)",
                            description=f"Patient systolic BP is {sbp:.0f} mmHg. Antihypertensive agent {med.generic_name} may precipitate shock.",
                            clinical_recommendation="Urgent clinician bed-side evaluation: hold antihypertensive therapies and evaluate perfusion.",
                            source="Hospital Critical Care Protocol",
                            evidence_level="Level A",
                        )
                    )

    # 7. Multimodal Cross-Checks: Laboratory Results (Electrolytes & Renal Function)
    if laboratory_results:
        k_val = laboratory_results.get("potassium") or laboratory_results.get("k")
        cr_val = laboratory_results.get("creatinine") or laboratory_results.get("serum_creatinine")
        egfr_val = laboratory_results.get("egfr")
        mg_val = laboratory_results.get("magnesium") or laboratory_results.get("mg")

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

        # Renal Impairment (Creatinine / eGFR) + DOACs / Digoxin
        has_renal_impairment = (cr_val is not None and cr_val >= 1.5) or (egfr_val is not None and egfr_val < 50)
        if has_renal_impairment:
            for med in entry_list:
                if "apixaban" in med.generic_name.lower():
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
                elif "digoxin" in med.generic_name.lower():
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

    # 8. Age-Related Beers Criteria
    if patient and patient.age is not None and patient.age >= 75:
        for med in entry_list:
            if "digoxin" in med.generic_name.lower():
                alerts.append(
                    SafetyAlert(
                        category="AGE_PRECAUTION",
                        severity="MODERATE",
                        title=f"AGS Beers Criteria Advisory: Digoxin in Geriatric Patient (Age {patient.age})",
                        description="Digoxin in patients aged >= 65 is associated with increased toxicity risk due to age-related decline in renal clearance.",
                        clinical_recommendation="Consider alternative rate-control therapies; if required, keep dose <= 0.125 mg daily.",
                        source="American Geriatrics Society Beers Criteria (2023 Update)",
                        evidence_level="Level B",
                    )
                )

    has_critical = any(a.severity == "CRITICAL" for a in alerts)

    return MedicationSafetyReport(
        patient_id=patient.patient_id if patient else "UNKNOWN",
        analyzed_medications=cleaned_meds,
        has_critical_alerts=has_critical,
        context_completeness=context_status,
        missing_context_fields=missing_fields,
        alerts=alerts,
    )

