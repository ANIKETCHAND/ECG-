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
) -> MedicationSafetyReport:
    """Evaluate drug interactions, contraindications, and allergy safety."""
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

    has_critical = any(a.severity == "CRITICAL" for a in alerts)

    return MedicationSafetyReport(
        patient_id=patient.patient_id if patient else "UNKNOWN",
        analyzed_medications=cleaned_meds,
        has_critical_alerts=has_critical,
        context_completeness=context_status,
        missing_context_fields=missing_fields,
        alerts=alerts,
    )

