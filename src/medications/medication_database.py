"""
Cardiovascular Medication Knowledge Base & Repository.
Strict adherence to Safety Rules:
1. No Autonomous Prescribing.
2. Authoritative, traceable sources (FDA Prescribing Info, DailyMed, AHA/ACC/ESC Guidelines).
3. No fabricated medications, doses, or interactions.
4. Distinguishes clinical considerations from patient prescriptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MedicationInteraction:
    interacting_drug: str
    severity: str  # CONTRAINDICATED, MAJOR, MODERATE, MINOR
    clinical_effect: str
    management_recommendation: str
    source: str
    evidence_level: str


@dataclass
class MedicationEntry:
    generic_name: str
    brand_names: List[str]
    drug_class: str
    mechanism_of_action: str
    cardiac_indications: List[str]
    contraindications: List[str]
    black_box_warnings: List[str]
    common_adverse_effects: List[str]
    serious_adverse_effects: List[str]
    qt_prolongation_risk: str  # HIGH, MODERATE, LOW, NONE
    renal_considerations: str
    hepatic_considerations: str
    pregnancy_category: str
    monitoring_parameters: List[str]
    known_interactions: List[MedicationInteraction]
    authoritative_source: str
    source_version: str
    last_verified: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Authoritative, curated cardiovascular medication repository
CARDIOVASCULAR_MEDICATION_REPOSITORY: Dict[str, MedicationEntry] = {
    "metoprolol": MedicationEntry(
        generic_name="Metoprolol",
        brand_names=["Lopressor", "Toprol-XL", "Betaloc"],
        drug_class="Beta-1 Selective Adrenergic Antagonist",
        mechanism_of_action="Competitively blocks beta-1 adrenergic receptors in cardiac tissue, decreasing heart rate, cardiac output, and blood pressure.",
        cardiac_indications=[
            "Premature Ventricular Contractions (symptomatic)",
            "Atrial Fibrillation (rate control)",
            "Hypertension",
            "Angina Pectoris",
            "Heart Failure with reduced ejection fraction (succinate formulation)",
        ],
        contraindications=[
            "Severe bradycardia (< 45 bpm)",
            "Second- or third-degree heart block (without functioning pacemaker)",
            "Cardiogenic shock",
            "Decompensated heart failure",
            "Sick sinus syndrome",
        ],
        black_box_warnings=[
            "Abrupt cessation can exacerbate angina or precipitate myocardial infarction."
        ],
        common_adverse_effects=["Bradycardia", "Fatigue", "Dizziness", "Hypotension", "Depression"],
        serious_adverse_effects=["Complete heart block", "Bronchospasm", "Acute heart failure exacerbation"],
        qt_prolongation_risk="NONE",
        renal_considerations="No initial dose adjustment required; eliminated primarily by hepatic metabolism.",
        hepatic_considerations="Metabolized heavily via CYP2D6; initiate at lower doses in severe hepatic impairment.",
        pregnancy_category="Category C (crosses placenta; monitor for fetal bradycardia)",
        monitoring_parameters=["Heart rate", "Blood pressure", "ECG PR interval", "Signs of heart failure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="MAJOR",
                clinical_effect="Additive bradycardia, sinus arrest, and severe AV block.",
                management_recommendation="Close hemodynamic and ECG monitoring; adjust doses as indicated by clinician.",
                source="FDA Prescribing Information / AHA Science Advisory",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Diltiazem",
                severity="MAJOR",
                clinical_effect="Additive negative inotropic and chronotropic effects.",
                management_recommendation="Avoid combination or monitor closely for bradycardia and heart failure.",
                source="ACC/AHA Guideline on Arrhythmia Management",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Metoprolol Succinate / DailyMed)",
        source_version="NDA-019962-Rev2024",
        last_verified="2026-06-15",
    ),
    "amiodarone": MedicationEntry(
        generic_name="Amiodarone",
        brand_names=["Cordarone", "Pacerone", "Nexterone"],
        drug_class="Class III Antiarrhythmic (Multichannel blocker: K+, Na+, Ca2+, Beta)",
        mechanism_of_action="Prolongs cardiac action potential duration and effective refractory period across all cardiac tissues.",
        cardiac_indications=[
            "Ventricular Arrhythmias (VT/VF refractory to other therapy)",
            "Atrial Fibrillation (rhythm control in structural heart disease)",
        ],
        contraindications=[
            "Severe sinus-node dysfunction / marked sinus bradycardia",
            "Second- or third-degree AV block without pacemaker",
            "Cardiogenic shock",
            "Known hypersensitivity to iodine",
        ],
        black_box_warnings=[
            "Pulmonary toxicity (hypersensitivity pneumonitis, pulmonary fibrosis).",
            "Severe hepatotoxicity.",
            "Proarrhythmic effects (QT prolongation / Torsades de Pointes).",
        ],
        common_adverse_effects=["Corneal microdeposits", "Photosensitivity", "Nausea", "Tremor", "Thyroid dysfunction"],
        serious_adverse_effects=["Pulmonary fibrosis", "Torsades de Pointes", "Hepatotoxicity", "Optic neuropathy"],
        qt_prolongation_risk="HIGH",
        renal_considerations="No dosage adjustment needed in renal failure; not dialyzable.",
        hepatic_considerations="Extensive hepatic metabolism; baseline and serial hepatic transaminases mandatory.",
        pregnancy_category="Category D (documented fetal harm including congenital goiter)",
        monitoring_parameters=["Baseline and serial ECG (QTc)", "Liver function tests", "Thyroid panel", "Chest X-ray / PFTs"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Warfarin",
                severity="MAJOR",
                clinical_effect="Inhibits CYP2C9, significantly increasing INR and bleeding risk.",
                management_recommendation="Reduce warfarin dose by 33% to 50% upon initiating amiodarone; monitor INR weekly.",
                source="FDA Prescribing Information / Chest Antithrombotic Guidelines",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Digoxin",
                severity="MAJOR",
                clinical_effect="Doubles serum digoxin concentration via P-glycoprotein inhibition.",
                management_recommendation="Reduce digoxin dose by 50% or discontinue; monitor serum digoxin levels.",
                source="DailyMed / FDA Prescribing Info",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Cordarone / DailyMed)",
        source_version="NDA-018972-Rev2025",
        last_verified="2026-05-10",
    ),
    "apixaban": MedicationEntry(
        generic_name="Apixaban",
        brand_names=["Eliquis"],
        drug_class="Direct Oral Anticoagulant (DOAC) — Factor Xa Inhibitor",
        mechanism_of_action="Directly and reversibly inhibits free and clot-bound Factor Xa, decreasing thrombin generation and thrombus development.",
        cardiac_indications=[
            "Nonvalvular Atrial Fibrillation (stroke and systemic embolism prevention)",
            "Deep Vein Thrombosis / Pulmonary Embolism treatment and prophylaxis",
        ],
        contraindications=[
            "Active pathological bleeding",
            "Severe hypersensitivity to apixaban",
            "Prosthetic mechanical heart valves",
        ],
        black_box_warnings=[
            "Premature discontinuation increases the risk of thrombotic events.",
            "Epidural or spinal hematomas may occur in patients receiving neuraxial anesthesia.",
        ],
        common_adverse_effects=["Bleeding (gingival, epistaxis, hematuria)", "Anemia", "Bruising"],
        serious_adverse_effects=["Intracranial hemorrhage", "Major gastrointestinal hemorrhage"],
        qt_prolongation_risk="NONE",
        renal_considerations="Dose reduction (to 2.5 mg BID) if patient meets at least two of: Age >= 80, Weight <= 60 kg, Serum Creatinine >= 1.5 mg/dL.",
        hepatic_considerations="Avoid in severe hepatic impairment (Child-Pugh C) or hepatic coagulopathy.",
        pregnancy_category="Not recommended during pregnancy due to intrinsic bleeding risks.",
        monitoring_parameters=["Hemoglobin/Hematocrit", "Renal function (eGFR/CrCl)", "Signs of occult bleeding"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Aspirin",
                severity="MODERATE",
                clinical_effect="Additive antiplatelet/anticoagulant bleeding risk.",
                management_recommendation="Combine only with verified clinical indication (e.g., recent ACS/PCI) for minimal duration.",
                source="AHA/ACC Atrial Fibrillation Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Eliquis / DailyMed)",
        source_version="NDA-202155-Rev2025",
        last_verified="2026-07-01",
    ),
}


class MedicationDatabase:
    """Provides querying and validation across authoritative cardiovascular medications."""

    def __init__(self, data_path: Optional[Path] = None):
        self._entries: Dict[str, MedicationEntry] = dict(CARDIOVASCULAR_MEDICATION_REPOSITORY)

    def search(self, query: str) -> List[MedicationEntry]:
        q = query.lower().strip()
        if not q:
            return list(self._entries.values())
        results = []
        for entry in self._entries.values():
            if (
                q in entry.generic_name.lower()
                or any(q in b.lower() for b in entry.brand_names)
                or q in entry.drug_class.lower()
                or any(q in ind.lower() for ind in entry.cardiac_indications)
            ):
                results.append(entry)
        return results

    def get_by_name(self, name: str) -> Optional[MedicationEntry]:
        clean = name.lower().strip()
        if clean in self._entries:
            return self._entries[clean]
        for entry in self._entries.values():
            if clean in [b.lower() for b in entry.brand_names]:
                return entry
        return None

    def list_all(self) -> List[MedicationEntry]:
        return list(self._entries.values())


GLOBAL_MEDICATION_DB = MedicationDatabase()

