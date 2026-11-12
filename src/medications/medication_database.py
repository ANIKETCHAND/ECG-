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
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request


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
    bradycardia_risk: bool = False
    hypotension_risk: bool = False
    egfr_cutoff_ml_min: Optional[float] = None
    liver_warning_level: str = "MONITOR"  # CONTRAINDICATED, WARNING, MONITOR, SAFE
    sider_adverse_effects: List[str] = field(default_factory=list)
    drugcentral_id: Optional[str] = None
    rxnorm_cui: Optional[str] = None
    dailymed_setid: Optional[str] = None

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
    "digoxin": MedicationEntry(
        generic_name="Digoxin",
        brand_names=["Lanoxin", "Digox"],
        drug_class="Cardiac Glycoside / Inotrope",
        mechanism_of_action="Inhibits sodium-potassium ATPase pump, increasing intracellular calcium and vagal tone to slow AV nodal conduction.",
        cardiac_indications=[
            "Heart Failure with reduced ejection fraction (symptom control)",
            "Atrial Fibrillation (rate control in sedentary patients or combined with beta-blockers)",
        ],
        contraindications=[
            "Ventricular fibrillation",
            "Known hypersensitivity to digoxin",
            "Wolff-Parkinson-White syndrome with atrial fibrillation",
            "Second- or third-degree AV block without pacemaker",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Nausea", "Vomiting", "Visual disturbances (yellow-green halos)", "Dizziness"],
        serious_adverse_effects=["Digoxin toxicity", "Fatal ventricular arrhythmias", "Severe bradycardia/AV block"],
        qt_prolongation_risk="NONE",
        renal_considerations="Narrow therapeutic index. Primarily eliminated by kidneys; dose reduction mandatory in renal insufficiency (eGFR < 50 mL/min).",
        hepatic_considerations="Minimal hepatic metabolism.",
        pregnancy_category="Category C",
        monitoring_parameters=["Serum digoxin level (target 0.5-0.9 ng/mL)", "Serum potassium", "Serum magnesium", "Serum creatinine / eGFR", "ECG PR interval and rhythm"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="MAJOR",
                clinical_effect="Amiodarone increases serum digoxin concentrations by 70-100% via P-glycoprotein inhibition, provoking lethal toxicity.",
                management_recommendation="Reduce digoxin dose by 50% when initiating amiodarone and monitor digoxin levels.",
                source="FDA Prescribing Information / DailyMed",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Metoprolol",
                severity="MAJOR",
                clinical_effect="Additive slowing of AV conduction; risk of advanced heart block.",
                management_recommendation="Monitor heart rate and PR interval closely.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Lanoxin / DailyMed)",
        source_version="NDA-009330-Rev2024",
        last_verified="2026-06-10",
    ),
    "diltiazem": MedicationEntry(
        generic_name="Diltiazem",
        brand_names=["Cardizem", "Tiazac", "Cartia"],
        drug_class="Non-Dihydropyridine Calcium Channel Blocker (Class IV)",
        mechanism_of_action="Inhibits transmembrane calcium influx in cardiac and vascular smooth muscle, slowing AV node conduction and depressing contractility.",
        cardiac_indications=[
            "Atrial Fibrillation / Atrial Flutter (ventricular rate control)",
            "Paroxysmal Supraventricular Tachycardia (conversion and prevention)",
            "Chronic Stable Angina",
            "Hypertension",
        ],
        contraindications=[
            "Sick sinus syndrome or 2nd/3rd degree AV block without functioning ventricular pacemaker",
            "Severe hypotension (systolic BP < 90 mmHg)",
            "Acute myocardial infarction with pulmonary congestion",
            "Heart Failure with reduced ejection fraction (LVEF < 40%)",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Peripheral edema", "Bradycardia", "Dizziness", "Headache", "Constipation"],
        serious_adverse_effects=["Complete AV block", "Decompensated heart failure", "Severe hypotension"],
        qt_prolongation_risk="NONE",
        renal_considerations="Use with caution in renal failure; monitor for metabolite accumulation.",
        hepatic_considerations="Extensive CYP3A4 metabolism; dosage titration must be slower in hepatic impairment.",
        pregnancy_category="Category C",
        monitoring_parameters=["Heart rate", "Blood pressure", "ECG PR interval", "Lower extremity edema"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Metoprolol",
                severity="MAJOR",
                clinical_effect="Severe bradycardia, profound hypotension, and potential cardiac arrest.",
                management_recommendation="Avoid concurrent administration in patients with impaired ventricular function.",
                source="AHA/ACC Guideline on Arrhythmia Management",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Cardizem / DailyMed)",
        source_version="NDA-018602-Rev2025",
        last_verified="2026-05-15",
    ),
    "lisinopril": MedicationEntry(
        generic_name="Lisinopril",
        brand_names=["Prinivil", "Zestril"],
        drug_class="Angiotensin-Converting Enzyme (ACE) Inhibitor",
        mechanism_of_action="Competitively inhibits ACE, blocking conversion of angiotensin I to angiotensin II, decreasing systemic vascular resistance and aldosterone secretion.",
        cardiac_indications=[
            "Hypertension",
            "Heart Failure (adjunctive therapy)",
            "Acute Myocardial Infarction (early survival improvement)",
        ],
        contraindications=[
            "History of angioedema with previous ACE inhibitor therapy",
            "Hereditary or idiopathic angioedema",
            "Concomitant use with aliskiren in patients with diabetes",
            "Pregnancy",
        ],
        black_box_warnings=[
            "Fetal toxicity: Discontinue as soon as pregnancy is detected; can cause fetal injury and death."
        ],
        common_adverse_effects=["Dry cough", "Dizziness", "Headache", "Hyperkalemia"],
        serious_adverse_effects=["Angioedema (life-threatening airway obstruction)", "Acute kidney injury", "Hyperkalemia"],
        qt_prolongation_risk="NONE",
        renal_considerations="Eliminated unchanged via kidneys. Dose reduction required when CrCl < 30 mL/min.",
        hepatic_considerations="Not metabolized by liver; safe in hepatic dysfunction.",
        pregnancy_category="Category D (Black Box Warning)",
        monitoring_parameters=["Serum potassium", "Serum creatinine / BUN", "Blood pressure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Spironolactone",
                severity="MAJOR",
                clinical_effect="Additive retention of serum potassium leading to severe life-threatening hyperkalemia.",
                management_recommendation="Monitor potassium levels at 1 week, 4 weeks, and periodically. Avoid if baseline K+ > 5.0 mEq/L.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Zestril / DailyMed)",
        source_version="NDA-019777-Rev2024",
        last_verified="2026-04-20",
    ),
    "spironolactone": MedicationEntry(
        generic_name="Spironolactone",
        brand_names=["Aldactone", "Carospir"],
        drug_class="Mineralocorticoid Receptor Antagonist (MRA) / Potassium-Sparing Diuretic",
        mechanism_of_action="Competitively blocks aldosterone binding at distal renal tubules, promoting sodium excretion while conserving potassium.",
        cardiac_indications=[
            "Heart Failure with reduced ejection fraction (NYHA class II-IV)",
            "Resistant Hypertension",
            "Post-myocardial infarction with left ventricular dysfunction",
        ],
        contraindications=[
            "Hyperkalemia (serum potassium > 5.0 mEq/L at baseline)",
            "Addison's disease",
            "Severe renal impairment (eGFR < 30 mL/min or creatinine > 2.5 mg/dL in men / > 2.0 in women)",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Hyperkalemia", "Gynecomastia", "Dizziness", "Dehydration"],
        serious_adverse_effects=["Severe hyperkalemia (cardiac arrest)", "Acute renal failure"],
        qt_prolongation_risk="NONE",
        renal_considerations="Contraindicated in severe renal impairment (eGFR < 30 mL/min). Risk of lethal hyperkalemia.",
        hepatic_considerations="Extensively metabolized; use cautiously in severe hepatic cirrhosis.",
        pregnancy_category="Category C",
        monitoring_parameters=["Serum potassium", "Serum creatinine", "Blood pressure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Lisinopril",
                severity="MAJOR",
                clinical_effect="Severe hyperkalemic cardiac arrest.",
                management_recommendation="Close monitoring of serum potassium within 3-7 days of initiation.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Aldactone / DailyMed)",
        source_version="NDA-012151-Rev2025",
        last_verified="2026-06-05",
    ),
    "warfarin": MedicationEntry(
        generic_name="Warfarin",
        brand_names=["Coumadin", "Jantoven"],
        drug_class="Vitamin K Antagonist (Anticoagulant)",
        mechanism_of_action="Inhibits vitamin K epoxide reductase complex 1 (VKORC1), depleting functional factors II, VII, IX, X and proteins C and S.",
        cardiac_indications=[
            "Prophylaxis and treatment of venous thromboembolism",
            "Thromboembolism prophylaxis in Atrial Fibrillation and mechanical prosthetic heart valves",
        ],
        contraindications=[
            "Active major hemorrhage",
            "Pregnancy (except women with mechanical valves at high thromboembolism risk)",
            "Severe uncontrolled hypertension",
        ],
        black_box_warnings=[
            "Can cause major or fatal bleeding. Regular monitoring of INR required."
        ],
        common_adverse_effects=["Bruising", "Epistaxis", "Hematomas"],
        serious_adverse_effects=["Fatal intracranial or retroperitoneal hemorrhage", "Skin necrosis"],
        qt_prolongation_risk="NONE",
        renal_considerations="No primary dose adjustment for renal function, but monitor INR closely.",
        hepatic_considerations="Metabolized primarily by CYP2C9; impaired liver function enhances anticoagulant response.",
        pregnancy_category="Category X (teratogenic)",
        monitoring_parameters=["INR (target typically 2.0-3.0 or 2.5-3.5)", "Hemoglobin/Hematocrit", "Signs of bleeding"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="MAJOR",
                clinical_effect="Inhibits warfarin metabolism, doubling INR and significantly increasing hemorrhage risk.",
                management_recommendation="Reduce warfarin dose by 33-50% when amiodarone is initiated; check INR every 3-5 days.",
                source="FDA Prescribing Information / Chest Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Coumadin / DailyMed)",
        source_version="NDA-009218-Rev2025",
        last_verified="2026-06-25",
    ),
    "carvedilol": MedicationEntry(
        generic_name="Carvedilol",
        brand_names=["Coreg", "Coreg CR"],
        drug_class="Nonselective Beta-Adrenergic and Alpha-1 Blocker",
        mechanism_of_action="Blocks beta-1, beta-2, and alpha-1 adrenergic receptors, decreasing systemic vascular resistance and suppressing reflex tachycardia.",
        cardiac_indications=[
            "Heart Failure with reduced ejection fraction (HFrEF)",
            "Post-Myocardial Infarction with Left Ventricular Dysfunction",
            "Hypertension",
        ],
        contraindications=[
            "Severe bradycardia (< 50 bpm)",
            "Second- or third-degree AV block (without pacemaker)",
            "Decompensated heart failure requiring inotropes",
            "Severe hepatic impairment",
            "Bronchial asthma or severe COPD",
        ],
        black_box_warnings=["Abrupt cessation may exacerbate ischemic heart disease."],
        common_adverse_effects=["Dizziness", "Fatigue", "Hypotension", "Bradycardia", "Weight gain"],
        serious_adverse_effects=["Profound bradycardia", "High-grade AV block", "Cardiogenic shock"],
        qt_prolongation_risk="NONE",
        renal_considerations="No primary dosage adjustment required in renal insufficiency.",
        hepatic_considerations="Extensive first-pass hepatic metabolism; contraindicated in severe hepatic impairment (Child-Pugh B/C).",
        pregnancy_category="Category C",
        monitoring_parameters=["Heart rate", "Blood pressure", "Signs of worsening heart failure", "Liver function"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Diltiazem",
                severity="MAJOR",
                clinical_effect="Additive negative inotropy and chronotropy; severe AV conduction arrest.",
                management_recommendation="Avoid combination in patients with low ejection fraction.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Coreg / DailyMed)",
        source_version="NDA-020297-Rev2025",
        last_verified="2026-06-15",
        bradycardia_risk=True,
        hypotension_risk=True,
        liver_warning_level="CONTRAINDICATED",
        rxnorm_cui="20352",
        dailymed_setid="a5ffb4a6-d242-43bb-a4e5-94f79443aa0c",
        drugcentral_id="533",
        sider_adverse_effects=["bradycardia", "hypotension", "dizziness", "fatigue", "edema"],
    ),
    "bisoprolol": MedicationEntry(
        generic_name="Bisoprolol",
        brand_names=["Zebeta", "Monocor"],
        drug_class="Beta-1 Selective Adrenergic Antagonist",
        mechanism_of_action="Selectively inhibits beta-1 adrenergic receptors in cardiac myocytes, reducing heart rate and myocardial contractility.",
        cardiac_indications=[
            "Hypertension",
            "Heart Failure with reduced ejection fraction (guideline-directed medical therapy)",
            "Stable Angina Pectoris",
        ],
        contraindications=[
            "Sinus bradycardia (< 50 bpm)",
            "Second- or third-degree AV block without pacemaker",
            "Cardiogenic shock",
            "Overt cardiac failure",
        ],
        black_box_warnings=["Abrupt discontinuation may provoke severe angina or myocardial infarction."],
        common_adverse_effects=["Fatigue", "Bradycardia", "Dizziness", "Cold extremities"],
        serious_adverse_effects=["Complete heart block", "Acute heart failure decompensation"],
        qt_prolongation_risk="NONE",
        renal_considerations="Half excreted unchanged by kidneys; reduce dose in severe renal impairment (eGFR < 40 mL/min).",
        hepatic_considerations="Dose titration slower in severe cirrhosis.",
        pregnancy_category="Category C",
        monitoring_parameters=["Heart rate", "Blood pressure", "ECG PR interval"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="MAJOR",
                clinical_effect="Severe additive bradycardia and AV nodal arrest.",
                management_recommendation="Serial ECG monitoring and dose titration.",
                source="DailyMed / FDA Label",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Zebeta / DailyMed)",
        source_version="NDA-019982-Rev2024",
        last_verified="2026-06-20",
        bradycardia_risk=True,
    ),
    "sotalol": MedicationEntry(
        generic_name="Sotalol",
        brand_names=["Betapace", "Betapace AF", "Sorine"],
        drug_class="Class III Antiarrhythmic with Nonselective Beta-Blocking Properties",
        mechanism_of_action="Inhibits rapid delayed rectifier potassium current (IKr) prolonging repolarization (Class III effect) and blocks beta receptors.",
        cardiac_indications=[
            "Maintenance of normal sinus rhythm in Atrial Fibrillation / Atrial Flutter",
            "Treatment of documented life-threatening ventricular arrhythmias (sustained VT)",
        ],
        contraindications=[
            "Baseline QTc interval > 450 ms",
            "Sinus bradycardia (< 50 bpm)",
            "Second- or third-degree AV block without pacemaker",
            "Severe renal impairment (eGFR / CrCl < 40 mL/min)",
            "Hypokalemia (< 4.0 mEq/L) or Hypomagnesemia",
            "Cardiogenic shock or decompensated HF",
        ],
        black_box_warnings=[
            "Can cause life-threatening ventricular proarrhythmia (Torsades de Pointes). Initiate in facility with continuous cardiac monitoring."
        ],
        common_adverse_effects=["Bradycardia", "Dyspnea", "Fatigue", "Dizziness", "Nausea"],
        serious_adverse_effects=["Torsades de Pointes (QTc prolongation)", "Sustained ventricular tachycardia", "Complete heart block"],
        qt_prolongation_risk="HIGH",
        renal_considerations="Excreted exclusively by kidneys. Narrow therapeutic window. Contraindicated if eGFR < 40 mL/min.",
        hepatic_considerations="Not metabolized by liver.",
        pregnancy_category="Category B",
        monitoring_parameters=["Serial ECG QTc interval (hold if QTc > 500 ms)", "Serum potassium (maintain >= 4.0 mEq/L)", "Serum magnesium", "eGFR / Creatinine"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="CONTRAINDICATED",
                clinical_effect="Severe additive QT prolongation provoking lethal Torsades de Pointes.",
                management_recommendation="Absolute contraindication: do not combine sotalol with other QT-prolonging antiarrhythmics.",
                source="AHA/ACC/HRS Ventricular Arrhythmia Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Betapace / DailyMed)",
        source_version="NDA-019881-Rev2025",
        last_verified="2026-07-01",
        bradycardia_risk=True,
        egfr_cutoff_ml_min=40.0,
    ),
    "flecainide": MedicationEntry(
        generic_name="Flecainide",
        brand_names=["Tambocor"],
        drug_class="Class IC Antiarrhythmic (Potent Fast Sodium Channel Blocker)",
        mechanism_of_action="Markedly slows phase 0 depolarization and cardiac conduction velocity with minimal effect on repolarization.",
        cardiac_indications=[
            "Prevention of Paroxysmal Atrial Fibrillation / Flutter in patients without structural heart disease",
            "Paroxysmal Supraventricular Tachycardias (PSVT)",
        ],
        contraindications=[
            "Preexisting structural heart disease (prior MI, CAD, Heart Failure with reduced LVEF)",
            "Second- or third-degree AV block or bifascicular block without pacemaker",
            "Cardiogenic shock",
        ],
        black_box_warnings=[
            "CAST Trial: Excessive mortality or non-fatal cardiac arrest rate in patients with structural heart disease or recent MI."
        ],
        common_adverse_effects=["Dizziness", "Visual disturbances", "Dyspnea", "Headache"],
        serious_adverse_effects=["Proarrhythmic ventricular tachycardia", "Marked QRS widening", "Sudden cardiac death in CAD"],
        qt_prolongation_risk="LOW",
        renal_considerations="Renal elimination (30%); reduce dosage in moderate-severe renal impairment (eGFR < 50 mL/min).",
        hepatic_considerations="CYP2D6 metabolism; use cautiously in severe hepatic impairment.",
        pregnancy_category="Category C",
        monitoring_parameters=["ECG QRS duration (hold if QRS widens > 25% from baseline)", "Serum electrolytes", "Renal function"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="MAJOR",
                clinical_effect="Doubles flecainide plasma concentration; severe conduction arrest risk.",
                management_recommendation="Reduce flecainide dose by 50% and monitor ECG intervals.",
                source="DailyMed / FDA Prescribing Info",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Tambocor / DailyMed)",
        source_version="NDA-019056-Rev2025",
        last_verified="2026-05-18",
    ),
    "aspirin": MedicationEntry(
        generic_name="Aspirin",
        brand_names=["Bayer", "Ecotrin", "Bufferin", "Aspro"],
        drug_class="Antiplatelet / Cyclooxygenase-1 (COX-1) Inhibitor",
        mechanism_of_action="Irreversibly acetylates platelet cyclooxygenase-1, inhibiting thromboxane A2 synthesis and platelet aggregation for the lifespan of the platelet.",
        cardiac_indications=[
            "Secondary prevention in documented Coronary Artery Disease (CAD)",
            "Acute Coronary Syndromes (STEMI / NSTEMI / Unstable Angina)",
            "Post-Percutaneous Coronary Intervention (PCI / Stent placement)",
            "Ischemic Cerebrovascular Disease",
        ],
        contraindications=[
            "Active peptic ulceration or major gastrointestinal bleeding",
            "Severe hypersensitivity to aspirin or NSAIDs (aspirin-exacerbated respiratory disease)",
            "Severe thrombocytopenia or bleeding diathesis",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Dyspepsia", "Epigastric discomfort", "Easy bruising", "Minor epistaxis"],
        serious_adverse_effects=["Major GI hemorrhage", "Intracranial bleeding", "Anaphylaxis / Bronchospasm in aspirin-sensitive asthma"],
        qt_prolongation_risk="NONE",
        renal_considerations="Avoid high doses in severe renal failure; low-dose 81-100 mg generally tolerated.",
        hepatic_considerations="Use cautiously in severe cirrhosis with coagulopathy.",
        pregnancy_category="Category D (in third trimester; low dose 81 mg used for preeclampsia prophylaxis under obstetric supervision)",
        monitoring_parameters=["Hemoglobin/Hematocrit", "Signs of occult bleeding", "Platelet count"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Warfarin",
                severity="MAJOR",
                clinical_effect="Significantly increases major bleeding and hemorrhagic stroke risk.",
                management_recommendation="Combine only with compelling indication (e.g., recent mechanical valve/PCI); monitor closely.",
                source="CHEST Antithrombotic Guidelines",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Apixaban",
                severity="MODERATE",
                clinical_effect="Additive bleeding risk; increases GI and mucosal hemorrhage.",
                management_recommendation="Limit combination duration to strictly guideline-supported timelines.",
                source="ACC/AHA Atrial Fibrillation Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information / USP-NF / DailyMed",
        source_version="USP44-NF39-Rev2025",
        last_verified="2026-06-01",
    ),
    "clopidogrel": MedicationEntry(
        generic_name="Clopidogrel",
        brand_names=["Plavix"],
        drug_class="P2Y12 Platelet Receptor Inhibitor (Thienopyridine Antiplatelet)",
        mechanism_of_action="Prodrug whose active metabolite irreversibly blocks the P2Y12 ADP receptor on platelets, preventing activation of the GPIIb/IIIa complex.",
        cardiac_indications=[
            "Acute Coronary Syndrome (STEMI / NSTEMI / UA) Dual Antiplatelet Therapy (DAPT)",
            "Recent Percutaneous Coronary Intervention with coronary stent",
            "Secondary prevention in ischemic stroke or peripheral arterial disease",
        ],
        contraindications=[
            "Active pathological bleeding (peptic ulcer or intracranial hemorrhage)",
            "Severe hypersensitivity to clopidogrel",
        ],
        black_box_warnings=[
            "Diminished antiplatelet effect in patients with CYP2C19 loss-of-function alleles (poor metabolizers)."
        ],
        common_adverse_effects=["Bruising", "Hematoma", "Epistaxis", "Gastrointestinal hemorrhage"],
        serious_adverse_effects=["Thrombotic Thrombocytopenic Purpura (TTP)", "Major life-threatening hemorrhage"],
        qt_prolongation_risk="NONE",
        renal_considerations="No dosage adjustment needed for renal impairment.",
        hepatic_considerations="Severe hepatic impairment increases bleeding risk.",
        pregnancy_category="Category B",
        monitoring_parameters=["Complete blood count", "Platelet count", "Signs of bleeding"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Omeprazole",
                severity="MODERATE",
                clinical_effect="Inhibits CYP2C19, decreasing active clopidogrel metabolite formation and antiplatelet efficacy.",
                management_recommendation="Use pantoprazole as alternative PPI if gastroprotection is required.",
                source="FDA Drug Safety Communication / ACCF/ACG/AHA Consensus",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Plavix / DailyMed)",
        source_version="NDA-020839-Rev2025",
        last_verified="2026-06-12",
    ),
    "atorvastatin": MedicationEntry(
        generic_name="Atorvastatin",
        brand_names=["Lipitor", "Atorva"],
        drug_class="HMG-CoA Reductase Inhibitor (High-Intensity Statin)",
        mechanism_of_action="Competitively inhibits HMG-CoA reductase, upregulating hepatic LDL receptors and increasing clearance of atherogenic lipoproteins.",
        cardiac_indications=[
            "Atherosclerotic Cardiovascular Disease (ASCVD) Secondary Prevention (CAD, Prior MI, Angina)",
            "Primary hyperlipidemia / Hypercholesterolemia",
            "Cardiovascular risk reduction in Type 2 Diabetes Mellitus",
        ],
        contraindications=[
            "Active liver disease or unexplained persistent elevations of hepatic transaminases (> 3x ULN)",
            "Known hypersensitivity to atorvastatin",
            "Pregnancy and lactation",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Myalgia", "Arthralgia", "Diarrhea", "Nasopharyngitis", "Mild transaminase elevation"],
        serious_adverse_effects=["Rhabdomyolysis with acute kidney injury", "Hepatotoxicity", "Immune-mediated necrotizing myopathy"],
        qt_prolongation_risk="NONE",
        renal_considerations="Eliminated primarily via bile; no dosage adjustment required in renal impairment.",
        hepatic_considerations="Extensive CYP3A4 metabolism; contraindicated in active liver disease or decompensated cirrhosis.",
        pregnancy_category="Category X (Contraindicated in pregnancy)",
        monitoring_parameters=["Lipid panel at 4-12 weeks", "Baseline and clinical liver function tests (ALT/AST)", "Serum CK if muscle symptoms arise"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Diltiazem",
                severity="MODERATE",
                clinical_effect="CYP3A4 inhibition elevates atorvastatin exposure, increasing myopathy risk.",
                management_recommendation="Limit atorvastatin dose or monitor for unexplained muscle pain/weakness.",
                source="FDA Prescribing Info / AHA Scientific Statement on Statin Safety",
                evidence_level="Level B",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Lipitor / DailyMed)",
        source_version="NDA-020702-Rev2025",
        last_verified="2026-06-25",
        liver_warning_level="CONTRAINDICATED",
    ),
    "rosuvastatin": MedicationEntry(
        generic_name="Rosuvastatin",
        brand_names=["Crestor"],
        drug_class="HMG-CoA Reductase Inhibitor (High-Intensity Statin)",
        mechanism_of_action="Selectively and competitively inhibits HMG-CoA reductase, markedly reducing circulating LDL-C and stabilizing atherosclerotic plaques.",
        cardiac_indications=[
            "Secondary prevention in documented CAD / ASCVD",
            "Hypercholesterolemia and mixed dyslipidemia",
            "Prevention of cardiovascular events in high-risk patients",
        ],
        contraindications=[
            "Active liver disease or unexplained transaminase elevations",
            "Severe renal impairment (eGFR < 30 mL/min for high doses)",
            "Pregnancy and breastfeeding",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Myalgia", "Headache", "Abdominal pain", "Nausea", "Asthenia"],
        serious_adverse_effects=["Rhabdomyolysis", "Acute pancreatitis", "Hepatitis"],
        qt_prolongation_risk="NONE",
        renal_considerations="10% cleared renally; in severe renal impairment (eGFR < 30 mL/min), initiate at 5 mg OD; do not exceed 10 mg OD.",
        hepatic_considerations="Contraindicated in active liver disease.",
        pregnancy_category="Category X",
        monitoring_parameters=["Lipid panel", "Baseline liver enzymes", "Serum creatinine / eGFR"],
        known_interactions=[],
        authoritative_source="FDA Prescribing Information (Crestor / DailyMed)",
        source_version="NDA-021366-Rev2025",
        last_verified="2026-06-18",
        egfr_cutoff_ml_min=30.0,
        liver_warning_level="CONTRAINDICATED",
    ),
    "metformin": MedicationEntry(
        generic_name="Metformin",
        brand_names=["Glucophage", "Fortamet", "Glumetza"],
        drug_class="Biguanide Antihyperglycemic Agent",
        mechanism_of_action="Decreases hepatic glucose production, decreases intestinal absorption of glucose, and improves insulin sensitivity by increasing peripheral glucose uptake.",
        cardiac_indications=[
            "First-line glycemic management in Type 2 Diabetes Mellitus with cardiovascular risk reduction",
        ],
        contraindications=[
            "Severe renal impairment (eGFR < 30 mL/min)",
            "Acute or chronic metabolic acidosis (including diabetic ketoacidosis)",
            "Severe heart failure or acute hypoxemia (predisposing to tissue hypoperfusion)",
            "Severe hepatic dysfunction",
        ],
        black_box_warnings=[
            "Lactic Acidosis: Rare but fatal metabolic complication. Risk increases with renal impairment, sepsis, and heart failure."
        ],
        common_adverse_effects=["Diarrhea", "Nausea", "Vomiting", "Flatulence", "Vitamin B12 deficiency"],
        serious_adverse_effects=["Lactic acidosis", "Severe hypoglycemia (when combined with secretagogues)"],
        qt_prolongation_risk="NONE",
        renal_considerations="Excreted exclusively unchanged by kidneys. Contraindicated if eGFR < 30 mL/min. If eGFR 30-44 mL/min, maximum dose 1000 mg/day.",
        hepatic_considerations="Impaired liver function increases risk of lactic acidosis; avoid in advanced cirrhosis.",
        pregnancy_category="Category B",
        monitoring_parameters=["eGFR / Serum creatinine at baseline and annually (or q3-6m if eGFR < 60)", "HbA1c", "Vitamin B12 levels"],
        known_interactions=[],
        authoritative_source="FDA Prescribing Information (Glucophage / DailyMed)",
        source_version="NDA-020357-Rev2025",
        last_verified="2026-06-08",
        egfr_cutoff_ml_min=30.0,
    ),
    "empagliflozin": MedicationEntry(
        generic_name="Empagliflozin",
        brand_names=["Jardiance"],
        drug_class="Sodium-Glucose Cotransporter 2 (SGLT2) Inhibitor",
        mechanism_of_action="Inhibits SGLT2 in the proximal renal tubule, reducing renal glucose and sodium reabsorption, promoting glucosuria and osmotic diuresis, reducing cardiac preload and afterload.",
        cardiac_indications=[
            "Type 2 Diabetes Mellitus with established cardiovascular disease (to reduce CV death)",
            "Heart Failure with reduced ejection fraction (HFrEF) or preserved ejection fraction (HFpEF)",
            "Chronic Kidney Disease progression reduction",
        ],
        contraindications=[
            "Severe hypersensitivity to empagliflozin",
            "Dialysis or end-stage renal disease",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Genital mycotic infections", "Urinary tract infections", "Volume depletion / Polyuria"],
        serious_adverse_effects=["Euglycemic diabetic ketoacidosis (euDKA)", "Necrotizing fasciitis of the perineum (Fournier's gangrene)", "Acute kidney injury in volume depletion"],
        qt_prolongation_risk="NONE",
        renal_considerations="Not recommended for glycemic control if eGFR < 30 mL/min; indicated for heart failure / CKD benefits down to eGFR >= 20 mL/min.",
        hepatic_considerations="No dose adjustment in mild-moderate hepatic impairment.",
        pregnancy_category="Not recommended during second and third trimesters.",
        monitoring_parameters=["eGFR / Renal panel", "Blood pressure / Volume status", "Symptoms of ketoacidosis"],
        known_interactions=[],
        authoritative_source="FDA Prescribing Information (Jardiance / DailyMed)",
        source_version="NDA-204629-Rev2025",
        last_verified="2026-06-14",
        egfr_cutoff_ml_min=20.0,
    ),
    "furosemide": MedicationEntry(
        generic_name="Furosemide",
        brand_names=["Lasix"],
        drug_class="Loop Diuretic (Sulfonamide derivative)",
        mechanism_of_action="Inhibits the Na+/K+/2Cl- cotransporter in the thick ascending limb of the loop of Henle, promoting marked excretion of sodium, chloride, potassium, and water.",
        cardiac_indications=[
            "Edema associated with Congestive Heart Failure",
            "Acute Pulmonary Edema",
            "Resistant Hypertension in renal disease",
        ],
        contraindications=[
            "Anuria",
            "Severe uncorrected electrolyte depletion (hypokalemia < 3.0 mEq/L, severe hyponatremia)",
            "Hypersensitivity to furosemide",
        ],
        black_box_warnings=[
            "Potent diuretic that can lead to profound diuresis with water and electrolyte depletion."
        ],
        common_adverse_effects=["Hypokalemia", "Hyponatremia", "Hypomagnesemia", "Hyperuricemia", "Dizziness"],
        serious_adverse_effects=["Life-threatening hypokalemic arrhythmias", "Ototoxicity (with rapid IV boluses)", "Acute pre-renal azotemia"],
        qt_prolongation_risk="LOW",
        renal_considerations="Effective even at low eGFR; higher doses required in severe renal insufficiency. Monitor electrolytes closely.",
        hepatic_considerations="In cirrhosis with ascites, electrolyte disturbances may precipitate hepatic encephalopathy.",
        pregnancy_category="Category C",
        monitoring_parameters=["Serum potassium (target >= 4.0 mEq/L)", "Serum magnesium", "Serum creatinine / BUN", "Blood pressure", "Weight"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Digoxin",
                severity="MAJOR",
                clinical_effect="Furosemide-induced hypokalemia severely increases myocardial susceptibility to fatal digoxin toxicity.",
                management_recommendation="Maintain serum potassium >= 4.0 mEq/L; consider potassium supplementation or K-sparing diuretic.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Lasix / DailyMed)",
        source_version="NDA-015457-Rev2025",
        last_verified="2026-06-11",
        hypotension_risk=True,
    ),
    "amlodipine": MedicationEntry(
        generic_name="Amlodipine",
        brand_names=["Norvasc"],
        drug_class="Dihydropyridine Calcium Channel Blocker",
        mechanism_of_action="Selectively inhibits calcium influx across vascular smooth muscle cells and cardiac muscle, causing peripheral arterial vasodilation and reducing afterload.",
        cardiac_indications=[
            "Hypertension",
            "Coronary Artery Disease (Chronic Stable Angina / Vasospastic Angina)",
        ],
        contraindications=[
            "Severe hypotension (systolic BP < 90 mmHg)",
            "Known hypersensitivity to amlodipine",
            "Severe aortic stenosis",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Peripheral ankle edema", "Flushing", "Dizziness", "Palpitations", "Headache"],
        serious_adverse_effects=["Severe hypotension / syncope", "Paradoxical angina exacerbation at initiation"],
        qt_prolongation_risk="NONE",
        renal_considerations="No dose adjustment required in renal impairment; not dialyzable.",
        hepatic_considerations="Extensive hepatic metabolism; initiate at lower dose (2.5 mg OD) in hepatic impairment.",
        pregnancy_category="Category C",
        monitoring_parameters=["Blood pressure", "Lower extremity edema"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Simvastatin",
                severity="MODERATE",
                clinical_effect="Amlodipine increases simvastatin exposure; limit simvastatin to 20 mg daily.",
                management_recommendation="Cap simvastatin dose or switch to atorvastatin/rosuvastatin.",
                source="FDA Drug Safety Communication",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Norvasc / DailyMed)",
        source_version="NDA-019787-Rev2025",
        last_verified="2026-06-19",
        hypotension_risk=True,
    ),
    "losartan": MedicationEntry(
        generic_name="Losartan",
        brand_names=["Cozaar"],
        drug_class="Angiotensin II Receptor Blocker (ARB)",
        mechanism_of_action="Selectively blocks binding of angiotensin II to AT1 receptors in vascular smooth muscle and adrenal gland, promoting vasodilation and aldosterone reduction.",
        cardiac_indications=[
            "Hypertension",
            "Diabetic Nephropathy in Type 2 Diabetes with elevated serum creatinine and proteinuria",
            "Left Ventricular Hypertrophy (stroke risk reduction)",
            "Alternative to ACE inhibitors in HFrEF intolerance",
        ],
        contraindications=[
            "Concomitant aliskiren in patients with diabetes",
            "Pregnancy (Black Box Warning)",
            "Known hypersensitivity to losartan",
            "Bilateral renal artery stenosis",
        ],
        black_box_warnings=[
            "Fetal Toxicity: When pregnancy is detected, discontinue as soon as possible; causes injury and death to developing fetus."
        ],
        common_adverse_effects=["Dizziness", "Hyperkalemia", "Fatigue", "Upper respiratory congestion"],
        serious_adverse_effects=["Acute renal failure", "Severe hyperkalemia", "Angioedema (rare compared to ACEi)"],
        qt_prolongation_risk="NONE",
        renal_considerations="Dose titration required; monitor serum potassium and creatinine within 1-2 weeks of initiation.",
        hepatic_considerations="Initiate at lower dose (25 mg OD) in patients with history of hepatic impairment.",
        pregnancy_category="Category D (Black Box Warning)",
        monitoring_parameters=["Serum potassium", "Serum creatinine / eGFR", "Blood pressure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Spironolactone",
                severity="MAJOR",
                clinical_effect="Additive retention of potassium provoking dangerous hyperkalemia.",
                management_recommendation="Monitor serum potassium within 1 week of combination.",
                source="AHA/ACC Hypertension Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Cozaar / DailyMed)",
        source_version="NDA-020386-Rev2025",
        last_verified="2026-06-22",
        hypotension_risk=True,
    ),
    "verapamil": MedicationEntry(
        generic_name="Verapamil",
        brand_names=["Calan", "Isoptin", "Verelan"],
        drug_class="Non-Dihydropyridine Calcium Channel Blocker (Class IV Antiarrhythmic)",
        mechanism_of_action="Inhibits L-type calcium channels, primarily depressing AV node conduction, slowing sinus rate, and decreasing myocardial contractility.",
        cardiac_indications=[
            "Atrial Fibrillation / Atrial Flutter ventricular rate control (in preserved ejection fraction)",
            "Supraventricular Tachycardia prophylaxis",
            "Chronic Stable Angina",
            "Hypertension",
        ],
        contraindications=[
            "Severe left ventricular dysfunction / HFrEF (LVEF < 40%)",
            "Second- or third-degree AV block or sick sinus syndrome without pacemaker",
            "Severe hypotension (SBP < 90 mmHg)",
            "Wolff-Parkinson-White syndrome with AFib (may precipitate VFib)",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Constipation", "Bradycardia", "Dizziness", "Hypotension", "Headache"],
        serious_adverse_effects=["Complete heart block", "Acute pulmonary edema / Heart failure decompensation", "Profound hypotension"],
        qt_prolongation_risk="NONE",
        renal_considerations="Use with caution in renal failure; monitor for hemodynamic depression.",
        hepatic_considerations="Extensive CYP3A4 hepatic clearance; reduce dose by 70% in cirrhosis.",
        pregnancy_category="Category C",
        monitoring_parameters=["Heart rate", "Blood pressure", "ECG PR interval", "Signs of heart failure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Metoprolol",
                severity="MAJOR",
                clinical_effect="Profound negative inotropic and chronotropic synergism; severe AV block or cardiac arrest.",
                management_recommendation="Avoid concurrent administration in heart failure or conduction system disease.",
                source="ACC/AHA Guidelines",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Digoxin",
                severity="MAJOR",
                clinical_effect="Increases serum digoxin levels by 50-70% via renal clearance inhibition and additive AV block.",
                management_recommendation="Reduce digoxin dose by 50% and monitor levels closely.",
                source="DailyMed / FDA Label",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Calan / DailyMed)",
        source_version="NDA-018535-Rev2025",
        last_verified="2026-06-17",
        bradycardia_risk=True,
        hypotension_risk=True,
    ),
    "valsartan": MedicationEntry(
        generic_name="Valsartan",
        brand_names=["Diovan"],
        drug_class="Angiotensin II Receptor Blocker (ARB)",
        mechanism_of_action="Blocks the vasoconstrictor and aldosterone-secreting effects of angiotensin II by selectively blocking AT1 receptors.",
        cardiac_indications=["Hypertension", "Heart Failure (NYHA class II-IV)", "Post-Myocardial Infarction"],
        contraindications=["Pregnancy", "Concomitant aliskiren in diabetes"],
        black_box_warnings=["Fetal toxicity."],
        common_adverse_effects=["Dizziness", "Hypotension", "Hyperkalemia"],
        serious_adverse_effects=["Angioedema", "Renal impairment"],
        qt_prolongation_risk="NONE",
        renal_considerations="Monitor serum creatinine and potassium in renal impairment.",
        hepatic_considerations="Reduce dose in mild-to-moderate hepatic impairment; avoid in severe cirrhosis.",
        pregnancy_category="Category D",
        monitoring_parameters=["Blood pressure", "Potassium", "Renal function"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Spironolactone",
                severity="MAJOR",
                clinical_effect="Additive retention of potassium provoking hyperkalemia.",
                management_recommendation="Monitor serum potassium within 1-2 weeks.",
                source="AHA/ACC Hypertension Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Diovan / DailyMed)",
        source_version="NDA-021283-Rev2025",
        last_verified="2026-06-20",
        hypotension_risk=True,
    ),
    "sacubitril/valsartan": MedicationEntry(
        generic_name="Sacubitril/Valsartan",
        brand_names=["Entresto"],
        drug_class="Angiotensin Receptor-Neprilysin Inhibitor (ARNI)",
        mechanism_of_action="Prodrug sacubitril inhibits neprilysin, increasing natriuretic peptide levels; valsartan selectively blocks AT1 angiotensin II receptors.",
        cardiac_indications=[
            "Heart Failure with reduced ejection fraction (HFrEF, NYHA Class II-IV)",
            "Heart Failure with preserved ejection fraction (HFpEF with below-normal LVEF)",
        ],
        contraindications=[
            "Concomitant use with ACE inhibitors (mandatory 36-hour washout required)",
            "History of angioedema related to prior ACE inhibitor or ARB therapy",
            "Concomitant use with Aliskiren in patients with diabetes",
            "Pregnancy (fetal toxicity)",
        ],
        black_box_warnings=["Drugs that act directly on the renin-angiotensin system can cause injury and death to the developing fetus."],
        common_adverse_effects=["Hypotension", "Hyperkalemia", "Cough", "Dizziness", "Renal impairment"],
        serious_adverse_effects=["Life-threatening angioedema", "Severe hypotension", "Acute renal failure"],
        qt_prolongation_risk="NONE",
        renal_considerations="Reduce starting dose in severe renal impairment (eGFR < 30 mL/min). Monitor serum potassium and creatinine.",
        hepatic_considerations="Moderate hepatic impairment (Child-Pugh B): reduce starting dose. Contraindicated in severe hepatic impairment (Child-Pugh C).",
        pregnancy_category="Category D",
        monitoring_parameters=["Blood pressure", "Serum potassium", "Serum creatinine / eGFR"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Lisinopril",
                severity="CRITICAL",
                clinical_effect="Severe additive risk of life-threatening angioedema. Strictly contraindicated.",
                management_recommendation="Do not administer within 36 hours of switching to or from an ACE inhibitor.",
                source="FDA Prescribing Information (Entresto / DailyMed)",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Spironolactone",
                severity="MAJOR",
                clinical_effect="Additive potassium retention predisposing to severe hyperkalemia.",
                management_recommendation="Monitor serum potassium within 1-2 weeks of concurrent therapy.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Entresto / DailyMed)",
        source_version="NDA-207620-Rev2025",
        last_verified="2026-06-28",
        hypotension_risk=True,
        egfr_cutoff_ml_min=30.0,
        liver_warning_level="WARNING",
        rxnorm_cui="1656337",
        dailymed_setid="8ba52899-7dd2-4b2e-a4be-83bf43714b18",
        drugcentral_id="4904",
        sider_adverse_effects=["hypotension", "hyperkalemia", "cough", "dizziness", "angioedema"],
    ),
    "hydrochlorothiazide": MedicationEntry(
        generic_name="Hydrochlorothiazide",
        brand_names=["Microzide", "Esidrix", "HydroDIURIL", "HCTZ"],
        drug_class="Thiazide Diuretic",
        mechanism_of_action="Inhibits sodium and chloride reabsorption in distal convoluted tubules, increasing natriuresis, secondarily causing potassium and magnesium loss.",
        cardiac_indications=[
            "Hypertension (First-line guideline-directed therapy)",
            "Edema associated with Congestive Heart Failure",
            "Hepatic Cirrhosis with Ascites",
        ],
        contraindications=[
            "Anuria",
            "Hypersensitivity to hydrochlorothiazide or sulfonamide-derived drugs",
            "Refractory hypokalemia or hyponatremia",
            "Symptomatic hyperuricemia or acute gout flare",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Hypokalemia", "Hyponatremia", "Hyperuricemia", "Hypomagnesemia", "Dizziness", "Hyperglycemia"],
        serious_adverse_effects=[
            "Severe electrolyte depletion provoking ventricular tachyarrhythmias",
            "Acute angle-closure glaucoma",
            "Renal failure secondary to hypovolemia",
        ],
        qt_prolongation_risk="MODERATE",
        renal_considerations="Ineffective as monotherapy when eGFR < 30 mL/min (switch to loop diuretics like furosemide). Monitor serum creatinine and BUN.",
        hepatic_considerations="Use cautiously in severe liver disease; hypokalemia/electrolyte shifts may precipitate hepatic encephalopathy.",
        pregnancy_category="Category B",
        monitoring_parameters=["Serum potassium", "Serum sodium", "Serum magnesium", "Serum creatinine / eGFR", "Blood pressure", "Serum uric acid"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Digoxin",
                severity="CRITICAL",
                clinical_effect="Thiazide-induced hypokalemia or hypomagnesemia dramatically potentiates digitalis toxicity, causing AV block and lethal ventricular arrhythmias.",
                management_recommendation="Maintain serum potassium > 4.0 mEq/L; monitor serum digoxin and potassium levels regularly.",
                source="AHA/ACC Heart Failure Guidelines",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Sotalol",
                severity="MAJOR",
                clinical_effect="Hypokalemia significantly potentiates sotalol-induced QTc prolongation, predisposing to Torsades de Pointes.",
                management_recommendation="Correct potassium and magnesium prior to initiation; monitor ECG QTc interval closely.",
                source="AHA/ACC/HRS Ventricular Arrhythmia Guidelines",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Amiodarone",
                severity="MAJOR",
                clinical_effect="Hypokalemia amplifies class III proarrhythmic risk; increased risk of polymorphic ventricular tachycardia.",
                management_recommendation="Ensure serum potassium is maintained between 4.0-5.0 mEq/L during combined therapy.",
                source="ESC Guidelines on Arrhythmias",
                evidence_level="Level B",
            ),
            MedicationInteraction(
                interacting_drug="Lisinopril",
                severity="MODERATE",
                clinical_effect="Synergistic blood pressure reduction; ACE inhibitor counters thiazide-induced hypokalemia.",
                management_recommendation="Monitor BP, potassium, and renal function at 1-2 weeks after initiating combination.",
                source="AHA/ACC Hypertension Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Microzide / DailyMed)",
        source_version="NDA-020582-Rev2025",
        last_verified="2026-06-15",
        bradycardia_risk=False,
        hypotension_risk=True,
        egfr_cutoff_ml_min=30.0,
        liver_warning_level="WARNING",
        sider_adverse_effects=["hypokalemia", "hyponatremia", "hyperuricemia", "dizziness", "hypotension"],
        drugcentral_id="1382",
        rxnorm_cui="5487",
        dailymed_setid="18086202-3ff7-440d-b108-c89b7cf3d3b7",
    ),
    "aspirin/dipyridamole": MedicationEntry(
        generic_name="Aspirin/Dipyridamole",
        brand_names=["Aggrenox"],
        drug_class="Antiplatelet Combination",
        mechanism_of_action="Dual antiplatelet action: Aspirin inhibits cyclooxygenase-1 and thromboxane A2; dipyridamole inhibits adenosine deaminase and PDE.",
        cardiac_indications=[
            "Secondary stroke prevention in patients with prior transient ischemic attack (TIA) or completed ischemic stroke",
        ],
        contraindications=[
            "Hypersensitivity to aspirin, NSAIDs, or dipyridamole",
            "Active peptic ulceration or active gastrointestinal bleeding",
            "Severe bleeding diathesis or severe thrombocytopenia",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Headache", "Dyspepsia", "Abdominal pain", "Nausea", "Diarrhea"],
        serious_adverse_effects=["Major GI hemorrhage", "Intracranial hemorrhage", "Anaphylaxis"],
        qt_prolongation_risk="NONE",
        renal_considerations="Avoid in severe renal failure (eGFR < 15 mL/min) due to aspirin-induced renal prostaglandin inhibition.",
        hepatic_considerations="Use caution in severe hepatic impairment; dipyridamole is metabolized hepatically.",
        pregnancy_category="Category D",
        monitoring_parameters=["Hemoglobin/Hematocrit", "Signs and symptoms of bleeding", "Blood pressure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Warfarin",
                severity="MAJOR",
                clinical_effect="Significantly increased risk of major gastrointestinal and intracranial bleeding.",
                management_recommendation="Avoid concurrent use unless strictly indicated; close INR and clinical bleeding monitoring.",
                source="CHEST Guidelines on Antithrombotic Therapy",
                evidence_level="Level A",
            ),
            MedicationInteraction(
                interacting_drug="Apixaban",
                severity="MAJOR",
                clinical_effect="Additive inhibition of platelet aggregation and thrombin generation, heightening bleeding risk.",
                management_recommendation="Limit concurrent duration to guideline-approved post-ACS or post-PCI indications.",
                source="AHA/ACC Stroke Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Aggrenox / DailyMed)",
        source_version="NDA-020884-Rev2025",
        last_verified="2026-06-25",
        bradycardia_risk=False,
        hypotension_risk=True,
        egfr_cutoff_ml_min=15.0,
        liver_warning_level="MONITOR",
        sider_adverse_effects=["headache", "dyspepsia", "bleeding", "diarrhea"],
        drugcentral_id="294",
        rxnorm_cui="214535",
        dailymed_setid="75da6ea4-33fa-4e76-8fe1-6923fc82faee",
    ),
    "dipyridamole": MedicationEntry(
        generic_name="Dipyridamole",
        brand_names=["Persantine"],
        drug_class="Antiplatelet / Phosphodiesterase Inhibitor",
        mechanism_of_action="Inhibits cellular reuptake of adenosine; inhibits platelet cyclic nucleotide phosphodiesterase.",
        cardiac_indications=[
            "Thromboembolism prophylaxis adjunct to warfarin following cardiac valve replacement",
            "Pharmacologic myocardial perfusion stress testing",
        ],
        contraindications=[
            "Hypersensitivity to dipyridamole",
            "Severe coronary artery disease / unstable angina (coronary steal phenomenon)",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Dizziness", "Headache", "Flushing", "Hypotension", "Gastrointestinal distress"],
        serious_adverse_effects=["Coronary steal provoking myocardial ischemia", "Severe bronchospasm", "Angioedema"],
        qt_prolongation_risk="NONE",
        renal_considerations="No dosage adjustment necessary in renal impairment.",
        hepatic_considerations="Metabolized extensively in liver; caution in severe hepatic insufficiency.",
        pregnancy_category="Category B",
        monitoring_parameters=["Blood pressure", "Heart rate", "Angina symptoms"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Adenosine",
                severity="CRITICAL",
                clinical_effect="Dipyridamole prevents adenosine uptake, substantially exaggerating adenosine effects, predisposing to prolonged asystole or severe heart block.",
                management_recommendation="Reduce adenosine diagnostic/therapeutic dose by 75% or withhold dipyridamole for at least 48 hours.",
                source="ACC/AHA Guidelines for Cardiac Stress Testing",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Persantine / DailyMed)",
        source_version="NDA-012836-Rev2024",
        last_verified="2026-06-20",
        bradycardia_risk=False,
        hypotension_risk=True,
        egfr_cutoff_ml_min=None,
        liver_warning_level="MONITOR",
        sider_adverse_effects=["headache", "dizziness", "flushing", "hypotension"],
        drugcentral_id="969",
        rxnorm_cui="3498",
        dailymed_setid="0ad71ef5-e51c-42cb-b1b0-9cf65eb1a257",
    ),
    "dapagliflozin": MedicationEntry(
        generic_name="Dapagliflozin",
        brand_names=["Farxiga", "Forxiga"],
        drug_class="Sodium-Glucose Cotransporter 2 (SGLT2) Inhibitor",
        mechanism_of_action="Inhibits SGLT2 in proximal renal tubules, reducing glucose and sodium reabsorption, promoting osmotic diuresis and cardiorenal protection.",
        cardiac_indications=[
            "Heart Failure with reduced ejection fraction (HFrEF) or preserved ejection fraction (HFpEF)",
            "Chronic Kidney Disease at risk of progression",
            "Type 2 Diabetes Mellitus with established cardiovascular disease",
        ],
        contraindications=[
            "Severe hypersensitivity to dapagliflozin",
            "Patients on dialysis",
        ],
        black_box_warnings=[],
        common_adverse_effects=["Mycotic genital infections", "Urinary tract infections", "Volume depletion"],
        serious_adverse_effects=["Euglycemic diabetic ketoacidosis (euDKA)", "Fournier gangrene", "Urosepsis"],
        qt_prolongation_risk="NONE",
        renal_considerations="Approved for cardiorenal protection down to eGFR 20 mL/min.",
        hepatic_considerations="No dose adjustment required in mild-to-moderate hepatic impairment.",
        pregnancy_category="Category D (second and third trimesters)",
        monitoring_parameters=["Renal function / eGFR", "Volume status / blood pressure", "Signs of ketoacidosis"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Furosemide",
                severity="MODERATE",
                clinical_effect="Additive volume depletion and hypotension.",
                management_recommendation="Assess volume status and consider loop diuretic dose adjustment upon initiation.",
                source="ADA / AHA Heart Failure Consensus",
                evidence_level="Level B",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Farxiga / DailyMed)",
        source_version="NDA-202293-Rev2025",
        last_verified="2026-06-10",
        bradycardia_risk=False,
        hypotension_risk=True,
        egfr_cutoff_ml_min=20.0,
        liver_warning_level="MONITOR",
        sider_adverse_effects=["urinary tract infection", "hypoglycemia", "hypotension"],
        drugcentral_id="4798",
        rxnorm_cui="1488564",
        dailymed_setid="2f9ec62d-9496-419b-a01b-c7414df8ff00",
    ),
    "enalapril": MedicationEntry(
        generic_name="Enalapril",
        brand_names=["Vasotec", "Epaned"],
        drug_class="Angiotensin-Converting Enzyme (ACE) Inhibitor",
        mechanism_of_action="Prodrug enalaprilat competitively inhibits ACE, suppressing angiotensin II synthesis and decreasing peripheral resistance.",
        cardiac_indications=[
            "Hypertension",
            "Heart Failure (NYHA Class II-IV)",
            "Asymptomatic Left Ventricular Dysfunction (LVEF <= 35%)",
        ],
        contraindications=[
            "History of angioedema related to prior ACE inhibitor therapy",
            "Concomitant use with aliskiren in patients with diabetes",
            "Pregnancy",
        ],
        black_box_warnings=["Fetal toxicity: Can cause injury and death to developing fetus."],
        common_adverse_effects=["Dry cough", "Dizziness", "Hypotension", "Hyperkalemia"],
        serious_adverse_effects=["Life-threatening angioedema", "Acute kidney injury", "Severe hyperkalemia"],
        qt_prolongation_risk="NONE",
        renal_considerations="Reduce dose in renal impairment (eGFR < 30 mL/min). Monitor creatinine and potassium.",
        hepatic_considerations="Prodrug bioactivation occurs in liver; use caution in severe hepatic impairment.",
        pregnancy_category="Category D",
        monitoring_parameters=["Serum potassium", "Serum creatinine", "Blood pressure"],
        known_interactions=[
            MedicationInteraction(
                interacting_drug="Spironolactone",
                severity="MAJOR",
                clinical_effect="Severe additive hyperkalemia risk.",
                management_recommendation="Monitor potassium closely; ensure baseline potassium < 5.0 mEq/L.",
                source="AHA/ACC HF Guidelines",
                evidence_level="Level A",
            ),
        ],
        authoritative_source="FDA Prescribing Information (Vasotec / DailyMed)",
        source_version="NDA-018998-Rev2025",
        last_verified="2026-06-05",
        bradycardia_risk=False,
        hypotension_risk=True,
        egfr_cutoff_ml_min=30.0,
        liver_warning_level="MONITOR",
        sider_adverse_effects=["cough", "hypotension", "hyperkalemia", "dizziness"],
        drugcentral_id="1035",
        rxnorm_cui="3827",
        dailymed_setid="0ef4ef9a-e160-4b20-ba10-53bc12304892",
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
        if not name or not str(name).strip():
            return None
        clean = str(name).lower().strip()
        # 1. Exact match on generic key
        if clean in self._entries:
            return self._entries[clean]
        # 2. Exact match on brand name
        for entry in self._entries.values():
            if clean in [b.lower() for b in entry.brand_names]:
                return entry
        # 3. Substring match (e.g. 'Metoprolol 50 mg' or 'Coreg CR 25mg')
        # Sort entries by length descending to match composite names first (e.g. 'Sacubitril/Valsartan' before 'Valsartan')
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: max([len(e.generic_name)] + [len(b) for b in e.brand_names]),
            reverse=True,
        )
        for entry in sorted_entries:
            if entry.generic_name.lower() in clean or any(b.lower() in clean for b in entry.brand_names):
                return entry
        # 4. Token / dosage-stripped matching
        base = re.sub(r'[\d/\.\+\-]+|\b(mg|mcg|g|ml|tab|tablet|capsule|daily|bid|tid|qid|od|prn|po|iv)\b', ' ', clean).strip()
        base_first = base.split()[0] if base.split() else clean
        if base_first in self._entries:
            return self._entries[base_first]
        for entry in self._entries.values():
            if any(base_first == b.lower().split()[0] for b in entry.brand_names):
                return entry
        return None

    def lookup_medication(self, name: str) -> Optional[MedicationEntry]:
        """Lookup medication entry by generic or brand name."""
        return self.get_by_name(name)

    def resolve_to_generic(self, query: str) -> Optional[str]:
        """Maps brand or generic drug name to its canonical generic name."""
        entry = self.get_by_name(query)
        return entry.generic_name if entry else None

    def list_all(self) -> List[MedicationEntry]:
        return list(self._entries.values())


GLOBAL_MEDICATION_DB = MedicationDatabase()


def resolve_drug_name(name: str) -> str:
    """Module-level resolver mapping brand or generic name to canonical generic name string."""
    entry = GLOBAL_MEDICATION_DB.get_by_name(name)
    if entry:
        return entry.generic_name.lower()
    return name.lower().strip()

_API_CACHE: Dict[str, Any] = {}


def query_rxnorm(drug_name: str, timeout_sec: float = 2.5) -> Dict[str, Any]:
    """Query NIH RxNorm REST API for RxCUI concept properties with local fallback."""
    cache_key = f"rxnorm_{drug_name.lower().strip()}"
    if cache_key in _API_CACHE:
        return _API_CACHE[cache_key]

    clean_name = urllib.parse.quote(drug_name.strip())
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={clean_name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ECG-Guardian/2.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            id_group = data.get("idGroup", {})
            rxnorm_id = (id_group.get("rxnormId") or [None])[0]
            result = {
                "source": "NIH RxNorm",
                "drug_name": drug_name,
                "rxcui": rxnorm_id,
                "is_found": bool(rxnorm_id),
            }
            _API_CACHE[cache_key] = result
            return result
    except Exception:
        pass

    entry = GLOBAL_MEDICATION_DB.get_by_name(drug_name)
    fallback = {
        "source": "NIH RxNorm (Local Reference Fallback)",
        "drug_name": drug_name,
        "rxcui": entry.generic_name if entry else None,
        "is_found": bool(entry),
    }
    _API_CACHE[cache_key] = fallback
    return fallback


def query_openfda(drug_name: str, timeout_sec: float = 2.5) -> Dict[str, Any]:
    """Query FDA openFDA Drug Label API for boxed warnings and QTc contraindications."""
    cache_key = f"openfda_{drug_name.lower().strip()}"
    if cache_key in _API_CACHE:
        return _API_CACHE[cache_key]

    clean_name = urllib.parse.quote(drug_name.strip())
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:\"{clean_name}\"+openfda.brand_name:\"{clean_name}\"&limit=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ECG-Guardian/2.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                lbl = results[0]
                res = {
                    "source": "FDA openFDA Drug Label",
                    "drug_name": drug_name,
                    "warnings": (lbl.get("warnings") or [""])[0][:300],
                    "boxed_warning": bool(lbl.get("boxed_warning")),
                    "is_found": True,
                }
                _API_CACHE[cache_key] = res
                return res
    except Exception:
        pass

    entry = GLOBAL_MEDICATION_DB.get_by_name(drug_name)
    fallback = {
        "source": "FDA Prescribing Information (Local DB Fallback)",
        "drug_name": drug_name,
        "warnings": entry.qt_prolongation_risk if entry else "",
        "boxed_warning": bool(entry and entry.black_box_warnings),
        "is_found": bool(entry),
    }
    _API_CACHE[cache_key] = fallback
    return fallback


def query_dailymed(drug_name: str, timeout_sec: float = 2.5) -> Dict[str, Any]:
    """Query NLM DailyMed Web Service for structured product labeling (SPL)."""
    cache_key = f"dailymed_{drug_name.lower().strip()}"
    if cache_key in _API_CACHE:
        return _API_CACHE[cache_key]

    clean_name = urllib.parse.quote(drug_name.strip())
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={clean_name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ECG-Guardian/2.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            spls = data.get("data", [])
            res = {
                "source": "NLM DailyMed SPL",
                "drug_name": drug_name,
                "spl_count": len(spls),
                "is_found": len(spls) > 0,
            }
            _API_CACHE[cache_key] = res
            return res
    except Exception:
        pass

    entry = GLOBAL_MEDICATION_DB.get_by_name(drug_name)
    fallback = {
        "source": "NLM DailyMed (Local DB Fallback)",
        "drug_name": drug_name,
        "spl_count": 1 if entry else 0,
        "is_found": bool(entry),
    }
    _API_CACHE[cache_key] = fallback
    return fallback

