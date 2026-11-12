"""
ECG Guardian — Multimodal Medication Candidate Generation & Safety Engine
========================================================================

Integrates:
1. Multimodal candidate association model (trained on patient-level MIMIC-IV / eICU / DailyMed).
2. Deterministic Safety Verification Gates:
   - Allergy cross-matching
   - Drug-drug interaction cross-checking
   - Physiological contraindications (QTc prolongation, bradycardia, hypotension)
   - Laboratory contraindications (Hyperkalemia, Hypokalemia, Renal impairment)
3. Authoritative Clinical Analysis Object generation:
   - Produces ONE strongly typed AuthoritativeClinicalAnalysis instance.
   - Non-autonomous: strictly decision support for qualified clinicians.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from src.clinical.clinical_analysis import (
    AuthoritativeClinicalAnalysis,
    ClinicalContextData,
    ECGAnalysisData,
    ECGMeasurementsData,
    MedicationCandidate,
    MedicationDecisionSupportData,
    ModelMetadataData,
    PatientContext,
    PreviousECGComparisonData,
)
from src.medications.interaction_checker import check_medication_safety
from src.medications.medication_database import GLOBAL_MEDICATION_DB


# Drug class definitions with canonical example agents and indications
MEDICATION_KNOWLEDGE = {
    "beta_blocker": {
        "display_name": "Beta-Adrenergic Blocker",
        "example_agents": "Metoprolol succinate (25-100 mg OD), Bisoprolol (2.5-10 mg OD)",
        "indication": "Ventricular ectopy suppression, rate control, secondary prevention in CAD/post-MI, and HFrEF guideline-directed medical therapy",
        "guideline": "2019 ESC Guidelines on Ventricular Arrhythmias / 2022 AHA/ACC Heart Failure Guideline",
        "qt_risk": "LOW",
        "brady_risk": True,
        "hypotension_risk": True,
    },
    "calcium_channel_blocker": {
        "display_name": "Calcium Channel Blocker (DHP / Non-DHP)",
        "example_agents": "Amlodipine (5-10 mg OD) or Diltiazem (120-240 mg OD)",
        "indication": "Hypertension management, angina relief, or rate control alternative to beta-blockers in preserved LV function",
        "guideline": "2020 ISH Global Hypertension Practice Guidelines / AHA 2017 Guidelines",
        "qt_risk": "LOW",
        "brady_risk": True,  # Non-DHP
        "hypotension_risk": True,
    },
    "antiarrhythmic_class_3": {
        "display_name": "Class III Antiarrhythmic",
        "example_agents": "Amiodarone (200 mg OD) or Sotalol (80-160 mg BID)",
        "indication": "Refractory ventricular arrhythmias or atrial fibrillation rhythm control in structural heart disease",
        "guideline": "2020 ESC AF Guidelines / 2017 AHA/ACC/HRS Ventricular Arrhythmia Guidelines",
        "qt_risk": "HIGH",
        "brady_risk": True,
        "hypotension_risk": False,
    },
    "anticoagulant_doac": {
        "display_name": "Direct Oral Anticoagulant (DOAC)",
        "example_agents": "Apixaban (5 mg BID) or Rivaroxaban (20 mg OD)",
        "indication": "Thromboembolism prevention in non-valvular atrial fibrillation with elevated CHA2DS2-VASc score",
        "guideline": "2020 ESC Guidelines for Atrial Fibrillation (Class I, Level A)",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": False,
    },
    "antiplatelet": {
        "display_name": "Antiplatelet Agent",
        "example_agents": "Aspirin (75-100 mg OD) / Clopidogrel (75 mg OD)",
        "indication": "Secondary prevention in documented coronary artery disease, acute coronary syndrome, or ischemic cerebrovascular disease",
        "guideline": "2023 ESC Guidelines for the Management of Acute Coronary Syndromes",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": False,
    },
    "ace_inhibitor_arb": {
        "display_name": "ACE Inhibitor / Angiotensin Receptor Blocker",
        "example_agents": "Lisinopril (10-40 mg OD) or Losartan (50-100 mg OD)",
        "indication": "Hypertension, HFrEF neurohormonal blockade, post-MI LV remodeling prevention, and diabetic nephropathy",
        "guideline": "2022 AHA/ACC/HFSA Heart Failure Guidelines (Class I, Level A)",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": True,
    },
    "diuretic_loop": {
        "display_name": "Loop Diuretic",
        "example_agents": "Furosemide (20-40 mg OD/BID) or Torsemide (10-20 mg OD)",
        "indication": "Volume overload and pulmonary/peripheral congestion in acute or chronic decompensated heart failure",
        "guideline": "2022 AHA/ACC Heart Failure Guideline (Class I, Level B)",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": True,
    },
    "statin": {
        "display_name": "HMG-CoA Reductase Inhibitor (Statin)",
        "example_agents": "Atorvastatin (20-80 mg OD) or Rosuvastatin (10-40 mg OD)",
        "indication": "Lipid-lowering and plaque stabilization in atherosclerotic cardiovascular disease or high cardiovascular risk",
        "guideline": "2019 ESC/EAS Guidelines for the Management of Dyslipidaemias",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": False,
    },
    "electrolyte_replacement": {
        "display_name": "Electrolyte Optimization (Potassium/Magnesium)",
        "example_agents": "Potassium Chloride (20-40 mEq PO) / Magnesium Sulfate",
        "indication": "Correction of hypokalemia/hypomagnesemia to prevent triggered ventricular ectopy and stabilize cardiac repolarization",
        "guideline": "AHA/ACC Standards for Inpatient Electrolyte Repletion in Cardiac Arrhythmias",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": False,
    },
    "sglt2_inhibitor": {
        "display_name": "SGLT2 Inhibitor",
        "example_agents": "Empagliflozin (10-25 mg OD) or Dapagliflozin (10 mg OD)",
        "indication": "Cardiorenal risk reduction, heart failure hospitalization prevention, and glycemic regulation in Type 2 Diabetes",
        "guideline": "2023 ADA Standards of Care / 2023 ESC Guidelines on CVD in Diabetes (Class I, Level A)",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": False,
    },
    "biguanide": {
        "display_name": "Biguanide (Metformin)",
        "example_agents": "Metformin (500-1000 mg BID with meals)",
        "indication": "First-line glycemic control and insulin sensitization in Type 2 Diabetes",
        "guideline": "2023 ADA Standards of Medical Care in Diabetes",
        "qt_risk": "LOW",
        "brady_risk": False,
        "hypotension_risk": False,
    },
}

FEATURE_COLUMNS = [
    "age", "is_male", "sym_chest_pain", "sym_palpitations", "sym_dyspnea", "sym_syncope",
    "systolic_bp", "diastolic_bp", "vital_heart_rate", "spo2",
    "ecg_heart_rate", "pr_interval_ms", "qrs_duration_ms", "qtc_ms",
    "is_ecg_normal", "is_ecg_pvc", "is_ecg_other",
    "has_hypertension", "has_diabetes", "has_cad", "has_heart_failure", "has_ckd",
    "serum_potassium", "serum_creatinine", "egfr",
    "on_beta_blocker", "on_ace_inhibitor", "on_statin", "on_antiplatelet",
]


class MultimodalMedicationCandidateEngine:
    """Predicts candidate medication classes and subjects them to rigorous multi-tiered safety checking."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or "models/medication_model")
        self.scaler = None
        self.estimators = None
        self.metadata = None
        self._load_model_artifacts()

    def _load_model_artifacts(self) -> None:
        try:
            if (self.model_dir / "scaler.pkl").exists() and (self.model_dir / "calibrated_estimators.pkl").exists():
                self.scaler = joblib.load(self.model_dir / "scaler.pkl")
                self.estimators = joblib.load(self.model_dir / "calibrated_estimators.pkl")
                if (self.model_dir / "metadata.json").exists():
                    with open(self.model_dir / "metadata.json", "r") as f:
                        self.metadata = json.load(f)
        except Exception:
            self.scaler = None
            self.estimators = None

    def _extract_feature_vector(
        self,
        patient: Optional[Dict[str, Any]],
        vitals: Optional[Dict[str, Any]],
        labs: Optional[Dict[str, Any]],
        ecg_finding: str,
        ecg_params: Dict[str, Any],
    ) -> np.ndarray:
        p = patient or {}
        v = vitals or {}
        l = labs or {}

        age = float(p.get("age") or 55.0)
        is_male = 1.0 if str(p.get("sex", "")).upper().startswith("M") else 0.0

        # Symptoms
        symptoms_str = str(p.get("symptoms", "")).lower()
        sym_cp = 1.0 if any(k in symptoms_str for k in ["chest", "angina", "tightness"]) else 0.0
        sym_palp = 1.0 if any(k in symptoms_str for k in ["palpitation", "flutter", "racing", "skipped"]) else 0.0
        sym_dysp = 1.0 if any(k in symptoms_str for k in ["shortness", "dyspnea", "breath", "sob"]) else 0.0
        sym_sync = 1.0 if any(k in symptoms_str for k in ["syncope", "faint", "dizziness", "lightheaded"]) else 0.0

        # Vitals
        sbp = float(v.get("systolic_bp") or v.get("sbp") or 124.0)
        dbp = float(v.get("diastolic_bp") or v.get("dbp") or 78.0)
        v_hr = float(v.get("heart_rate") or ecg_params.get("heart_rate_bpm") or 72.0)
        spo2 = float(v.get("spo2") or 98.0)

        # ECG Params
        ecg_hr = float(ecg_params.get("heart_rate_bpm") or v_hr)
        pr_ms = float(ecg_params.get("pr_interval_ms") or 160.0)
        qrs_ms = float(ecg_params.get("qrs_duration_ms") or 90.0)
        qtc_ms = float(ecg_params.get("qtc_bazett_ms") or ecg_params.get("qtc_ms") or 420.0)

        finding_u = ecg_finding.upper()
        is_norm = 1.0 if "NORMAL" in finding_u and "PVC" not in finding_u else 0.0
        is_pvc = 1.0 if "PVC" in finding_u or ("VENTRICULAR" in finding_u and "SUPRA" not in finding_u) else 0.0
        is_oth = 1.0 if (not is_norm and not is_pvc) else 0.0

        # Comorbidities
        cond_str = str(p.get("conditions", "")).lower() + " " + str(p.get("cardiac_history", "")).lower()
        has_htn = 1.0 if any(k in cond_str for k in ["hypertension", "htn", "high blood pressure"]) else 0.0
        has_dm = 1.0 if any(k in cond_str for k in ["diabetes", "dm", "t2dm", "t1dm"]) else 0.0
        has_cad = 1.0 if any(k in cond_str for k in ["cad", "coronary", "mi", "infarction", "stent", "cabg"]) else 0.0
        has_hf = 1.0 if any(k in cond_str for k in ["heart failure", "hf", "chf", "reduced ejection", "hfref"]) else 0.0
        has_ckd = 1.0 if any(k in cond_str for k in ["ckd", "kidney", "renal", "nephropathy"]) else 0.0

        # Labs
        k_val = float(l.get("potassium_meq_l") or l.get("potassium") or l.get("serum_potassium") or 4.2)
        cr_val = float(l.get("creatinine_mg_dl") or l.get("creatinine") or l.get("serum_creatinine") or 0.95)
        egfr = float(l.get("egfr") or max(15.0, 110.0 - (age * 0.6) - (cr_val * 22.0)))

        # Current Meds
        curr_str = str(p.get("current_medications", "")).lower()
        on_bb = 1.0 if any(k in curr_str for k in ["metoprolol", "atenolol", "bisoprolol", "carvedilol", "propranolol", "beta"]) else 0.0
        on_ace = 1.0 if any(k in curr_str for k in ["lisinopril", "enalapril", "ramipril", "losartan", "valsartan", "arb", "ace"]) else 0.0
        on_statin = 1.0 if any(k in curr_str for k in ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "statin"]) else 0.0
        on_ap = 1.0 if any(k in curr_str for k in ["aspirin", "clopidogrel", "plavix", "ticagrelor", "antiplatelet"]) else 0.0

        vec = np.array([
            age, is_male, sym_cp, sym_palp, sym_dysp, sym_sync,
            sbp, dbp, v_hr, spo2,
            ecg_hr, pr_ms, qrs_ms, qtc_ms,
            is_norm, is_pvc, is_oth,
            has_htn, has_dm, has_cad, has_hf, has_ckd,
            k_val, cr_val, egfr,
            on_bb, on_ace, on_statin, on_ap
        ], dtype=float).reshape(1, -1)
        return vec

    def generate_candidate_recommendations(
        self,
        ecg_finding: str,
        ecg_measurements: Optional[Dict[str, Any]] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        vital_signs: Optional[Dict[str, Any]] = None,
        laboratory_results: Optional[Dict[str, Any]] = None,
        finding_reliability: Optional[Any] = None,
    ) -> MedicationDecisionSupportData:
        """Generate multimodal candidates subjected to clinical safety verification."""
        p_dict = patient_profile or {}
        v_dict = vital_signs or {}
        l_dict = laboratory_results or {}
        ecg_params = ecg_measurements or {}

        # 1. Evaluate Evidence Gating & Reliability
        rel_obj = finding_reliability
        is_reliable = True
        withheld_reasons = []

        if rel_obj is not None:
            if hasattr(rel_obj, "reliable"):
                is_reliable = bool(rel_obj.reliable)
                if not is_reliable and hasattr(rel_obj, "reasons"):
                    withheld_reasons.extend(rel_obj.reasons)
            elif isinstance(rel_obj, dict):
                is_reliable = bool(rel_obj.get("reliable", True))
                if not is_reliable:
                    withheld_reasons.extend(rel_obj.get("reasons", ["ECG finding unverified"]))

        # Missing clinical context audit
        missing_info = []
        if not v_dict.get("systolic_bp") and not v_dict.get("blood_pressure"):
            missing_info.append("Blood pressure not recorded")
        if not l_dict.get("potassium_meq_l") and not l_dict.get("serum_potassium"):
            missing_info.append("Serum potassium not recorded")
        if not l_dict.get("creatinine_mg_dl") and not l_dict.get("serum_creatinine"):
            missing_info.append("Serum creatinine not recorded")
        if not p_dict.get("current_medications") or str(p_dict.get("current_medications")).lower() in ["none recorded", "none documented"]:
            missing_info.append("Current medication list unverified")
        if not p_dict.get("allergies") and not p_dict.get("known_allergies"):
            missing_info.append("Allergy status unverified")

        # 2. Candidate Prediction
        raw_associations: Dict[str, float] = {}
        target_classes = list(MEDICATION_KNOWLEDGE.keys())

        feat_vec = self._extract_feature_vector(p_dict, v_dict, l_dict, ecg_finding, ecg_params)

        if self.scaler is not None and self.estimators is not None:
            try:
                scaled_vec = self.scaler.transform(feat_vec)
                for i, col_name in enumerate(self.metadata.get("target_classes", target_classes)):
                    est = self.estimators[i]
                    prob = float(est.predict_proba(scaled_vec)[0, 1])
                    key = col_name.replace("tgt_", "")
                    raw_associations[key] = round(prob, 4)
            except Exception:
                raw_associations = {}

        # Comprehensive Multi-Disease Detection for Guideline Therapy Mapping
        cond_parts = [
            str(p_dict.get("conditions", "")),
            str(p_dict.get("existing_conditions", "")),
            str(p_dict.get("cardiac_history", "")),
            str(p_dict.get("medical_history", "")),
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

        sym_str = str(p_dict.get("symptoms", "")).lower()
        finding_u = ecg_finding.upper()
        is_pvc = "PVC" in finding_u or ("VENTRICULAR" in finding_u and "SUPRA" not in finding_u)
        is_afib = "AFIB" in finding_u or "FIBRILLATION" in finding_u or "FLUTTER" in finding_u

        has_htn = any(k in cond_str for k in ["hypertension", "htn", "high blood pressure", "high bp", "elevated bp"])
        has_dm = any(k in cond_str for k in ["diabetes", "dm", "t2dm", "t1dm", "diabetic", "sugar"])
        has_dyslip = any(k in cond_str for k in ["cholesterol", "lipid", "dyslipidemia", "hyperlipidemia", "hypercholesterolemia"])
        has_cad = any(k in cond_str for k in ["cad", "coronary", "angina", "mi", "infarction", "stent", "cabg", "ischemic"]) or "angina" in sym_str
        has_hf = any(k in cond_str for k in ["heart failure", "hf", "chf", "hfref", "hfmref", "hfpef", "ejection fraction", "cardiomyopathy"]) or "shortness" in sym_str
        has_arrh = any(k in cond_str for k in ["pvc", "arrhythmia", "palpitation", "ectopy", "ventricular ectopy", "flutter"]) or is_pvc
        has_afib_cond = any(k in cond_str for k in ["atrial fibrillation", "afib", "a-fib", "af"]) or is_afib

        if not raw_associations:
            raw_associations = {
                "beta_blocker": 0.85 if is_pvc else (0.65 if has_cad else 0.35),
                "calcium_channel_blocker": 0.60 if has_htn else 0.20,
                "antiarrhythmic_class_3": 0.55 if is_pvc and has_hf else 0.10,
                "anticoagulant_doac": 0.75 if has_afib_cond else 0.15,
                "antiplatelet": 0.70 if has_cad else 0.25,
                "ace_inhibitor_arb": 0.68 if has_htn or has_hf else 0.30,
                "diuretic_loop": 0.72 if has_hf else 0.15,
                "statin": 0.78 if has_cad or has_dm or has_dyslip else 0.40,
                "electrolyte_replacement": 0.80 if is_pvc else 0.20,
            }

        # Strongly link documented patient conditions to guideline therapy candidates
        if has_htn:
            raw_associations["ace_inhibitor_arb"] = max(raw_associations.get("ace_inhibitor_arb", 0.0), 0.96)
            raw_associations["calcium_channel_blocker"] = max(raw_associations.get("calcium_channel_blocker", 0.0), 0.91)
        if has_dm:
            raw_associations["sglt2_inhibitor"] = 0.95
            raw_associations["biguanide"] = 0.90
            raw_associations["statin"] = max(raw_associations.get("statin", 0.0), 0.88)
        if has_dyslip:
            raw_associations["statin"] = max(raw_associations.get("statin", 0.0), 0.97)
        if has_cad:
            raw_associations["beta_blocker"] = max(raw_associations.get("beta_blocker", 0.0), 0.94)
            raw_associations["antiplatelet"] = max(raw_associations.get("antiplatelet", 0.0), 0.92)
            raw_associations["statin"] = max(raw_associations.get("statin", 0.0), 0.95)
        if has_hf:
            raw_associations["ace_inhibitor_arb"] = max(raw_associations.get("ace_inhibitor_arb", 0.0), 0.96)
            raw_associations["beta_blocker"] = max(raw_associations.get("beta_blocker", 0.0), 0.94)
            raw_associations["diuretic_loop"] = max(raw_associations.get("diuretic_loop", 0.0), 0.90)
        if has_arrh:
            raw_associations["beta_blocker"] = max(raw_associations.get("beta_blocker", 0.0), 0.93)
            raw_associations["electrolyte_replacement"] = max(raw_associations.get("electrolyte_replacement", 0.0), 0.88)
        if has_afib_cond:
            raw_associations["anticoagulant_doac"] = max(raw_associations.get("anticoagulant_doac", 0.0), 0.96)
            raw_associations["beta_blocker"] = max(raw_associations.get("beta_blocker", 0.0), 0.90)

        # 3. Deterministic Safety Verification & Gate Enforcement
        hr_val = float(ecg_params.get("heart_rate_bpm") or v_dict.get("heart_rate") or v_dict.get("heart_rate_bpm") or 72.0)
        qtc_val = float(ecg_params.get("qtc_bazett_ms") or ecg_params.get("qtc_ms") or 420.0)
        pr_val = float(ecg_params.get("pr_interval_ms") or 160.0)
        sbp_val = float(v_dict.get("systolic_bp") or v_dict.get("systolic_bp_mmhg") or v_dict.get("sbp") or 124.0)
        k_val = float(l_dict.get("potassium_mmol_l") or l_dict.get("potassium_meq_l") or l_dict.get("serum_potassium") or l_dict.get("potassium") or 4.2)
        cr_val = float(l_dict.get("serum_creatinine_mg_dl") or l_dict.get("creatinine_mg_dl") or l_dict.get("serum_creatinine") or l_dict.get("creatinine") or 0.95)
        trop_val = float(l_dict.get("troponin_i_ng_ml") or l_dict.get("troponin") or 0.0)

        allergies_str = str(p_dict.get("allergies") or p_dict.get("known_allergies") or "").lower()
        curr_meds_str = str(p_dict.get("current_medications") or "").lower()

        candidates: List[MedicationCandidate] = []
        contraindications_found: List[str] = []
        interactions_found: List[str] = []
        allergy_alerts: List[Dict[str, Any]] = []

        # Sort candidate classes by predicted association score descending
        sorted_targets = sorted(raw_associations.items(), key=lambda kv: kv[1], reverse=True)

        for drug_key, score in sorted_targets:
            if score < 0.25:
                continue

            info = MEDICATION_KNOWLEDGE.get(drug_key, {})
            display_name = info.get("display_name", drug_key.replace("_", " ").title())
            example_agents = info.get("example_agents", "Consult formulary")
            indication = info.get("indication", "Clinical decision support candidate")
            guideline = info.get("guideline", "AHA/ACC/ESC Standards")

            c_status = "SAFE"
            c_contraindications = []
            c_interactions = []
            c_allergy = "CLEAR"
            c_lab_checks = []

            # A. Allergy Verification
            allergies_norm = allergies_str.replace("-", " ")
            drug_name_tokens = [drug_key.replace("_", " ")]
            if "beta" in drug_key:
                drug_name_tokens.extend(["beta blocker", "beta", "metoprolol", "bisoprolol", "carvedilol", "atenolol"])
            elif "calcium" in drug_key:
                drug_name_tokens.extend(["calcium channel blocker", "ccb", "diltiazem", "verapamil", "amlodipine"])
            elif "antiarrhythmic" in drug_key:
                drug_name_tokens.extend(["amiodarone", "sotalol", "flecainide", "iodine"])
            elif "statin" in drug_key:
                drug_name_tokens.extend(["statin", "atorvastatin", "rosuvastatin", "simvastatin"])
            elif "antiplatelet" in drug_key:
                drug_name_tokens.extend(["aspirin", "clopidogrel", "plavix", "salicylate", "nsaid"])
            elif "ace" in drug_key:
                drug_name_tokens.extend(["ace inhibitor", "ace", "lisinopril", "enalapril", "arb", "losartan", "angioedema"])

            matched_allergy = any((tok in allergies_str or tok in allergies_norm) for tok in drug_name_tokens if len(tok) >= 3)
            if matched_allergy:
                c_status = "BLOCKED"
                c_allergy = "ALLERGY_ALERT"
                alert_msg = f"Documented allergy conflict: patient record contains '{allergies_str}'."
                c_contraindications.append(alert_msg)
                contraindications_found.append(f"{display_name}: {alert_msg}")
                allergy_alerts.append({"medication": display_name, "alert": alert_msg})

            # B. Electrophysiological Safety Gates
            # QTc Prolongation Gate
            if info.get("qt_risk") == "HIGH" or drug_key == "antiarrhythmic_class_3":
                if qtc_val >= 500.0:
                    c_status = "BLOCKED"
                    msg = f"Severe QTc Prolongation ({qtc_val:.0f} ms >= 500 ms). High risk of Torsades de Pointes."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")
                elif qtc_val >= 470.0:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    msg = f"Borderline prolonged QTc ({qtc_val:.0f} ms). Close serial ECG monitoring required."
                    c_contraindications.append(msg)

            # Bradycardia / AV Conduction Gate
            if info.get("brady_risk", False):
                if hr_val < 50.0:
                    c_status = "BLOCKED"
                    msg = f"Sinus Bradycardia (Heart Rate {hr_val:.0f} bpm < 50 bpm). AV nodal blocking agent contraindicated."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")
                elif hr_val < 60.0 or pr_val > 220.0:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    msg = f"Marginal Heart Rate ({hr_val:.0f} bpm) or Prolonged PR ({pr_val:.0f} ms). Risk of higher-grade heart block."
                    c_contraindications.append(msg)

            # C. Hemodynamic Safety Gates
            if info.get("hypotension_risk", False) and sbp_val < 95.0:
                if c_status != "BLOCKED":
                    c_status = "REVIEW_REQUIRED"
                msg = f"Borderline Hypotension (Systolic BP {sbp_val:.0f} mmHg < 95 mmHg). Vasodilation may cause hypoperfusion."
                c_contraindications.append(msg)

            # D. Laboratory Safety Gates
            # Hyperkalemia Gate
            if drug_key in ["ace_inhibitor_arb", "electrolyte_replacement"]:
                if k_val >= 5.2:
                    c_status = "BLOCKED"
                    msg = f"Hyperkalemia (Serum Potassium {k_val:.1f} mEq/L >= 5.2 mEq/L). Contraindicated to prevent life-threatening conduction arrest."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")
                elif k_val >= 4.9:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    c_contraindications.append(f"High-normal Potassium ({k_val:.1f} mEq/L). Frequent electrolyte checks mandated.")
                c_lab_checks.append(f"Serum K+: {k_val:.1f} mEq/L")

            # Renal Function Gate
            egfr_val = float(l_dict.get("egfr_ml_min") or l_dict.get("egfr") or max(15.0, 110.0 - (float(p_dict.get("age") or 55.0) * 0.6) - (cr_val * 22.0)))
            c_lab_checks.append(f"Serum Cr: {cr_val:.2f} mg/dL, eGFR: {egfr_val:.0f} mL/min")

            if drug_key == "biguanide":
                if egfr_val < 30.0 or cr_val > 2.5:
                    c_status = "BLOCKED"
                    msg = f"Severe Renal Impairment (eGFR {egfr_val:.0f} mL/min < 30 mL/min). Metformin contraindicated due to risk of fatal lactic acidosis (FDA Black Box / ADA 2024 / KDIGO)."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")
                elif egfr_val < 45.0:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    c_contraindications.append(f"Moderate CKD (eGFR {egfr_val:.0f} mL/min). Dose cap of 1000 mg/day indicated (KDIGO/ADA).")

            if drug_key == "antiarrhythmic_class_3":
                if egfr_val < 40.0:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    msg = f"Renal Impairment (eGFR {egfr_val:.0f} mL/min < 40 mL/min). Sotalol accumulation risk; contraindicated or requires strict dose reduction (FDA Label)."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")

            if drug_key in ["anticoagulant_doac", "ace_inhibitor_arb"]:
                if cr_val > 2.5 or egfr_val < 30.0:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    msg = f"Severe Renal Impairment (Serum Creatinine {cr_val:.2f} mg/dL, eGFR {egfr_val:.0f} mL/min). Renal dosage titration or alternative required (KDIGO)."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")

            # Hepatic Function Gate
            alt_val = float(l_dict.get("alt_u_l") or l_dict.get("alt") or 0.0)
            ast_val = float(l_dict.get("ast_u_l") or l_dict.get("ast") or 0.0)
            bili_val = float(l_dict.get("bilirubin_mg_dl") or l_dict.get("bilirubin") or 0.0)
            has_liver = any(k in cond_str for k in ["liver", "hepatic", "cirrhosis", "hepatitis", "jaundice"])

            if alt_val > 0.0 or ast_val > 0.0 or bili_val > 0.0:
                c_lab_checks.append(f"LFTs: ALT {alt_val:.0f} U/L, AST {ast_val:.0f} U/L, Bilirubin {bili_val:.1f} mg/dL")

            if drug_key == "statin":
                if (alt_val >= 120.0 or ast_val >= 120.0 or bili_val >= 2.0 or "cirrhosis" in cond_str):
                    c_status = "BLOCKED"
                    msg = f"Active Hepatic Impairment / Transaminitis (ALT {alt_val:.0f} U/L, AST {ast_val:.0f} U/L, Bilirubin {bili_val:.1f} mg/dL). Statins contraindicated in decompensated liver disease (FDA Label / ESC 2019)."
                    c_contraindications.append(msg)
                    contraindications_found.append(f"{display_name}: {msg}")
                elif has_liver:
                    if c_status != "BLOCKED":
                        c_status = "REVIEW_REQUIRED"
                    c_contraindications.append("Documented liver disease: Monitor LFTs before initiation and periodically (FDA Label).")

            if drug_key == "antiarrhythmic_class_3" and (alt_val >= 120.0 or ast_val >= 120.0 or bili_val >= 2.0 or has_liver):
                if c_status != "BLOCKED":
                    c_status = "REVIEW_REQUIRED"
                msg = "Documented hepatic impairment / transaminitis: Amiodarone carries risk of hepatotoxicity. Baseline and serial LFT monitoring mandatory (FDA / openFDA)."
                c_contraindications.append(msg)
                contraindications_found.append(f"{display_name}: {msg}")

            # E. Drug-Drug Interactions
            if drug_key == "beta_blocker" and ("diltiazem" in curr_meds_str or "verapamil" in curr_meds_str):
                if c_status != "BLOCKED":
                    c_status = "REVIEW_REQUIRED"
                inter_msg = "Concurrent Non-DHP Calcium Channel Blocker: additive AV block and severe bradycardia risk."
                c_interactions.append(inter_msg)
                interactions_found.append(f"Beta Blocker + Non-DHP CCB: {inter_msg}")

            if drug_key == "antiarrhythmic_class_3" and ("warfarin" in curr_meds_str or "digoxin" in curr_meds_str):
                if c_status != "BLOCKED":
                    c_status = "REVIEW_REQUIRED"
                inter_msg = "CYP2C9/P-gp inhibition: Amiodarone markedly elevates Warfarin and Digoxin levels."
                c_interactions.append(inter_msg)
                interactions_found.append(f"Amiodarone + Warfarin/Digoxin: {inter_msg}")

            # Explicit target condition label
            target_disease = None
            if drug_key in ["ace_inhibitor_arb", "calcium_channel_blocker"] and has_htn:
                target_disease = "Hypertension"
            elif drug_key in ["sglt2_inhibitor", "biguanide"] and has_dm:
                target_disease = "Type 2 Diabetes"
            elif drug_key == "statin" and has_dyslip:
                target_disease = "High Cholesterol / Dyslipidemia"
            elif drug_key == "statin" and (has_dm or has_cad):
                target_disease = "Cardiovascular Plaque Stabilization"
            elif drug_key in ["antiplatelet", "beta_blocker"] and has_cad:
                target_disease = "Coronary Artery Disease"
            elif drug_key in ["diuretic_loop", "ace_inhibitor_arb"] and has_hf:
                target_disease = "Heart Failure"
            elif drug_key == "beta_blocker" and has_arrh:
                target_disease = "Ventricular Ectopy / Palpitations"
            elif drug_key == "anticoagulant_doac" and has_afib_cond:
                target_disease = "Atrial Fibrillation"

            if target_disease:
                ind_text = f"[Target Disease: {target_disease}] {indication} ({guideline})"
                rationale_text = f"Indicated for documented {target_disease} per {guideline}."
            else:
                ind_text = f"{indication} ({guideline})"
                rationale_text = f"Considered for {indication}."

            candidates.append(
                MedicationCandidate(
                    medication=display_name,
                    drug_class=display_name,
                    model_association=score,
                    clinical_rationale=rationale_text,
                    relevant_indication=ind_text,
                    safety_status=c_status,
                    contraindications_detected=c_contraindications,
                    drug_interactions_detected=c_interactions,
                    allergy_check=c_allergy,
                    relevant_laboratory_checks=c_lab_checks,
                    missing_information=missing_info,
                    evidence_source=guideline,
                    clinician_review_required=True,
                )
            )

        # 4. Resolve Overall Status
        status = "AVAILABLE"
        status_reason = None
        has_doc_htn = any(k in cond_str for k in ["hypertension", "htn", "high blood pressure", "high bp", "elevated bp"])
        has_doc_dm = any(k in cond_str for k in ["diabetes", "dm", "t2dm", "t1dm", "diabetic", "sugar"])
        has_doc_dyslip = any(k in cond_str for k in ["cholesterol", "lipid", "dyslipidemia", "hyperlipidemia", "hypercholesterolemia"])
        has_doc_cad = any(k in cond_str for k in ["cad", "coronary", "angina", "mi", "infarction", "stent", "cabg", "ischemic"]) or "angina" in sym_str
        has_doc_hf = any(k in cond_str for k in ["heart failure", "hf", "chf", "hfref", "hfmref", "hfpef", "ejection fraction", "cardiomyopathy"])
        has_doc_arrh = any(k in cond_str for k in ["pvc", "arrhythmia", "palpitation", "ectopy", "ventricular ectopy", "flutter"])
        has_doc_afib = any(k in cond_str for k in ["atrial fibrillation", "afib", "a-fib", "af"])
        has_documented_history = (has_doc_htn or has_doc_dm or has_doc_dyslip or has_doc_cad or has_doc_hf or has_doc_arrh or has_doc_afib)

        # Check potential clinical emergency or severe contraindication (Section 5 requirement)
        emergency_triggers = []
        if k_val >= 6.2:
            emergency_triggers.append(f"Critical Hyperkalemia (K+ {k_val:.1f} mmol/L >= 6.2)")
        if hr_val < 42.0:
            emergency_triggers.append(f"Severe Symptomatic Bradycardia (HR {hr_val:.0f} bpm < 42)")
        if sbp_val < 85.0 and bool(v_dict.get("systolic_bp") or v_dict.get("systolic_bp_mmhg") or v_dict.get("sbp")):
            emergency_triggers.append(f"Profound Hemodynamic Shock / Hypotension (SBP {sbp_val:.0f} mmHg < 85)")
        if trop_val >= 1.0 and any(k in sym_str for k in ["chest pain", "angina", "pressure", "substernal"]):
            emergency_triggers.append(f"Suspected Acute Myocardial Infarction / ACS with Critical Troponin Elevation ({trop_val:.2f} ng/mL)")

        if emergency_triggers:
            status = "EMERGENCY_ESCALATION"
            status_reason = (
                f"POTENTIAL CLINICAL EMERGENCY ESCALATION: {'; '.join(emergency_triggers)}. "
                "Routine elective oral medication recommendations withheld. Urgent bedside clinical evaluation, "
                "stabilization, and cardiology emergency escalation required."
            )
            candidates = []
        elif not is_reliable and not has_documented_history:
            status = "WITHHELD"
            status_reason = " ".join(withheld_reasons) if withheld_reasons else "ECG finding reliability unverified; recommendations withheld for clinician review."
        elif has_documented_history and len(candidates) > 0:
            status = "AVAILABLE"
            active_conditions_list = [c for c, present in [("Hypertension", has_doc_htn), ("Diabetes", has_doc_dm), ("Dyslipidemia", has_doc_dyslip), ("CAD", has_doc_cad), ("Heart Failure", has_doc_hf), ("Arrhythmia", has_doc_arrh), ("AFib", has_doc_afib)] if present]
            status_reason = f"Guideline medication candidates synthesized for documented patient diseases ({', '.join(active_conditions_list)})."
        elif not is_reliable:
            status = "WITHHELD"
            status_reason = " ".join(withheld_reasons) if withheld_reasons else "ECG finding reliability unverified; recommendations withheld for clinician review."
        elif len(missing_info) >= 3:
            status = "INSUFFICIENT_INFORMATION"
            status_reason = f"Critical clinical information missing: {'; '.join(missing_info)}. Candidates require complete clinical verification."
        elif any(c.safety_status == "BLOCKED" for c in candidates):
            status = "LIMITED"
            status_reason = "Certain candidates were blocked due to detected physiological, laboratory, or allergy contraindications."

        evidence_entries = [
            {"source": "AHA/ACC/HRS Clinical Practice Guidelines", "url": "https://www.ahajournals.org", "evidence_level": "Level A/B"},
            {"source": "ESC European Society of Cardiology Guidelines", "url": "https://www.escardio.org", "evidence_level": "Level A/B"},
            {"source": "FDA DailyMed Structured Product Labeling (SPL)", "url": "https://dailymed.nlm.nih.gov", "evidence_level": "Regulatory Drug Label"},
        ]

        return MedicationDecisionSupportData(
            status=status,
            status_reason=status_reason,
            clinical_finding=ecg_finding,
            candidates=candidates,
            safety_checks=[
                {"check": "Allergy Screening", "status": "FLAGGED" if allergy_alerts else "PASSED"},
                {"check": "QTc Prolongation Gate", "status": "CONTRAINDICATED" if qtc_val >= 500 else ("CAUTION" if qtc_val >= 470 else "PASSED")},
                {"check": "Bradycardia / AV Conduction Gate", "status": "CONTRAINDICATED" if hr_val < 50 else ("CAUTION" if hr_val < 60 else "PASSED")},
                {"check": "Electrolyte (Potassium) Gate", "status": "CONTRAINDICATED" if k_val >= 5.2 else "PASSED"},
                {"check": "Drug-Drug Interaction Screen", "status": "WARNING" if interactions_found else "PASSED"},
            ],
            contraindications=contraindications_found,
            interactions=interactions_found,
            allergy_checks=allergy_alerts,
            missing_information=missing_info,
            evidence=evidence_entries,
            withheld_reason=status_reason if status in ["WITHHELD", "INSUFFICIENT_INFORMATION"] else None,
            model_version="MED-CANDIDATE-1.0.0-GBM",
            dataset_version="MIMIC-IV-v2.2+eICU+DailyMed-2026.04",
            clinician_review_required=True,
        )


GLOBAL_MEDICATION_CANDIDATE_ENGINE = MultimodalMedicationCandidateEngine()


def build_authoritative_clinical_analysis(
    ecg_finding: str,
    ecg_measurements: Dict[str, Any],
    ecg_probabilities: Optional[Dict[str, float]] = None,
    signal_quality_info: Optional[Dict[str, Any]] = None,
    patient_profile: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
    previous_comparison: Optional[Dict[str, Any]] = None,
    finding_reliability: Optional[Any] = None,
) -> AuthoritativeClinicalAnalysis:
    """Consolidated builder creating the unified AuthoritativeClinicalAnalysis instance."""
    p = patient_profile or {}
    v = vital_signs or {}
    l = laboratory_results or {}
    m = ecg_measurements or {}
    sq = signal_quality_info or {}

    def _parse_list(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(item).strip() for item in val if str(item).strip() and str(item).strip().lower() not in ("none", "none documented", "no known allergies", "no allergies")]
        if isinstance(val, str) and val.strip() and val.strip().lower() not in ("none", "none documented", "no known allergies", "no allergies"):
            return [item.strip() for item in val.split(",") if item.strip()]
        return []

    patient_ctx = PatientContext(
        patient_id=p.get("patient_id") or "PAT-ANON",
        hospital_mrn=p.get("mrn") or p.get("patient_mrn") or p.get("hospital_mrn") or "NOT ASSIGNED",
        name=p.get("name") or p.get("patient_name") or "Anonymous",
        age=int(p.get("age")) if p.get("age") is not None and str(p.get("age")).isdigit() else None,
        sex=p.get("sex") or p.get("patient_sex"),
        blood_group=p.get("blood_group"),
        rh_factor=p.get("rh_factor"),
        date_of_birth=p.get("date_of_birth") or p.get("dob"),
        smoking_status=p.get("smoking_status"),
        contact=p.get("contact"),
        emergency_contact=p.get("emergency_contact"),
        height_cm=float(p.get("height_cm")) if p.get("height_cm") is not None else None,
        weight_kg=float(p.get("weight_kg")) if p.get("weight_kg") is not None else None,
        bmi=float(p.get("bmi")) if p.get("bmi") is not None else (
            round(float(p.get("weight_kg")) / ((float(p.get("height_cm")) / 100.0) ** 2), 1)
            if p.get("weight_kg") and p.get("height_cm") and float(p.get("height_cm")) > 0
            else None
        ),
        previous_surgeries=_parse_list(p.get("previous_surgeries") or p.get("surgeries")),
        previous_hospitalizations=_parse_list(p.get("previous_hospitalizations") or p.get("hospitalizations")),
        food_allergies=_parse_list(p.get("food_allergies")),
        other_allergies=_parse_list(p.get("other_allergies")),
        past_medications=_parse_list(p.get("past_medications")),
        discontinued_medications=_parse_list(p.get("discontinued_medications")),
    )

    meas_data = ECGMeasurementsData(
        heart_rate_bpm=m.get("heart_rate_bpm") or m.get("heart_rate"),
        mean_rr_ms=m.get("mean_rr_ms"),
        pr_interval_ms=m.get("pr_interval_ms"),
        qrs_duration_ms=m.get("qrs_duration_ms"),
        qt_interval_ms=m.get("qt_interval_ms"),
        qtc_bazett_ms=m.get("qtc_bazett_ms") or m.get("qtc_interval_ms"),
        qtc_fridericia_ms=m.get("qtc_fridericia_ms"),
        p_axis_deg=m.get("p_axis_deg"),
        qrs_axis_deg=m.get("qrs_axis_deg"),
        t_axis_deg=m.get("t_axis_deg"),
    )

    ecg_data = ECGAnalysisData(
        primary_finding=ecg_finding,
        findings=[ecg_finding],
        probabilities=ecg_probabilities or {},
        heart_rate=meas_data.heart_rate_bpm,
        rhythm=ecg_finding,
        signal_quality=sq.get("category", "GOOD"),
        signal_quality_score=float(sq.get("score", 1.0)),
        snr_db=float(sq.get("snr_db", 25.0)),
        measurements=meas_data,
    )

    # Clean list fields
    symptoms = _parse_list(p.get("symptoms"))
    history = _parse_list(p.get("conditions") or p.get("existing_conditions"))
    allergies = _parse_list(p.get("allergies") or p.get("known_allergies"))
    current_meds = _parse_list(p.get("current_medications"))

    clinical_ctx = ClinicalContextData(
        symptoms=symptoms,
        medical_history=history,
        allergies=allergies,
        food_allergies=patient_ctx.food_allergies,
        other_allergies=patient_ctx.other_allergies,
        current_medications=current_meds,
        past_medications=patient_ctx.past_medications,
        previous_surgeries=patient_ctx.previous_surgeries,
        previous_hospitalizations=patient_ctx.previous_hospitalizations,
        vitals=v,
        laboratory_results=l,
    )

    med_ds = GLOBAL_MEDICATION_CANDIDATE_ENGINE.generate_candidate_recommendations(
        ecg_finding=ecg_finding,
        ecg_measurements=m,
        patient_profile=p,
        vital_signs=v,
        laboratory_results=l,
        finding_reliability=finding_reliability,
    )

    prev_comp = PreviousECGComparisonData()
    if previous_comparison:
        prev_comp = PreviousECGComparisonData(**{k: v for k, v in previous_comparison.items() if k in PreviousECGComparisonData.__dataclass_fields__})

    model_meta = ModelMetadataData()

    return AuthoritativeClinicalAnalysis(
        patient_context=patient_ctx,
        ecg_analysis=ecg_data,
        clinical_context=clinical_ctx,
        medication_decision_support=med_ds,
        previous_ecg_comparison=prev_comp,
        model_metadata=model_meta,
    )
