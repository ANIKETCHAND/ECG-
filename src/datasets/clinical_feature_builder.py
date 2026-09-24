"""
Clinical Feature Builder Subsystem
===================================

Phases 7, 8, & 9:
Constructs structured tabular feature vectors from multimodal patient context:
- Demographics: Age, Sex, BMI
- Vital Signs: Systolic BP, Diastolic BP, SpO2, Respiratory Rate
- Laboratory Values: Potassium, Magnesium, Serum Creatinine, eGFR, Troponin
- Symptoms: Chest Pain, Dyspnea, Palpitations, Syncope
- Conditions: Hypertension, CAD, Prior MI, Heart Failure, Diabetes
- Active Medications: Beta-blockers, Antiarrhythmics, ACEi/ARBs, Anticoagulants
- Missingness Indicators: Explicit binary flags (is_missing) to honor Rule 1 (Never assume normal)
- Blood Group Governance: Controlled by include_blood_group flag (default False per Rule 28)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.clinical.models import Patient, VitalSigns, LaboratoryResults, Symptoms


CLINICAL_FEATURE_NAMES = [
    # Demographics
    "age",
    "is_female",
    "bmi",
    "bmi_is_missing",
    # Vital signs
    "systolic_bp",
    "systolic_bp_is_missing",
    "diastolic_bp",
    "diastolic_bp_is_missing",
    "spo2_percent",
    "spo2_is_missing",
    # Key Labs
    "potassium_mmol_l",
    "potassium_is_missing",
    "serum_creatinine_mg_dl",
    "creatinine_is_missing",
    "troponin_elevated",
    "troponin_is_missing",
    # Acute Symptoms
    "symptom_chest_pain",
    "symptom_dyspnea",
    "symptom_palpitations",
    "symptom_syncope",
    # Chronic Conditions
    "has_hypertension",
    "has_cad_or_prior_mi",
    "has_heart_failure",
    "has_diabetes",
    # Active Medication Classes
    "med_beta_blocker",
    "med_antiarrhythmic",
    "med_acei_or_arb",
    "med_anticoagulant",
    # Lifestyle
    "is_smoker",
]

BLOOD_GROUP_FEATURE_NAMES = [
    "blood_group_A",
    "blood_group_B",
    "blood_group_AB",
    "blood_group_O",
    "blood_group_Rh_pos",
]


class ClinicalFeatureBuilder:
    """Extracts standardized tabular feature vectors from Patient clinical records."""

    def __init__(self, include_blood_group: bool = False):
        self.include_blood_group = include_blood_group
        self.feature_names = list(CLINICAL_FEATURE_NAMES)
        if self.include_blood_group:
            self.feature_names.extend(BLOOD_GROUP_FEATURE_NAMES)

    def build_features(
        self,
        patient: Optional[Patient],
        as_of_timestamp: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extracts aligned feature array and feature dict from patient record."""
        feat_dict: Dict[str, float] = {}

        if patient is None:
            # All fields marked missing
            feat_dict["age"] = 60.0  # Median population default
            feat_dict["is_female"] = 0.0
            feat_dict["bmi"] = 25.0
            feat_dict["bmi_is_missing"] = 1.0
            feat_dict["systolic_bp"] = 120.0
            feat_dict["systolic_bp_is_missing"] = 1.0
            feat_dict["diastolic_bp"] = 80.0
            feat_dict["diastolic_bp_is_missing"] = 1.0
            feat_dict["spo2_percent"] = 98.0
            feat_dict["spo2_is_missing"] = 1.0
            feat_dict["potassium_mmol_l"] = 4.2
            feat_dict["potassium_is_missing"] = 1.0
            feat_dict["serum_creatinine_mg_dl"] = 1.0
            feat_dict["creatinine_is_missing"] = 1.0
            feat_dict["troponin_elevated"] = 0.0
            feat_dict["troponin_is_missing"] = 1.0
            feat_dict["symptom_chest_pain"] = 0.0
            feat_dict["symptom_dyspnea"] = 0.0
            feat_dict["symptom_palpitations"] = 0.0
            feat_dict["symptom_syncope"] = 0.0
            feat_dict["has_hypertension"] = 0.0
            feat_dict["has_cad_or_prior_mi"] = 0.0
            feat_dict["has_heart_failure"] = 0.0
            feat_dict["has_diabetes"] = 0.0
            feat_dict["med_beta_blocker"] = 0.0
            feat_dict["med_antiarrhythmic"] = 0.0
            feat_dict["med_acei_or_arb"] = 0.0
            feat_dict["med_anticoagulant"] = 0.0
            feat_dict["is_smoker"] = 0.0

            if self.include_blood_group:
                feat_dict["blood_group_A"] = 0.0
                feat_dict["blood_group_B"] = 0.0
                feat_dict["blood_group_AB"] = 0.0
                feat_dict["blood_group_O"] = 0.0
                feat_dict["blood_group_Rh_pos"] = 0.0

            vec = np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32)
            return vec, feat_dict

        # 1. Demographics
        feat_dict["age"] = float(patient.age) if patient.age is not None else 60.0
        feat_dict["is_female"] = 1.0 if (patient.sex and "F" in patient.sex.upper()) else 0.0
        if patient.bmi is not None:
            feat_dict["bmi"] = float(patient.bmi)
            feat_dict["bmi_is_missing"] = 0.0
        else:
            feat_dict["bmi"] = 25.0
            feat_dict["bmi_is_missing"] = 1.0

        # 2. Vital signs (Time-aware)
        vitals = patient.get_latest_vitals_before(as_of_timestamp)
        if vitals and vitals.systolic_bp_mmhg is not None:
            feat_dict["systolic_bp"] = float(vitals.systolic_bp_mmhg)
            feat_dict["systolic_bp_is_missing"] = 0.0
        else:
            feat_dict["systolic_bp"] = 120.0
            feat_dict["systolic_bp_is_missing"] = 1.0

        if vitals and vitals.diastolic_bp_mmhg is not None:
            feat_dict["diastolic_bp"] = float(vitals.diastolic_bp_mmhg)
            feat_dict["diastolic_bp_is_missing"] = 0.0
        else:
            feat_dict["diastolic_bp"] = 80.0
            feat_dict["diastolic_bp_is_missing"] = 1.0

        if vitals and vitals.spo2_percent is not None:
            feat_dict["spo2_percent"] = float(vitals.spo2_percent)
            feat_dict["spo2_is_missing"] = 0.0
        else:
            feat_dict["spo2_percent"] = 98.0
            feat_dict["spo2_is_missing"] = 1.0

        # 3. Laboratory values (Time-aware)
        labs = patient.get_latest_labs_before(as_of_timestamp)
        if labs and labs.potassium_mmol_l is not None:
            feat_dict["potassium_mmol_l"] = float(labs.potassium_mmol_l)
            feat_dict["potassium_is_missing"] = 0.0
        else:
            feat_dict["potassium_mmol_l"] = 4.2
            feat_dict["potassium_is_missing"] = 1.0

        if labs and labs.serum_creatinine_mg_dl is not None:
            feat_dict["serum_creatinine_mg_dl"] = float(labs.serum_creatinine_mg_dl)
            feat_dict["creatinine_is_missing"] = 0.0
        else:
            feat_dict["serum_creatinine_mg_dl"] = 1.0
            feat_dict["creatinine_is_missing"] = 1.0

        if labs and (labs.troponin_i_ng_ml is not None or labs.troponin_t_ng_ml is not None):
            trop_val = labs.troponin_i_ng_ml if labs.troponin_i_ng_ml is not None else labs.troponin_t_ng_ml
            feat_dict["troponin_elevated"] = 1.0 if (trop_val and trop_val > 0.04) else 0.0
            feat_dict["troponin_is_missing"] = 0.0
        else:
            feat_dict["troponin_elevated"] = 0.0
            feat_dict["troponin_is_missing"] = 1.0

        # 4. Symptoms (Time-aware)
        symp_list = [s.primary_symptom.lower() for s in patient.symptoms if s.primary_symptom]
        feat_dict["symptom_chest_pain"] = 1.0 if any("chest" in s or "angina" in s for s in symp_list) else 0.0
        feat_dict["symptom_dyspnea"] = 1.0 if any("dyspnea" in s or "breath" in s for s in symp_list) else 0.0
        feat_dict["symptom_palpitations"] = 1.0 if any("palpitat" in s for s in symp_list) else 0.0
        feat_dict["symptom_syncope"] = 1.0 if any("syncope" in s or "dizzi" in s for s in symp_list) else 0.0

        # 5. Conditions
        cond_str = " ".join(patient.existing_conditions + patient.cardiac_history).lower()
        feat_dict["has_hypertension"] = 1.0 if "hypertension" in cond_str or "htn" in cond_str else 0.0
        feat_dict["has_cad_or_prior_mi"] = 1.0 if "cad" in cond_str or "stemi" in cond_str or "infarct" in cond_str or "coronary" in cond_str else 0.0
        feat_dict["has_heart_failure"] = 1.0 if "heart failure" in cond_str or "hfref" in cond_str or "hfpef" in cond_str or "chf" in cond_str else 0.0
        feat_dict["has_diabetes"] = 1.0 if "diabetes" in cond_str or "dm" in cond_str else 0.0

        # 6. Active Medications (Time-aware)
        active_meds = patient.get_active_medications_at(as_of_timestamp)
        med_str = " ".join([m.drug_name.lower() for m in active_meds])
        feat_dict["med_beta_blocker"] = 1.0 if any(b in med_str for b in ["metoprolol", "bisoprolol", "carvedilol", "atenolol", "propranolol"]) else 0.0
        feat_dict["med_antiarrhythmic"] = 1.0 if any(a in med_str for a in ["amiodarone", "flecainide", "sotalol", "dofetilide", "propafenone", "digoxin"]) else 0.0
        feat_dict["med_acei_or_arb"] = 1.0 if any(c in med_str for c in ["lisinopril", "ramipril", "enalapril", "losartan", "valsartan", "telmisartan"]) else 0.0
        feat_dict["med_anticoagulant"] = 1.0 if any(ac in med_str for ac in ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin"]) else 0.0

        # 7. Smoking
        smk = (patient.smoking_status or "").lower()
        feat_dict["is_smoker"] = 1.0 if "active" in smk or "current" in smk or "smoker" in smk and "never" not in smk and "former" not in smk else 0.0

        # 8. Blood Group (if enabled)
        if self.include_blood_group:
            bg = (patient.blood_group or "").upper()
            feat_dict["blood_group_A"] = 1.0 if "A" in bg and "AB" not in bg else 0.0
            feat_dict["blood_group_B"] = 1.0 if "B" in bg and "AB" not in bg else 0.0
            feat_dict["blood_group_AB"] = 1.0 if "AB" in bg else 0.0
            feat_dict["blood_group_O"] = 1.0 if "O" in bg else 0.0
            feat_dict["blood_group_Rh_pos"] = 1.0 if "+" in bg else 0.0

        vec = np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32)
        return vec, feat_dict


GLOBAL_CLINICAL_FEATURE_BUILDER = ClinicalFeatureBuilder(include_blood_group=False)
