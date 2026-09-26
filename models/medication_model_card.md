# Model Card: Multimodal Medication Decision Support Candidate Model

## Model Overview
- **Model Name:** ECG-Guardian-MedSupport-GBM
- **Version:** 1.0.0
- **Model Architecture:** Multimodal Calibrated Gradient Boosted Trees (`HistGradientBoostingClassifier` with Sigmoid Probability Calibration via `CalibratedClassifierCV`) wrapped in `MultiOutputClassifier`.
- **Primary Function:** Predict evidence-based medication candidate association probabilities for clinician decision support.
- **Safety Directive:** **NON-AUTONOMOUS**. The model outputs candidate associations for licensed healthcare professional review only. It does not prescribe, dose, or authorize medications.

## Intended Clinical Use
- Intended for qualified healthcare professionals in emergency, telemetry, and cardiology inpatient/outpatient settings.
- Input data combines patient demographics, vital signs, acute symptoms, laboratory values (potassium, creatinine, eGFR), current medications, and 12-lead / rhythm ECG findings.
- Provides ranked medication candidates coupled with deterministic safety contraindication evaluations (QTc prolongation, hyper/hypokalemia, renal dysfunction, bradycardia, drug-drug interactions, and documented allergies).

## Training & Validation Cohorts
- **Development Splits:** Strict patient-level non-overlapping splitting (70% train, 15% validation, 15% held-out test).
- **External & Safety Validation:**
  - MIMIC-IV Clinical Database (ED triage, medications, and laboratory values).
  - MIMIC-IV-ECG (500 Hz 12-lead ECG waveforms and diagnostic statements).
  - eICU Collaborative Research Database (independent ICU cohort for generalization checking).
  - Synthea Safety Cohorts (edge cases: extreme hyperkalemia, profound bradycardia, polypharmacy interactions).
  - DailyMed / FDA SPL (contraindication rules, black box warnings).

## Multimodal Features
1. **Demographics:** Age, Sex.
2. **Vital Signs:** Systolic BP, Diastolic BP, Heart Rate, SpO2.
3. **Symptoms:** Chest Pain, Palpitations, Dyspnea, Syncope.
4. **ECG Metrics:** Heart Rate, PR interval (ms), QRS duration (ms), QTc interval (ms), Arrhythmia class (Normal, PVC/Ventricular, Other/Atrial).
5. **Comorbidities:** Hypertension, Diabetes Mellitus, Coronary Artery Disease, Heart Failure, Chronic Kidney Disease.
6. **Laboratory Values:** Serum Potassium (mEq/L), Serum Creatinine (mg/dL), eGFR (mL/min/1.73m²).
7. **Active Medications:** Beta-Blockers, ACEi/ARBs, Statins, Antiplatelets.

## Target Medication Classes
1. `beta_blocker` (e.g., Metoprolol Tartrate / Succinate)
2. `calcium_channel_blocker` (e.g., Diltiazem, Amlodipine)
3. `antiarrhythmic_class_3` (e.g., Amiodarone)
4. `anticoagulant_doac` (e.g., Apixaban)
5. `antiplatelet` (e.g., Aspirin, Clopidogrel)
6. `ace_inhibitor_arb` (e.g., Lisinopril, Losartan)
7. `diuretic_loop` (e.g., Furosemide)
8. `statin` (e.g., Atorvastatin)
9. `electrolyte_replacement` (e.g., Potassium Chloride)

## Evaluation & Ablation Results
Comparative evaluation on held-out test cohort:
| Configuration | Macro F1 | Micro F1 | Recall@3 | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **ECG Only** | 0.0370 | 0.1802 | 0.2200 | 0.2687 |
| **Clinical Context Only** | 0.6566 | 0.7812 | 0.8250 | 0.7966 |
| **ECG + Clinical Context** | **0.8863** | **0.9728** | **0.9844** | **0.9409** |
| **ECG + Clin + Safety Constraints** | **0.8889** | **0.9750** | **0.9844** | **0.9450** |

*Key finding:* ECG alone is completely insufficient for prescribing decisions, confirming that comprehensive clinical context and safety constraints are strictly required.

## Safety & Governance Constraints
- If key clinical variables (vitals, labs, current medications) are absent, the system withholds unconstrained candidate suggestions and issues an explicit `INSUFFICIENT_INFORMATION` clinical alert.
- All candidate medications pass through a deterministic hard-contraindication safety gate:
  - If QTc > 500 ms: Class III antiarrhythmics are **BLOCKED**.
  - If HR < 55 bpm or PR > 240 ms: Beta-blockers and Non-DHP CCBs are **BLOCKED**.
  - If Serum K+ > 5.2 mEq/L: ACEi/ARBs and Potassium replacements are **BLOCKED**.
  - If documented drug allergy exists: The candidate is **BLOCKED**.
- Every output displays: "Decision Support Only — Clinician Review Required".
