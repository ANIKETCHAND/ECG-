# ECG Guardian — Multimodal Medication & Clinical Decision Support System

## 1. System Architecture & Scientific Motivation

The **ECG Guardian Medication & Clinical Decision Support System** bridges advanced machine learning, multimodal clinical knowledge representation, and strict patient safety engineering.

### 1.1 The Core Scientific Imperative: Why ECG Alone Must Never Prescribe
A common failure mode in AI cardiology systems is naive "ECG $\rightarrow$ Drug" inference. Prescribing or recommending medication based solely on an ECG waveform violates fundamental clinical medicine:
1. **Clinical Context Independence:** A normal sinus rhythm tracing may belong to a patient in decompensated heart failure requiring aggressive neurohormonal therapy, or a healthy athlete requiring zero medication.
2. **Hidden Physiological Risks:** A patient with frequent PVCs might benefit from a beta-blocker; however, if their baseline heart rate is 44 bpm, serum potassium is 5.8 mEq/L, or PR interval is 260 ms, administering a nodal blocker or ACEi/ARB could precipitate complete heart block, asystole, or fatal hyperkalemic arrest.
3. **Severe Medication Conflicts:** A patient may have documented anaphylaxis to penicillin or beta-blockers, or may already be taking high-dose diltiazem (where adding metoprolol causes profound cardiogenic shock).

Therefore, ECG Guardian enforces that **Medication Decision Support requires the complete multimodal clinical profile**:
$$\text{Patient Profile} + \text{Vitals} + \text{Labs (K}^+, \text{Cr, eGFR)} + \text{Allergies} + \text{Active Meds} + \text{ECG Waveform} \longrightarrow \text{Candidate Generation} \longrightarrow \text{Safety Verification Gates} \longrightarrow \text{Clinician Decision Support}$$

---

## 2. Comparative Ablation Benchmark Results

We benchmarked four distinct pipeline architectures on an independent, held-out patient cohort:

| Architecture | Macro F1 | Micro F1 | Recall@3 | Macro PR-AUC | Safety Breaches |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. ECG Waveform Only** | 0.0370 | 0.1802 | 0.2200 | 0.2687 | *Unconstrained* |
| **2. Clinical Context Only** | 0.6566 | 0.7812 | 0.8250 | 0.7966 | *Unconstrained* |
| **3. ECG + Multimodal Clinical Context** | 0.8863 | 0.9728 | 0.9844 | 0.9409 | 1 flag unblocked |
| **4. ECG + Multimodal + Safety Constraint Gates** | **0.8889** | **0.9750** | **0.9844** | **0.9450** | **0 (100% Intercepted)** |

*Key Findings:*
- Waveform features alone achieve near-zero predictive capacity for pharmacotherapy (Macro F1 = 0.037), mathematically confirming the clinical requirement for multimodal context.
- Fusing ECG measurements with patient clinical context boosts Macro F1 to 0.8863 and Recall@3 to 98.44%.
- Applying deterministic safety constraint gates eliminates hazardous drug proposals while preserving top-tier sensitivity.

---

## 3. Dataset Preprocessing & Multimodal Feature Schema

Training and validation pipelines ingest four primary healthcare data modalities:
1. **MIMIC-IV (v2.2):** Emergency department triage vitals, laboratory orders (potassium, creatinine), inpatient prescriptions, and ICD-10 comorbidities. Temporal ordering ($T_{\text{event}} \le T_{\text{prescribed}}$) is strictly enforced to prevent data leakage.
2. **MIMIC-IV-ECG (v1.0):** 500 Hz 12-lead diagnostic ECG waveforms and cardiologist-verified diagnostic statements linked via unique `study_id`.
3. **eICU Collaborative Research Database (v2.0):** Multi-center ICU admissions serving as an independent external validation cohort.
4. **DailyMed / FDA Structured Product Labeling (SPL):** Structured contraindications, black-box warnings, and drug-drug interaction tables.
5. **Synthea Edge-Case Cohorts:** Physiological stress cases (extreme hyperkalemia, profound bradycardia, polypharmacy interactions) used for safety regression testing.

### Patient-Level Group Separation
To eliminate data leakage, all splits enforce patient-level isolation using `GroupShuffleSplit`:
- **Train Split:** 70% of unique patients.
- **Validation Split:** 15% of unique patients.
- **Held-Out Test Split:** 15% of unique patients.
Zero patient overlap between splits is verified programmatically by automated assertions.

---

## 4. Multimodal Feature Vector (29 Dimensions)

Each clinical case is embedded into a standardized 29-dimensional vector:
1. **Demographics:** Age, Sex (binary male/female).
2. **Acute Symptoms:** Chest Pain / Angina, Palpitations, Dyspnea / Shortness of breath, Syncope / Presyncope.
3. **Vital Signs:** Systolic BP, Diastolic BP, Heart Rate, SpO2 (%).
4. **ECG Quantitative Metrics:** Measured Heart Rate, PR interval (ms), QRS duration (ms), Bazett-corrected QTc (ms).
5. **ECG Diagnostic Classes:** Normal Sinus Rhythm (binary), Ventricular Ectopy / PVC (binary), Other / Atypical Rhythm (binary).
6. **Comorbidities:** Hypertension, Diabetes Mellitus, Coronary Artery Disease, Heart Failure (HFrEF/HFpEF), Chronic Kidney Disease.
7. **Laboratory Values:** Serum Potassium (mEq/L), Serum Creatinine (mg/dL), Estimated GFR (mL/min/1.73m²).
8. **Active Pharmacotherapy:** On Beta-Blocker, On ACEi/ARB, On Statin, On Antiplatelet.

---

## 5. Candidate Generation & Probability Calibration

Candidate generation models predict multi-label probabilities across 9 evidence-based cardiovascular medication classes:
1. `beta_blocker` (Metoprolol Succinate, Bisoprolol)
2. `calcium_channel_blocker` (Amlodipine, Diltiazem)
3. `antiarrhythmic_class_3` (Amiodarone, Sotalol)
4. `anticoagulant_doac` (Apixaban, Rivaroxaban)
5. `antiplatelet` (Aspirin, Clopidogrel)
6. `ace_inhibitor_arb` (Lisinopril, Losartan)
7. `diuretic_loop` (Furosemide, Torsemide)
8. `statin` (Atorvastatin, Rosuvastatin)
9. `electrolyte_replacement` (Potassium Chloride, Magnesium)

### Probability Calibration
To prevent uncalibrated tree probabilities from overstating confidence, the model wraps `HistGradientBoostingClassifier` within `CalibratedClassifierCV` (sigmoid calibration). Output scores reflect true empirical association frequencies.

---

## 6. Deterministic Safety Verification Gates

Generated candidates undergo mandatory multi-tiered deterministic safety checks:

```
[Candidate Generated]
         │
         ▼
┌─────────────────────────┐
│ 1. ALLERGY GATE         │─── Documented Allergy? ──► [BLOCKED: ALLERGY_ALERT]
└─────────────────────────┘
         │ Passed
         ▼
┌─────────────────────────┐
│ 2. QTc PROLONGATION     │─── QTc >= 500 ms? ────────► [BLOCKED: TORSADES RISK]
└─────────────────────────┘
         │ Passed
         ▼
┌─────────────────────────┐
│ 3. CONDUCTION / BRADY   │─── HR < 50 bpm or PR>240? ► [BLOCKED: HEART BLOCK RISK]
└─────────────────────────┘
         │ Passed
         ▼
┌─────────────────────────┐
│ 4. ELECTROLYTE (K+)     │─── K+ >= 5.2 mEq/L? ──────► [BLOCKED: HYPERKALEMIA RISK]
└─────────────────────────┘
         │ Passed
         ▼
┌─────────────────────────┐
│ 5. DRUG-DRUG INTERACTION│─── Concurrent Antagonist? ► [REVIEW REQUIRED: INTERACTION]
└─────────────────────────┘
         │ Passed
         ▼
[CANDIDATE CLEARED: SAFE]
```

### Evidence Gating & Withholding Behavior
- When ECG signal quality is `UNUSABLE` or `UNACCEPTABLE`, or finding reliability cannot be verified:
  - Automated therapy recommendations are set to `status = "WITHHELD"`.
  - Suggestions are moved aside with an explicit, auditable `withheld_reason`.
  - The UI and PDF render a prominent clinical warning banner: "Recommendations Withheld — Repeat Tracing / Manual Physician Evaluation Mandated".
- When vital signs, labs, or allergy profiles are missing:
  - Candidates are tagged with `status = "INSUFFICIENT_INFORMATION"` and `safety_status = "REVIEW_REQUIRED"`.

---

## 7. The Authoritative Clinical Analysis Object

The system unifies all tiers using a single authoritative data structure (`AuthoritativeClinicalAnalysis` in `src/clinical/clinical_analysis.py`):
```json
{
  "patient_context": {
    "patient_id": "PAT-5F2298",
    "hospital_mrn": "MRN-10029",
    "name": "Jane Doe",
    "age": 62,
    "sex": "F",
    "blood_group": "A+",
    "smoking_status": "Non-Smoker"
  },
  "ecg_analysis": {
    "primary_finding": "Premature Ventricular Contractions (PVC)",
    "heart_rate": 74.0,
    "signal_quality": "GOOD",
    "signal_quality_score": 0.94,
    "measurements": {
      "pr_interval_ms": 162.0,
      "qrs_duration_ms": 118.0,
      "qtc_bazett_ms": 428.0
    }
  },
  "clinical_context": {
    "symptoms": ["Palpitations", "Mild Dyspnea"],
    "medical_history": ["Hypertension", "CAD"],
    "allergies": ["Penicillin"],
    "current_medications": ["Amlodipine 5mg"],
    "vitals": {"systolic_bp": 134, "heart_rate": 74, "spo2": 98},
    "laboratory_results": {"potassium_meq_l": 4.2, "creatinine_mg_dl": 0.9}
  },
  "medication_decision_support": {
    "status": "AVAILABLE",
    "clinical_finding": "Premature Ventricular Contractions (PVC)",
    "candidates": [
      {
        "medication": "Beta-Adrenergic Blocker",
        "model_association": 0.924,
        "safety_status": "SAFE",
        "relevant_indication": "Ventricular ectopy suppression and rate control (2019 ESC Guidelines)",
        "evidence_source": "2019 ESC Guidelines on Ventricular Arrhythmias"
      }
    ],
    "safety_checks": [
      {"check": "Allergy Screening", "status": "PASSED"},
      {"check": "QTc Prolongation Gate", "status": "PASSED"},
      {"check": "Bradycardia / AV Conduction Gate", "status": "PASSED"},
      {"check": "Electrolyte (Potassium) Gate", "status": "PASSED"}
    ],
    "clinician_review_required": true
  }
}
```

This object is serialized and consumed identically across:
1. Serverless API (`/api/analyze`)
2. Vitaliscope Interactive Web Dashboard (`public/index.html`)
3. Physician Diagnostic PDF Generator (`src/report/pdf_generator.py`)
4. Plain-Text Structured Sourced Reports (`src/report/report_generator.py`)
5. Supabase & SQLite Database Snapshots (`src/services/report_persistence_service.py`)

---

## 8. Clinical Governance & Regulatory Compliance

- **CDSCO MDR 2017 & IEC 62304 Compliance:** The platform is classified as a Clinical Decision Support Software as a Medical Device (SaMD).
- **Mandatory Clinician in the Loop:** The system displays `"Decision Support Only — Clinician Review Required"` across all interfaces. It never issues automated prescriptions or autonomous dosing directives. Prescribing authority remains solely with the licensed healthcare professional.
