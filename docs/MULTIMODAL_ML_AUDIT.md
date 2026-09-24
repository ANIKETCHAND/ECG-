# Phase 0: Multimodal Machine Learning & Clinical System Audit

**Document Identifier:** `DOC-MM-AUDIT-001`  
**System Name:** ECG GUARDIAN — Patient-Aware Multimodal ECG Model & Hospital Report System  
**Audit Date:** September 2026  
**Auditor:** Antigravity Autonomous Bioengineering & AI Specialist Agent  
**Compliance & Governance References:**  
- **IEC 62304:2006+A1:2015**: Medical device software — Software life cycle processes (Class B SaMD)  
- **ISO 14971:2019**: Application of risk management to medical devices  
- **CDSCO Medical Devices Rules (MDR) 2017**: Class B Medical Device Software  
- **AAMI/ANSI EC57:2012**: Testing and reporting performance results of cardiac rhythm and ST-segment algorithms  
- **FDA Guidance (2023)**: Marketing Submission Recommendations for AI/ML-Enabled Device Software Functions  
- **PhysioNet Credentialed Data Use Agreement 1.5.0** (MIMIC-IV & MIMIC-IV-ECG Governance)  

---

## 1. Executive Summary & Purpose

This audit serves as the formal baseline assessment for transitioning **ECG GUARDIAN** from a standalone ECG waveform beat classifier into a **Patient-Aware Multimodal Clinical Decision-Support Platform**.

The target vision integrates patient demographics, blood group, vital signs, laboratory findings, longitudinal history, medication regimens, and clinician review into an evidence-based clinical report. In strict accordance with **Rule 1 (No Fabrication)** and **Rule 2 (No Black-Box Monolith)**, the multimodal redesign must decouple signal analysis from clinical context, enforce temporal look-back safety, eliminate data leakage, and maintain source-attributed transparency.

### Core Audit Findings at a Glance
1. **Existing Test Suite**: **172 automated unit and integration tests are 100% passing** across authentication, database, signal quality gating, feature extraction, ML inference, and report generation.
2. **Current Model**: Random Forest (100 estimators) trained on 4 records ($10,152$ heartbeats) and tested on 3 patient records ($6,807$ heartbeats) from the MIT-BIH Arrhythmia Database. On the held-out test evaluation set, it achieves $71.67\%$ accuracy, $70.51\%$ weighted F1-score, and $82.1\%$ Normal F1, but suffers a **$0.0\%$ recall failure on Class "Other"** (supraventricular ectopies).
3. **Current Input Modality**: Strictly 1D single-lead ECG signal (Modified Lead II / MLII). Zero demographic, vital, lab, or history features are currently utilized in machine learning.
4. **Current Database & Clinical Schema**: SQLite database (`data/hospital_clinical.db`) supports basic patient demographics, blood group, and free-text clinical history fields. However, it completely lacks structured observation models, timestamps for clinical events, vital sign tracking, and lab panels.
5. **Current Report Generator**: Generates comprehensive PDF reports with waveform strips, cardiac intervals, AI findings, and physician sign-offs. However, patient fields (blood group, conditions, medications, allergies) are not dynamically mapped into the final document, and statements lack discrete source-attribution metadata.

---

## 2. Current System Architecture

The existing repository is organized into distinct submodules under `src/`, with dual frontend interfaces and dual backend services:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   USER INTERFACES                                      │
│  • Streamlit Clinical Dashboard (app.py: 1,359 LOC): Localhost :8501                   │
│    - Role-based portal (Admin, Cardiologist, Physician, Technician, Nurse)             │
│    - Interactive waveform plots (Plotly), beat inspection, PDF report downloads        │
│  • Web Application Prototype (public/index.html: 490 LOC): Tailwind CSS + Plotly       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                   BACKEND SERVICES                                     │
│  • Serverless / REST API (api/index.py: 221 LOC): FastAPI on Localhost :8000           │
│    - /api/health, /api/sample, /api/analyze, /api/review                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                               PERSISTENCE & SECURITY                                   │
│  • Database Layer (src/database/db_manager.py): SQLite (data/hospital_clinical.db)     │
│    - Multi-tenant architecture (Hospitals, Doctors, Patients, Records, Reviews, Reports)│
│  • Authentication (src/auth/auth_manager.py): PBKDF2-HMAC-SHA256 (100k rounds) + RBAC  │
│  • Tamper-Evident Audit (src/audit/audit_logger.py): Cryptographic SHA-256 hash chains  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                             SIGNAL PROCESSING & ML PIPELINE                            │
│  • Ingestion & Detection (src/ecg_input/): PDF, scanned image, digital (CSV/TXT/NPY)   │
│  • Quality Gatekeeper (src/quality/quality_gate.py): SNR, baseline drift, powerline    │
│  • Preprocessing (src/preprocessing.py): Butterworth 0.5–40 Hz bandpass, median filter │
│  • Peak Detection & Segmentation (src/peak_detection.py, src/segmentation.py):         │
│    Adaptive refractory SciPy peak finding; beat windows: [-0.2s, +0.4s]                │
│  • Feature Extraction (src/feature_extraction.py): 28 handcrafted features             │
│  • Inference (src/prediction.py, src/ml/inference/): Scikit-learn Random Forest        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                           CLINICAL INTELLIGENCE & REPORTING                            │
│  • Clinical Decision Support (src/clinical/): Evidence-backed recommendations (ACC/AHA)│
│  • Medication Safety (src/medications/): 10 cardiovascular drugs, interaction checker │
│  • Longitudinal & Machine Comparison (src/longitudinal/, src/comparison/)              │
│  • Report Generation (src/report/): Structured dict, JSON/TXT export, ReportLab PDF    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Current Datasets & Local Data Inventory

### 3.1 Local Data Files
- `data/processed/train_dataset.csv` ($1.11$ MB): $10,152$ annotated beats from MIT-BIH records `100`, `106`, `200`, `213`.
- `data/processed/test_dataset.csv` ($333$ KB): $6,807$ annotated beats (evaluation subset: $600$ beats) from MIT-BIH records `101`, `119`, `208`.
- `data/processed/dataset_info.json` & `split_info.json`: Detailed split metadata and class distributions.
- `data/raw/`: Contains `.gitkeep`. The raw WFDB binary files (`.hea`, `.dat`, `.atr`) are downloaded on demand and not permanently stored in the repository.

### 3.2 Registered Datasets in `src/datasets/registry.py`
The repository defines metadata, licensing, and task suitability for 12 public and credentialed datasets:
1. **MIT-BIH Arrhythmia Database (`mit_bih_arrhythmia`)**: Open PhysioNet, 48 records, 360 Hz, MLII/V1.
2. **MIT-BIH Supraventricular Database (`mit_bih_svdb`)**: Open PhysioNet, 78 records, 128 Hz, SVEB benchmark.
3. **MIT-BIH Atrial Fibrillation Database (`mit_bih_afdb`)**: Open PhysioNet, 25 records, 250 Hz, rhythm AF episodes.
4. **MIT-BIH Normal Sinus Rhythm Database (`mit_bih_nsrdb`)**: Open PhysioNet, 18 records, 128 Hz.
5. **MIT-BIH Long-Term Database (`mit_bih_ltdb`)**: Open PhysioNet, 7 24-hr records, 128 Hz.
6. **PTB-XL Diagnostic Database (`ptb_xl`)**: Open PhysioNet (CC-BY 4.0), 21,837 12-lead ECGs, 100/500 Hz.
7. **PTB Diagnostic Database (`ptbdb`)**: Open PhysioNet, 549 records, 15 channels (12 leads + Frank XYZ), 1000 Hz.
8. **PTB-XL+ Companion Database (`ptb_xl_plus`)**: Open PhysioNet (CC-BY 4.0), fiducial annotations.
9. **MIT-BIH ST Change Database (`mit_bih_stdb`)**: Open PhysioNet, 28 records, 360 Hz.
10. **European ST-T Database (`european_st_t`)**: Open PhysioNet, 90 records, 250 Hz.
11. **MIMIC-IV-ECG (`mimic_iv_ecg`)**: PhysioNet Credentialed 1.5.0, ~800,000 12-lead ECGs matched to EHR.
12. **MIMIC-IV Clinical Database (`mimic_iv_clinical`)**: PhysioNet Credentialed 1.5.0, ICU/EMR longitudinal records.

---

## 4. Current Labels & Class Distributions

The active model classifies heartbeats into **3 classes** defined in `src/label_mapping.py`:

| Canonical Class | Source AAMI / MIT-BIH Symbols | Clinical Significance | Train Beats ($N=10,152$) | Test Beats ($N=6,807$) | Test Proportion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Normal`** | `N`, `L`, `R`, `e`, `j` | Normal Sinus Rhythm, Bundle Branch Block | $8,130$ ($80.08\%$) | $4,989$ ($73.29\%$) | $73.29\%$ |
| **`PVC`** | `V`, `W`, `F` | Premature Ventricular Contraction, Fusion | $1,931$ ($19.02\%$) | $1,809$ ($26.58\%$) | $26.58\%$ |
| **`Other`** | `A`, `a`, `J`, `S`, `s`, `Q`, `P`, `!`, `+`, `~` | Supraventricular Ectopy (SVEB), Paced, Atypical | $91$ ($0.90\%$) | $9$ ($0.13\%$) | $0.13\%$ |

### Deficiencies in Label Formulation
- Severe class imbalance: Class `Other` comprises only $0.9\%$ of training data and $0.13\%$ of test data.
- Task conflation: Supraventricular ectopy, atrial premature contractions, and paced beats are grouped into an unlearnable catch-all bucket.
- Single-lead limitation: Inability to evaluate chamber-specific pathology or 12-lead ischemic statements (e.g., STEMI, anterior MI, LVH).

---

## 5. Current Feature Extraction Pipeline

The pipeline extracts **28 handcrafted features** per individual cardiac cycle window ($-0.2\text{s}$ pre-R to $+0.4\text{s}$ post-R):

### 5.1 The 28 Feature Dimensions
1. **Time-Domain Statistical (12)**: `mean`, `std`, `min`, `max`, `range`, `median`, `energy`, `rms`, `mav` (mean absolute value), `snr`, `zero_crossing_rate`, `autocorr_first_peak`.
2. **Morphological Waveform (6)**: `r_peak_amplitude`, `p_wave_amplitude`, `t_wave_amplitude`, `peak_to_peak_amplitude`, `max_slope`, `qrs_width_samples`.
3. **Spectral / Frequency-Domain (7)**: `total_power`, `lf_power` ($0.5$–$1.0\text{ Hz}$), `hf_power` ($1.0$–$10.0\text{ Hz}$), `vhf_power` ($>10\text{ Hz}$), `dominant_frequency`, `max_power`, `spectral_entropy`.
4. **R-R Dynamics & Timing (3)**: `pre_rr` (preceding interval), `post_rr` (compensatory pause), `local_rr_ratio` ($\text{pre\_rr} / \text{mean\_rr}$).

### 5.2 Top Predictive Features by Gini Importance
1. `local_rr_ratio`: **21.69%** (Primary prematurity detector)
2. `pre_rr`: **13.39%** (Preceding coupling interval)
3. `autocorr_first_peak`: **12.26%** (Loss of periodic regularity)
4. `spectral_entropy`: **6.90%** (Frequency dispersion in aberrant conduction)
5. `max_power`: **6.30%** (Low-frequency power concentration)

---

## 6. Current Model & Validated Performance

### 6.1 Artifacts
- Primary Model: `models/classifier.pkl` (Scikit-Learn `RandomForestClassifier`, 100 trees, `max_depth=None`)
- Normalizer: `models/scaler.pkl` (`StandardScaler`, fitted exclusively on `X_train`)
- Baseline Model: `models/baseline_classifier.pkl` (`LogisticRegression`, `max_iter=1000`)
- Registry Metadata: `models/metadata.json`

### 6.2 Empirical Evaluation Results
Executed via `training/evaluate_model.py` on the held-out patient evaluation test set:

| Evaluation Metric | Random Forest (Primary) | Logistic Regression (Baseline) | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **71.67%** | 55.17% | RF demonstrates superior overall discernment |
| **Macro F1-Score** | **41.81%** | 41.90% | Depressed by complete Class 'Other' failure |
| **Weighted F1-Score** | **70.51%** | 60.26% | Acceptable on dominant clinical classes |
| **Macro Precision** | **40.04%** | 45.04% | Conservative positive predictive performance |
| **Macro Recall** | **44.16%** | 50.22% | Baseline catches more outliers at cost of precision |
| **Normal Beat F1** | **82.1%** (Prec: 82.2%, Rec: 82.0%) | 65.8% (Prec: 92.2%, Rec: 51.2%) | High reliability on normal sinus cycles |
| **PVC Beat F1** | **43.3%** (Prec: 38.0%, Rec: 50.5%) | 54.5% (Prec: 39.4%, Rec: 88.3%) | Moderate sensitivity to ventricular ectopy |
| **Class 'Other' F1** | **0.0%** (Prec: 0.0%, Rec: 0.0%) | 5.4% (Prec: 3.5%, Rec: 11.1%) | **Catastrophic zero-detection failure** |

---

## 7. Current Patient Database & Clinical Fields

### 7.1 Database Schema (`src/database/db_manager.py`)
The SQLite schema persists patient demographics in the `patients` table:
```sql
CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    hospital_mrn TEXT UNIQUE NOT NULL,
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
    name TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    contact TEXT,
    created_at TEXT NOT NULL,
    date_of_birth TEXT,
    emergency_contact TEXT,
    blood_group TEXT,
    known_allergies TEXT,
    existing_conditions TEXT,
    current_medications TEXT,
    previous_cardiac_history TEXT,
    family_history TEXT,
    smoking_status TEXT,
    other_relevant_history TEXT,
    past_medical_history TEXT,
    other_clinical_information TEXT,
    updated_at TEXT
);
```

### 7.2 Patient Field Audit & Clinical Limitations
1. **Unstructured Data**: `known_allergies`, `existing_conditions`, and `current_medications` are stored as loose free-text strings (e.g., `"Metoprolol 50mg, Aspirin"`), preventing programmatic querying and drug-drug safety parsing.
2. **Missing Vital Signs**: No columns or child tables exist for Heart Rate, Blood Pressure (Systolic/Diastolic), Oxygen Saturation ($SpO_2$), Body Temperature, or Respiratory Rate.
3. **Missing Laboratory Results**: No schema exists for Potassium, Magnesium, High-Sensitivity Troponin I/T, Serum Creatinine, eGFR, or BNP.
4. **Missing Symptom Tracking**: No fields record acute symptoms (e.g., Angina, Dyspnea, Syncope, Palpitations, Dizziness) or their onset durations.
5. **No Temporal Awareness**: Clinical fields lack observation timestamps (`recorded_at`), creating a major risk of look-ahead bias and temporal leakage.
6. **No Observation Provenance**: Data does not track who entered the value or whether it came from EHR integration, patient self-reporting, or triage intake.

---

## 8. Current Medication Safety Functionality

### 8.1 Active Implementation (`src/medications/`)
- **Knowledge Base (`medication_database.py`)**: Authoritative repository of 10 curated cardiovascular drugs: Metoprolol, Amiodarone, Digoxin, Lisinopril, Amlodipine, Furosemide, Atorvastatin, Warfarin, Apixaban, and Clopidogrel. Each contains FDA indications, contraindications, black-box warnings, and QT prolongation risks.
- **Safety Checker (`interaction_checker.py`)**:
  - Scans pairwise drug-drug interactions.
  - Cross-references known conditions (e.g., Beta-blockers contraindicated in severe bradycardia $<45\text{ bpm}$ or 2nd/3rd degree AV block).
  - Checks drug-allergy conflicts.
  - Triggers `"INSUFFICIENT CLINICAL CONTEXT"` if patient record is unlinked.

### 8.2 Deficiencies for Multimodal System
- **No Dosage or Frequency Checking**: Cannot verify whether a dosage is toxic or subtherapeutic.
- **No Renal/Hepatic Adjustment Logic**: Does not compute dose reductions based on lab values (eGFR/creatinine clearance).
- **No Temporal Relevance**: Cannot determine if a drug was actively taken at the exact timestamp of the ECG recording versus discontinued weeks prior.
- **String Token Matching**: Free-text string parsing easily misses variations, brand names, or misspellings.

---

## 9. Current Report Generator & Presentation Structure

### 9.1 Report Generator (`src/report/report_generator.py`)
Compiles a consolidated dictionary with the following sections:
1. `hospital_info`: Facility name, department, ID.
2. `patient_info`: Extracted from file metadata (`patient_name`, `patient_age`, `patient_sex`, `recording_date`).
3. `input_info`: Modality, sampling rate, lead, duration.
4. `signal_quality`: Category, score, SNR (dB), artifact warnings.
5. `cardiac_parameters`: HR, Mean RR, PR, QRS, QT, QTc, axes, machine interpretation.
6. `ai_analysis`: Primary pattern, class probabilities, beat breakdown.
7. `ai_evidence`: Aberrant beat details, coupling intervals, compensatory pauses.
8. `machine_comparison`: Concordance between machine interpretation and AI model.
9. `longitudinal_comparison`: Interval drift from prior ECGs.
10. `clinical_decision_support`: Urgency, triage guidance, guidelines.
11. `medication_safety`: Active alerts, contraindications, drug interactions.
12. `clinician_review`: Attending physician name, registration number, interpretation, seal.

### 9.2 PDF Generation (`src/report/pdf_generator.py`)
Generates publication-quality doctor-facing and patient-facing PDFs using ReportLab.

### 9.3 Deficiencies Compared to Reference Standard
- **Disconnected Patient Demographics**: While `PatientRecord` contains `blood_group`, `smoking_status`, `known_allergies`, `existing_conditions`, and `current_medications`, `generate_structured_report()` does not pull these from the database.
- **Missing Vital Signs & Labs**: No tables exist in the report for vitals (BP, SpO2, Temp) or lab panels (Electrolytes, Cardiac Biomarkers).
- **Lacks Explicit Source Attribution**: Individual parameters do not display provenance tags (e.g., `[ECG-DERIVED]`, `[RECORDED HISTORY]`, `[AI FINDING]`, `[CLINICIAN-ENTERED]`).

---

## 10. Data Leakage Risks & Prevention Protocols

| Leakage Dimension | Risk Severity | Mechanism of Risk in Multimodal ECG | Mandatory Safeguard in Redesign |
| :--- | :--- | :--- | :--- |
| **Patient Overlap** | **CRITICAL** | If multiple ECGs from the same patient exist across training and test splits, the model memorizes patient-specific morphological signatures. | **Strict Patient-Level Hashing**: Splits must partition by `patient_id`, guaranteeing $\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$. |
| **Temporal Look-Ahead** | **CRITICAL** | Using discharge diagnoses (assigned days after admission) or post-ECG lab/medication orders to predict current ECG rhythm creates false superhuman accuracy. | **Point-in-Time Constraint**: For any ECG at timestamp $T_{\text{ecg}}$, feature builders may ONLY ingest clinical events where $T_{\text{event}} \le T_{\text{ecg}}$. |
| **Preprocessing Leakage** | **HIGH** | Fitting normalizers (e.g., `StandardScaler`, imputation models) across the entire combined dataset leaks test set statistics into training. | **Pipeline Encapsulation**: Scalers, imputers, and encoders must be fitted strictly inside training folds. |
| **Shortcut / Confounder** | **HIGH** | If the multimodal model is trained on a hospital cohort where older patients predominantly have arrhythmias, it may predict arrhythmias from `age` alone without inspecting the ECG. | **Decoupled 3-Model Architecture**: Evaluate ECG-only, Clinical-only, and Multimodal models side-by-side to detect shortcut learning. |

---

## 11. Multimodal Dataset Discovery & Feasibility Assessment

To support multimodal training, public and credentialed datasets were evaluated against the required clinical variables:

| Dataset | ECG Waveforms | Patient ID | Age & Sex | Blood Group | Symptoms | Existing Diseases | Medications | Vital Signs | Labs | Temporal Links |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MIT-BIH Arrhythmia** | Yes (2 leads) | Yes (48 recs) | Header only | **NO** | **NO** | **NO** | Partial comment | **NO** | **NO** | Static only |
| **PTB-XL** | Yes (12 leads) | Yes (18,885) | Yes (in CSV) | **NO** | **NO** | SCP-ECG only | **NO** | **NO** | **NO** | Recording date |
| **MIMIC-IV-ECG + MIMIC-IV** | Yes (12 leads, ~800k) | Yes (~160k) | Yes | **NO** | Unstructured | Yes (ICD-9/10) | Yes (Prescriptions) | Yes (chartevents) | Yes (labevents) | **Full Exact Timestamps** |

### Key Scientific Finding on Blood Group
**No major public ECG benchmark (MIT-BIH, PTB-XL, or MIMIC-IV) records patient blood group.**  
Under **Rule 1 (Zero Fabrication)** and **Rule 28 (Feature Governance)**:
- Blood group must be stored in the hospital clinical record and displayed on the clinical report when provided by the hospital or patient.
- Blood group must **NOT** be fabricated into public ECG training sets.
- Blood group must **NOT** be used as a predictive feature in ML models unless a valid, empirically audited clinical dataset containing blood group is legitimately connected.

---

## 12. Gap Analysis & Roadmap for Subsequent Phases

```text
┌─────────────────────────────────┬──────────────────────────────────┬─────────────────────────────────┐
│ System Capability               │ Current Baseline State           │ Target Redesign State           │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ Clinical Data Models            │ Flat dataclass & basic SQLite    │ Full structured models: Vitals, │
│                                 │ table with free-text fields      │ Labs, Symptoms, History, Timing │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ Temporal Governance             │ No event timestamps              │ Point-in-time filtering:        │
│                                 │ (only patient created_at)        │ T_event <= T_ecg enforced       │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ Multimodal Model Architecture   │ Single Random Forest on 28 ECG   │ Decoupled architecture:         │
│                                 │ waveform features                │ Model A (ECG) + Model B (Clin)  │
│                                 │                                  │ -> Model C (Multimodal Fusion)  │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ Model Benchmarking              │ Single RF vs Logistic Baseline   │ 3-Model comparison (A, B, C)    │
│                                 │ on beat arrhythmia               │ with feature ablation matrix    │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ Medication Safety Engine        │ 10 drugs, pairwise check,        │ Renal/hepatic lab checks,       │
│                                 │ unlinked context warning         │ temporal active-regimen filter  │
├─────────────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ Report Integration & Provenance │ Report dictionary lacks clinical │ 12-section reference report     │
│                                 │ fields & source metadata tags    │ with explicit provenance tags   │
└─────────────────────────────────┴──────────────────────────────────┴─────────────────────────────────┘
```

---

## 13. Audit Sign-Off & Phase 0 Completion

- **Audit Status:** **COMPLETED**
- **Existing Tests Status:** **172/172 PASSED**
- **Existing Baseline Evaluation:** **VERIFIED**
- **Data Leakage Safeguards:** **DOCUMENTED**
- **Phase 0 Execution Gate:** **HALTED per Phase 30 Execution Directive.**
