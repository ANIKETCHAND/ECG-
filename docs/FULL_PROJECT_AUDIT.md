# ECG GUARDIAN — Comprehensive Phase 0 Repository Audit

**Audit Date:** September 24, 2026  
**Audited Systems:** Core Algorithm Pipeline, Ingestion Engine, ML Subsystem, Clinical Persistence, RBAC, Decision Support, Medication Safety, and User Interfaces.  
**Governing Standard:** ECG Guardian Multi-Dataset ML & Hospital Platform Master Specification.

---

## 1. Executive Summary

This document establishes the official **Phase 0 Baseline Audit** for transforming the existing ECG analysis codebase into **ECG GUARDIAN**: an enterprise hospital-grade ECG clinical decision-support platform with an autonomous multi-dataset machine learning intelligence architecture.

### Key Audit Findings:
1. **Signal Processing & Ingestion Strength:** The repository has an exceptionally well-engineered biomedical signal processing backbone: multi-modality file detection (PDF, images, CSV, TXT, NPY, WFDB, EDF), deterministic 2-stage median baseline removal, 0.5–40 Hz Butterworth bandpass filtering, adaptive R-peak detection with refractory enforcement, and quantitative signal quality gating (SNR, clipping, baseline drift, 50/60 Hz powerline, motion artifacts).
2. **Current Production ML Model:** An authentic Random Forest classifier (`ECG-RF-1.0.0`, 100 trees) trained on 28 morphological, spectral, and R-R interval features from the MIT-BIH Arrhythmia Database with zero data fabrication and strict record-level patient isolation (10,152 training beats across records 100, 106, 200, 213; 6,807 unseen testing beats across records 101, 119, 208).
3. **Current Test Status:** **167/167 automated pytest unit and integration tests pass** across all existing modules.
4. **Primary Architectural Deficiencies for Target Platform:**
   - **Multi-Tenant Isolation Missing:** No `hospitals` table; `hospital_id` is absent from `patients`, `ecg_records`, `analysis_results`, and `clinician_reviews`.
   - **Missing Roles:** RBAC lacks `HOSPITAL_ADMIN`, `NURSE`, `ECG_TECHNICIAN`, and `PATIENT` tenant-scoped roles.
   - **Single Model Scope:** While multi-task task definitions exist (`TASK_BEAT_ARRHYTHMIA`, `TASK_AF_DETECTION`, `TASK_12LEAD_DIAGNOSTIC`, `TASK_ST_ANALYSIS`), only the beat arrhythmia model is actively trained and registered; others are research placeholders.
   - **Missing Dual Reporting:** Current reporting emits a single consolidated research PDF; separate patient-friendly and doctor-technical reports do not exist.
   - **Medication Knowledge Base:** Currently contains a static 2-medication sample (Metoprolol, Amiodarone); lacks full cardiovascular formulary, structured contraindication registry, and external openFDA/DailyMed ingestion.

---

## 2. Existing Architecture Assessment

### 2.1 File System & Modules Layout
```text
ECG--main/
├── app.py                         # Streamlit interactive application (1607 lines)
├── api/                           # Vercel serverless FastAPI endpoints (index.py)
├── configs/                       # JSON configurations for preprocessing and label mappings
│   ├── label_mappings/            # mit_bih_arrhythmia.json, mit_bih_afdb.json, ptb_xl.json
│   └── preprocessing/             # beat_arrhythmia_v1.json, af_v1.json, ptbxl_12lead_v1.json
├── data/                          # Persistent storage & dataset staging
│   ├── hospital_clinical.db       # Primary SQLite clinical database
│   ├── audit_trail.db             # Append-only cryptographic audit ledger
│   ├── datasets/                  # Staging directory for external datasets (currently empty)
│   ├── processed/                 # Feature extraction & split metadata
│   ├── raw/                       # Staging directory for raw WFDB records
│   └── splits/                    # Deterministic split manifests (beat_arrhythmia/v1.json)
├── docs/                          # Architecture documentation & model cards
├── models/                        # Serialized model artifacts & registry
│   ├── production/                # classifier.pkl (ECG-RF-1.0.0), scaler.pkl, metadata.json
│   ├── candidate/                 # classifier.pkl (ECG-RF-2.0.0), deep_classifier.pkl, scaler.pkl
│   └── registry/                  # catalog.json (lifecycle status tracking)
├── public/                        # Static web landing assets (index.html)
├── regulatory/                    # Software lifecycle, risk management, intended use documentation
├── reports/                       # Generated audit reports and evaluation JSONs
├── sample_ecgs/                   # Ready-to-test clinical PDFs and digital CSVs
├── src/                           # Modular core business logic & algorithms
│   ├── alerts/                    # Clinical and technical alert dispatcher
│   ├── audit/                     # Cryptographic hash-chained audit logger
│   ├── auth/                      # PBKDF2 password hashing & RBAC permission checks
│   ├── clinical/                  # Clinical decision support & guideline engine
│   ├── comparison/                # AI vs. Machine printed comparison engine
│   ├── database/                  # SQLite connection manager & ORM-like entities
│   ├── datasets/                  # Dataset registry, downloader, inspector, patient index
│   ├── ecg_core/                  # Standardized immutable ECG recording dataclasses
│   ├── ecg_input/                 # Multi-modality file ingestion & image OCR
│   ├── evidence/                  # Beat-level visual and statistical evidence extraction
│   ├── longitudinal/              # Serial ECG comparison and trend tracker
│   ├── measurements/              # Deterministic clinical interval measurement engine
│   ├── medications/               # Medication database and interaction checker
│   ├── ml/                        # ML tasks, preprocessing pipelines, model registry
│   ├── quality/                   # Signal quality copilot and gatekeeper
│   ├── report/                    # Structured dict and PDF report generators
│   ├── review/                    # Clinician review, modification, and sealing
│   └── safety/                    # Signal quality gates & failure mode barriers
├── tests/                         # Pytest test suite (31 test files, 167 passing tests)
└── training/                      # Scripts for downloading, preparing, and training ML models
```

---

## 3. Existing ECG Pipeline

### 3.1 Ingestion Flow
1. **Modality Identification (`src/ecg_input/input_detector.py`):**
   - Automatically sniffs binary magic bytes and file extensions.
   - Categorizes into: `DIGITAL_SIGNAL` (CSV, TXT, NPY), `REPORT_PDF` (PDF), `REPORT_IMAGE` (JPG, PNG, TIFF, BMP), or `BENCHMARK_DEMO` (MIT-BIH records).
2. **Signal Extraction:**
   - **Digital:** Parsed via pandas/numpy with automatic delimiter sniffing (comma, tab, space, semicolon).
   - **PDF:** Selectable text extraction via PyPDF; isolates machine-printed measurements (Heart Rate, PR, QRS, QT/QTc, electrical axes).
   - **Image:** OpenCV grayscale thresholding, grid line detection/subtraction, column-wise center-of-mass trace extraction.
3. **Signal Quality Copilot & Gatekeeper (`src/quality/quality_gate.py` & `src/signal_quality.py`):**
   - Evaluates:
     - Signal-to-Noise Ratio (SNR dB).
     - Rail saturation & clipping percentage.
     - Baseline wander spectral power (< 0.5 Hz).
     - Powerline interference ratio (50/60 Hz notch band).
     - High-frequency muscle tremor and motion spikes.
   - Categorizes signal into: `GOOD`, `ACCEPTABLE`, `POOR`, or `UNUSABLE`.
   - **Safety Rule Enforced:** If `UNUSABLE`, AI inference is strictly blocked with explicit rejections recorded; no fake normal is ever produced.
4. **Preprocessing Pipeline (`src/preprocessing.py`):**
   - Baseline removal: 2-stage running median filter (window 1: 0.2s for P-QRS complexes, window 2: 0.6s for T waves).
   - Filtering: 0.5–40.0 Hz 4th-order zero-phase Butterworth bandpass filter.
   - Normalization: Z-score normalization (`(signal - mean) / std`).
5. **Peak Detection & Beat Windowing (`src/peak_detection.py`, `src/segmentation.py`):**
   - R-Peak detection using derivative, squaring, moving window integration, and adaptive thresholding.
   - Refractory period: $\ge 300\text{ ms}$ refractory window preventing double-counting of tall T-waves.
   - Segmentation window: $-0.2\text{ s}$ pre-R to $+0.4\text{ s}$ post-R (216 samples at 360 Hz).
6. **Feature Extraction (`src/feature_extraction.py`):**
   - 28 deterministic features computed per beat across 4 domains:
     - *Time-domain:* Mean, Standard Deviation, Min, Max, Range, Median, Energy, RMS, Mean Absolute Value (MAV), Zero-crossing rate, Max slope.
     - *Morphological:* R-peak amplitude, P-wave amplitude, T-wave amplitude, Peak-to-peak amplitude, QRS width in samples.
     - *Frequency-domain:* Total power, LF power (0.5–5 Hz), HF power (5–15 Hz), VHF power (15–40 Hz), Dominant frequency, Max power spectral density, Spectral entropy.
     - *R-R Dynamics:* Pre-RR interval, Post-RR interval, Local RR ratio (`pre_rr / post_rr`).
7. **Inference & Calibration:**
   - Standardization using pre-fitted `StandardScaler`.
   - Multi-class probability generation via Random Forest estimator tree voting.

---

## 4. Existing Datasets & Models

### 4.1 Datasets
- **MIT-BIH Arrhythmia Database (PhysioNet):**
  - Canonical records utilized: 100, 101, 106, 119, 200, 208, 213.
  - Total labeled cardiac beats: 16,959.
  - Class distribution:
    - Normal (`Normal`): 13,119 beats (77.36%)
    - Premature Ventricular Contractions (`PVC`): 3,740 beats (22.05%)
    - Other Ectopic (`Other`): 100 beats (0.59%)
- **Data Partitions:**
  - Training records (patient-level): 100, 106, 200, 213 (10,152 beats).
  - Validation record: 200.
  - Testing records (unseen patients): 101, 119, 208 (6,807 beats).
- **Physical Dataset Status:**
  - Raw PhysioNet records are not bundled in git; metadata is stored in `data/processed/dataset_info.json`.
  - Downloading subsystem in `src/datasets/downloader.py` can fetch from PhysioNet on-demand.

### 4.2 Models & Performance
| Model Identifier | Architecture | Test Accuracy | Weighted F1 | PVC Recall | PVC Specificity | AUROC | ECE | Lifecycle Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ECG-RF-1.0.0`** | Random Forest (100 trees) | 97.86% | 97.92% | 99.89% | 97.52% | 0.9969 | 0.1277 | **PRODUCTION** |
| **`ECG-RF-2.0.0-cand`** | Balanced Random Forest (150 trees) | 98.33% | 98.31% | 99.67% | 98.06% | 0.9973 | 0.0989 | **VALIDATED** |
| **`ECG-MLP-1.0.0-cand`** | Waveform Deep Representation MLP | 95.96% | 96.04% | 94.31% | 97.06% | 0.9924 | 0.0243 | **EXPERIMENTAL** |
| **`ECG-LR-1.0.0`** | Logistic Regression | 95.21% | 95.50% | 95.80% | 95.98% | 0.9922 | 0.0306 | **BASELINE** |

---

## 5. Existing Frontend & Backend

### 5.1 Frontend (`app.py` & `public/index.html`)
- **Technology:** Streamlit 1.42+ single-file application with Plotly visualization.
- **Current Views:**
  - Clinician Workspace: File uploader, MIT-BIH demo selector, signal preview, quality copilot cards, inference cards, clinician sign-off modal, download buttons.
  - Patient Health Portal: Basic demographic viewer and signed report download.
- **Limitations:**
  - Layout is monolithic rather than structured around hospital operational roles.
  - No hospital-level dashboard, no doctor management UI, no hospital onboarding flow.

### 5.2 Backend & API (`api/index.py` & `src/`)
- **Technology:** Python backend with FastAPI for serverless deployment and Streamlit execution server for local usage.
- **Endpoints in `api/index.py`:**
  - `GET /api/health`
  - `GET /api/sample`
  - `POST /api/analyze`
  - `POST /api/review`
- **Limitations:**
  - Endpoints do not validate `hospital_id` or authenticate multi-tenant bearer tokens.

---

## 6. Existing Database & Persistence

### 6.1 Database Engine (`src/database/db_manager.py`)
- SQLite3 database located at `data/hospital_clinical.db`.
- Configured with WAL (`PRAGMA journal_mode=WAL`) and foreign key enforcement.

### 6.2 Current Tables:
1. `patients`: `patient_id`, `hospital_mrn`, `name`, `age`, `sex`, `contact`, `created_at`, `date_of_birth`, `emergency_contact`, `blood_group`, `known_allergies`, `existing_conditions`, `current_medications`, `previous_cardiac_history`, `family_history`, `smoking_status`, `other_relevant_history`, `updated_at`.
2. `ecg_records`: `record_id`, `patient_id`, `device`, `sampling_rate`, `lead_names`, `duration_sec`, `file_hash`, `source_format`, `signal_quality`, `quality_score`, `uploaded_by`, `uploaded_at`, `raw_data_path`, `workflow_status`, `priority`, `assigned_doctor`.
3. `analysis_results`: `analysis_id`, `record_id`, `model_id`, `model_version`, `preprocessing_version`, `lead_analyzed`, `signal_quality`, `quality_score`, `prediction`, `probabilities_json`, `heart_rate_bpm`, `mean_rr_ms`, `detected_beats_count`, `warnings_json`, `limitations_json`, `processing_time_ms`, `analyzed_at`.
4. `clinician_reviews`: `review_id`, `analysis_id`, `record_id`, `clinician_id`, `clinician_name`, `clinician_role`, `registration_number`, `agreement_status`, `clinician_interpretation`, `clinical_notes`, `reviewed_at`.
5. `clinical_reports`: `report_id`, `record_id`, `analysis_id`, `review_id`, `report_type`, `status`, `file_path`, `report_sha256`, `generated_at`.

### 6.3 Audit Trail Ledger (`src/audit/audit_logger.py`)
- SQLite3 database at `data/audit_trail.db` with table `audit_trail`:
  - `sequence_id`, `timestamp`, `event_type`, `user_id`, `username`, `user_role`, `action`, `details_json`, `status`, `patient_id`, `record_id`, `ip_address`, `previous_hash`, `entry_hash`.
  - Cryptographic SHA-256 hash chaining prevents log alteration.

---

## 7. Existing Security & Access Control

- **Authentication (`src/auth/auth_manager.py`):**
  - Password hashing: PBKDF2-HMAC-SHA256 with 100,000 iterations and random salt.
  - Active roles: `ADMIN`, `DOCTOR`, `CARDIOLOGIST`, `TECHNICIAN`, `RESEARCHER`.
  - Permissions matrix enforced per action.
- **Deficiencies:**
  - User accounts are stored in an in-memory dictionary rather than persisted in the database.
  - No `hospital_id` attached to users or sessions.
  - Missing `HOSPITAL_ADMIN`, `NURSE`, `ECG_TECHNICIAN`, and `PATIENT` roles.

---

## 8. Deficiencies & Gap Analysis Against Target Platform

| Functional Domain | Current Repository State | Target ECG Guardian Platform State | Deficit Severity |
| :--- | :--- | :--- | :--- |
| **Multi-Tenancy** | Single-tenant database | Strict tenant isolation with `hospitals` table and `hospital_id` foreign key on all data | **CRITICAL** |
| **Hospital Onboarding** | None | Full hospital onboarding: hospital name, registration number, address, contact, and hospital admin setup | **CRITICAL** |
| **Doctor & Staff Management**| Hardcoded demo credentials in memory | Persistent database tables for doctors, technicians, nurses; verified registration numbers; deactivate/activate | **CRITICAL** |
| **Hospital Worklist** | Basic query view inside `app.py` | Dedicated hospital triage worklist with statuses: `UPLOADED`, `VALIDATING`, `QUALITY_CHECK`, `READY`, `ANALYZING`, `AI_COMPLETE`, `REVIEW_REQUIRED`, `DOCTOR_REVIEW`, `SIGNED`, `REJECTED`, `ERROR` | **HIGH** |
| **Multi-Model Intelligence** | Single production model (beat arrhythmia) | Multi-model system: Quality Gate, Beat Arrhythmia, AF Rhythm, 12-Lead Multi-Label, ST/T Analysis | **HIGH** |
| **Medication Knowledge** | Static 2-medication dictionary | Authoritative formulary, structured contraindications, allergy verification, drug-drug interactions, source tracking | **HIGH** |
| **Autonomous Prescribing Prevention**| Disclaimers present in code | Strict architectural boundary: medication considerations presented only as clinical safety considerations; doctor authorization mandatory | **MANDATORY SAFETY** |
| **Dual Reporting** | Single research PDF generated | Distinct **Patient Report** (simple language, safety, doctor plan) and **Doctor Report** (technical findings, AI vs. machine, measurements, signature) | **HIGH** |
| **Patient Portal** | Read-only profile viewer | Patient dashboard with medical history, allergies, conditions, signed reports, doctor instructions | **MEDIUM** |
| **Datasets Staged Locally** | Sample files and test fixtures only | Automated multi-dataset downloader (`training/download_datasets.py`) with licensing and checksum validation | **MEDIUM** |

---

## 9. Data Leakage & Security Risks

### 9.1 Data Leakage Risks
1. **Patient Leakage:** While the MIT-BIH split currently isolates patients 100, 106, 200, 213 (train) from 101, 119, 208 (test), ingesting new datasets (PTB-XL, MIT-BIH AF, SVDB) carries the risk of patient cross-contamination if recording IDs rather than patient IDs are used for splitting.
2. **Preprocessing Leakage:** Feature normalization parameters (e.g. `StandardScaler` mean and variance) must be computed exclusively on `X_train` and applied out-of-sample to `X_val` and `X_test`.

### 9.2 Security & Multi-Tenancy Risks
1. **Cross-Tenant Data Exposure:** Without `hospital_id` filtering at the SQL query level, queries could leak patient records across hospitals.
2. **In-Memory User State:** In-memory users in `auth_manager.py` reset upon server restart. User records, passwords, and sessions must reside in the database.
3. **Session Spoofing:** Switching active users via Streamlit sidebar selectbox must be replaced with authenticated session management.

---

## 10. Phased Migration Plan

```text
Phase 0: Full Repository Audit [COMPLETED]
   │
Phase 1: Target Architecture Specification (docs/TARGET_ARCHITECTURE.md)
   │
Phase 2: Multi-Tenant Hospital Relational Schema & Tenant Isolation
   │
Phase 3: Hospital Registration & Hospital Admin Onboarding
   │
Phase 4: Doctor, Technician, and Staff Management (RBAC)
   │
Phase 5: Patient Clinical Management (Allergies, Conditions, Medications)
   │
Phase 6: Patient Dashboard & Secure Health Portal
   │
Phase 7: Hospital ECG Worklist & Status State Machine
   │
Phase 8: Universal Multi-Modality ECG Ingestion Engine
   │
Phase 9: ECG Quality Copilot & Gatekeeper
   │
Phases 10–16: Dataset Management System, Licensing & Standard Format
   │
Phases 17–26: Multi-Task ML Intelligence, Training Automation & Model Registry
   │
Phases 27–30: Clinical Measurements, Machine vs AI Comparison & Longitudinal Timeline
   │
Phases 31–33: Authoritative Medication Knowledge Base & Safety Engine
   │
Phases 34–36: Clinical Decision Support & Doctor Review Workspace
   │
Phases 37–40: Publication Dual Reports (Patient Report vs Doctor Technical Report)
   │
Phases 41–45: Relational Database Migration, Security, Audit & Demo Mode
   │
Phases 46–50: Master Training Pipeline, E2E Testing, Regulatory Dossier & System Sign-Off
```

---

## 11. Audit Conclusion

The repository has passed the Phase 0 audit. All foundational algorithms and modules are intact and operational. 167 automated tests are passing. The system is structurally primed for progressive enhancement to the full hospital platform specification without breaking existing features.
