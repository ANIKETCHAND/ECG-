# ECG GUARDIAN: Comprehensive Architecture & Repository Audit
**Document ID:** AUD-ECG-GUARDIAN-2026-001  
**Project Name:** ECG GUARDIAN  
**Repository:** [https://github.com/ANIKETCHAND/ECG-](https://github.com/ANIKETCHAND/ECG-)  
**Phase:** Phase 1 — Repository Audit & Architecture Baseline  
**Date:** September 2026  
**Status:** Approved Engineering Baseline Audit  

---

## Statutory Regulatory & Clinical Disclosure

> [!CAUTION]
> **MANDATORY STATUTORY DISCLAIMER:**
> **ECG GUARDIAN** is currently an investigational software engineering platform and prototype. It is:
> - **NOT CDSCO Approved** (Central Drugs Standard Control Organisation, India) under the Medical Devices Rules (MDR) 2017.
> - **NOT FDA Approved or Cleared** (US Food and Drug Administration) under 510(k) or De Novo pathways.
> - **NOT CE Marked** under European Union Medical Device Regulation (EU MDR 2017/745).
> - **NOT Clinically Validated** in prospective clinical trials.
> - **NOT Authorized for Autonomous Diagnosis** of any disease, cardiac pathology, or physiological state.
> - **NOT a Replacement for a Physician**, cardiologist, or qualified healthcare professional.
>
> All AI-derived parameters and classifications are provisional algorithmic decision support outputs subject to mandatory qualified physician review, modification, and sign-off.

---

## Global Architectural Rules (Governing All Phases)

Every component and phase of the ECG GUARDIAN platform adheres to the following inviolable principles:

- **RULE 1 — Never Fabricate Data:** Never fabricate ECG signals, patient demographics, clinical measurements, diagnoses, model probabilities, performance metrics, clinical validation, or regulatory approvals.
- **RULE 2 — Never Convert Failure into Normal:** Never implement `if error: prediction = "Normal"`. System failures, unreadable data, or pipeline errors strictly resolve to **NO RESULT**.
- **RULE 3 — Never Generate Synthetic ECG Data to Replace Missing Patient Data:** If waveform extraction or digitization fails, the pipeline outputs **NO AI ANALYSIS**.
- **RULE 4 — Never Assume Missing ECG Metadata:** Never silently assume 360 Hz, 12 leads, or 10-second duration unless explicitly established by digital headers, calibrations, or confirmed operator entry.
- **RULE 5 — Never Call Model Probability "Diagnostic Confidence":** Use strictly **Model probability**; statistical softmax outputs must never be conflated with clinical diagnostic certainty.
- **RULE 6 — Do Not Invent Disease Classes:** The application strictly exposes only the classes supported by the actual trained and validated model (`Normal Sinus Rhythm`, `Premature Ventricular Contraction (PVC)`).
- **RULE 7 — Clinician Remains Responsible for Final Interpretation:** The AI output remains strictly separate from the physician's signed clinical interpretation.
- **RULE 8 — Do Not Claim Clinical Validation:** No claim of clinical efficacy or diagnostic accuracy in human populations may be made until appropriate prospective institutional trials have been executed.
- **RULE 9 — Preserve Reproducibility:** Every analysis records: `ECG ID`, `Model version`, `Preprocessing version`, `Software version`, and cryptographic `Timestamp`.
- **RULE 10 — Test Every Phase:** Every phase must implement, test, fix, document, and verify before proceeding to the subsequent phase.

---

## 1. CURRENT SYSTEM

### 1.1 Repository Structure
The repository is organized as follows:
```text
E:\CODE\AI-ECG-Analyzer
├── app.py                      # Streamlit interactive application orchestrator
├── requirements.txt            # Production dependencies
├── requirements-all.txt        # Full local development dependencies
├── PROJECT_DOCUMENTATION.md    # Initial project documentation
├── README.md                   # Repository overview and deployment guide
├── api/
│   ├── index.py                # Serverless FastAPI endpoint
│   └── requirements.txt        # Serverless dependency profile
├── src/
│   ├── data_loader.py          # MIT-BIH PhysioNet WFDB record loader
│   ├── preprocessing.py        # Baseline wander median filter & Butterworth bandpass
│   ├── signal_quality.py       # SNR, baseline drift, 50/60 Hz powerline, quality scoring
│   ├── peak_detection.py       # Scipy find_peaks with 300 ms refractory window
│   ├── segmentation.py         # [-0.2s, +0.4s] cardiac cycle extraction
│   ├── feature_extraction.py   # 28 time, morphology, frequency, and RR features
│   ├── label_mapping.py        # AAMI EC57 to 3-class mapping (Normal, PVC, Other)
│   ├── prediction.py           # Window-level Random Forest inference
│   ├── evaluation.py           # Multi-metric evaluation (Accuracy, F1, Precision, Recall)
│   ├── visualization.py        # Plotly & Matplotlib waveform strip plotting
│   ├── ecg_core/               # ECGRecording and ECGAnalysisResult domain entities
│   ├── ecg_input/              # Format detection, CSV/NPY, PDF text, OpenCV trace isolation
│   ├── safety/                 # Signal Quality Gatekeeper (GOOD, ACCEPTABLE, POOR, UNUSABLE)
│   ├── measurements/           # Deterministic measurement engine (HR, RR, QRS, QTc)
│   ├── inference/              # Decoupled inference engine (ECG-RF-1.0.0)
│   ├── auth/                   # PBKDF2 authentication & RBAC manager
│   ├── database/               # SQLite patient, ECG, and review database manager
│   ├── audit/                  # Cryptographic SHA-256 hash-chained audit logger
│   └── report/                 # ReportLab PDF and JSON export engine
├── models/
│   ├── classifier.pkl          # Trained Random Forest classifier (100 trees)
│   ├── baseline_classifier.pkl # Trained Logistic Regression baseline
│   ├── scaler.pkl              # StandardScaler fitted on training records only
│   ├── metadata.json           # Model configuration, hyperparameters, feature names
│   ├── production/             # Active production model registry
│   ├── validation/             # Validation test artifacts
│   ├── archived/               # Superseded model versions
│   └── registry/               # Registry catalog
├── training/
│   ├── prepare_dataset.py      # Feature matrix extraction from raw records
│   ├── train_test_split.py     # Patient-level split logic (no patient overlap)
│   ├── train_model.py          # Model training and artifact serialization
│   └── evaluate_model.py       # Record-level validation metrics calculation
├── data/
│   ├── raw/                    # MIT-BIH benchmark records (.dat, .hea, .atr)
│   ├── processed/              # Processed train/test feature matrices
│   ├── hospital_clinical.db    # Relational transactional database
│   └── audit_trail.db          # Immutable audit database
├── docs/                       # Architectural & regulatory documentation
└── tests/                      # 72 automated pytest test suites
```

### 1.2 Current Machine Learning Model
- **Algorithm:** `RandomForestClassifier` (Scikit-Learn).
- **Ensemble:** 100 decision trees (`n_estimators=100`, `random_state=42`).
- **Feature Vector:** 28 engineered features per heartbeat.
- **Normalization:** `StandardScaler` fitted strictly on training partition (`models/scaler.pkl`).
- **Baseline Model:** L2-regularized `LogisticRegression` (`models/baseline_classifier.pkl`).
- **Active Model Identifier:** `ECG-RF-1.0.0`.

### 1.3 Current Dataset & Partitioning
- **Source:** MIT-BIH Arrhythmia Database (PhysioNet).
- **Lead Evaluated:** Modified Lead II (MLII).
- **Sampling Rate:** 360 Hz.
- **Record-Level Patient Partitioning (Zero Patient Leakage):**
  - **Train Set (4 records, 10,152 heartbeats):** Records `100`, `106`, `200`, `213`.
  - **Unseen Test Set (3 records, 6,807 heartbeats):** Records `101`, `119`, `208`.
- **Test Performance on Unseen Patients:**
  - Overall Accuracy: **97.86%**
  - Normal Sinus Rhythm (Support: 4,989): Precision 99.88%, Recall 97.29%, **F1: 98.57%**
  - PVC Ventricular Ectopy (Support: 1,809): Precision 93.58%, Recall 99.89%, **F1: 96.63%**
  - Other Class (Support: 9): Precision 0.00%, Recall 0.00%, **F1: 0.00%**

### 1.4 Current Signal Processing Pipeline
1. **Baseline Wander Removal:** Cascaded median filters (200 ms kernel removes P-QRS-T complexes; 600 ms kernel smooths baseline drift; subtraction removes wander).
2. **Bandpass Filtering:** 3rd-order Butterworth bandpass filter (0.5 Hz to 40.0 Hz) via forward-backward zero-phase filtering (`scipy.signal.filtfilt`).
3. **Signal Quality Gatekeeper:** Evaluates SNR (dB), baseline drift ratio, 50/60 Hz powerline harmonic ratio, clipping percentage, and flatline detection. Outputs categorical grade: `GOOD`, `ACCEPTABLE`, `POOR`, `UNUSABLE`.
4. **R-Peak Detection:** Adaptive amplitude thresholding with a physiologically constrained 300 ms refractory period.
5. **Beat Segmentation:** Isolates temporal window $[-0.20\text{s}, +0.40\text{s}]$ (216 samples at 360 Hz) centered on each detected R-peak.
6. **28-Feature Extraction:** Computes 11 time-domain, 7 morphological, 7 frequency-domain (Welch PSD), and 3 R-R interval dynamic features.
7. **Deterministic Measurements:** Computes Heart Rate (BPM), mean R-R interval (ms), QRS duration (ms), Bazett-corrected QTc (ms), Fridericia-corrected QTc (ms), and frontal electrical axis (requiring multilead I & II; safely rejects single-lead axis calculation).

---

## 2. CURRENT LIMITATIONS

1. **Single-Lead MLII Operational Constraint:**  
   The classifier was trained strictly on Modified Lead II (MLII). It cannot evaluate 12-lead spatial vectors, precordial ischemic changes (V1–V6), or limb lead electrical axes. Attempting to classify other leads using this model is clinically invalid.
2. **Zero Sensitivity for Class "Other":**  
   Due to extreme class imbalance in the training data, the model scored 0.0% precision and recall on class `Other`. The model cannot be claimed to detect general ectopic or supraventricular arrhythmias; it is strictly an adjunctive detector for Normal Sinus Rhythm vs. PVC.
3. **Lack of Explainable Beat Evidence:**  
   When the model identifies PVC ectopy, it outputs window-level probabilities without highlighting or isolating the specific aberrant beats (e.g., Beat #37, Beat #84) with local morphology, coupling intervals, and pre/post-RR ratios.
4. **No Automated Machine vs. AI Comparison:**  
   While machine-printed text is extracted from PDF reports, the system lacks an automated semantic comparison engine to detect discrepancies between the ECG machine's printed interpretation and the AI model's finding.
5. **Absence of Longitudinal Comparison:**  
   The current architecture does not track patient timelines (e.g., Jan vs. Mar vs. Sep) or calculate delta metrics ($\Delta\text{HR}$, $\Delta\text{PR}$, $\Delta\text{QRS}$, $\Delta\text{QTc}$, rhythm shift, ectopic burden changes).
6. **Format Gaps in Clinical Ingestion:**  
   Native healthcare interchange standards such as DICOM Waveform (SOP Class `1.2.840.10008.5.1.4.1.1.9.1.1`), HL7 aECG XML, and European Data Format (EDF/EDF+) are not yet fully supported.
7. **Coupled Monolithic Deployment:**  
   Streamlit serves as both the presentation layer and the application coordinator, which limits enterprise microservice scalability, containerized job queuing, and cloud orchestration.

---

## 3. TARGET ECG GUARDIAN SYSTEM

### 3.1 Target End-to-End Workflow
```text
ECG MACHINE
    │
    ▼
ECG FILE (DICOM, HL7, EDF, XML, CSV, TXT, NPY, PDF, Image)
    │
    ▼
ECG INGESTION (Format-specific loaders & device catalog)
    │
    ▼
INPUT VALIDATION (File integrity, magic bytes, duration, schema)
    │
    ▼
ECG QUALITY COPILOT (Lead-level noise, drift, clipping, flatline)
    │
    ▼
SIGNAL TRUST CHECK (GOOD / ACCEPTABLE -> Proceed; UNUSABLE -> HALT: NO AI RESULT)
    │
    ▼
AI ECG ANALYSIS (Decoupled inference engine with locked models)
    │
    ▼
AI EVIDENCE ENGINE (Beat-level attribution, morphometrics, coupling intervals)
    │
    ▼
ECG MACHINE vs AI COMPARISON (Semantic extraction & alignment)
    │
    ▼
DISAGREEMENT DETECTION (AGREE, MINOR_DIFFERENCE, SIGNIFICANT_DISAGREEMENT)
    │
    ▼
PREVIOUS ECG COMPARISON (Longitudinal trend & delta analysis)
    │
    ▼
CLINICIAN REVIEW (Mandatory affirmative sign-off: Accept / Reject / Modify)
    │
    ▼
FINAL REPORT (Cryptographically sealed multi-page PDF & JSON)
    │
    ▼
AUDIT TRAIL (Tamper-evident append-only SHA-256 hash chaining)
```

### 3.2 Target Architectural Subsystems
1. **Ingestion & Domain Core (`src/ecg_input/`, `src/ecg_core/`):** Unified `ECGRecording` domain entity with full metadata, manufacturer registry, and immutable raw data fingerprints.
2. **ECG Quality Copilot (`src/quality/`):** Independent lead-by-lead signal quality assessment with actionable clinical feedback.
3. **Refactored ML Engine (`src/ml/`):** Decoupled inference, reproducible patient-level training pipelines, and versioned model registry.
4. **ECG Evidence Engine (`src/evidence/`):** Per-beat evidence attribution linking model findings to specific cardiac cycles with interactive zoom and morphology metrics.
5. **ECG Machine vs AI Verification (`src/comparison/`):** Rule-based and semantic disagreement detector comparing machine-printed text against AI predictions.
6. **Longitudinal ECG & Patient History (`src/longitudinal/`):** Timeline comparison calculating physiological deltas without hallucinating clinical conclusions.
7. **Hospital Backend & Multi-Tenancy (`backend/`, `frontend/`, `database/`):** FastAPI REST backend, PostgreSQL/SQLite persistence, role-based access control, and clinician review workflow.
8. **Security, Privacy & Audit Trail (`src/audit/`, `src/auth/`):** OWASP Top 10 hardening, PBKDF2 password hashing, zero PHI in logs, and SHA-256 audit chaining.
9. **Regulatory Readiness Framework (`regulatory/`):** IEC 62304 Class B software lifecycle, ISO 14971 risk management, and formal clinical validation plans.

---

## 4. MIGRATION PLAN (Phases 1 to 12)

The migration is executed strictly phase-by-phase. Each phase must be implemented, tested, verified, and documented before the subsequent phase commences:

| Phase | Title | Scope & Key Deliverables | Stop Condition |
| :---: | :--- | :--- | :--- |
| **Phase 1** | **Repository Audit & Architecture** | Full repository audit, system baseline, test suite baseline, creation of `docs/ARCHITECTURE_AUDIT.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/INTENDED_USE.md`. | Complete audit and 100% passing test baseline. |
| **Phase 2** | **ECG Input & Ingestion Engine** | Comprehensive format loaders (CSV, TXT, NPY, WFDB, EDF, XML, JSON, DICOM, PDF, Image), unified `ECGRecording` domain entity, anti-hallucination validation gates. | All input types either produce valid `ECGRecording` or clear `NO RESULT`. |
| **Phase 3** | **ECG Quality Copilot** | Lead-level artifact, noise, drift, clipping, flatline checks in `src/quality/`. Quality categorizations: `GOOD`, `ACCEPTABLE`, `POOR`, `UNUSABLE`. | AI inference strictly blocked on `UNUSABLE` inputs. |
| **Phase 4** | **Refactor ML Pipeline** | Decouple ML engine into `src/ml/` with independent inference engine, structured output schema, and UI independence. | ML inference callable independently from frontend. |
| **Phase 5** | **Model Registry & Training Infrastructure** | Patient-level splitting, multi-metric evaluation (AUROC, AUPRC, calibration), formal `MODEL_CARD.md`, model registry (`models/production/`, `validation/`, `archived/`). | No model promoted without patient-level validation & model card. |
| **Phase 6** | **ECG Evidence Engine** | `src/evidence/` identifying aberrant beat indexes (e.g. Beat #37, Beat #84) with coupling intervals, morphology, and rhythm context. | Every explanation traces directly to signal data. |
| **Phase 7** | **Machine vs AI Verification** | `src/comparison/` extracting machine-printed interpretation, comparing against AI finding, flagging `SIGNIFICANT_DISAGREEMENT`. | System flags discrepancy without guessing which is correct. |
| **Phase 8** | **Longitudinal ECG & Patient History** | `src/longitudinal/` tracking patient ECG history and computing compatible metric deltas ($\Delta\text{HR}$, $\Delta\text{QRS}$, $\Delta\text{QTc}$). | Only compatible available metrics compared. |
| **Phase 9** | **Hospital Backend & Patient Management** | Relational schemas for patients, ECGs, analyses, and reports; secure REST API; multi-user isolation. | Full multi-user relational persistence functioning. |
| **Phase 10** | **Clinician Dashboard & Review** | Clinical review interface with mandatory Accept/Reject/Modify sign-off, registration number recording, and sealing. | Clinician able to override and seal interpretation. |
| **Phase 11** | **Reporting, Audit & Security** | Publication-grade PDF/JSON reports, append-only SHA-256 audit chaining, OWASP security controls, zero PHI logging. | Full cryptographic audit verification passing. |
| **Phase 12** | **Regulatory Readiness & Final Acceptance** | Regulatory documentation (`regulatory/`), risk management (ISO 14971), clinical validation plan, failure-mode test suite, final end-to-end acceptance test. | All 42 acceptance criteria verified. |

---

## 5. Audit Verification & Baseline Sign-Off

- **Current Test Suite Status:** **72 passed, 0 failed, 3 warnings** (`pytest tests/ -v`).
- **Codebase Integrity:** Working tree clean, zero uncommitted changes, baseline synchronized with remote `https://github.com/ANIKETCHAND/ECG-`.
- **Phase 1 Verdict:** **AUDIT COMPLETE & VERIFIED.** Ready for Phase 2 initiation upon user approval.
