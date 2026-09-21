# ECG GUARDIAN: Target System Architecture
**Document ID:** ARCH-ECG-GUARDIAN-2026-001  
**Project Name:** ECG GUARDIAN  
**Reference Standards:** CDSCO MDR 2017 (India), IEC 62304:2006/Amd 1:2015, ISO 14971:2019, ISO 27799 / IEC 81001-5-1  
**Software Classification:** Software as a Medical Device (SaMD), Non-invasive Clinical Decision Support  
**Status:** Approved Architectural Blueprint  

---

## 1. Architectural Philosophy & Guiding Principles

ECG GUARDIAN is engineered as an **ECG safety, verification, evidence, longitudinal comparison, and clinician-review platform** designed to provide robust clinical decision support.

The platform is designed around the central question chain:
```text
CAN WE TRUST THE ECG?
        │
        ▼
WHAT DOES THE AI DETECT?
        │
        ▼
WHAT ECG EVIDENCE SUPPORTS IT?
        │
        ▼
DOES THE MACHINE AGREE?
        │
        ▼
WHAT CHANGED FROM PREVIOUS ECG?
        │
        ▼
WHAT DOES THE CLINICIAN DECIDE?
        │
        ▼
CAN WE REPRODUCE THE RESULT?
```

### Core Tenet
> **"NO RELIABLE INPUT = NO AI RESULT"**  
> Under no circumstances does the system fabricate, guess, or synthesize signals or classifications when input data is missing, corrupted, or unsupported. Every AI finding is strictly an adjunctive recommendation; final interpretation rests with the qualified clinician.

---

## 2. End-to-End Hospital Data Workflow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ECG GUARDIAN PIPELINE WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                [ ECG MACHINE ]
                                      │
                                      ▼
                                 [ ECG FILE ]
               (DICOM Waveform, HL7 aECG, EDF+, XML, CSV, TXT, NPY, PDF, Image)
                                      │
                                      ▼
                              [ ECG INGESTION ]
               (Format sniffers, header parsers, device registry)
                                      │
                                      ▼
                             [ INPUT VALIDATION ]
            (Magic bytes, schema check, min duration >= 1.5s, bounds check)
                                      │
                                      ▼
                           [ ECG QUALITY COPILOT ]
              (Lead-by-lead SNR, baseline drift, clipping, flatline)
                                      │
                                      ▼
                            [ SIGNAL TRUST CHECK ]
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
              [ UNUSABLE Signal ]               [ GOOD / ACCEPTABLE ]
                     │                                 │
                     ▼                                 ▼
             [ NO AI ANALYSIS ]                [ AI ECG ANALYSIS ]
           (Safety barrier halt)          (Decoupled inference engine)
                                                       │
                                                       ▼
                                             [ AI EVIDENCE ENGINE ]
                                       (Aberrant beat index attribution,
                                        coupling intervals, morphology)
                                                       │
                                                       ▼
                                        [ ECG MACHINE vs AI COMPARISON ]
                                       (Extract machine-printed findings)
                                                       │
                                                       ▼
                                          [ DISAGREEMENT DETECTION ]
                                       (AGREE / MINOR / SIGNIFICANT)
                                                       │
                                                       ▼
                                         [ PREVIOUS ECG COMPARISON ]
                                        (Longitudinal timeline & deltas)
                                                       │
                                                       ▼
                                             [ CLINICIAN REVIEW ]
                                         (Mandatory physician review:
                                          Accept / Modify / Reject)
                                                       │
                                                       ▼
                                               [ FINAL REPORT ]
                                        (Cryptographically sealed PDF)
                                                       │
                                                       ▼
                                               [ AUDIT TRAIL ]
                                         (Tamper-evident hash chain)
```

---

## 3. Subsystem Architecture & Module Mapping

The codebase transitions from a single-tier script into modular, decoupled packages:

```text
ecg_guardian/
├── src/
│   ├── ecg_core/             # Core clinical domain models (ECGRecording, ECGAnalysisResult)
│   ├── ecg_input/            # Multi-format ingestion loaders & device registry
│   ├── quality/              # ECG Quality Copilot & Signal Trust Gatekeeper
│   ├── ml/                   # Refactored machine learning pipeline
│   │   ├── preprocessing/    # Baseline median filter & zero-phase bandpass filter
│   │   ├── peak_detection/   # Physiologically constrained R-peak detectors
│   │   ├── segmentation/     # Individual cardiac cycle window extraction
│   │   ├── feature_extraction/# 28-feature extraction engine
│   │   ├── models/           # Scikit-Learn Random Forest & baseline models
│   │   ├── inference/        # Standalone, UI-independent inference engine
│   │   └── validation/       # Outlier traps and statistical sanity checks
│   ├── measurements/         # Deterministic biomedical measurement engine (HR, RR, QRS, QTc)
│   ├── evidence/             # ECG Evidence Engine (Beat attribution & morphology)
│   ├── comparison/           # Machine interpretation extraction & disagreement detector
│   ├── longitudinal/         # Longitudinal ECG timeline comparison & delta tracker
│   ├── database/             # Relational database models (Patients, ECGs, Reviews)
│   ├── auth/                 # PBKDF2 authentication & Role-Based Access Control (RBAC)
│   ├── audit/                # Cryptographic append-only SHA-256 audit logger
│   └── report/               # Publication-grade ReportLab PDF and JSON export engine
├── models/
│   ├── production/           # Hash-locked active model artifacts (ECG-RF-1.0.0)
│   ├── validation/           # Validation datasets and performance benchmarks
│   ├── archived/             # Historical model artifacts
│   └── registry/             # Model catalog metadata & specifications
├── backend/                  # FastAPI REST API services
├── frontend/                 # Interactive clinical dashboard (Streamlit / React)
├── regulatory/               # IEC 62304 / ISO 14971 compliance documentation
└── tests/                    # Comprehensive unit, integration, and security test suite
```

---

## 4. Core Data Entities

### 4.1 ECGRecording (`src/ecg_core/models.py`)
Immutable record representing an ingested ECG:
- `record_id`: Unique identifier (e.g. `REC-2026-A48F91`)
- `patient_id`: Associated hospital patient ID
- `sampling_rate`: Acquisition frequency in Hertz (Hz)
- `duration`: Recording duration in seconds
- `lead_names`: Ordered list of available leads (e.g. `["I", "II", "V1"]`)
- `number_of_leads`: Integer count of leads
- `signals`: Calibrated floating-point voltage array in millivolts ($\text{mV}$)
- `units`: String voltage units (`"mV"`)
- `acquisition_time`: ISO-8601 acquisition timestamp
- `device`: Device model identifier from registry
- `manufacturer`: Equipment manufacturer (e.g. GE, Philips, Contec)
- `source_format`: Raw format (`DICOM`, `WFDB`, `CSV`, `PDF`, `IMAGE`)
- `data_hash`: SHA-256 cryptographic fingerprint of raw voltage array
- `quality_metrics`: Dictionary of signal quality scores

### 4.2 ECGAnalysisResult (`src/ecg_core/models.py`)
Immutable analytical finding generated by the platform:
- `analysis_id`: Unique analysis identifier
- `record_id`: Linked `ECGRecording` identifier
- `model_id`: Active model identifier (e.g. `ECG-RF-1.0.0`)
- `prediction`: Primary rhythm finding (`Normal Sinus Rhythm`, `Premature Ventricular Contraction`)
- `model_probabilities`: Normalized class probability mapping
- `detected_r_peaks`: Array of integer sample indices corresponding to R-peaks
- `detected_beats_count`: Total number of segmented heartbeats
- `heart_rate_bpm`: Calculated ventricular rate
- `mean_rr_ms`: Mean R-R interval in milliseconds
- `signal_quality`: Categorical rating (`GOOD`, `ACCEPTABLE`, `POOR`, `UNUSABLE`)
- `limitations`: Explicit clinical operational boundaries
- `warnings`: Technical warnings (e.g. baseline drift, high noise)
- `evidence_beats`: Aberrant beat indices identified by the Evidence Engine
- `machine_comparison`: Disagreement status against machine interpretation

### 4.3 ClinicianReview (`src/ecg_core/models.py`)
Physician review and sign-off entity:
- `review_id`: Unique review identifier (e.g. `REV-2026-CDE022`)
- `analysis_id`: Linked `ECGAnalysisResult`
- `clinician_name`: Full legal name of reviewing physician
- `clinician_role`: Professional title (`CARDIOLOGIST`, `ELECTROPHYSIOLOGIST`, `PHYSICIAN`)
- `registration_number`: Medical council registration number (e.g. `MCI-DEL-2024-9988`)
- `agreement_status`: Mandatory status (`CONFIRMED`, `MODIFIED`, `REJECTED`)
- `clinician_interpretation`: Physician's authoritative diagnosis
- `clinical_notes`: Bedside notes, clinical correlation, treatment plan
- `reviewed_at`: ISO-8601 review timestamp
- `review_hash`: Cryptographic SHA-256 seal of the review and analysis

---

## 5. Cross-Cutting Governance

### 5.1 Security & Access Control
- **Authentication:** Salted PBKDF2-HMAC-SHA256 password hashing (100,000 iterations).
- **Authorization:** Strict Role-Based Access Control (RBAC):
  - `DOCTOR`: View patients, run ECG analysis, review/sign reports.
  - `CARDIOLOGIST`: Full clinical privileges, review/override AI findings, seal reports.
  - `TECHNICIAN`: Ingest ECGs, inspect signal quality, create patient records.
  - `ADMIN`: User management, system health monitoring, audit inspection.
  - `RESEARCHER`: De-identified dataset inspection, model evaluation.
- **Privacy (Zero-PHI in Logs):** Patient identifiers, names, and raw medical data are strictly excluded from console logs, system logs, and external telemetry.

### 5.2 Cryptographic Audit Trail
- Every critical system event (Login, Ingestion, Quality Analysis, AI Inference, Disagreement Detection, Review Sign-Off) is committed to an **append-only audit log**.
- Each entry contains: `log_id`, `timestamp`, `user_id`, `event_type`, `record_id`, `details`, and `previous_hash`.
- The `current_hash` is computed as $\text{SHA-256}(\text{details} + \text{previous\_hash})$, creating an unbreakable, tamper-evident cryptographic chain.

---

## 6. Verification & Architectural Readiness

- **Current Operational Status:** Fully verified baseline with 72 automated pytest tests.
- **Decoupled Engines:** Model inference, deterministic measurements, quality gatekeeper, and cryptographic audit are architected to operate independently of any specific frontend.
- **Deployment Modalities:** Dual-mode support for local clinical workstations (Streamlit), edge serverless microservices (FastAPI), and future Dockerized hospital enterprise deployments.
