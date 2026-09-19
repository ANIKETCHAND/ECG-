# Target Hospital-Oriented System Architecture
**Document ID:** ARCH-ECG-2026-001  
**Regulatory Framework:** Medical Devices Rules, 2017 (CDSCO, India), IEC 62304:2006/Amd 1:2015 (Medical Device Software - Software Life Cycle Processes), ISO 14971:2019  
**Software Classification:** Software as a Medical Device (SaMD), Non-invasive Diagnostic Decision Support  
**Status:** Approved Architectural Blueprint  

---

## 1. System Philosophy & Design Principles

The target architecture transitions the AI ECG Analyzer from a standalone local script into an **enterprise hospital-grade SaMD platform**. The architecture prioritizes:
1. **Patient Safety Over Availability**: If an input is ambiguous, corrupted, or unsupported, the system fails safe (**NO AI RESULT**).
2. **Strict Non-Hallucination**: Mathematical extraction only; zero synthetic signals, zero imputed measurements, zero fabricated diagnoses.
3. **Decoupled Autonomous Layers**: UI, API, database, and inference engine run as isolated services with clean interfaces.
4. **Mandatory Clinician Oversight**: AI never signs a report. AI findings are presented as provisional inputs to a human physician review workflow.
5. **Cryptographic Traceability**: Every analytical inference is tied to an immutable data hash, model version hash, preprocessing version, and operator identity.

---

## 2. End-to-End Hospital Data Pipeline

```text
                           HOSPITAL USER (Clinician / Tech)
                                         │
                                         ▼
                           [ 1. AUTHENTICATION & RBAC ]
                    JWT / MFA / Role Check (Technician, Doctor, Admin)
                                         │
                                         ▼
                           [ 2. PATIENT MANAGEMENT ]
                 Patient ID Linkage / Enterprise Master Patient Index
                                         │
                                         ▼
                             [ 3. ECG INGESTION ]
              Multi-format Ingestion (CSV, TXT, NPY, EDF, WFDB, PDF, DICOM)
                                         │
                                         ▼
                            [ 4. INPUT VALIDATION ]
                  File Integrity / Magic Bytes / Schema Verification
                                         │
                                         ▼
                         [ 5. SIGNAL QUALITY GATE ]
                    SNR / Drift / Artifacts / Clipping Check
                    Decision: GOOD | ACCEPTABLE | POOR | UNUSABLE
                      (If UNUSABLE ──► HALT: NO AI RESULT)
                                         │ (If Valid)
                                         ▼
                           [ 6. ECG PREPROCESSING ]
                 Causal/Zero-Phase Bandpass (0.5–40 Hz) + Median Filter
                                         │
                                         ▼
                       [ 7. MEASUREMENTS & FEATURES ]
             Deterministic Detection: R-Peaks, RR Intervals, Heart Rate
                                         │
                                         ▼
                            [ 8. AI MODEL INFERENCE ]
                 Locked Model Registry (e.g. ECG-RF-1.0.0 via Joblib)
                   Single-Lead MLII Only (Validated Bounds Only)
                                         │
                                         ▼
                           [ 9. RESULT VALIDATION ]
                 Sanity Checks / Outlier Traps / Model Probability
                                         │
                                         ▼
                          [ 10. CLINICIAN REVIEW ]
                 Mandatory Separation: AI Finding vs Physician Sign-off
                   Physician: [ Confirm ] | [ Modify ] | [ Reject ]
                                         │
                                         ▼
                          [ 11. FINAL CLINICAL REPORT ]
                    Cryptographically Sealed PDF / JSON / FHIR
                                         │
                                         ▼
                           [ 12. AUDIT LOGGING ]
                  Immutable Append-Only Audit Trail (WORM Storage)
```

---

## 3. Cross-Cutting Systems

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CROSS-CUTTING GOVERNANCE                        │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 1. SECURITY       │ 2. AUDIT TRAIL    │ 3. MODEL REGISTRY              │
│ - TLS 1.3 forced  │ - Immutable logs  │ - Hash-locked model weights    │
│ - Bcrypt hashing  │ - User + Action   │ - Versioned Model Cards        │
│ - JWT RBAC tokens │ - Data hash tied  │ - Deployment approvals        │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ 4. RISK MGMT      │ 5. DATA PRIVACY   │ 6. QUALITY MANAGEMENT          │
│ - ISO 14971 gates │ - DISHA / HIPAA   │ - IEC 62304 Class B processes  │
│ - Fail-safe traps │ - De-id engine    │ - Full Traceability Matrix     │
│ - Hazard logging  │ - Zero PHI in logs│ - CI/CD regression testing     │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

---

## 4. Multi-Service Architecture Breakdown

The system is organized into decoupled services communicating via documented REST APIs:

```text
                              INTERNET / HOSPITAL INTRANET
                                           │
                                           ▼
                                 [ NGINX REVERSE PROXY ]
                                  TLS 1.3 / Rate Limiting
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
             [ HOSPITAL CLINICAL UI ]               [ BACKEND REST API ]
            React / Next.js or Streamlit              FastAPI (Python 3.11+)
            • Clinician Worklist                    • Authentication & RBAC
            • Interactive Waveform Viewer           • Ingestion & Device Registry
            • Review & Sign-Off Portal              • Job Queue Dispatcher
            • Audit Log Inspector                   • Report Compilation
                        │                                     │
                        │                                     ▼
                        │                            [ ASYNC WORKER POOL ]
                        │                               Celery / Redis
                        │                           • Signal Quality Gate
                        │                           • Deterministic Measurements
                        │                           • ML Inference Service
                        │                                     │
                        ▼                                     ▼
           ┌────────────────────────┐            ┌────────────────────────┐
           │   POSTGRESQL DATABASE   │            │   OBJECT STORE / S3    │
           │ • Users & RBAC Roles    │            │ • Encrypted Raw ECGs   │
           │ • Patient Registry      │            │ • Verified Signal Data │
           │ • Analysis Runs         │            │ • Final Sealed PDFs    │
           │ • Signed Reports        │            │ • Immutable Audit Logs │
           │ • Audit Event Trail     │            │                        │
           └────────────────────────┘            └────────────────────────┘
```

---

## 5. Core Subsystem Responsibilities

### 5.1 Ingestion & Device Registry (`src/ecg_input/`, `src/ecg_core/`)
- **Unified Object Model**: Converts all inbound ECG formats into an internal `ECGRecording` instance with immutable metadata (sampling rate, duration, lead configuration, calibration units, device serial, data hash).
- **Device Registry (`device_registry.py`)**: Catalogs known hospital ECG devices (e.g. GE MAC series, Philips PageWriter, Schiller, Welch Allyn) and validates formats against tested device profiles.
- **Strict Format Guard**: Rejects unvalidated file types and prompts for missing metadata rather than assuming defaults.

### 5.2 Signal Quality Gate (`src/safety/signal_quality_gate.py`)
- Evaluates raw physiological voltage series against pre-inference thresholds:
  - Missing data / NaN / Inf ratio ($= 0\%$).
  - Saturation / rail clipping check ($< 1\%$ total duration).
  - Powerline interference ratio (50 Hz / 60 Hz).
  - Baseline drift variance.
  - Overall SNR threshold ($\ge 12\text{ dB}$ for `GOOD`, $\ge 6\text{ dB}$ for `ACCEPTABLE`).
- Output Categories:
  - `GOOD`: Pass to full AI inference and measurements.
  - `ACCEPTABLE`: Pass with explicit clinical noise warning flag.
  - `POOR`: Measurements computed with warning; **AI classification suppressed**.
  - `UNUSABLE`: **HALT PIPELINE**. No analysis, no measurements, no AI result.

### 5.3 Deterministic Measurements Engine (`src/measurements/`)
- Calculates physiological metrics using deterministic, peer-reviewed biomedical engineering algorithms:
  - Heart Rate (BPM) derived from validated R-R intervals with refractory constraints ($\ge 300\text{ ms}$).
  - Mean, Median, and SDNN of R-R intervals.
  - QRS duration via derivative thresholding on confirmed R-peaks.
- Explicit Error State: If any parameter cannot be reliably measured, returns `Not reliably measurable` (never an interpolated or default number).

### 5.4 Machine Learning Inference Service (`src/ml/`, `src/inference/`)
- Encapsulates model execution strictly within tested operational boundaries.
- **Model Registry**: Enforces model versioning (e.g., `ECG-RF-1.0.0`).
- **Lead Compatibility Gate**: Confirms that the input signal represents Modified Lead II (MLII) before running inference. If a 12-lead signal is provided without an isolated Lead II strip, returns `UNSUPPORTED_CONFIGURATION`.
- **Honest Probability Output**: Model probabilities are returned as statistical distributions (e.g. `{"Normal": 0.94, "PVC": 0.05, "Other": 0.01}`) with mandatory non-equivalence disclaimer.

### 5.5 Clinician Review & Sign-Off Portal (`src/review/`)
- Presents a side-by-side display of:
  1. Patient metadata and recording acquisition time.
  2. Technical signal quality score and warnings.
  3. Interactive multi-scale waveform with pan/zoom.
  4. Extracted deterministic measurements.
  5. Provisional AI model classification.
- Requires affirmative physician interaction:
  - Select diagnosis agreement or provide clinical override.
  - Enter physician name, registration number (e.g. State Medical Council ID), and clinical notes.
  - Apply electronic sign-off timestamp.

### 5.6 Audit & Governance Logging (`src/audit/`)
- Implements an append-only, tamper-evident audit logger recording every system transaction:
  - `LOGIN`, `PATIENT_CREATE`, `ECG_UPLOAD`, `INFERENCE_RUN`, `REPORT_GENERATE`, `REPORT_VIEW`, `CLINICIAN_SIGN_OFF`, `SYSTEM_CONFIG_CHANGE`.
- Stores event hashes to guarantee non-repudiation.

---

## 6. Regulatory Traceability & Lifecycle Mapping (IEC 62304 / MDR 2017)

All system modules map directly to lifecycle requirements:
- **Software Safety Classification**: Class B (Non-serious injury possible if incorrect decision is made without clinician review; mitigated to Class A through mandatory physician sign-off gate).
- **Verification Protocols**: Automated test suite (`pytest`) enforcing $100\%$ pass rate on safety-critical gates prior to deployment.
- **Software Change Protocol**: Strict semantic versioning (`MAJOR.MINOR.PATCH`) with locked model weights and mandatory regulatory reassessment upon model architecture updates.
