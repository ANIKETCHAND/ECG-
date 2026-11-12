# ECG Guardian — Known System Limitations & Clinical Boundaries

> [!WARNING] **INVESTIGATIONAL HEALTHCARE SOFTWARE DISCLAIMER**
> ECG Guardian is an investigational software platform and clinical decision support system (CDSS / SaMD). It is **NOT** cleared or certified by CDSCO (India), US FDA, or European Notified Bodies as an autonomous medical diagnostic device. Under no circumstances should this software be utilized for autonomous patient diagnosis, acute coronary triage, emergency alarm dispatch, or automated medication prescribing without licensed physician oversight.

---

## 1. Clinical & Diagnostic Scope Limitations

### 1.1 Single-Lead (Lead II) Rhythm Strip Boundary
- **Restriction**: The algorithmic pipeline and production machine learning model (`ECG-RF-2.0.0-PROD`) are engineered and validated strictly on single-lead electrocardiograms (predominantly modified Lead II).
- **Prohibited Use Case**: The software cannot detect, evaluate, or localize acute myocardial infarction (STEMI / NSTEMI), reciprocal ST depressions, acute coronary syndrome (ACS), or anatomical vascular territory ischemia.
- **Clinical Policy**: The user interface and API explicitly prohibit generating myocardial infarction diagnoses from single-lead inputs.

### 1.2 Non-Autonomous Decision Support Boundary
- **Mandatory Clinician Review**: No AI inference or interval measurement is considered diagnostic or released into the finalized patient record until a licensed physician (`CARDIOLOGIST`, `PHYSICIAN`, or `MEDICAL_DIRECTOR`) reviews, confirms, and cryptographically signs the record.
- **Patient Dossier Isolation**: Patients have access only to finalized, signed reports (`report_status: "FINAL"`). Preliminary, unsealed AI hypotheses are withheld from patient portals to prevent inappropriate self-interpretation or health anxiety.

### 1.3 Excluded Patient Populations
- **Pediatric & Neonatal Cohorts**: Validated exclusively on adult subjects (age >= 18). Algorithmic interval thresholds (PR, QRS, QT/QTc) and heart rate norms are not calibrated for pediatric or neonatal electrophysiology.
- **Implanted Cardiac Devices**: The model is not trained or validated on paced rhythms (ventricular or dual-chamber pacing), cardiac resynchronization therapy (CRT), or automated implantable cardioverter-defibrillator (ICD) discharge morphology. Pacing spikes may distort feature extraction.

---

## 2. Biomedical Signal Processing & Optical Ingestion Limitations

### 2.1 Optical ECG Scans & Photographs
- **Grid Calibration Absence**: Optical scans, photographs, and rasterized PDF ECGs frequently lack verified physical millimeter grid calibration. Camera angle skew, zoom level, and optical distortion prevent deterministic voltage calibration.
- **Enforced Safeguards**:
  - Optical waveform extractions are permanently tagged with `is_calibrated: false`, `can_evaluate_st: false`, and `calibration_status: "UNCALIBRATED_OPTICAL_EXTRACT"`.
  - Voltage-dependent metrics (ST-segment elevation/depression, Sokolow-Lyon LVH criteria) are strictly withheld.
  - Optical signals are standardized via z-score normalization (`(signal - median) / std`) to preserve relative morphological timing while preventing amplitude misrepresentation.
  - QT interval prolongation alerts on optical scans are preliminary warnings; confirmation via manual caliper measurement on a standard calibrated 25 mm/s strip is mandatory.

### 2.2 Ingress Signal Quality & Noise Artifacts
- **Quality Gating**: Recordings with high muscle tremor (EMG), electrode detachment, extreme baseline wander, or rail saturation (>98% clipping) trigger `UNUSABLE` or `POOR` Signal Quality Indices (SQI).
- **Abstention Policy**: When an ECG is classified as `UNUSABLE`, AI inference and interval calculation are refused. The platform never fabricates a "Normal" rhythm for low-quality or corrupt signals.
- **Sampling Frequency**: Incoming signals are resampled to 360 Hz via polyphase filtering. Recordings acquired below 200 Hz may experience attenuation of high-frequency morphological notches.

---

## 3. Machine Learning & Statistical Boundaries

### 3.1 Dataset Generalization & Domain Shift
- **Training Cohort**: The production classifier is trained and evaluated on the MIT-BIH Arrhythmia Database (48 ambulatory Holter records, 360 Hz, Lead II).
- **Generalization Boundary**: Model performance has not been prospectively validated across diverse demographic cohorts, inpatient intensive care monitors, or 12-lead resting devices. Performance on external databases (PTB-XL, Chapman-Shaoxing, MIMIC-IV-ECG) must be independently proven before cross-cohort claims are asserted.

### 3.2 Premature Ventricular Contraction (PVC) False Discovery Rate
- **Measured Evaluation**: On the 14 held-out unseen MIT-BIH test records (32,446 heartbeats):
  - Recall / Sensitivity: **85.17%**
  - Precision: **62.99%**
  - False Positive Rate: **37.01%**
- **Clinical Consequence**: More than one in three PVC beat alerts is a false positive. Supraventricular aberrancy, bundle branch block morphology, and motion spikes can mimic ventricular ectopy.
- **Mitigation**: PVC counts and alerts are presented as advisory triage indicators requiring visual clinician verification prior to therapeutic intervention.

### 3.3 Secondary Task Placeholders
- **Model Scope**: While task definitions for Atrial Fibrillation detection (`TASK_AF_DETECTION`), 12-Lead Diagnostic Classification (`TASK_12LEAD_DIAGNOSTIC`), and ST-Segment Analysis (`TASK_ST_ANALYSIS`) exist in the codebase, only the beat arrhythmia model (`ECG-RF-2.0.0-PROD`) is trained and registered for active inference. Secondary task modules remain research placeholders and are gated from production diagnostic execution.

---

## 4. Clinical Pharmacology & Decision Support Boundaries

### 4.1 Mandatory Laboratory Withholding
- **Zero-Assumption Lab Policy**: The medication decision support engine strictly prohibits assuming normal laboratory constants.
- **Enforced Withholding**:
  - If serum potassium ($K^+$) is unrecorded, potassium-altering and QT-prolonging drugs are withheld with candidate status set to `REVIEW_REQUIRED` and mandatory advisory to obtain STAT electrolyte labs.
  - If serum creatinine or eGFR is unrecorded, renally cleared agents (e.g., Digoxin, DOACs, SGLT2i) are withheld pending renal clearance verification.
- **Hypokalemia Contraindication Gate**: Serum potassium < 3.5 mEq/L strictly blocks antiarrhythmic and torsadogenic candidates due to ventricular fibrillation / Torsades de Pointes risk.

### 4.2 Formulary & Guideline Coverage
- **Formulary Boundary**: The current clinical pharmacology engine covers essential cardiovascular drug classes (beta-blockers, calcium channel blockers, antiarrhythmics, ACEi/ARBs, anticoagulants, statins).
- **Uncovered Domains**: Rare orphan cardiac medications, pediatric dosing schedules, pregnancy-specific teratogenicity adjustments, and complex oncology-cardiology chemotherapy interactions are outside current engine coverage.

---

## 5. Architecture, Tenancy, & Operational Boundaries

### 5.1 Client-Side Token Storage
- **Current State**: Active session JWT tokens are maintained in browser `localStorage` to support client-side single-page application routing.
- **Limitation**: `localStorage` is vulnerable to token extraction if a cross-site scripting (XSS) vulnerability is introduced.
- **Roadmap Item**: Transition to HTTP-only, `SameSite=Strict` secure cookies is scheduled for Phase 2 infrastructure hardening.

### 5.2 Serverless Execution Timeouts
- **Compute Boundary**: The API runs on serverless container environments (Vercel Functions) with execution limits capped at 10–15 seconds.
- **Impact**: While 10-second rhythm strips process in <1.2 seconds, 24-hour ambulatory Holter recordings or high-resolution batch ingestions cannot run synchronously within serverless invocations and require background worker queues (Celery/Redis or BullMQ).

### 5.3 Local Dual Persistence vs Cloud Synchronization
- **Offline Resiliency**: In the event of a cloud database outage, the platform falls back to local relational SQLite persistence (`src/database/db_manager.py`).
- **Limitation**: Cross-device synchronization and centralized hospital analytics require active network connectivity to the primary Supabase PostgreSQL instance. Multi-workstation local synchronization is not supported without the central database.

---

## 6. Regulatory & Compliance Status

| Jurisdiction / Standard | Formal Status | Evidence / Verification |
| :--- | :--- | :--- |
| **CDSCO MDR 2017 (India)** | Investigational Pre-Submission Dossier Prepared (Class B SaMD) | Technical Dossier drafted; formal clinical trial clearance pending |
| **IEC 62304:2006/AMD 1:2015** | Architecture & test harness aligned (Class B software) | 573 automated regression tests passing |
| **ISO 14971:2019** | Risk management file and FMEA hazard mitigations implemented | FMEA ledger documented in regulatory plan |
| **ISO 13485:2016** | Quality management procedures mapped | Internal document control established |
| **DPDP Act 2025 (India)** | Electronic patient consent endpoints & audit logging operational | `/api/patients/{id}/consent` & immutable audit trail |
| **US FDA 510(k)** | Not Cleared | No marketing clearance exists; strictly research / investigational |
| **CE Mark (EU MDR 2017/745)** | Not Certified | Class IIa clinical evaluation pending |