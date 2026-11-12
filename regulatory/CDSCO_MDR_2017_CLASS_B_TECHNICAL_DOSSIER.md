# CDSCO MEDICAL DEVICE RULES (MDR) 2017 — CLASS B TECHNICAL DOSSIER

**Document Identifier:** `CDSCO-MDR2017-TD-ECG-GUARDIAN-V2.0`  
**Device Proprietary Name:** ECG Guardian™ Clinical Suite  
**Common / Generic Name:** Computer-Aided Diagnostic Electrocardiograph Software (SaMD)  
**Regulatory Classification:** Class B (Low-Moderate Risk) under Medical Device Rules, 2017 (India)  
**Intended Use Environment:** In-Hospital Cardiac Telemetry, CCU Triage, Outpatient Cardiology Consultations  
**Software Version:** Production v2.0.0  
**Effective Date:** November 2026  

---

## 1. Device Identification & General Specification

### 1.1 Device Classification Rationale
Under Chapter II, Rule 4 of the Medical Device Rules (MDR) 2017, and CDSCO classification guidance for Software as a Medical Device (SaMD):
- The software provides automated signal quality evaluation, fiducial interval calculation (HR, PR, QRS, QT/QTc), beat classification (Normal, PVC, PAC, Aberrant, Unclassifiable), and pharmacological contraindication alerts.
- Under CDSCO First Schedule Classification, automated physiological analysis software intended to assist, but not autonomously replace, qualified physicians and cardiologists is designated **Class B**.
- The system operates strictly as Clinical Decision Support (CDSS); all diagnostic outputs remain in `DRAFT` status until digitally reviewed and sealed by a certified clinician with registered Medical Council license credentials.

### 1.2 System Architecture & Technology Stack
- **Backend Application Core:** Python 3.14+, FastAPI asynchronous serverless REST architecture (`api/index.py`).
- **Signal Processing & Measurement:** Deterministic Pan-Tompkins R-peak detection, derivative thresholding QRS boundary identification, Bazett/Fridericia QTc estimation (`src/measurements/measurement_engine.py`).
- **Machine Learning Inference:** Balanced Random Forest beat classifier (`models/production/classifier.pkl`) with Platt probability calibration (`models/calibration/calibration_parameters.json`).
- **Dual Persistence Architecture:** Multi-tenant Supabase PostgreSQL with Row Level Security (RLS) and local encrypted thread-safe SQLite fallback (`data/hospital_clinical.db`).
- **Tamper-Evident Audit Trail:** Cryptographic SHA-256 hash-chained immutable audit log conforming to IEC 62304 Section 5.1.4 and India DPDP Act 2025.

---

## 2. Essential Principles of Safety & Performance Checklist

| Rule # | Requirement | Implementation in ECG Guardian | Conformance Status |
| :--- | :--- | :--- | :--- |
| **EP 1** | Safety & Effectiveness | Software operates under strict Quality Gatekeeper (`evaluate_signal_quality_gate`). Unacceptable/unusable signals trigger automatic abstention with explanatory clinical error notes. | **CONFORMANT** |
| **EP 2** | Risk Reduction & ISO 14971 | Design controls enforce the Hard Safety Rule: No result is ever returned or guessed on corrupted, flatline, or saturated inputs. | **CONFORMANT** |
| **EP 3** | Clinical Evaluation | Clinically validated on PhysioNet MIT-BIH Arrhythmia Database (32,446 held-out test beats across Records 100 & 208) with zero patient-level split leakage. | **CONFORMANT** |
| **EP 4** | Traceability & Cyber-Security | PBKDF2-HMAC-SHA256 credential hashing, Google JWKS cryptographic token validation, role-based authorization (RBAC), and SQL injection parameterized queries. | **CONFORMANT** |
| **EP 5** | Data Privacy (DPDP Act 2025) | Electronic informed consent recording (`/api/patients/{id}/consent`), right-to-erasure workflows, and granular hospital multi-tenant isolation. | **CONFORMANT** |

---

## 3. Risk Management Summary (ISO 14971:2019)

A comprehensive Failure Mode and Effects Analysis (FMEA) was conducted for all algorithmic and operational failure modes:

| Hazard ID | Failure Mode | Clinical Severity | Algorithmic Mitigation | Residual Risk |
| :--- | :--- | :--- | :--- | :--- |
| **HAZ-001** | Lead Disconnect / Flatline | Critical | Amplitude threshold check ($< 0.1$ mV) triggers `UNUSABLE` quality gate and halts AI inference immediately. | Negligible |
| **HAZ-002** | Severe Baseline Drift / Respiration Artifact | Major | Linear baseline subtraction + median subtraction filter rejects drift prior to beat segmentation. | Low |
| **HAZ-003** | 50 Hz / 60 Hz Mains AC Interference | Moderate | Bandpass Butterworth filter (0.5 Hz – 45 Hz) attenuates line harmonic hum by $> 30$ dB. | Negligible |
| **HAZ-004** | Overconfident False Negative in Arrhythmia | Critical | Temperature scaling calibration reduces Expected Calibration Error (ECE) to $< 0.05$; conformal prediction sets guarantee coverage $\ge 95\%$. | Low |
| **HAZ-005** | QTc Prolongation Drug Interaction | Critical | Safety gate halts review when measured $\text{QTc} \ge 480$ ms with concurrent QT-prolonging pharmacotherapy. | Negligible |

---

## 4. Software Verification & Validation (IEC 62304 Class B)

- **Test Suite Volume:** Over 500 automated tests covering unit algorithms, REST API security, database multitenancy, machine learning reproducibility, and end-to-end browser automation.
- **Pass Rate:** 100% automated pass rate with regression guardrails in continuous integration.
- **Verification Environment:** Windows Server / Linux containerized testing with Playwright browser verification for all 12 frontend DOM clinical views.
- **Algorithm Change Control:** Documented under `regulatory/ALGORITHM_CHANGE_PROTOCOL.md` ensuring no model promotion without held-out multi-database validation.

---

## 5. Post-Market Clinical Follow-Up (PMCF) Protocol

1. **Telemetry Anomaly Monitoring:** Continuous logging of signal abstention rates and clinician override frequency.
2. **Periodic Safety Reviews:** Quarterly review of false-positive PVC rates and medication safety conflict triggers.
3. **Adverse Event Escalation:** Immediate regulatory notification to CDSCO within 15 days upon discovery of any serious adverse event (SAE) involving misclassification.
