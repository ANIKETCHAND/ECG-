# ECG Guardian — Master System Architecture Specification

## 1. System Topology & Overview

ECG Guardian is an enterprise-grade, hospital-oriented ECG analysis and clinical workflow platform. It combines biomedical signal processing, machine-learning assisted arrhythmia triage, clinical pharmacology decision support, and multi-tenant hospital administrative workflows under an authoritative, zero-trust security perimeter.

```
+----------------------------------------------------------------------------------------------------+
|                                           PRESENTATION LAYER                                       |
|  - Modern Web SPA / Patient & Clinician Portal (public/index.html)                                  |
|  - Streamlit Diagnostic & Research Dashboard (app.py)                                              |
+----------------------------------------------------------------------------------------------------+
                                               │
                                 HTTPS / WSS / JWT Bearer Tokens
                                               ▼
+----------------------------------------------------------------------------------------------------+
|                                      API & SECURITY GATEWAY                                        |
|  - FastAPI Serverless API (api/index.py, api/security.py)                                          |
|  - Cryptographic JWT Token Engine & Role-Based Access Control (RBAC)                                |
|  - Zero-Trust Ingress Security & Hospital Tenant Isolation (hospital_id scoping)                   |
|  - DPDP Act 2025 Electronic Consent & Regulatory Gating                                            |
+----------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+----------------------------------------------------------------------------------------------------+
|                                     CORE CLINICAL ENGINE TIER                                      |
|  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────────┐   |
|  │  Biomedical Processing  │  │   AI Arrhythmia Model   │  │   Multimodal Medication Engine    │   |
|  │  - Quality assessment   │  │   - MIT-BIH Random      │  │   - Multimodal candidate ranking  │   |
|  │  - Waveform extraction  │  │     Forest (ECG-RF-     │  │   - Mandatory lab withholding     │   |
|  │  - Uncalibrated flags   │  │     2.0.0-PROD)         │  │   - Drug-drug interaction gate    │   |
|  │  - Pan-Tompkins QRS     │  │   - Zero-leakage split  │  │   - Physiological contraindicat.  │   |
|  └─────────────────────────┘  └─────────────────────────┘  └───────────────────────────────────┘   |
|                                              │                                                     |
|                                              ▼                                                     |
|                     Authoritative Clinical Analysis Object (ACAO)                                  |
|                     (src/clinical/clinical_analysis.py - Single Source of Truth)                   |
+----------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+----------------------------------------------------------------------------------------------------+
|                                    PERSISTENCE & STORAGE TIER                                      |
|  - Primary: Supabase PostgreSQL Managed Database (Multi-Tenant, Row-Level Security RLS)            |
|  - Storage: S3 Private Encrypted Object Storage with Pre-Signed Ephemeral URLs                      |
|  - Cache / Fallback: Local Encrypted Relational SQLite Persistence                                 |
|  - Audit Trail: Immutable Append-Only Audit Logging with Cryptographic Hashes                      |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Principles

### 2.1 Authoritative Clinical Analysis Object (ACAO) as Single Source of Truth
To prevent inconsistencies across user interfaces, export pipelines, and downstream analytics, all diagnostic processing produces a single strongly typed `AuthoritativeClinicalAnalysis` structure.
- **Components**: `PatientContext`, `ECGMeasurementsData`, `ECGAnalysisData`, `MedicationDecisionSupportData`, `PreviousECGComparisonData`, `ModelMetadataData`.
- **Integrity**: Every analysis run computes an SHA-256 integrity hash of its raw input signal and generated clinical fields.
- **Two-Tier Report Separation**:
  1. *Technical Doctor Report*: Complete beat classifications, interval durations (PR, QRS, QT, QTc), confidence scores, and raw signal quality metrics.
  2. *Patient-Friendly Report*: Readable summary, lifestyle advisories, and emergency escalation alerts; explicitly displays only clinician-approved interpretations.

### 2.2 Deny-by-Default Zero-Trust Security Boundary
- All endpoints under `/api/patients`, `/api/reports`, `/api/review`, and `/api/auth/me` enforce cryptographic JWT authentication.
- Strict authorization (`ECG_STRICT_AUTH=1`) is enforced across production deployments and security gates.
- Anonymous requests receive immediate `HTTP 401 Unauthorized`.
- Role-based permissions prevent patient accounts from sealing reports or accessing cross-patient data (`HTTP 403 Forbidden`).

### 2.3 Non-Autonomous Decision Support Boundary
- The platform explicitly disclaims autonomous medical device status.
- AI inferences are categorized as *Investigational Clinical Decision Support*.
- No diagnostic finding is finalized or released to a patient dossier until a licensed clinician reviews, confirms, and cryptographically signs the record.

---

## 3. Data Flow Architecture

1. **Ingestion & Validation**:
   - The user submits an ECG (raw numerical waveform, EDF, MIT-BIH, DICOM-ECG, PDF, or image).
   - Ingress gate inspects sampling frequency (target: 360 Hz), amplitude calibration, clipping, lead configuration, and SNR.
   - If optical image is uploaded, `extract_waveform_from_image` detects trace morphology, tags `is_calibrated: false`, disables ST-segment evaluation, and preserves true unscaled amplitude.
2. **Arrhythmia Classification**:
   - Validated beats are extracted via Pan-Tompkins QRS detection and morphologically analyzed.
   - Evaluated by `ECG-RF-2.0.0-PROD` (trained on MIT-BIH Arrhythmia Database with strict record-level separation).
3. **Pharmacology & Laboratory Gating**:
   - Medication candidate engine cross-references patient age, sex, comorbidities, current prescriptions, allergies, and vital signs.
   - Mandatory laboratory checks: If serum potassium or renal biomarkers (creatinine, eGFR) are absent, the engine withholds recommendations with a mandatory laboratory warning.
4. **Clinician Review & Sealing**:
   - An attending physician reviews the AI findings, modifies or confirms the diagnosis, signs with their medical registration number, and commits the report.
   - Audit logger writes an immutable signed record to the database audit table.

---

## 4. Multi-Tenant Isolation & Resiliency

- **Tenancy Scoping**: Every clinical table (`patients`, `ecg_studies`, `ecg_reports`, `hospital_memberships`) is partitioned by `hospital_id`.
- **Dual Persistence Strategy**:
  - Primary production persistence uses Supabase PostgreSQL with Row Level Security (RLS).
  - Secondary fallback local SQLite persistence guarantees uninterrupted emergency room operation during network partition or cloud outage.
  - Serverless container recycle on Vercel does not destroy committed clinical records.
