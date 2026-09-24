# ECG GUARDIAN — PHASE 0: REPOSITORY & DATABASE AUDIT
**Document Version:** 1.0.0  
**Regulatory References:** IEC 62304 Section 5.3 (Software Architecture), CDSCO MDR 2017, ISO 27799 / HIPAA Security Rule  
**Audit Date:** 2026-09-24  
**Auditor:** DeepMind Antigravity AI  

---

## 1. Executive Summary

This repository audit provides an exhaustive inspection of the existing **ECG Guardian** codebase to analyze the current data persistence, report generation, and session lifecycle. 

The investigation confirms the primary problem:
> **The ECG report is generated successfully, but after the analysis/session the report is not persisted in a persistent, queryable cloud database. The user cannot reopen previous ECG reports after a page refresh or in subsequent sessions.**

Currently:
- Reports exist only as temporary in-memory Python dictionaries (`report_data` in [app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L900-L926)) or client-side JavaScript memory ([public/index.html](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/public/index.html#L266-L269)).
- While a local SQLite database ([src/database/db_manager.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/database/db_manager.py)) exists, its `clinical_reports` table only stores sparse reference metadata (IDs, SHA-256) and **never stores the structured report snapshot JSON (`report_data`) or the PDF binary**.
- There is **no Supabase integration** configured anywhere in the project.
- There is **no History navigation page or endpoint** allowing users to query, search, view, or re-download previously generated reports.
- On browser refresh or session reset, all analysis state is completely lost.

---

## 2. Core Repository Architecture Inventory

| Architectural Layer | Implementation Technology | Primary File Path(s) | Current State & Responsibilities |
| :--- | :--- | :--- | :--- |
| **Primary Frontend** | Streamlit (Python) | [app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py) | Full-featured clinical dashboard, multi-tenant UI, waveform visualization, CDS display, review sign-off. |
| **Secondary Web Portal** | HTML5, Vanilla JS, Tailwind CSS CDN, Plotly.js | [public/index.html](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/public/index.html) | Lightweight serverless web client calling FastAPI endpoints. |
| **Backend API** | FastAPI / Uvicorn (ASGI) | [api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py) | Serverless REST API endpoints for `/api/health`, `/api/sample`, `/api/analyze`, `/api/report/pdf`, `/api/review`. |
| **Database Layer** | SQLite (Local file `data/hospital_clinical.db`) | [src/database/db_manager.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/database/db_manager.py) | Local relational tables for hospitals, doctors, patients, ecg_records, analysis_results, clinician_reviews, clinical_reports. |
| **Authentication & RBAC** | In-memory / PBKDF2-HMAC-SHA256 | [src/auth/auth_manager.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/auth/auth_manager.py) | Role-Based Access Control matrix (`HOSPITAL_ADMIN`, `CARDIOLOGIST`, `DOCTOR`, `ECG_TECHNICIAN`, `NURSE`, `RESEARCHER`, `PATIENT`). |
| **ECG Ingestion & Parsing** | Python (NumPy, SciPy, PyPDF2/pdfplumber, OpenCV) | [src/ecg_input/](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/ecg_input/) | Modality detection (digital signal, printed PDF, photo/image) and digital parsing. |
| **Signal Quality Gate** | Python algorithmic SNR & drift | [src/safety/signal_quality_gate.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/safety/signal_quality_gate.py) | Strict barrier: `ACCEPTABLE`, `POOR`, `UNUSABLE`. Rejects unusable signals before inference. |
| **ML Inference Engine** | Scikit-learn (RandomForestClassifier, 28 features) | [src/inference/inference_engine.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/inference/inference_engine.py), [src/prediction.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/prediction.py) | Arrhythmia classification across 10,152 trained beats (Normal, PVC, Other). |
| **Measurements Engine** | Pan-Tompkins & fiducial algorithms | [src/measurements/measurement_engine.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/measurements/measurement_engine.py) | HR, RR, PR, QRS duration, QT, QTc Bazett/Fridericia intervals. |
| **Clinical Decision Support** | Rule-based guideline engine (AHA/ACC/ESC) | [src/clinical/recommendation_engine.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/clinical/recommendation_engine.py) | Actionable urgency recommendations, contraindication warnings, evidence links. |
| **Medication Safety** | Multimodal drug interaction & conflict engine | [src/medications/interaction_checker.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/medications/interaction_checker.py) | Drug-drug interactions, drug-QTc prolongation, drug-bradycardia, electrolyte alerts. |
| **Report Generator** | Structured Python clinical aggregator | [src/report/report_generator.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/report/report_generator.py) | Assembles 12-section standardized multimodal clinical report dictionary. |
| **PDF Generator** | ReportLab | [src/report/pdf_generator.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/report/pdf_generator.py) | Generates publication-grade Doctor and Patient summary PDFs with criteria indicators. |
| **Audit Logger** | SQLite & JSONL | [src/audit/audit_logger.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/audit/audit_logger.py) | Records clinical events, clinician reviews, sign-offs, and compliance traces. |

---

## 3. Detailed Answers to Audit Questions

### 1. Where the ECG analysis result is generated?
- **Streamlit ([app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py)):**
  - **MIT-BIH Demo Pipeline (Lines 660–680):** When selecting an MIT-BIH benchmark record, `predict_ecg(signal_data, sampling_rate)` from [src/prediction.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/prediction.py) is called.
  - **File Upload Pipeline (Lines 830–865):** When an ECG file (CSV, JSON, DAT, Image, PDF) is uploaded, `predict_ecg(signal_data, sampling_rate, ...)` executes bandpass filtering, Pan-Tompkins R-peak detection, beat segmentation, 28-feature extraction, and Random Forest classification.
  - Output is assigned to in-memory variable `ai_results` containing: `predicted_class`, `probabilities`, `heart_rate_bpm`, `mean_rr_sec`, `beat_count`, `signal_quality`, `quality_score`, `detected_peaks`.
- **FastAPI ([api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py#L132-L235)):**
  - In route `POST /api/analyze`:
    - Signal quality is evaluated via `evaluate_signal_quality_gate(sig_arr, req.fs, req.lead)` ([src/safety/signal_quality_gate.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/safety/signal_quality_gate.py)).
    - Inference runs via `run_ecg_inference(rec, model_id=ACTIVE_MODEL_ID)` ([src/inference/inference_engine.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/inference/inference_engine.py)).
    - Measurements are computed via `compute_ecg_measurements(sig_arr, req.fs, ...)` ([src/measurements/measurement_engine.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/measurements/measurement_engine.py)).
    - Medication safety is checked via `check_medication_safety(...)` ([src/medications/interaction_checker.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/medications/interaction_checker.py)).
    - CDS is evaluated via `GLOBAL_CDS_ENGINE.evaluate_finding(...)` ([src/clinical/recommendation_engine.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/clinical/recommendation_engine.py)).

### 2. Where the report object is generated?
- **Core Function:** `generate_structured_report(...)` in [src/report/report_generator.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/report/report_generator.py#L24-L350).
- **Streamlit Invocations:**
  - In [app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L900-L925), `report_data` is generated immediately following analysis by consolidating:
    - `input_info`, `ai_results`, `extracted_measurements`, `waveform_status`, `clinician_review`, `cds_report`, `medication_safety`, `patient_profile`, `vital_signs`, and `laboratory_results`.
  - In Patient View ([app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L603-L607)), `pat_rep` is generated dynamically on-the-fly from historical records.
- **FastAPI Invocations:**
  - In [api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py#L296-L311) when `req.include_full_report=True`.
  - In [api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py#L325-L340) inside `generate_pdf_endpoint()`.

### 3. Where the PDF is generated?
- **Core Implementation:** [src/report/pdf_generator.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/report/pdf_generator.py):
  - `generate_doctor_report(report_data, waveform, fs, r_peaks)` (Lines 15–460): Publication-grade 2-page clinical report with hospital branding, patient demographics, clinical history, 12-lead measurements, beat segmentation, waveform strip with R-peak markers, AI classification, medication safety warnings, CDS recommendations, physician signature panel, and criteria footers.
  - `generate_patient_report(report_data, waveform, fs, r_peaks)` (Lines 462–620): Patient-friendly, plain-language summary.
- **Frontend Calling Points:**
  - [app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L1429-L1440): `doctor_pdf_bytes = generate_doctor_report(...)` and `patient_pdf_bytes = generate_patient_report(...)`, streamed directly to Streamlit download buttons `st.download_button`.
  - [api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py#L340): Generated on the fly and returned as raw bytes `Response(content=pdf_bytes, media_type="application/pdf")`.

### 4. Whether any database currently exists?
- **Yes, a local SQLite database exists:**
  - Class: `DatabaseManager` in [src/database/db_manager.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/database/db_manager.py).
  - File location: `data/hospital_clinical.db` (configurable via `DATABASE_PATH`).
  - Schema defines: `hospitals`, `doctors`, `staff`, `patients`, `ecg_records`, `analysis_results`, `clinician_reviews`, `clinical_reports`, `vital_signs`, `laboratory_results`, `patient_symptoms`.
- **Critical Flaws of the Existing SQLite Database:**
  1. **No Report Snapshot Storage:** The `clinical_reports` table only has columns: `report_id`, `record_id`, `analysis_id`, `review_id`, `report_type`, `status`, `file_path`, `report_sha256`, `generated_at`. It **does not have a `report_data` JSONB/TEXT column**. The structured 12-section report is never saved to the database.
  2. **No Binary PDF Storage:** `file_path` is left null or points to a non-existent local file. No PDF binary is stored.
  3. **No Cloud / Serverless Persistence:** The database is an ephemeral SQLite file on local disk. When hosted on serverless platforms (like Vercel), disk writes are ephemeral and destroyed upon container recycling.
  4. **No Remote Multi-User Access:** Cannot be queried by remote clients or frontends without shared local filesystem access.

### 5. Whether Supabase is already partially configured?
- **No.**
- Grep search across the entire repository for `supabase` yields **0 matches**.
- No `supabase-py` package in `requirements.txt`.
- No `supabase/` migrations folder exists.
- No `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, or `SUPABASE_SERVICE_ROLE_KEY` in `.env` or `.env.example`.

### 6. Whether authentication already exists?
- **Yes, a local mock/RBAC system exists:**
  - Class: `AuthManager` in [src/auth/auth_manager.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/src/auth/auth_manager.py).
  - Seeded users: `admin`, `dr.sharma` (Cardiologist), `dr.patel` (Doctor), `tech.singh` (ECG Tech), `patient.raj` (Patient).
  - Roles defined: `HOSPITAL_ADMIN`, `CARDIOLOGIST`, `DOCTOR`, `ECG_TECHNICIAN`, `NURSE`, `RESEARCHER`, `PATIENT`.
  - Passwords hashed using PBKDF2-HMAC-SHA256 (100,000 rounds).
  - In [app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L48), `AUTH_MANAGER` governs UI access and sign-off permission (`report:sign_off`).
- **Limitation:** It is purely local/in-memory with hardcoded initial credentials, completely decoupled from Supabase Auth (`auth.users`).

### 7. What frontend framework is being used?
- **Primary:** **Streamlit** ([app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py)).
  - Runs on port 8501.
  - Interactive widgets for file upload, parameter selection, Plotly chart rendering, clinician sign-off forms, and PDF downloads.
- **Secondary:** **HTML5 + Vanilla JavaScript + Tailwind CSS (CDN) + Plotly.js (CDN)** ([public/index.html](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/public/index.html)).
  - Serves as the static web client for Vercel deployment, calling the FastAPI backend.

### 8. What backend framework is being used?
- **FastAPI / Uvicorn** ([api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py)).
  - Configured for serverless deployment on Vercel (`vercel.json`) and local server execution.
  - Provides REST API endpoints with Pydantic schema validation.

### 9. How the current report is passed from backend to frontend?
- **In Streamlit ([app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py)):**
  - Backend logic and frontend presentation are co-located in the same Python process.
  - `report_data` is a Python dictionary computed directly during the script run and bound to local variables and Streamlit component scopes.
  - PDF generation occurs in-memory, passing raw bytes directly to `st.download_button(data=doctor_pdf_bytes, ...)`.
- **In FastAPI / Web Portal ([api/index.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/api/index.py) + [public/index.html](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/public/index.html)):**
  - Client sends JSON via `fetch('/api/analyze', { method: 'POST', body: JSON.stringify(payload) })`.
  - Backend responds with JSON payload containing analysis, quality, CDS, and criteria metadata.
  - Web client parses JSON in JavaScript and updates DOM elements.
  - PDF is requested via `fetch('/api/report/pdf', { method: 'POST', ... })` which streams the binary blob to the browser.

### 10. Where the report disappears after page refresh?
The report vanishes at multiple levels due to the lack of an immutable persistence and history layer:

1. **Streamlit Execution Model ([app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L619-L626)):**
   - On page reload (F5) or browser session disconnection, Streamlit restarts script execution from Line 1.
   - Script variables are reinitialized:
     ```python
     signal_data: Optional[np.ndarray] = None
     sampling_rate: float = float(sampling_rate_setting)
     extracted_measurements = None
     ai_results = None
     ```
   - If no file is actively present in `st.file_uploader`, the analysis block does not execute.
   - The generated `report_data` dictionary and the generated PDF bytes cease to exist.
2. **SQLite Database Missing Snapshot Data:**
   - Although [app.py](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/app.py#L1402) calls `DB_MANAGER.save_report(...)`, it only inserts IDs and status into `clinical_reports`.
   - The structured JSON (`report_data`) is **never saved** to `clinical_reports`.
   - The PDF bytes are **never saved** to disk or storage.
   - `DB_MANAGER.get_report(report_id)` (even if called) cannot reconstruct the waveform, the criteria indicators, the medication alerts, or the detailed findings because that data was never persisted.
3. **Absence of a History Route / View:**
   - There is no "History" navigation tab in Streamlit or the Web Portal.
   - The UI provides no mechanism to query past reports by date, patient, finding, or doctor, nor any screen to render a stored report snapshot.
4. **Web Portal Client Memory ([public/index.html](file:///c:/Users/Lenovo/Downloads/ECG--main/ECG--main/public/index.html#L266)):**
   - In the web portal, `currentSignal`, `currentAnalysisId`, and DOM text contents reside exclusively in browser RAM.
   - On refresh, the browser loads a blank initial state with `loadSample('normal')`.

---

## 4. Current Data Flow Diagram

```text
CURRENT FLOW (EPHEMERAL / LOSS ON REFRESH)

User Browser
    │
    ▼ (Upload file / Select demo)
Streamlit app.py / Web Portal index.html
    │
    ├──► 1. Preprocessing & Quality Gate
    │       └─► quality_res (In-Memory)
    │
    ├──► 2. Feature Extraction (28 features)
    │       └─► beat_features (In-Memory)
    │
    ├──► 3. ML Inference (Random Forest)
    │       └─► ai_results (In-Memory)
    │
    ├──► 4. Clinical Context & CDS Engine
    │       └─► cds_rec (In-Memory)
    │
    ├──► 5. Medication Interaction Checker
    │       └─► med_safety_eval (In-Memory)
    │
    ├──► 6. report_generator.generate_structured_report()
    │       └─► report_data DICTIONARY (In-Memory Variable Only)
    │
    ├──► 7. pdf_generator.generate_doctor_report()
    │       └─► PDF Bytes (In-Memory Bytes Only)
    │
    ├──► 8. UI Display & Download Button
    │
    ▼ [USER PRESSES REFRESH (F5) OR CLOSES TAB]
💥 In-Memory Variables Destroyed
💥 SQLite clinical_reports has only empty stub (no snapshot JSON, no PDF)
💥 NO History UI exists to view or reopen past reports
💥 ALL REPORT DATA DISAPPEARS!
```

---

## 5. Target Architecture Flow (Supabase Persistence & History)

```text
TARGET FLOW (IMMUTABLE PERSISTENCE & HISTORY)

User Browser
    │
    ▼ (Upload ECG)
Backend / Streamlit Ingestion
    │
    ▼
Create `ecg_recordings` row (status: UPLOADED)
    │
    ▼
Execute ML Analysis & Signal Quality Gate
    │
    ▼
Save to `ai_analyses`, `ai_evidence`, `ecg_measurements`
    │
    ▼
Generate Immutable Report Snapshot:
    │   {
    │     "patient": {...},
    │     "ecg": {...},
    │     "measurements": {...},
    │     "quality": {...},
    │     "ai_analysis": {...},
    │     "cds": {...},
    │     "medication_safety": {...},
    │     "criteria_used": {...}
    │   }
    │
    ├──► 1. Insert into Supabase `reports` (report_number: 'ECG-2026-XXXXXX', report_data: JSONB)
    │
    ├──► 2. Generate PDF & Upload to Supabase Storage (`ecg-reports/reports/{patient_id}/{report_id}/report.pdf`)
    │
    ├──► 3. Update `reports.pdf_storage_path` with secure storage URI
    │
    ├──► 4. Log Audit Event in `audit_logs` (`REPORT_CREATED`)
    │
    ▼
Display Report in UI (Active View)
    │
    ▼ [USER REFRESHES (F5) OR REOPENS LATER]
Navigate to "History" Page
    │
    ▼
Query Supabase `reports` (Server-Side Filtered & RLS Protected)
    │
    ├──► Click "View Report" ──► Load Immutable JSON Snapshot (NO ML RERUN)
    │
    └──► Click "Download PDF" ──► Secure Signed URL from Supabase Storage
```

---

## 6. Audit Conclusion & Next Phase Readiness

- **Phase 0 Audit is Complete.**
- All 10 questions have been addressed with exact code line references.
- No modifications have been made to the ML pipeline or inference logic.
- We are prepared to proceed to **Phase 1 — Supabase Configuration** upon instruction.
