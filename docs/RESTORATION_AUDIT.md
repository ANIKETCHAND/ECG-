# ECG GUARDIAN — PHASE 0 RESTORATION AUDIT REPORT

**Date:** 2026-09-25  
**Version:** Audit Baseline v1.0.0  
**Repository:** `ANIKETCHAND/ECG-` (`e:\CODE\AI-ECG-Analyzer`)  
**Compliance Standards:** CDSCO MDR 2017 & IEC 62304 / ISO 13485 (Clinical Decision Support Architecture)

---

## 1. Executive Summary

This audit assesses the state of the ECG Guardian clinical decision-support platform following recent Vercel serverless deployment attempts. The audit establishes a comprehensive freeze of the repository across all components: frontend interfaces, backend endpoints, ML pipelines, serialization artifacts, database/persistence systems, and test coverage.

### Key Audit Findings:
1. **Core Unit/System Tests Are Fully Passing:** 198/198 pytest tests pass, and 7/7 serverless API endpoint tests pass. The internal algorithmic components (preprocessing, R-peak detection, Pan-Tompkins filtering, signal quality gatekeeper, medication safety, and clinical decision support) remain mathematically and clinically intact.
2. **Current Model Artifact Degradation:** The serialized primary model artifact in `models/classifier.pkl` exhibits poor generalization on unseen patient test records (`accuracy: 67.58%`, `PVC recall: 0.0%`, `Other recall: 0.0%`). An unweighted Random Forest retrained on the exact same train/test split achieves **97.69%**, and HistGradientBoosting achieves **97.83%**. The deployed `.pkl` file suffered degradation during hyperparameter modifications prior to deployment.
3. **Frontend Data Synchronization Gaps in Pure Web Stack (`public/index.html`):** While the Streamlit application (`app.py`) historically derived all charts dynamically, the standalone web interface created for Vercel (`public/index.html`) contains several hardcoded/static fallbacks (static class probabilities `[72.6, 7.9, 19.5]`, synthetic beat overlay loops, static beat feature table values, and hardcoded history table rows) rather than dynamically binding to the authoritative `/api/analyze` response object.
4. **Dual Entry Point Divergence:** The repository currently has two distinct application frontends:
   - Streamlit Platform (`app.py`, 1,626 lines) running locally on port 8501.
   - Serverless FastAPI + Static Web Portal (`api/index.py` + `public/index.html`) deployed to Vercel.
5. **Class Imbalance in Benchmark Dataset:** The MIT-BIH dataset split has severe class imbalance for the "Other" class (only 91 train beats vs. 9 test beats), causing models to struggle on minority class recall without cost-sensitive learning or focal loss.

---

## 2. Current Architecture Overview

```
                                      ┌──────────────────────────────────────────────┐
                                      │              USER / CLINICIAN                │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                             ┌───────────────────────────────┴───────────────────────────────┐
                             │                                                               │
                             ▼                                                               ▼
             ┌───────────────────────────────┐                               ┌───────────────────────────────┐
             │       LOCAL STREAMLIT         │                               │     VERCEL SERVERLESS WEB     │
             │   app.py (Port 8501)          │                               │   public/index.html (Port 80) │
             └───────────────┬───────────────┘                               └───────────────┬───────────────┘
                             │                                                               │
                             │                                                               ▼
                             │                                               ┌───────────────────────────────┐
                             │                                               │       FASTAPI BACKEND         │
                             │                                               │   api/index.py (/api/*)       │
                             │                                               └───────────────┬───────────────┘
                             │                                                               │
                             └───────────────────────────────┬───────────────────────────────┘
                                                             │
                                                             ▼
                                     ┌───────────────────────────────────────────────┐
                                     │            ECG GUARDIAN PIPELINE              │
                                     ├───────────────────────────────────────────────┤
                                     │ 1. Ingestion Engine (CSV, WFDB, JSON, PDF)    │
                                     │ 2. Signal Quality Gate (SNR, drift, noise)    │
                                     │ 3. Preprocessing (Bandpass 0.5-40Hz, detrend) │
                                     │ 4. R-Peak Detection (Pan-Tompkins detector)   │
                                     │ 5. Beat Segmentation (-200ms to +400ms)       │
                                     │ 6. 28 Extracted Morphological Features        │
                                     │ 7. ML Arrhythmia Classifier (RF / Scaler)     │
                                     │ 8. Deterministic Measurements (QRS, QT/QTc)   │
                                     │ 9. Multimodal Medication Safety Check         │
                                     │ 10. Clinical Decision Support (AHA/ACC)       │
                                     │ 11. Dual PDF Generator (Doctor & Patient)     │
                                     └───────────────────────┬───────────────────────┘
                                                             │
                             ┌───────────────────────────────┴───────────────────────────────┐
                             │                                                               │
                             ▼                                                               ▼
             ┌───────────────────────────────┐                               ┌───────────────────────────────┐
             │        LOCAL STORAGE          │                               │       SUPABASE CLOUD          │
             │ - data/hospital_clinical.db   │                               │ - PostgreSQL Database         │
             │ - data/audit_trail.db         │                               │ - 'ecg-reports' Storage Bucket│
             │ - reports/*.pdf, *.json       │                               │ - RLS Security Policies       │
             └───────────────────────────────┘                               └───────────────────────────────┘
```

---

## 3. Component Inventory & Audit Details

### 3.1 Frontend Entry Points
| Entry Point | Technology | State | Audit Notes |
|---|---|---|---|
| `app.py` | Streamlit 1.28+ | Functional locally | The original full hospital platform. Uses Plotly, session state, patient registration, and local SQLite/Supabase. Requires full dependencies (`requirements-streamlit.txt`). |
| `public/index.html` | Pure HTML/CSS/Plotly | Deployed on Vercel | Standalone pure web implementation. Reconstructed to match original Streamlit styling. Contains static fallbacks that need dynamic data binding to `/api/analyze`. |
| `public/design-preview.html` | Compiled Tailwind/React | Static asset | Backup mockup provided by user for UI reference. |

### 3.2 Backend Entry Points & API Endpoints
| Endpoint | Method | Function in `api/index.py` | State | Audit Notes |
|---|---|---|---|---|
| `GET /api/health` | GET | `health_check` | Functional | Returns status `HEALTHY`, version `1.0.0`, timestamp. |
| `GET /api/sample` | GET | `get_sample_data` | Functional | Returns realistic ECG samples (`normal`, `pvc`, `noisy`) with sampling rate 360Hz. |
| `POST /api/analyze` | POST | `analyze_ecg` | Functional | Orchestrates quality gate, inference, measurements, CDS, medication safety, and report persistence. |
| `POST /api/report/pdf` | POST | `generate_pdf_endpoint` | Functional | Runs analysis and returns binary PDF (`application/pdf`). |
| `POST /api/review` | POST | `record_clinician_review` | Functional | Records clinician sign-off with cryptographic hash and persistence. |
| `GET /api/reports` | GET | `list_reports_endpoint` | Functional | Lists persistent reports with search/status filters. |
| `GET /api/reports/{id}` | GET | `get_report_endpoint` | Functional | Retrieves immutable snapshot without re-running ML inference. |
| `GET /api/reports/{id}/pdf` | GET | `get_report_pdf_endpoint` | Functional | Downloads historical report PDF. |

### 3.3 Core Processing Pipeline Flow
```
Uploaded ECG Raw File
  ↓
[ecg_input_engine.py / parseEcgContent()]
  Extracts 1D signal array & sampling rate (fs)
  ↓
[signal_quality_gate.py]
  Evaluates SNR (dB), baseline drift, 50/60Hz powerline harmonics, motion artifact
  HALT if UNUSABLE (Patient Safety Guarantee)
  ↓
[preprocessing.py]
  Bandpass filtering (0.5 - 40 Hz), median filter baseline removal, z-score normalization
  ↓
[peak_detection.py]
  Pan-Tompkins gradient & refractory peak detector → detected R-peaks
  ↓
[segmentation.py]
  Extracts heartbeat segments in [-200ms, +400ms] window centered at each R-peak
  ↓
[feature_extraction.py]
  Computes 28 morphological, spectral, and temporal features per beat
  ↓
[inference_engine.py]
  StandardScaler transform → Classifier predict & predict_proba
  ↓
[measurement_engine.py]
  Deterministic Heart Rate, Mean R-R, QRS duration, QT, QTc (Bazett / Fridericia)
  ↓
[interaction_checker.py]
  Multimodal drug-drug, drug-allergy, QT-prolongation, electrolyte safety crosschecks
  ↓
[recommendation_engine.py]
  AHA/ACC/ESC clinical decision support guidance & urgency tiering
  ↓
[report_persistence_service.py]
  Generates immutable Report ID (ECG-YYYY-XXXXXX), dual-writes to Supabase/SQLite
  ↓
AUTHORITATIVE CLINICAL ANALYSIS OBJECT
```

---

## 4. Model & Dataset Audit

### 4.1 Deployed Artifacts in `models/`
- `models/classifier.pkl`: Serialized `RandomForestClassifier`.
  - **Audit Evaluation:** Accuracy = **67.58%**, Macro F1 = **26.89%**.
  - **Root Cause:** Artifact suffers from severe under-prediction of PVC and Other classes due to suboptimal regularization parameters (`max_depth=16, min_samples_leaf=2, class_weight='balanced'`).
- `models/baseline_classifier.pkl`: Serialized `LogisticRegression`.
  - **Audit Evaluation:** Accuracy = **97.27%**, Macro F1 = **68.27%**.
- `models/scaler.pkl`: Serialized `StandardScaler` fitted on 28 features.
- `models/metadata.json`: Feature definitions, classes (`Normal`, `Other`, `PVC`), and split metadata.

### 4.2 Available Datasets
| Dataset | Location | Records | Total Beats | Class Distribution | Split Strategy |
|---|---|---|---|---|---|
| **MIT-BIH Arrhythmia (Train)** | `data/processed/train_dataset.csv` | 100, 106, 200, 213 | 10,152 | Normal: 8,130 (80.1%)<br>PVC: 1,931 (19.0%)<br>Other: 91 (0.9%) | Patient/Record-level separation (zero leakage) |
| **MIT-BIH Arrhythmia (Test)** | `data/processed/test_dataset.csv` | 101, 119, 208 | 6,807 | Normal: 4,989 (73.3%)<br>PVC: 1,809 (26.6%)<br>Other: 9 (0.1%) | Patient/Record-level separation (unseen test subjects) |
| **Raw PhysioNet Records** | `data/raw/` & `data/datasets/mit_bih_arrhythmia/raw/` | 100, 101, 106, 119, 200, 208, 213 | 7 records | `.dat`, `.hea`, `.atr` raw binary WFDB files |

### 4.3 Pilot Retraining Benchmark (Zero-Leakage Test Evaluation)
To identify the baseline capability of the existing feature set before Phase 10:
- Baseline Logistic Regression: **97.27%** test accuracy
- Fresh Unweighted Random Forest (100 trees): **97.69%** test accuracy (Normal F1: 98.45%, PVC F1: 96.60%)
- HistGradientBoostingClassifier: **97.83%** test accuracy (Normal F1: 98.54%, PVC F1: 96.39%)
- *Observation on Target >= 99%:* The "Other" class in the current test set contains only 9 samples. Achieving >=99% test accuracy legitimately requires advanced feature engineering, threshold optimization, class balancing, and potentially multi-lead / augmentation techniques without any data leakage.

---

## 5. Identified Gaps, Regressions & Action Items

### 5.1 Hardcoded / Static Fallbacks in Web Frontend (`public/index.html`)
| Element | Current State | Required State |
|---|---|---|
| AI Probabilities Chart | Displays hardcoded `[72.6, 7.9, 19.5]` on initial render | Must dynamically bind to `ai_classification.probabilities` returned by `/api/analyze` |
| Beat Segmentation Overlay | Uses synthetic mathematical cosine wave generator | Must slice and display actual beat segments from `signal` around detected `r_peaks` |
| Beat Feature Table | Static HTML table (`2.421 mV`, `0.352 s`, `5.160 mV`, etc.) | Must populate dynamically from calculated morphological features |
| Signal Quality Table | Static values (`-9.7 dB`, `Clean`, `Suppressed`, `Minimal`) | Must populate from `signal_quality` metadata in analysis response |
| Patient Context Considered | Hardcoded text (`Age (15y) • Sex (F)...`) | Must reflect actual registered patient age, sex, vitals, labs |
| ECG History Archive | Static HTML table rows in `tbody` | Must fetch live reports from `GET /api/reports` |
| Patient Portal View | Static welcome text for `raj` | Must adapt dynamically to registered patient data |

### 5.2 Serverless Environment Limitations
- **Read-Only Filesystem:** On Vercel, the default filesystem is read-only except `/tmp`.
  - *Current Status:* `/tmp` fallbacks have already been implemented for `DatabaseManager`, `AuditLogger`, and `ReportPersistenceService`.
- **Bundle Size:** Vercel functions have a strict 500MB uncompressed limit.
  - *Current Status:* Requirements split into slim `requirements.txt` (364MB) for Vercel and full `requirements-streamlit.txt` for local development.

### 5.3 Last Known Good Implementation
- **Git Commit:** `79ab9b0` (*"feat: complete ECG Guardian hospital clinical platform, multimodal ML system, criteria indicators, and Supabase persistence"*).
  - All 198 pytest tests passed.
  - Full Streamlit UI functioned cleanly with Supabase persistence.
  - All subsequent changes were deployment-driven adaptations for Vercel.

---

## 6. Phase 0 Audit Conclusion & Next Phase Readiness

The repository is frozen and documented. No code modifications were made during this audit.

### Prerequisites for Phase 1:
1. Create a safe backup tag/branch: `git branch backup/pre-restoration-baseline`.
2. Confirm environment variable template (`.env.example`) completeness across Development, Staging, and Production.
3. Preserve all raw dataset files, training scripts, and model registry artifacts.

**Phase 0 is complete. Ready for Phase 1 upon confirmation.**
