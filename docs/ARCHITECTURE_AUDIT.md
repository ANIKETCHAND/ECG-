# Comprehensive Architecture & Engineering Audit
**Document ID:** AUD-ECG-2026-001  
**Author:** AI ECG Platform Engineering & Regulatory Team  
**Reference Standards:** Medical Devices Rules (MDR) 2017 (CDSCO India), IEC 62304:2006/Amd 1:2015, ISO 14971:2019, IMDRF SaMD N12  
**Baseline Date:** September 2026  
**Status:** Approved Engineering Baseline Audit  

---

## Executive Summary

This document establishes the technical, clinical, and regulatory baseline of the **AI ECG Analyzer** codebase prior to its architectural transition into a hospital-oriented, medical-device-ready platform. 

> [!CAUTION]
> **REGULATORY NOTICE:** This software is currently an unvalidated research prototype. It is **NOT** approved by CDSCO (India), FDA (USA), EMA/CE-mark (EU), or any statutory medical device regulatory agency. It is **NOT** certified for autonomous clinical diagnosis or clinical decision-making.

---

## 1. Current Architecture

The current repository follows a monolithic, single-tier script structure built primarily around Streamlit (`app.py`), directly orchestrating:
- Signal loading and file format sniffers (`src/ecg_input/`)
- Preprocessing and quality scoring (`src/preprocessing.py`, `src/signal_quality.py`)
- Beat detection and segmentation (`src/peak_detection.py`, `src/segmentation.py`)
- Morphological and rhythm feature extraction (`src/feature_extraction.py`)
- Synchronous model loading and inference (`src/prediction.py`)
- Direct in-memory report generation (`src/report/`)
- Front-end rendering (`app.py`, `src/visualization.py`)

### Architectural Block Diagram (As-Is):
```text
[ Browser / User ]
        │ (HTTP Port 8501)
        ▼
 [ Streamlit Server (Python Runtime) ]
   ├── app.py (UI Layout, Session State, Orchestration)
   ├── src/ecg_input/ (Format Detection & CSV/PDF/Image loaders)
   ├── src/preprocessing.py (Median filter & Butterworth Bandpass)
   ├── src/peak_detection.py (SciPy find_peaks)
   ├── src/segmentation.py (Window extraction)
   ├── src/feature_extraction.py (28 manual features)
   ├── src/prediction.py (Joblib load of models/classifier.pkl)
   ├── src/report/ (ReportLab PDF & JSON generator)
   └── data/raw/ (Local WFDB files)
```

---

## 2. Current Data Flow

1. **Ingestion**: User uploads a file via Streamlit file uploader, or selects an MIT-BIH demo record.
2. **Detection**: `input_detector.py` checks file suffix and initial magic bytes to categorize modality (`DIGITAL_SIGNAL`, `REPORT_PDF`, `REPORT_IMAGE`).
3. **Signal Parsing**:
   - Digital CSV/TXT/NPY: `signal_loader.py` reads first column as voltage.
   - PDF: `pdf_processor.py` extracts text and looks for embedded raster images.
   - Image: `image_processor.py` suppresses red/pink grid lines, `waveform_extractor.py` traces center-of-mass dark pixels.
4. **Validation**: `extraction_validation.py` checks duration ($\ge 1.5$s) and variance.
5. **Noise Filtering**: `preprocess_pipeline` applies:
   - Median filter baseline correction (200 ms and 600 ms windows).
   - 0.5–40 Hz 3rd-order Butterworth bandpass filter.
   - Z-score normalization: $\hat{s} = \frac{s - \mu}{\sigma}$.
6. **Quality Assessment**: `calculate_signal_quality` computes SNR, baseline drift, 50/60 Hz powerline interference, and outlier ratios.
7. **R-Peak Detection**: `detect_r_peaks` runs `scipy.signal.find_peaks` with height and distance thresholds (min distance = $0.3 \times f_s$).
8. **Beat Segmentation**: Extracts $[-0.2\text{s}, +0.4\text{s}]$ window around each R-peak.
9. **Feature Extraction**: Computes 28 time, morphology, frequency, and RR interval metrics per beat.
10. **Inference**: Scikit-Learn `RandomForestClassifier` predicts class probabilities per beat; majority voting sets window-level pattern.
11. **Reporting**: Assembled into memory and downloaded via Streamlit buttons.

---

## 3. Current Machine Learning Model

- **Architecture**: `RandomForestClassifier` (Scikit-Learn).
- **Ensemble Size**: 100 decision trees (`n_estimators=100`).
- **Splitting Criterion**: Gini impurity.
- **Max Depth**: Unlimited (`None`).
- **Baseline Model**: L2-regularized `LogisticRegression`.
- **Serialization**: Python Pickle via `joblib` (`models/classifier.pkl`, `models/scaler.pkl`).
- **Artifact Coupling**: The pickled models depend directly on the specific Python version (Python 3.14.6) and Scikit-Learn version installed in the host environment.

---

## 4. Current Dataset

- **Primary Source**: MIT-BIH Arrhythmia Database (PhysioNet).
- **Lead Evaluated**: Single lead, Modified Lead II (MLII).
- **Sampling Frequency**: Native 360 Hz.
- **Partitioning**: Strictly partitioned at the patient/record level (no patient leakage):
  - **Train Set (4 records, 10,152 beats)**: Record 100, Record 106, Record 200, Record 213.
  - **Test Set (3 records, 6,807 beats)**: Record 101, Record 119, Record 208.
- **Total Unique Patients Represented**: 7 patients.
- **Audit Finding**: While patient leakage between train and test sets was strictly prevented, 7 patients from an American ambulatory ECG dataset from 1980 represents an extremely small sample. It cannot represent global hospital patient demographics, pediatric populations, or patients with acute ischemic syndromes.

---

## 5. Current Classes

The model classifies three mutually exclusive categories mapped from AAMI EC57 standards:
1. **Normal (`Normal`)**: Normal sinus rhythm beats (`N`, `L`, `R`).
2. **PVC (`PVC`)**: Premature Ventricular Contraction / Ventricular Ectopy (`V`, `E`).
3. **Other (`Other`)**: Supraventricular ectopic, atrial premature, paced, or fusion beats (`A`, `a`, `J`, `S`, `F`, `f`, `j`).

> [!WARNING]
> **CRITICAL CLINICAL BOUNDARY:** The model **DOES NOT** detect:
> - Atrial Fibrillation (AFib)
> - ST-Segment Elevation Myocardial Infarction (STEMI / Heart Attack)
> - Non-ST Elevation Myocardial Infarction (NSTEMI)
> - Left/Right Bundle Branch Block (LBBB / RBBB) as independent diagnoses
> - Long QT Syndrome
> - Brugada Syndrome
> - Hyperkalemia or electrolyte disorders
> 
> Claiming or implying that this software detects "heart disease" or "cardiac abnormalities generally" is a severe regulatory and clinical safety violation.

---

## 6. Current Preprocessing Pipeline

- **Baseline Wander Correction**: Two-stage cascading median filter (200 ms kernel removes P-QRS-T complexes; 600 ms kernel smooths baseline drift; difference subtracted from raw signal).
- **Bandpass Filter**: 3rd-order Butterworth bandpass filter ($0.5\text{ Hz} - 40.0\text{ Hz}$) using forward-backward zero-phase filtering (`scipy.signal.filtfilt`).
- **Normalization**: Z-score standardization.
- **Limitation**: `filtfilt` is a non-causal batch filter that requires the entire signal window in memory. It is suitable for batch offline processing but requires causal restructuring if adapted to real-time telemetry streaming.

---

## 7. Current Feature Extraction

28 hand-crafted engineered features:
- **Time Domain (11)**: Mean, standard deviation, min, max, range, median, signal energy, RMS amplitude, mean absolute value (MAV), SNR, zero-crossing rate.
- **Morphology (7)**: Autocorrelation first peak, R-peak amplitude, P-wave amplitude estimate, T-wave amplitude estimate, peak-to-peak amplitude, maximum derivative slope, QRS width in samples.
- **Frequency Domain (7)**: Total spectral power, low-frequency power (0–4 Hz), high-frequency power (4–15 Hz), very-high-frequency power (15–40 Hz), dominant frequency, peak spectral power, spectral entropy.
- **R-R Interval Dynamics (3)**: Pre-RR interval, Post-RR interval, Local RR ratio ($\frac{\text{pre-RR}}{\text{local mean RR}}$).
- **Audit Finding**: Features heavily depend on accurate R-peak detection. If R-peak detection fails or shifts due to bundle branch block or tall T-waves, all 28 features propagate significant error.

---

## 8. Current Performance

Evaluated strictly on the 6,807 unseen test beats:

| Metric | Random Forest (Current) | Logistic Regression (Baseline) |
| :--- | :--- | :--- |
| **Overall Accuracy** | 97.86% | 95.21% |
| **Weighted F1-Score** | 97.92% | 95.50% |
| **Normal Precision / Recall / F1** | 99.88% / 97.29% / **98.57%** (Support: 4,989) | 96.6% / 96.8% / 96.7% |
| **PVC Precision / Recall / F1** | 93.58% / 99.89% / **96.63%** (Support: 1,809) | 91.8% / 93.4% / 92.6% |
| **Other Precision / Recall / F1** | **0.00% / 0.00% / 0.00%** (Support: 9) | 0.00% / 0.00% / 0.00% |
| **Macro F1-Score** | **65.07%** | 66.61% |

### Critical Performance Audit Finding:
The class `Other` has a test sample size of only 9 beats in the test records, on which the model scored **0.0% precision and 0.0% recall**. The high overall accuracy of 97.86% is driven entirely by the abundance of `Normal` and `PVC` beats. 
In a regulatory submission (e.g. CDSCO Form MD-40 or FDA 510(k)), claiming support for class `Other` would be rejected due to zero clinical sensitivity. **The model must be documented as validated exclusively for Normal Sinus Rhythm vs. Ventricular Ectopy (PVC), with `Other` explicitly flagged as unvalidated.**

---

## 9. Current Limitations

1. **Single-Lead Limitation**: Designed and trained strictly on Lead II representations. Cannot analyze 12-lead spatial vectors, precordial leads (V1–V6), or limb lead vectors.
2. **Fixed Sampling Rate Expectation**: Model features were extracted at 360 Hz. Digital signals at 250 Hz or 500 Hz must be resampled, introducing interpolation variance.
3. **No Batch Job Queue**: Synchronous execution blocks the Streamlit thread.
4. **No Multi-Tenancy**: Single-user desktop mode only.
5. **No Long-Term Persistence**: Analysis results are lost when browser refreshes unless manually downloaded.

---

## 10. Current Input Formats

- **Digital**: CSV, TXT, NPY. Delimiter autodetection is implemented.
- **Documents**: PDF (extracts text and embedded images via `pypdf`).
- **Images**: Scanned JPG/PNG (OpenCV thresholding).
- **Missing Clinical Standards**: Does **NOT** yet parse native DICOM Waveform (SOP Class 1.2.840.10008.5.1.4.1.1.9.1.1), HL7 aECG (FDA XML standard), or European Data Format (EDF/EDF+).

---

## 11. Current Reporting System

- Generates:
  1. Multi-page PDF via ReportLab with embedded matplotlib waveforms.
  2. Structured JSON dump.
  3. Plain-text summary.
- **Limitation**: Reports are generated client-side upon button click. There is no server-side archival, no cryptographically signed hash, no digital signature, and no clinician review sign-off workflow.

---

## 12. Current Security & Patient Privacy

- **Authentication**: None. Anyone who accesses the port can view or run analyses.
- **Authorization**: No role-based access control (RBAC).
- **Session Management**: Native Streamlit memory.
- **Encryption in Transit**: Plain HTTP (no forced TLS/HTTPS).
- **Encryption at Rest**: Uploaded files are held in temporary system RAM or unencrypted filesystem.
- **Audit Logging**: No audit trails exist for user logins, record views, or report generation.
- **Data Privacy**: No HIPAA/DISHA compliance features, no de-identification pipeline, no configurable retention policy.

---

## 13. Current Patient Data Handling

- No relational schema or patient registry.
- Patient names and demographics are parsed from PDF text if present, but never validated against an Enterprise Master Patient Index (EMPI) or hospital Electronic Medical Record (EMR).

---

## 14. Current Testing Suite

- 43 automated unit tests in `tests/` covering:
  - Signal preprocessing and normalization
  - R-peak detection and refractory period validation
  - Beat segmentation
  - 28-feature extraction
  - Inference edge cases (NaNs, empty signals)
  - Input format detection
  - PDF measurement extraction
  - Waveform extraction validation gate
  - Report serializers (PDF, JSON, TXT)
- **Gap**: No security tests, no authentication tests, no database integrity tests, no DICOM/HL7 tests, no API integration tests, and no stress/load testing.

---

## 15. Current Deployment Architecture

- Local execution via `python -m streamlit run app.py --server.port 8501`.
- No Docker containerization, no reverse proxy, no database service, no background task workers (Celery/Redis), no healthcheck endpoints, and no automated CI/CD pipeline.

---

## 16. Comprehensive Gap Analysis: College Prototype vs. Hospital Medical Software

| Engineering & Regulatory Domain | Current Prototype State | Target Hospital Platform (MDR 2017 / IEC 62304 / ISO 14971) | Gap Severity |
| :--- | :--- | :--- | :--- |
| **Regulatory Standing** | Academic project; claims educational use only | Designed under CDSCO MDR 2017 SaMD principles, ready for future clinical trial protocols | **CRITICAL** |
| **System Architecture** | Monolithic Streamlit application | Multi-tier microservices (FastAPI backend + PostgreSQL + Redis queue + Clinical Web UI) | **HIGH** |
| **Data Model** | Raw numpy arrays and ad-hoc dictionaries | Unified, immutable `ECGRecording` schema with metadata, device registry, and audit hashes | **HIGH** |
| **Clinical Interoperability** | Basic CSV, TXT, PDF, image parsing | Native DICOM Waveform, HL7 aECG, EDF+, and FHIR DiagnosticReport integration | **HIGH** |
| **Signal Quality Gate** | Basic SNR and wander metric | Strict, mandatory fail-safe quality gatekeeper: `POOR` or `UNUSABLE` $\rightarrow$ **NO AI RESULT** | **CRITICAL** |
| **Model Independence** | Coupled to Streamlit memory | Decoupled ML Inference Microservice with explicit model registry and versioning | **HIGH** |
| **Model Card & Transparency** | JSON metadata file only | Formal Regulatory Model Cards (`MODEL_CARD.md`) with explicit operational boundaries | **MEDIUM** |
| **Training & Leakage** | Patient-level split on 7 records | Multi-dataset registry (MIT-BIH + PTB-XL), patient-level verification, CI/CD retraining locks | **HIGH** |
| **Safety & Anti-Hallucination** | Validation heuristics in input parser | System-wide hard rule: `NO RELIABLE INPUT = NO RESULT`; strictly zero synthetic signals | **CRITICAL** |
| **Clinician Interaction** | Static display of predictions | Mandatory separation of AI output and Clinician Review with agreement/disagreement sign-off | **CRITICAL** |
| **User Access & RBAC** | Open public access | Secure JWT authentication with strict roles (Technician, Doctor, Cardiologist, Admin, Researcher) | **CRITICAL** |
| **Security & Privacy** | Plaintext, no HTTPS, no audit logs | OWASP Top 10 compliance, TLS 1.3, encrypted storage, immutable tamper-evident audit logs | **CRITICAL** |
| **Software Lifecycle (IEC 62304)** | Ad-hoc git commits | Traceable lifecycle documentation (Requirements $\rightarrow$ Architecture $\rightarrow$ Verification $\rightarrow$ Validation) | **HIGH** |
| **Risk Management (ISO 14971)** | Implicit code checks | Formal Hazard Analysis, Risk Register, Risk Mitigations, and Verification Evidence | **CRITICAL** |
| **Deployment & Containers** | Manual local python command | Hardened Docker compose with PostgreSQL, Redis, API, and Nginx reverse proxy | **MEDIUM** |

---

## 17. Conclusion & Roadmap

The existing codebase contains robust, mathematically sound algorithms for signal preprocessing, R-peak detection, and feature extraction that can and should be preserved. However, the system architecture, data models, security, and governance must be rebuilt to meet medical-device software engineering standards.

Phase 1 establishes this baseline. The subsequent phases will build the modular foundation without compromising clinical honesty or patient safety.
