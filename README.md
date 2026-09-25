# ❤️ AI ECG Analyzer

**Universal AI-Powered Clinical ECG Abnormality Detection & Reporting System**

> ⚠️ **EDUCATIONAL AND SCIENTIFIC RESEARCH USE ONLY**  
> This software is intended strictly for educational and scientific research purposes. It is **NOT** a certified medical diagnostic device and must **NOT** be used to make clinical decisions, self-diagnose, or replace certified healthcare professional evaluation.

---

## 📌 Project Overview

**AI ECG Analyzer** is a bioengineering and computer science research platform that unifies clinical ECG report documents, scanned waveform images, and high-frequency digital recordings into an automated, zero-data-fabrication analysis pipeline:

1. **Universal Multi-Format Ingestion**:
   - **Clinical ECG Documents (`.pdf`)**: Extracts selectable text, patient demographics, and machine-printed clinical measurements (Heart Rate, PR interval, QRS duration, QT/QTc, P-QRS-T axes, and printed interpretation) using `pypdf` without data hallucination.
   - **Scanned ECG Images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`)**: Automatically isolates grid lines, cleans artifacts, extracts 1D voltage traces via column-wise center-of-mass analysis, and strictly validates continuity.
   - **Digital Signals (`.csv`, `.txt`, `.npy`)**: Auto-detects delimiters, isolates ECG voltage columns, normalizes sampling frequency, and extracts features.
   - **MIT-BIH Benchmark Demo Mode**: Instant access to verified expert physician annotations from the MIT-BIH Arrhythmia Database.
2. **Zero-Hallucination & Clinical Safety Gates**:
   - Strictly separates **Printed Machine Interpretation (Source Information)** from **AI Model Predictions**.
   - If an uploaded document or image does not contain an extractable 1D waveform trace of sufficient fidelity, the system presents the extracted printed parameters and halts AI waveform classification with an honest explanation—**never inventing signals or diagnoses**.
3. **Signal Quality & Preprocessing**:
   - Median filter baseline removal + 0.5–40 Hz Butterworth bandpass filtering.
   - Quantitative Signal-to-Noise Ratio (SNR), baseline drift detection, 50/60 Hz powerline interference check, and motion artifact categorization (`GOOD`, `ACCEPTABLE`, `POOR`).
4. **Beat Segmentation & 28-Feature Extraction**:
   - Adaptive R-peak detection with physiologically constrained refractory periods (≥300 ms).
   - Individual cardiac cycle extraction (-0.2s pre-R, +0.4s post-R) and 28 morphological, spectral, and R-R dynamic features.
5. **Machine Learning Abnormality Classification**:
   - Classifies **Normal Sinus Beats**, **Premature Ventricular Contractions (PVC)**, and **Other Ectopic Beats** using an authentic Random Forest classifier (100 trees, strict record-level patient split).
6. **Publication-Grade Multi-Format Reporting**:
   - **📄 Publication PDF Report**: Formatted clinical research report generated via ReportLab with embedded high-resolution waveform strips, parameter tables, and medical disclaimers.
   - **📊 Machine-Readable JSON (`.json`)**: Structured export ready for EMR or database integration.
   - **📝 Clinical Summary Text (`.txt`)**: Formatted plain-text summary.

---

## 🏗️ System Architecture

```text
                                  USER ECG UPLOAD
                     (PDF • JPG • JPEG • PNG • CSV • TXT • NPY)
                                       │
                                       ▼
                       [src/ecg_input/input_detector.py]
                           Detects File Modality & Format
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       [Digital Signal]                             [Report Image / PDF]
  [src/ecg_input/signal_loader.py]              [image_processor & pdf_processor]
  • Auto-detects delimiters (CSV/TXT)           • Text/OCR extraction (pypdf & regex)
  • Identifies voltage vs time cols             • Extracts patient info & machine measurements
  • Detects / prompts sampling rate             • Grid & trace isolation (cv2)
                │                                             │
                │                               ┌─────────────┴─────────────┐
                │                               ▼                           ▼
                │                    [Waveform Extractor]          [Printed Measurements Only]
                │                    • Trace skeletonization       • PR, QRS, QT, QTc, Axes, HR
                │                    • Calibration detection       • Explicitly labeled as
                │                    • Validation gate:             "Machine Interpretation"
                │                      (If low confidence ->         (Not AI diagnosis)
                │                       Prompt digital file)
                └───────────────────────┬───────────────────────────────────┘
                                        ▼
                             [Unified ECG Ingestion]
                   (Validated 1D voltage series or Structured Metadata)
                                        │
                                        ▼
                          [Signal Quality Assessment]
                       (GOOD / ACCEPTABLE / POOR Gatekeeper)
                                        │
                                        ▼
                       [Existing Preprocessing Pipeline]
                       (Median Filter + Butterworth Bandpass)
                                        │
                                        ▼
                         [Existing Detection & Features]
                     (R-Peaks, Beat Windows, 28 Measurements)
                                        │
                                        ▼
                         [Trained Random Forest Model]
                   (Normal / PVC / Other Class Probabilities)
                                        │
                                        ▼
                         [src/report/report_generator.py]
                          [src/report/pdf_generator.py]
             • Professional On-Screen Dashboard with Progress Steps
             • Downloadable Research PDF Report (ReportLab)
             • Downloadable JSON & Plain-Text Summaries
```

---

## 📊 Actual Machine Learning Performance

> **How to read this table.** These figures are a record of one evaluation run on
the partitions below, not a permanent claim. Re-run `python training/train_all.py`
to regenerate every number from scratch; the pipeline writes the metrics it
actually measured and will not fill in a value it did not compute. Any cell it
could not measure is written as `not measured`.

Models were trained and evaluated using a **strict record-level split** (no patient overlap):
- **Training Records**: `100`, `106`, `200`, `213` (10,152 heartbeats)
- **Unseen Testing Records**: `101`, `119`, `208` (6,807 heartbeats)

### Test Performance on Unseen Patients

| Metric | Random Forest (Primary) | Logistic Regression (Baseline) |
| :--- | :--- | :--- |
| **Accuracy** | **97.86%** | 95.21% |
| **Weighted F1-Score** | **97.92%** | 95.50% |
| **Normal Beat F1-Score** | **98.6%** (Support: 4,989) | 96.7% |
| **PVC Beat F1-Score** | **96.6%** (Support: 1,809) | 92.6% |
| **Macro F1-Score** | **65.07%** | 66.61% |

---

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **Signal Processing**: SciPy, NumPy, OpenCV (`cv2`)
- **Document & PDF Processing**: `pypdf`, `reportlab`, `pillow`
- **Machine Learning**: Scikit-learn, Joblib
- **Data Analysis**: Pandas
- **Visualization**: Plotly Express, Plotly Graph Objects, Matplotlib
- **Web Interface**: Streamlit
- **Testing**: Pytest (202 automated tests) + GitHub Actions CI
- **Optional**: PyTorch (1-D CNN), SHAP (attribution), Tesseract (scanned-report OCR)

---

## 🚀 Quick Start & Installation

### 1. Clone or Open the Repository
```bash
git clone https://github.com/ANIKETCHAND/ECG-.git
cd ECG-
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Run the full suite (202 tests) covering input detection, PDF parsing, image processing, waveform extraction, model prediction, calibration, OCR, and report generation:
```bash
python -m pytest -q            # full suite
python -m pytest tests/ -v     # verbose
```

### 5. Launch the Streamlit App
```bash
streamlit run scripts/app_streamlit_legacy.py
```
Open your browser at `http://localhost:8501`.

### 6. Launch the FastAPI Service (optional)
```bash
uvicorn api.index:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

### 7. Optional Capabilities (optional)
Core features need no extra packages. Install these to unlock the corresponding capability:
```bash
pip install -r requirements-optional.txt
```
| Capability | Package | Behaviour when absent |
| :--- | :--- | :--- |
| Shapley per-beat attribution | `shap` | Falls back to labelled baseline-deviation attribution |
| OCR of scanned reports | `pytesseract` + Tesseract engine | Returns `OCR_UNAVAILABLE` with instructions |
| Rasterising image-only PDFs | `pypdfium2` | Returns `RASTERISER_UNAVAILABLE` with instructions |
| Real 1-D CNN on raw waveforms | `torch` | `ECGConv1DClassifier.fit()` raises `PyTorchNotAvailable` |

**No capability is faked.** When a package is missing the system reports the capability as unavailable and names the package; it never substitutes a different algorithm under the same label.

---

## 🔐 Model Provenance & Training Data Integrity

The project's clinical tenet is **"NO RELIABLE INPUT = NO AI RESULT"**, and it
applies to training as strictly as it does to inference.

Earlier revisions of the training scripts silently generated `numpy.random`
samples whenever a PhysioNet dataset was absent and then registered the resulting
artifact under a clinical-sounding model id. Weights fitted to random numbers
carry no biological signal. That behaviour is now structurally impossible:

* Missing dataset → training **aborts** with `PlaceholderTrainingNotAllowed` and
  prints exactly how to fetch the real data.
* Explicit opt-in (`ECG_ALLOW_PLACEHOLDER_TRAINING=1` or `--allow-placeholder`) →
  training proceeds, but the artifact is stamped `SYNTHETIC_PLACEHOLDER` and
  `ModelRegistry.load_model()` **refuses to serve it**.
* Every registry entry carries a `provenance` block (`REAL_DATASET`,
  `SYNTHETIC_PLACEHOLDER`, `DEMO_FIXTURE`, `UNVERIFIED_LEGACY`) recording where
  its training data came from.
* External cross-dataset validation now requires a real external cohort. When
  none is present the report says `NOT_EVALUATED` and contains no numbers — the
  previous constant-offset "domain shift" figures are gone.

Inspect the current state of every model:
```bash
python -c "import json; d=json.load(open('models/registry/catalog.json'));\
[print(f\"{k:34s} {v['status']:22s} {v.get('provenance',{}).get('data_status')}\") for k,v in d.items()]"
```

> **Legacy artifacts.** Reports produced before this work was done contain figures
derived from the synthetic fallbacks. They are listed and invalidated in
`reports/experiments/INVALIDATED_BY_PROVENANCE_AUDIT.md`; do not cite numbers from
them.

---

## 🎯 Probability Calibration & Conformal Abstention

A Random Forest's `predict_proba` output is a vote frequency, not a probability.
Two post-hoc layers make the reported confidence honest:

1. **Calibration** — per-class one-vs-rest isotonic or Platt scaling, followed by
   renormalisation, so "PVC 0.80" means roughly four in five.
2. **Conformal abstention** — a Mondrian split-conformal predictor returning a
   *prediction set* with finite-sample coverage `1 - alpha`. When the set holds
   more than one label the model has not separated them, and the platform says so
   instead of emitting a confident single label.

Fit them with:
```bash
python training/fit_calibration.py --alpha 0.10
```

The fitter splits the held-out partition **by record** into a calibration half and
a disjoint evaluation half, then applies a deployment gate: if the calibration map
measurably makes the evaluation partition *worse*, the artifact is retained for
inspection but `deployment_recommended: false` is recorded and inference keeps
serving the raw probabilities. A calibration step that hurts is reported as such,
not shipped.

---

## 🔎 Explainability: SHAP and Honest Fallbacks

`src/evidence/shap_evidence.py` produces per-beat Shapley attributions when both
`shap` and a supported tree model are available, and otherwise returns
baseline-deviation attribution **labelled as such** (`method` field:
`shap_tree` vs `baseline_deviation_fallback`). The two answer different
questions — "what moved this model's output" versus "how does this beat differ
from the patient's other beats" — and the artifact never conflates them.

---

## 🖼️ Scanned Reports: OCR and Digitisation Accuracy

`src/ecg_input/ocr.py` reads printed text from scanned ECG reports, filling the
gap where a PDF has no text layer. Availability is reported explicitly
(`OCR_UNAVAILABLE`, `RASTERISER_UNAVAILABLE`, `EMPTY_TEXT`, `LOW_CONFIDENCE`).

Image-to-waveform extraction is measured rather than assumed:
```bash
python training/validate_digitization.py --duration 10 --heart-rate 75
```
This synthesises an ECG with known ground truth, renders it as a calibrated
25 mm/s, 10 mm/mV strip, runs the production extractor, and reports beat-timing
error in milliseconds and morphology correlation. It explicitly records
`amplitude_recovery: NOT_CALIBRATED`, because the extractor rescales the trace so
its peak equals 1.5 mV — so absolute voltages are not recoverable from an image.

---

## 🔄 Operational Capabilities

### Shadow mode — candidate vs production
```python
from ml.shadow import run_shadow_comparison
report = run_shadow_comparison(feature_matrix)   # candidate is never served
print(report.agreement_rate, report.recommendation)
```
Runs ``ECG-RF-2.0.0-candidate`` alongside production on the same beats and logs
the disagreements to the tamper-evident audit trail. Each model uses its own
scaler, placeholder candidates are refused outright, and the candidate's output
is never returned to a caller.

### Simulated real-time streaming
```python
from src.streaming.simulated_stream import run_streaming_demo
result = run_streaming_demo(duration_s=30, heart_rate_bpm=78)
print(result["summary"]["estimated_heart_rate_bpm"])   # 78.0
```
Classifies beats incrementally as windows arrive, with a refractory guard so
overlapping windows do not re-report the same beat. Replacing the window
generator with a BLE or serial source is the only change needed to go live.

### FHIR R4 export
```python
from src.report.fhir_export import to_fhir_bundle
bundle = to_fhir_bundle(structured_report)   # Patient + Observation* + DiagnosticReport
```
Only heart rate is given a LOINC coding, because that is the one mapping this
project can assert offline. Every interval is emitted without a fabricated code
and the bundle declares that terminology validation is pending.

### Medication x longitudinal QTc
```python
from src.longitudinal.medication_qtc_link import assess_qtc_medication_association
report = assess_qtc_medication_association(ecg_history, medications)
```
Joins the medication-safety and longitudinal modules: pairs serial QTc values with
medication start dates and reports whether a rise is *temporally consistent* with
starting a QT-prolonging agent. It reports association, never causation, and
returns `INSUFFICIENT_DATA` when either side of the start date is unobserved.

---

## 🔐 API Security Configuration

The serverless API is configurable by environment variable:

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `ECG_API_KEY` | unset | When set, data-bearing endpoints require the `X-API-Key` header. `/health` reports `auth: DISABLED` when unset so an unauthenticated deployment is visible. |
| `ECG_CORS_ORIGINS` | none | Comma-separated allow-list. No cross-origin access unless requested; `*` must be explicit. |
| `ECG_MAX_UPLOAD_MB` | 25 | Request body ceiling. |
| `ECG_MAX_SIGNAL_SAMPLES` | 5000000 | Maximum samples accepted per analysis. |

---

## 💡 Testing with Sample Files

The repository includes ready-to-use sample ECG files in `sample_ecgs/`:
- `sample_clinical_ecg_report.pdf`: A complete 12-lead clinical ECG report PDF with printed measurements.
- `normal_ecg_sample.csv`: Digital single-lead ECG recording showing Normal Sinus Rhythm.
- `pvc_arrhythmia_sample.csv`: Digital single-lead ECG recording demonstrating Premature Ventricular Contractions.

---

## 📁 Project Structure

```text
AI-ECG-Analyzer/
├── README.md                      # This guide
├── PROJECT_DOCUMENTATION.md       # Detailed multi-chapter technical documentation
├── requirements.txt               # Core dependencies
├── requirements-optional.txt      # torch / shap / pytesseract / pypdfium2
├── docker-compose.yml             # Containerised deployment
├── vercel.json                    # Serverless deployment configuration
│
├── .github/workflows/ci.yml       # CI: test suite + training-integrity guards
│
├── api/                           # Vercel serverless clinical API
│   ├── index.py                   # FastAPI endpoints (analyze, upload, report, review)
│   └── security.py                # API key, CORS allow-list, size limits
│
├── public/index.html              # Browser portal (Plotly telemetry + sign-off)
│
├── sample_ecgs/                   # Ready-to-test clinical sample files
│   ├── sample_clinical_ecg_report.pdf
│   ├── normal_ecg_sample.csv
│   └── pvc_arrhythmia_sample.csv
│
├── src/
│   ├── preprocessing.py           # Median filter, bandpass, Z-score normalization
│   ├── peak_detection.py          # Adaptive R-peak detection (refractory period)
│   ├── segmentation.py            # Beat window extraction (-0.2s to +0.4s)
│   ├── feature_extraction.py      # 28 time, frequency, and RR rhythm features
│   ├── label_mapping.py           # AAMI / 3-class standardized symbol mapping
│   ├── data_loader.py             # MIT-BIH loader via WFDB
│   │
│   ├── ecg_core/                  # Domain entities (ECGRecording, ECGAnalysisResult)
│   ├── ecg_input/                 # Multi-format ingestion
│   │   ├── input_detector.py      # Format & modality sniffing (magic bytes)
│   │   ├── csv_loader.py          # Digital: CSV/TXT/NPY/JSON
│   │   ├── wfdb_loader.py         # WFDB records
│   │   ├── edf_loader.py          # EDF+ / XML / DICOM waveform loaders
│   │   ├── ocr.py                 # Scanned-report OCR (pytesseract, optional)
│   │   ├── pdf_processor.py       # pypdf text layer, with OCR fallback
│   │   ├── image_processor.py     # OpenCV grid isolation & preprocessing
│   │   ├── waveform_extractor.py  # 1D trace extraction from images
│   │   ├── measurement_extractor.py # Printed-measurement regex parser
│   │   └── extraction_validation.py # Anti-hallucination validation gate
│   │
│   ├── quality/                   # Signal quality gate (rule-based, deterministic)
│   ├── safety/                    # Pre-inference quality gatekeeper
│   ├── measurements/              # PR / QRS / QT / QTc measurement engine
│   ├── inference/                 # Production inference engine
│   ├── ml/
│   │   ├── calibration.py         # Calibration + conformal abstention
│   │   ├── models/registry.py     # Lifecycle + provenance gates
│   │   ├── models/deep_1d_cnn.py  # Feature MLP + optional PyTorch 1D-CNN
│   │   └── inference/unified_api.py # Multi-task dispatch
│   ├── evidence/                  # Evidence engine, SHAP attribution, waveform snippets
│   ├── comparison/                # AI vs ECG-machine interpretation
│   ├── clinical_context/          # Patient context assembly
│   ├── clinical/                  # Recommendation engine
│   ├── medications/               # Interaction, allergy, contraindication checks
│   ├── longitudinal/              # Previous-ECG comparison & trend analysis
│   ├── review/                    # Clinician review & sign-off
│   ├── audit/                     # Audit trail
│   ├── auth/                      # Role-based access control
│   ├── database/                  # Supabase / local persistence
│   └── report/                    # Structured, JSON/TXT, and PDF report generation
│
├── training/                      # Training & validation pipelines
│   ├── guards.py                  # Data-integrity guards (no fabrication)
│   ├── real_features.py           # Grounded AF / ST / 12-lead / quality features
│   ├── train_arrhythmia.py        # Task A: beat arrhythmia
│   ├── train_af.py                # Task B: atrial fibrillation
│   ├── train_ptbxl.py             # Task C: 12-lead multi-label
│   ├── train_st.py                # Task D: ST-segment ischaemia
│   ├── train_quality.py           # Task E: quality gate
│   ├── train_waveform_cnn.py      # Task G: raw-waveform 1-D CNN (optional torch)
│   ├── fit_calibration.py         # Calibration + conformal fitting
│   ├── external_validation.py     # Cross-dataset validation (reports NOT_EVALUATED)
│   ├── validate_digitization.py   # Image digitisation round-trip accuracy
│   ├── splitting/                 # Patient-level splitter (leakage rejection)
│   └── evaluation/                # Clinical metrics
│
├── models/
│   ├── production/                # Active model artifacts + metadata
│   ├── candidate/                 # Candidate artifacts
│   ├── registry/catalog.json      # Lifecycle status + data provenance per model
│   └── calibration/calibration.json # Fitted calibration + conformal quantiles
│
├── supabase/migrations/           # Postgres schema, RLS policies, tenancy hardening
│
└── tests/                         # 202 tests across <40 modules
    ├── test_inference_engine.py
    ├── test_api_serverless.py
    ├── test_model_registry_and_training.py
    ├── test_evidence_engine.py
    ├── test_report_generation.py
    └── ... (ingestion, quality, comparison, medication safety, longitudinal,
              persistence, auth, audit, failure modes)
```

## 🚀 Deployment Guide

### Option 1: Vercel Serverless Deployment (Production Edge API & Clinical Web Portal)
This project is pre-configured for **instant serverless deployment on Vercel**:
- **Frontend Portal**: `public/index.html` (Interactive ECG telemetry with Plotly.js, sample waveform tests, safety gatekeeper, and physician sign-off).
- **Backend API**: `api/index.py` (FastAPI serverless microservice executing signal quality gating, inference, and deterministic measurements).

**Steps to Deploy to Vercel:**
1. Push your repository to GitHub: `https://github.com/ANIKETCHAND/ECG-`
2. Go to [vercel.com](https://vercel.com) and log in.
3. Click **"Add New..."** ➔ **"Project"**.
4. Import your `ANIKETCHAND/ECG-` repository.
5. In the project configuration:
   - **Framework Preset**: Leave as *Other* (Vercel automatically detects `vercel.json` and `api/index.py`).
   - **Root Directory**: `./`
6. Click **Deploy**. Vercel will build and provision your edge application with an automatic `*.vercel.app` domain.

---

### Option 2: Streamlit Community Cloud (Full Python Dashboard)
Streamlit requires persistent WebSockets and a stateful Python process:
1. Navigate to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account and select repository: `ANIKETCHAND/ECG-`.
3. Set **Main file path** to `scripts/app_streamlit_legacy.py`.
4. Click **Deploy**.

---

## ⚠️ Medical Disclaimer

This application is strictly for **educational and scientific research purposes**. It has not been approved by the US Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory body. It must **not** be used for clinical decision-making or medical diagnosis.