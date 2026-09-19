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
- **Testing**: Pytest (43 automated tests)

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
Verify that all 43 tests pass across input detection, PDF parsing, image processing, waveform extraction, model prediction, and report generation:
```bash
python -m pytest tests/ -v
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

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
├── app.py                         # Upgraded interactive Streamlit dashboard
├── requirements.txt               # Dependencies including pypdf, reportlab, opencv
├── README.md                      # Comprehensive project guide
├── PROJECT_DOCUMENTATION.md      # Detailed 16-chapter technical documentation
│
├── sample_ecgs/                   # Ready-to-test clinical sample files
│   ├── sample_clinical_ecg_report.pdf
│   ├── normal_ecg_sample.csv
│   └── pvc_arrhythmia_sample.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # MIT-BIH record loader via WFDB
│   ├── preprocessing.py          # Median filter, bandpass, Z-score normalization
│   ├── signal_quality.py         # Quantitative SNR and artifact assessment
│   ├── peak_detection.py         # Adaptive R-peak detection (refractory period)
│   ├── segmentation.py           # Beat window extraction (-0.2s to +0.4s)
│   ├── feature_extraction.py     # 28 time, frequency, and RR rhythm features
│   ├── label_mapping.py          # AAMI / 3-class standardized symbol mapping
│   ├── prediction.py             # End-to-end inference pipeline
│   ├── evaluation.py             # Metric computation and confusion matrices
│   ├── visualization.py          # Plotly waveform and probability visualizers
│   │
│   ├── ecg_input/                 # Unified Multi-Format Ingestion System
│   │   ├── __init__.py
│   │   ├── input_detector.py      # Format & modality detector
│   │   ├── signal_loader.py       # Universal CSV/TXT/NPY signal loader
│   │   ├── pdf_processor.py       # pypdf text & metadata extraction
│   │   ├── image_processor.py     # OpenCV grid isolation & trace preprocessing
│   │   ├── measurement_extractor.py # Clinical measurement regex parser
│   │   ├── waveform_extractor.py  # 1D waveform trace extractor
│   │   └── extraction_validation.py # Anti-hallucination validation gate
│   │
│   └── report/                    # Multi-Format Professional Reporting
│       ├── __init__.py
│       ├── report_generator.py    # Structured report compiler & JSON/TXT export
│       └── pdf_generator.py       # Publication-grade ReportLab PDF generator
│
├── models/
│   ├── classifier.pkl             # Trained Random Forest classifier
│   ├── baseline_classifier.pkl    # Trained Logistic Regression baseline
│   ├── scaler.pkl                 # StandardScaler fitted on train set only
│   └── metadata.json              # Model configuration and feature importances
│
├── data/
│   ├── raw/                       # MIT-BIH records (.dat, .hea, .atr)
│   └── processed/                 # Train/test datasets with zero leakage
│
└── tests/
    ├── test_preprocessing.py      # Signal filtering tests
    ├── test_peak_detection.py     # R-peak detection tests
    ├── test_segmentation.py       # Beat segmentation tests
    ├── test_features.py           # Feature calculation tests
    ├── test_prediction.py         # End-to-end ML prediction tests
    ├── test_input_detection.py    # Multi-format detection tests
    ├── test_pdf_processing.py     # Clinical PDF extraction tests
    ├── test_image_processing.py   # OpenCV image tests
    ├── test_waveform_extraction.py # Anti-hallucination validation tests
    └── test_report_generation.py  # PDF/JSON/TXT export tests
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
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

---

## ⚠️ Medical Disclaimer

This application is strictly for **educational and scientific research purposes**. It has not been approved by the US Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory body. It must **not** be used for clinical decision-making or medical diagnosis.