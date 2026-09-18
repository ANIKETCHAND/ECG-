# ❤️ AI ECG Analyzer

**AI-Based ECG Signal Quality Assessment & Cardiac Abnormality Detection**

> ⚠️ **EDUCATIONAL AND RESEARCH PURPOSES ONLY**  
> This software is intended strictly for educational and scientific research purposes. It is **NOT** a certified medical diagnostic device and must **NOT** be used to make clinical decisions or diagnose medical conditions.

---

## 📌 Project Overview

**AI ECG Analyzer** is a bioengineering and computer science research prototype designed to:
1. **Assess ECG Signal Quality**: Evaluates SNR, baseline wander, 50/60 Hz powerline interference, and motion artifacts (`GOOD`, `ACCEPTABLE`, `POOR`).
2. **Detect R-Peaks & Segment Beats**: Automatically identifies R-peaks using refractory period validation (≥300 ms) and extracts individual cardiac cycles (-0.2s to +0.4s).
3. **Extract Morphological & Rhythm Features**: Computes 28 quantitative time-domain, frequency-domain, and R-R interval features.
4. **Classify Beat Abnormalities**: Distinguishes **Normal beats**, **Premature Ventricular Contractions (PVC)**, and **Other ectopic beats** with an authentic machine learning model.
5. **Prevent Data Leakage**: Enforces strict patient/record-level train-test splits on the MIT-BIH Arrhythmia Database.
6. **Interactive Research Dashboard**: Visualizes live waveforms, signal quality metrics, feature importances, and provides direct side-by-side comparison with physician reference annotations.

---

## 🏗️ System Architecture

```text
                             ECG INPUT
                                 |
              +------------------+------------------+
              |                                     |
     MIT-BIH Record Selection                 Custom CSV/NPY Upload
              |                                     |
              +------------------+------------------+
                                 |
                                 v
                            DATA LOADER
                                 |
                                 v
                           PREPROCESSING
                 [Median Filter + Butterworth Bandpass]
                                 |
                                 v
                       Z-SCORE NORMALIZATION
                                 |
                                 v
                     SIGNAL QUALITY ASSESSMENT
                 [SNR, Drift, 50/60Hz, Artifacts]
                     (GOOD / ACCEPTABLE / POOR)
                                 |
                                 v
                         R-PEAK DETECTION
                     (Refractory Period ≥ 0.3s)
                                 |
                                 v
                       HEARTBEAT SEGMENTATION
                        [-0.2s, R-Peak, +0.4s]
                                 |
                                 v
                     FEATURE EXTRACTION (28 Dims)
         [Time Domain | ECG Morphology | FFT Power | RR Dynamics]
                                 |
                                 v
                        SAVED ML CLASSIFIER
                        (Random Forest Model)
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
         Prediction        Probabilities     Feature Importance
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                        STREAMLIT DASHBOARD
```

---

## 📊 Actual Machine Learning Results

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

### Top Random Forest Feature Importances
1. `local_rr_ratio` (0.2169) — Ratio of preceding RR to local mean RR
2. `pre_rr` (0.1339) — Preceding RR interval in seconds
3. `autocorr_first_peak` (0.1226) — Waveform autocorrelation
4. `spectral_entropy` (0.0690) — Energy distribution across frequency spectrum
5. `max_power` (0.0630) — Peak spectral power density

---

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **Signal Processing**: SciPy, NumPy
- **Physiological Data Access**: WFDB (PhysioNet)
- **Machine Learning**: Scikit-learn, Joblib
- **Data Analysis**: Pandas
- **Visualization**: Plotly, Matplotlib
- **Web Interface**: Streamlit
- **Testing**: Pytest

---

## 🚀 Quick Start & Installation

### 1. Clone or Open the Repository
```bash
git clone https://github.com/yourusername/AI-ECG-Analyzer.git
cd AI-ECG-Analyzer
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

---

## ⚙️ Workflow Execution

### Step 1: Prepare Dataset (Record-Level Split)
Downloads and extracts features from MIT-BIH records with zero data leakage:
```bash
python training/train_test_split.py --train-records 100 106 200 213 --test-records 101 119 208
```

### Step 2: Train Machine Learning Models
Trains Random Forest and baseline Logistic Regression models, saving weights and metadata:
```bash
python training/train_model.py
```

### Step 3: Evaluate on Unseen Patients
Computes metrics, confusion matrices, and feature importance plots:
```bash
python training/evaluate_model.py
```

### Step 4: Run Unit Tests
Verifies preprocessing, peak detection, segmentation, feature extraction, and prediction:
```bash
python -m pytest -v
```

### Step 5: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
AI-ECG-Analyzer/
│
├── app.py                         # Interactive Streamlit dashboard
├── requirements.txt               # Project dependencies
├── README.md                      # Project overview and instructions
├── PROJECT_DOCUMENTATION.md      # Detailed 16-chapter technical documentation
├── test_system.py                 # Multi-record end-to-end integration test
├── .gitignore                     # Git ignore file
│
├── data/
│   ├── raw/                       # MIT-BIH records (.dat, .hea, .atr)
│   └── processed/
│       ├── train_dataset.csv      # Training beats (records 100, 106, 200, 213)
│       ├── test_dataset.csv       # Testing beats (records 101, 119, 208)
│       ├── split_info.json        # Patient-level partition metadata
│       └── dataset_info.json      # Dataset summary and feature list
│
├── models/
│   ├── classifier.pkl             # Trained Random Forest classifier
│   ├── baseline_classifier.pkl    # Trained Logistic Regression baseline
│   ├── scaler.pkl                 # StandardScaler fitted on train set only
│   └── metadata.json              # Model configuration and feature importances
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # MIT-BIH record loading via WFDB
│   ├── preprocessing.py          # Median filter, bandpass, Z-score normalization
│   ├── signal_quality.py         # Quantitative SNR and artifact assessment
│   ├── peak_detection.py         # SciPy-based adaptive R-peak detection
│   ├── segmentation.py           # Beat window extraction (-0.2s to +0.4s)
│   ├── feature_extraction.py     # 28 time, frequency, and RR rhythm features
│   ├── label_mapping.py          # AAMI / 3-class standardized symbol mapping
│   ├── prediction.py             # End-to-end inference pipeline
│   ├── evaluation.py             # Metric computation and confusion matrices
│   └── visualization.py          # Plotly waveform and probability visualizers
│
├── training/
│   ├── prepare_dataset.py        # Dataset ingestion pipeline
│   ├── train_test_split.py       # Patient-level train/test separation
│   ├── train_model.py            # Model training & artifact serialization
│   └── evaluate_model.py         # Evaluation and report generation
│
├── reports/
│   ├── evaluation_report.json    # Genuine metric report
│   └── figures/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_baseline.png
│       ├── class_distribution.png
│       └── feature_importance.png
│
└── tests/
    ├── test_preprocessing.py     # Preprocessing stability and filtering tests
    ├── test_peak_detection.py     # R-peak detection & refractory period tests
    ├── test_segmentation.py       # Beat segmentation and padding tests
    ├── test_features.py           # Feature calculation and robustness tests
    └── test_prediction.py         # End-to-end pipeline and edge-case tests
```

---

## ⚠️ Medical Disclaimer

This application is strictly for **educational and scientific research purposes**. It has not been approved by the US Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory body. It must **not** be used for clinical decision-making or medical diagnosis.