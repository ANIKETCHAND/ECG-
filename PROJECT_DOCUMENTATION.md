# AI ECG Analyzer — Project Documentation

> ⚠️ **EDUCATIONAL AND RESEARCH PURPOSES ONLY**
> This software system is developed strictly for educational and academic research purposes. It is **NOT** a certified medical diagnostic device and must **NOT** be used to make clinical decisions or replace professional medical consultation.

---

## Table of Contents
1. [Introduction](#chapter-1--introduction)
2. [Problem Statement](#chapter-2--problem-statement)
3. [Objectives](#chapter-3--objectives)
4. [Existing Approach](#chapter-4--existing-approach)
5. [Proposed System](#chapter-5--proposed-system)
6. [System Architecture](#chapter-6--system-architecture)
7. [Dataset Description & Attribution](#chapter-7--dataset)
8. [Signal Preprocessing Pipeline](#chapter-8--preprocessing)
9. [Signal Quality Assessment & R-Peak Detection](#chapter-9--signal-processing)
10. [Feature Extraction](#chapter-10--feature-extraction)
11. [Machine Learning Methodology](#chapter-11--machine-learning)
12. [Actual Experimental Results](#chapter-12--results)
13. [Interactive Dashboard](#chapter-13--dashboard)
14. [Limitations](#chapter-14--limitations)
15. [Future Scope](#chapter-15--future-scope)
16. [Medical Disclaimer](#chapter-16--medical-disclaimer)

---

## Chapter 1 — Introduction

An electrocardiogram (ECG/EKG) non-invasively records the electrical depolarization and repolarization potentials of the heart muscle across cardiac cycles. It is one of the most vital diagnostic tools in clinical cardiology for identifying arrhythmias, myocardial ischemia, conduction defects, and ectopic rhythms.

However, interpreting ambulatory (Holter) or telemetry ECG recordings often requires cardiologists to visually review tens of thousands of consecutive heartbeats per patient per day. Automated, computer-assisted signal processing and machine learning provide transparent screening tools to assist medical researchers and students in understanding arrhythmia patterns.

### Motivation
Early detection of ventricular arrhythmias, such as Premature Ventricular Contractions (PVCs) and runs of ventricular ectopy, is critical for assessing cardiovascular risk. A reproducible, transparent machine learning pipeline trained on standard benchmark physiological databases enables explainable, beat-by-beat rhythm analysis.

---

## Chapter 2 — Problem Statement

Manual ECG interpretation presents significant real-world challenges:
- **Time Intensity**: Holter recordings spanning 24 to 48 hours contain roughly 100,000 heartbeats per patient.
- **Inter-Observer Variability**: Diagnostic concordance across clinical experts can vary, especially for ambiguous ectopic beats.
- **Signal Quality Degradation**: Ambulatory ECG signals frequently suffer from baseline wander, motion artifacts, electrode loose contact, and electromagnetic 50/60 Hz powerline interference.
- **Data Leakage in ML**: Naive train/test splitting at the individual beat level causes severe information leakage when beats from the same patient appear in both training and test sets.

This project addresses these challenges through a modular Python system that incorporates automated signal quality assessment, patient-level train/test separation, and transparent classical machine learning.

---

## Chapter 3 — Objectives

1. **Load Standard Benchmark ECG Data**: Integrate the MIT-BIH Arrhythmia Database via WFDB.
2. **Implement Scientifically Valid Preprocessing**: Provide baseline wander removal (median filtering), Butterworth bandpass filtering (5–15 Hz), and Z-score normalization.
3. **Automated Signal Quality Assessment (SQA)**: Classify signal quality into `GOOD`, `ACCEPTABLE`, or `POOR` using SNR, baseline instability, powerline interference, and motion artifact metrics.
4. **Adaptive R-Peak Detection & Segmentation**: Detect R-peaks with physiological refractory period validation (≥300 ms) and segment cardiac cycles (-0.2s to +0.4s).
5. **Multi-Domain Feature Extraction**: Extract 28 morphological, spectral, and rhythm features (including pre-RR, post-RR, and local RR ratios).
6. **Prevent Data Leakage**: Enforce strict patient/record-level train-test splits (Train: records 100, 106, 200, 213; Test: records 101, 119, 208).
7. **Train & Compare Genuine Classifiers**: Train a Random Forest classifier alongside a baseline Logistic Regression model without data fabrication.
8. **Interactive Streamlit Dashboard**: Build an interactive web application displaying live waveforms, signal quality metrics, feature importances, beat segmentation, and research demo comparison.

---

## Chapter 4 — Existing Approach

### Traditional Manual ECG Analysis
ECG strips are printed or viewed on calibrated grid paper (25 mm/s, 10 mm/mV). Clinicians measure PR intervals, QRS durations, QT intervals, and ST segments manually.

### Conventional Automated Approaches
- Commercial Holter algorithms rely primarily on heuristic thresholding and template matching.
- Many research publications report over-optimistic test accuracies (>99%) due to naive random heartbeat splitting that leaks identical patient morphology into the test set.
- Many black-box deep learning models lack feature transparency and do not assess whether input signal quality is sufficient before performing inference.

---

## Chapter 5 — Proposed System

The proposed **AI ECG Analyzer** system addresses these drawbacks by providing:
1. **Signal Gatekeeper**: A Signal Quality Assessment (SQA) module that evaluates SNR and artifacts before relying on ML predictions.
2. **Robust Signal Processing**: Median baseline removal, zero-phase bandpass filtering, and physiological refractory peak filtering.
3. **Leak-Free Record Split**: Models are trained and tested on strictly disjoint patient records.
4. **Physiologically Grounded Features**: Combines QRS morphology with local RR rhythm dynamics to capture premature ectopic beats.
5. **Interactive Visualization**: Complete visibility into raw vs filtered waveforms, per-beat segmentation overlays, and feature importances.

---

## Chapter 6 — System Architecture

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
                                 |
              +------------------+------------------+
              |                                     |
     Median Filter (Baseline)           Butterworth Bandpass (5-15 Hz)
              |                                     |
              +------------------+------------------+
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
                                 |
         +-------------+---------+---------+-------------+
         |             |                   |             |
         v             v                   v             v
     Waveform    Quality Metrics     AI Prediction    Demo Compare
```

---

## Chapter 7 — Dataset

### MIT-BIH Arrhythmia Database
- **Source**: PhysioNet (Goldberger et al., 2000). URL: `https://physionet.org/content/mitdb/1.0.0/`
- **Sampling Frequency**: 360 Hz (11-bit resolution over a ±5 mV range).
- **Leads**: Modified limb lead II (MLII) and modified lead V1.
- **Reference Annotations**: Each beat was independently annotated by two or more cardiologists.

### Record Partitioning (Patient-Level Split)
To prevent data leakage, individual patient records were allocated to disjoint sets:
- **Training Records**: `100`, `106`, `200`, `213` (Total: 10,152 beats)
- **Testing Records**: `101`, `119`, `208` (Total: 6,807 beats)

### Standardized Class Mapping
Following AAMI standards, annotation symbols were mapped to three clinically meaningful classes:
- **Normal (`Normal`)**: Normal sinus beats (`N`), bundle branch block beats (`L`, `R`), nodal escape beats (`e`, `j`).
- **Ventricular Ectopic (`PVC`)**: Premature ventricular contractions (`V`), ventricular escape beats (`W`).
- **Other / Ectopic (`Other`)**: Supraventricular ectopic beats (`A`, `a`, `J`, `S`), fusion beats (`F`), and unclassifiable beats (`Q`).
- Non-beat markers (rhythm changes `+`, noise `~`, artifacts `|`) are filtered out.

### Class Distribution
| Partition | Records | Total Beats | Normal | PVC | Other |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 100, 106, 200, 213 | 10,152 | 8,130 (80.1%) | 1,931 (19.0%) | 91 (0.9%) |
| **Test Set** | 101, 119, 208 | 6,807 | 4,989 (73.3%) | 1,809 (26.6%) | 9 (0.1%) |
| **Combined** | 7 Records | 16,959 | 13,119 (77.4%) | 3,740 (22.1%) | 100 (0.6%) |

---

## Chapter 8 — Preprocessing

The preprocessing module (`src/preprocessing.py`) executes sequentially:
1. **Missing-Value Handling**: Linear interpolation with backward/forward fill for missing sample indices.
2. **Baseline Drift Correction**: A median filter with a 121-sample kernel (~336 ms at 360 Hz) estimates the low-frequency baseline wander, which is subtracted from the raw signal.
3. **Bandpass Filtering**: A 4th-order zero-phase Butterworth bandpass filter with cutoff frequencies at 5 Hz and 15 Hz attenuates high-frequency muscle tremor and low-frequency motion noise.
4. **Z-Score Normalization**: Removes patient-specific amplitude offset by centering to mean zero and scaling to unit variance:
   $$z(t) = \frac{x(t) - \mu}{\sigma}$$

---

## Chapter 9 — Signal Processing

### Signal Quality Assessment (`src/signal_quality.py`)
Signal quality is evaluated across four quantitative physical indicators:
- **SNR Estimation**: Compares power in the QRS signal band (5–15 Hz) to the baseline noise band (0–2 Hz) via Welch's power spectral density.
- **Baseline Wander Detection**: Thresholds low-frequency PSD power (< 2 Hz).
- **Powerline Interference**: Checks spectral energy at 50 Hz and 60 Hz bins.
- **Motion Artifacts**: Evaluates variance spikes across 500 ms windows.
- **Composite Score**: Normalized from 0.0 to 1.0, categorized into `GOOD` ($\ge 0.80$), `ACCEPTABLE` ($0.50 \le s < 0.80$), or `POOR` ($< 0.50$).

### R-Peak Detection (`src/peak_detection.py`)
- Detects local maxima using prominence thresholding on the filtered signal.
- Enforces a minimum refractory period of 300 ms ($\approx 108$ samples at 360 Hz) to eliminate false detections from elevated T-waves.

### Heartbeat Segmentation (`src/segmentation.py`)
- For each validated R-peak, extracts a window from $t - 0.2\text{ s}$ (72 samples pre-peak) to $t + 0.4\text{ s}$ (144 samples post-peak), resulting in 217 samples per heartbeat cycle.
- Applies mirror/edge padding for beats occurring near the start or end of a record.

---

## Chapter 10 — Feature Extraction

For every segmented beat, 28 distinct features are computed (`src/feature_extraction.py`):

1. **Time-Domain Statistics (12)**: `mean`, `std`, `min`, `max`, `range`, `median`, `energy`, `rms`, `mav`, `snr`, `zero_crossing_rate`, `autocorr_first_peak`.
2. **ECG Morphological Features (6)**: `r_peak_amplitude`, `p_wave_amplitude`, `t_wave_amplitude`, `peak_to_peak_amplitude`, `max_slope`, `qrs_width_samples`.
3. **Frequency-Domain Spectral Features (7)**: `total_power`, `lf_power` (0.5–1 Hz), `hf_power` (1–10 Hz), `vhf_power` (>10 Hz), `dominant_frequency`, `max_power`, `spectral_entropy`.
4. **Rhythm & RR Dynamics (3)**:
   - `pre_rr`: Interval in seconds from preceding R-peak to current R-peak.
   - `post_rr`: Interval in seconds from current R-peak to succeeding R-peak.
   - `local_rr_ratio`: Ratio of `pre_rr` to running local mean RR.

---

## Chapter 11 — Machine Learning

### Model 1: Random Forest Classifier (Primary)
- `n_estimators`: 100 decision trees
- `max_depth`: 16
- `class_weight`: `"balanced"` (inversely proportional to class frequencies to counter class imbalance)
- `random_state`: 42

### Model 2: Logistic Regression (Baseline Comparison)
- Regularized multinomial logistic regression (`max_iter=1000`, `class_weight="balanced"`).

### Data Standardization
- A `StandardScaler` is fitted **strictly on the training partition** and then applied to transform test and inference samples, guaranteeing zero data leakage.

---

## Chapter 12 — Results

### Actual Test Set Performance (Completely Unseen Patient Records: 101, 119, 208)

| Evaluation Metric | Random Forest (Primary) | Logistic Regression (Baseline) |
| :--- | :--- | :--- |
| **Overall Accuracy** | **97.86%** | 95.21% |
| **Weighted F1-Score** | **97.92%** | 95.50% |
| **Macro Precision** | 64.48% | 64.73% |
| **Macro Recall** | 65.73% | 74.75% |
| **Macro F1-Score** | 65.07% | 66.61% |

### Detailed Per-Class Breakdown (Random Forest)
| Beat Class | Precision | Recall | F1-Score | Support (Beats) |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 99.9% | 97.3% | 98.6% | 4,989 |
| **PVC** | 93.6% | 99.9% | 96.6% | 1,809 |
| **Other** | 0.0% | 0.0% | 0.0% | 9 |

### Random Forest Top Feature Importances
1. `local_rr_ratio` (0.2169) — Ratio of pre-RR to average RR
2. `pre_rr` (0.1339) — Preceding RR interval
3. `autocorr_first_peak` (0.1226) — Waveform periodicity
4. `spectral_entropy` (0.0690) — Energy dispersion across frequencies
5. `max_power` (0.0630) — Peak spectral density

*Cardiological interpretation*: Premature Ventricular Contractions are physiologically characterized by an early/premature contraction (short `pre_rr`, low `local_rr_ratio`), followed by a compensatory pause, and a wide, bizarre QRS complex that alters autocorrelation and spectral entropy.

---

## Chapter 13 — Dashboard

The Streamlit dashboard (`app.py`) provides 9 functional modules:
1. **ECG Waveform & R-Peak Viewer**: Interactive Plotly time-series with zoom/pan and peak markers.
2. **Signal Quality Assessment**: Quantitative cards and tabular breakdown of SNR, baseline drift, and artifacts.
3. **AI Abnormality Prediction**: Primary classification badge and percentage probability bars.
4. **Heartbeat Segments Overlay**: Overlay of individual segmented beats aligned at the R-peak with average profile.
5. **Morphological Features Table**: Quantitative summary of amplitude, energy, and entropy.
6. **Model Transparency Card**: Training record details, sample sizes, and honest test metrics.
7. **Feature Importance Plot**: Interactive bar chart displaying top discriminating features.
8. **Research Demo Mode**: Direct side-by-side comparison with expert MIT-BIH physician annotations.
9. **Report Export**: Downloadable plain-text clinical research summary report.

---

## Chapter 14 — Limitations

1. **Single-Lead Limitation**: MIT-BIH recordings use modified limb lead II. 12-lead ECGs provide richer spatial vector information not captured by single-lead data.
2. **Minority Class Representation**: While Normal and PVC classes have thousands of samples, supraventricular and fusion beats (`Other`) represent under 1% of the data.
3. **Non-Clinical Validation**: The system is an educational prototype and has not undergone clinical validation or regulatory clearance (e.g., FDA 510(k) or CE mark).

---

## Chapter 15 — Future Scope

1. **1D Convolutional Neural Networks (1D CNN)**: End-to-end feature learning directly from raw waveforms without manual feature engineering.
2. **Recurrent / Attention Architectures**: Bidirectional LSTMs and Transformers to model long-range cardiac rhythm dependencies.
3. **Multi-Lead ECG Analysis**: Extending the pipeline to 12-lead standard clinical databases (e.g., PTB-XL).
4. **Wearable & Real-Time Streaming**: Integration with Bluetooth Low Energy (BLE) ECG sensors for streaming mobile monitoring.
5. **Model Explainability**: Integrating SHAP / LIME to provide beat-by-beat attribution maps.

---

## Chapter 16 — Medical Disclaimer

> **IMPORTANT DISCLAIMER**
> This software is intended strictly for educational and scientific research purposes. It is **not** a certified medical diagnostic device and should **never** be used to make clinical diagnoses, guide medical treatments, or replace professional medical consultation.