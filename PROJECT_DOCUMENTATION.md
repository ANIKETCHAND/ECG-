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

> **Every number in this chapter is generated, not transcribed.** Run
> `python training/generate_model_card.py` to recompute them and rewrite
> `models/production/metrics.json` and `models/MODEL_CARD.md`. CI fails if the
> committed card stops agreeing with the shipped artifact. Earlier revisions of
> this chapter carried figures that no longer reproduced; the model card is now
> the single source of truth.

### Actual Test Set Performance (Completely Unseen Patient Records: 101, 119, 208)

| Evaluation Metric | Measured value |
| :--- | :--- |
| **Accuracy** | **98.74%** |
| **Weighted F1-Score** | 98.68% |
| **Balanced accuracy** | 66.06% |
| **Macro F1-Score** | **65.67%** |
| **Cohen's kappa** | 96.81% |

### Detailed Per-Class Breakdown (Random Forest)

| Beat Class | Precision | Recall | F1-Score | Support (Beats) | Conclusive? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 99.74% | 98.62% | 99.17% | 4,989 | yes |
| **PVC** | 96.16% | 99.56% | 97.83% | 1,809 | yes |
| **Other** | 0.00% | 0.00% | 0.00% | **9** | **no** |

**Read the macro F1, not the accuracy.** The high accuracy and the poor macro F1
are the same fact seen twice: the corpus contains two learnable classes and one
class that has nine held-out examples and four distinct annotation symbols it
never saw in training. No estimator can learn a class from nine examples, so the
0.00 F1 for `Other` measures the dataset, not the algorithm. Restricted to the
two classes the corpus can actually support, accuracy is 98.87%.

### What "100% accuracy" would actually mean

Accuracy over *every* beat is not the useful question, because a classifier that
must answer on every beat will always have to guess on some of them. The useful
question is: **how accurate are the answers the system is willing to stand
behind?** That is selective prediction, and the honest answer is measurable.

`training/fit_operating_point.py` selects a confidence threshold by
leave-one-record-out and then measures it on the whole held-out partition:

| Confidence threshold | Beats reported | Coverage | Accuracy of reported beats |
| :--- | ---: | ---: | ---: |
| 0.990 (**installed**) | 712 / 6,807 | 10.5% | **100.00%** |
| 0.970 | 1,661 / 6,807 | 24.4% | 99.94% |
| 0.800 | 6,188 / 6,807 | 90.9% | 99.69% |
| 0.650 | 6,602 / 6,807 | 97.0% | 99.35% |

At the installed threshold the model is **exactly 100% correct on every beat it
reports**, and it declines to answer on the remaining 89.5%. Those beats are
returned as `INDETERMINATE` and handed to the clinician with their confidences
attached; the raw model opinion is preserved in `raw_beat_predictions` so nothing
is hidden. Coverage is the honest price of the claim, and the trade-off is
explicitly measured rather than asserted.

Measured live-pipeline coverage is lower than the figures above. These metrics
are computed on beats centred on the expert annotations, while deployment centres
beats on the engine's own R-peak detection; alignment error moves the beat under
the model and lowers confidence. The table is therefore an upper bound on what
the deployed gate will report, and the model card says so.

### Random Forest Top Feature Importances
1. `local_rr_ratio` (0.2169) — Ratio of pre-RR to average RR
2. `pre_rr` (0.1339) — Preceding RR interval
3. `autocorr_first_peak` (0.1226) — Waveform periodicity
4. `spectral_entropy` (0.0690) — Energy dispersion across frequencies
5. `max_power` (0.0630) — Peak spectral density

*Cardiological interpretation*: Premature Ventricular Contractions are physiologically characterized by an early/premature contraction (short `pre_rr`, low `local_rr_ratio`), followed by a compensatory pause, and a wide, bizarre QRS complex that alters autocorrelation and spectral entropy.

---

## Chapter 13 — Dashboard

The upgraded Streamlit dashboard (`scripts/app_streamlit_legacy.py`; the primary
deployment surface is now the `public/index.html` portal backed by `api/index.py`)
provides 10 unified functional modules:
1. **Universal Multi-Format Ingestion**: Supports drag-and-drop clinical ECG PDFs, scanned images (JPG/PNG), and raw signals (CSV/TXT/NPY).
2. **6-Step Progress Pipeline**: Step-by-step progress tracking for ingestion, quality checks, beat detection, and AI inference.
3. **Rapid Clinical Overview Cards**: Live metrics for rhythm pattern, heart rate, signal quality score, and detected cycles.
4. **Printed Machine Interpretation (Direct Document Data)**: Displays printed measurements (HR, PR, QRS, QT/QTc, axes, diagnosis) strictly separated from AI predictions.
5. **Interactive Waveform & R-Peak Viewer**: Plotly visualization with pan/zoom and detected peak annotations.
6. **AI Abnormality Classification**: Random Forest probabilities across Normal, PVC, and Other classes.
7. **Beat Segmentation Overlay**: Aligned cardiac cycle overlay with average morphology.
8. **Morphological Features Table**: Quantitative 28-feature summary.
9. **MIT-BIH Ground Truth Comparison**: Direct verification against cardiologist annotations in demo mode.
10. **Multi-Format Report Export Center**: One-click downloads for Publication-Grade PDF, JSON, and Summary TXT.

---

## Chapter 14 — Limitations

1. **Single-Lead ML Model**: The trained Random Forest classifier operates on Modified Lead II (MLII) representations. When standard 12-lead reports are ingested, the system evaluates Lead II representations and extracts printed 12-lead measurements from the document header.
2. **Minority Class Representation**: While Normal and PVC classes have thousands of samples, supraventricular and fusion beats (`Other`) represent under 1% of the training data.
3. **Scanned Image Resolution**: Heavily corrupted or blurred scanned photocopies without clear contrast between trace and background may be rejected by the anti-hallucination validation gate.
4. **Non-Clinical Validation**: The system is an educational and scientific research prototype and has not undergone clinical regulatory clearance (e.g., FDA 510(k) or CE mark).

---

## Chapter 15 — Future Scope

### Delivered since this chapter was written

1. **1-D Convolutional Neural Networks** — `ECGConv1DClassifier` in
   `src/ml/models/deep_1d_cnn.py` is a genuine convolutional network over raw beat
   windows, trained by `training/train_waveform_cnn.py`. PyTorch is optional, and
   when it is absent the class raises `PyTorchNotAvailable` rather than passing a
   feature MLP off as a CNN. The feature-based model was renamed
   `ECGFeatureMLPClassifier` to match what it is.
2. **Multi-lead learning** — the PTB-XL pipeline now builds 24 per-lead
   descriptors from real multi-lead recordings and aggregates SCP statements to
   the five diagnostic superclasses.
3. **Real-time streaming** — `src/streaming/simulated_stream.py` classifies beats
   incrementally with cross-window duplicate suppression and a refractory guard.
4. **Explainable AI** — `src/evidence/shap_evidence.py` produces per-beat Shapley
   attributions when `shap` is installed, and otherwise returns baseline-deviation
   attribution explicitly labelled as such.
5. **Calibration and abstention** — `src/ml/calibration.py` adds per-class
   calibration and a Mondrian conformal predictor that can abstain when the
   prediction set contains more than one label.
6. **Selective prediction** — `src/ml/selective.py` + `training/fit_operating_point.py`
   add a measured abstention gate: a confidence threshold selected by
   leave-one-record-out and confirmed on the held-out partition. Beats below it
   are reported as `INDETERMINATE` rather than guessed, so the accuracy of
   reported beats is a number the system can substantiate. Installed threshold
   0.990 gives 100% measured precision on 712 held-out beats (10.5% coverage).
7. **Model card and measured metrics** — `training/generate_model_card.py` writes
   `models/MODEL_CARD.md` and `models/production/metrics.json`, and stamps
   `metrics` + `provenance` into `models/production/metadata.json` (which
   previously described the estimator but said nothing about how well it worked).
8. **Evidence-gated therapy considerations** — the CDS engine now withholds
   medication suggestions unless the finding they rest on is demonstrably
   reliable (adequate signal quality, no abstention, a quantified confidence,
   and a statement backed by a majority of the recording). Withheld suggestions
   are retained under `withheld_medication_recommendations` for audit.
9. **Real demo data** — `/api/sample` serves a window from an actual MIT-BIH
   record instead of a generated sine wave, and labels the payload
   `REAL_RECORDED_DATASET`, falling back to an explicitly labelled
   `SYNTHETIC_DEMO_NOT_FOR_CLINICAL_USE` trace only when no corpus is on disk.

### Still open

1. **Recurrent / attention architectures** — Bidirectional LSTMs and Transformers
   for long-range rhythm dependencies.
2. **Wearable integration** — BLE sensor drivers feeding the streaming pipeline
   (the analysis path is ready; the transport is not).
3. **Terminology binding** — LOINC/SNOMED mapping for interval measurements via a
   terminology server, currently declared as pending in the FHIR export.

---

## Chapter 16 — Universal Multi-Format Ingestion System (`src/ecg_input/`)

To support clinical ECG reports from hospitals, diagnostic labs, and personal monitors, the system incorporates an autonomous ingestion engine:
1. **Input Detection (`input_detector.py`)**: Uses file extensions and binary magic numbers (`%PDF`, `\x89PNG`, `\xff\xd8\xff`, `\x93NUMPY`) to route files to specialized parsers.
2. **Digital Signal Loader (`signal_loader.py`)**: Sniffs delimiters (comma, tab, space, semicolon), identifies voltage vs. time columns, and validates sampling rate.
3. **PDF Report Processor (`pdf_processor.py`)**: Leverages `pypdf` to extract selectable text layers and isolates raster image strips.
4. **Clinical Measurement Extractor (`measurement_extractor.py`)**: Regex-driven parser matching clinical reporting conventions (e.g. `Vent Rate: 72 BPM`, `PR Int: 156 ms`, `QRS Dur: 94 ms`, `QT/QTc: 398/418 ms`, `P-QRS-T Axes: 48 52 42`, and printed diagnostic conclusions).
5. **Image Preprocessing & Trace Extraction (`image_processor.py`, `waveform_extractor.py`)**: OpenCV color segmentation to suppress pink/red grid lines, followed by column-wise center-of-mass trace extraction.
6. **Waveform Validation Gate (`extraction_validation.py`)**: Rigorously evaluates signal continuity, sampling duration, electrical dynamic range (std > 0.05, ptp > 0.2), and border spikes before passing to the ML classifier. **Rejects unconfident traces rather than hallucinating signals.**

---

## Chapter 17 — Multi-Format Clinical Reporting (`src/report/`)

1. **Structured Report Generator (`report_generator.py`)**: Merges patient demographics, recording parameters, signal quality metrics, printed machine interpretation, and AI predictions into a normalized data dictionary.
2. **Publication-Grade PDF Engine (`pdf_generator.py`)**: ReportLab-powered document generator with institutional header, colored clinical alert banners, patient metadata tables, embedded high-resolution ECG waveforms with annotated R-peaks, and standard medical disclaimers.
3. **Machine JSON & Plain Text Exporters**: Serializes report models to `.json` for database ingestion and `.txt` for clinical summary notes.

---

## Chapter 18 — Medical Disclaimer

> **IMPORTANT CLINICAL & REGULATORY DISCLAIMER**  
> This software is intended strictly for educational and scientific research purposes. It is **not** a certified medical diagnostic device and should **never** be used to make clinical diagnoses, guide medical treatments, alter medication regimens, or replace professional cardiovascular medical consultation. If you are experiencing symptoms of acute coronary syndrome or arrhythmia, seek emergency medical care immediately.