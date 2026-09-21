# Phase 0: Machine Learning System Audit — ECG GUARDIAN
**Document Identifier:** `DOC-ML-AUDIT-001`  
**System Name:** ECG GUARDIAN  
**Audit Date:** September 2026  
**Auditor:** Antigravity Autonomous Agent  
**Compliance Standard References:**  
- IEC 62304:2006+A1:2015 (Medical device software — Software life cycle processes)  
- ISO 14971:2019 (Application of risk management to medical devices)  
- AAMI/ANSI EC57:2012 (Testing and reporting performance results of cardiac rhythm algorithms)  
- FDA Guidance: Marketing Submission Recommendations for AI/ML-Enabled Device Software Functions (2023)  

---

## 1. Executive Summary

This audit establishes the baseline technical and scientific state of the machine-learning subsystems within ECG GUARDIAN. The existing system contains a functioning single-lead beat-level arrhythmia classifier (Random Forest baseline) trained and evaluated on 7 records from the MIT-BIH Arrhythmia Database.

While the existing pipeline enforces patient-level separation and pre-inference quality gating, it is currently limited to a single dataset, a single lead (Lead II / MLII), and a narrow clinical scope (distinguishing Normal Sinus Rhythm from Premature Ventricular Contractions). To evolve into a serious hospital-ready, multi-dataset platform, the system requires an autonomous dataset manager, standardized multi-lead representations, explicit label mappers, dedicated multi-task pipelines, and deep neural architectures.

---

## 2. Current Architecture & Pipeline Flow

The current ML lifecycle is architecturally decoupled across the following modules:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & DATA STRUCTURE                                               │
│    • Universal Dispatcher: src/ecg_input/signal_loader.py (load_any_ecg)   │
│    • Core Data Model: src/ecg_core/models.py (ECGRecording dataclass)       │
│    • Supported formats: CSV, TXT, NPY, JSON, WFDB, EDF/EDF+, HL7 XML, DICOM │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. MANDATORY PRE-INFERENCE QUALITY GATEKEEPER                               │
│    • Gatekeeper Module: src/quality/quality_gate.py                         │
│    • Metrics: SNR (dB), 50/60 Hz powerline, drift ratio, clipping ratio    │
│    • Safety Interlock: UNUSABLE/POOR -> Pipeline halted with NO_RESULT      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. PREPROCESSING PIPELINE                                                   │
│    • Module: src/preprocessing.py & src/ml/preprocessing/                   │
│    • Bandpass filter: 0.5 – 40.0 Hz (3rd order Butterworth)                 │
│    • Baseline wander removal: Low-cut highpass / median subtraction         │
│    • Normalization: Z-score normalization (zero mean, unit variance)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. R-PEAK DETECTION & CARDIAC SEGMENTATION                                  │
│    • Peak Detector: src/peak_detection.py & src/ml/peak_detection/          │
│    • Method: Adaptive threshold SciPy find_peaks with refractory period     │
│    • Segmentation: src/segmentation.py & src/ml/segmentation/               │
│    • Window: -0.2s (pre-R) to +0.4s (post-R) = 217 samples at 360 Hz        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. FEATURE EXTRACTION (28 Engineered Dimensions)                            │
│    • Module: src/feature_extraction.py & src/ml/feature_extraction/         │
│    • Generates 2D NumPy feature matrix (n_beats, 28)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. MODEL INFERENCE & VERSIONING                                             │
│    • Engine: src/ml/inference/inference_engine.py (run_ecg_ml_inference)    │
│    • Production Artifacts: models/classifier.pkl, models/scaler.pkl         │
│    • Model ID: ECG-RF-1.0.0 (RandomForestClassifier, 100 Estimators)        │
│    • Baseline Artifact: models/baseline_classifier.pkl (LogisticRegression) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. CLINICAL EVIDENCE & EXPLAINABILITY ENGINE                                │
│    • Module: src/evidence/evidence_engine.py                                │
│    • Extracts: Aberrant beat index, coupling interval (ms), compensatory    │
│      pause ratio, QRS duration, feature z-scores, and waveform snippets     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Current Datasets & Local Data Inventory

| Item | Current Status | Notes |
| :--- | :--- | :--- |
| **Integrated Datasets** | 1 dataset: **MIT-BIH Arrhythmia Database** | Only dataset currently used in training/testing |
| **Storage Location** | `data/raw/` (PhysioNet format: `.hea`, `.dat`, `.atr`) | 7 records stored locally |
| **Training Records** | `100`, `106`, `200`, `213` | 4 patients ($10,152$ beats) |
| **Held-Out Test Records** | `101`, `119`, `208` | 3 patients ($6,807$ beats) |
| **Sampling Rate** | $360\text{ Hz}$ | Uniform clock across records |
| **Electrode Leads** | Modified Lead II (MLII) | Channel 0 across all 7 records |
| **Processed Datasets** | `data/processed/train_dataset.csv`<br>`data/processed/test_dataset.csv` | Feature matrices with ground-truth labels |

---

## 4. Current Label Mapping & Class Distribution

The current model uses a **3-Class Grouping** defined in `src/label_mapping.py`:

| Canonical Class | Source MIT-BIH Symbols | Clinical Meaning | Train Beats | Test Beats | Test Support % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Normal`** | `N`, `L`, `R`, `e`, `j` | Normal Sinus Conduction / Bundle Branch Conduction | 8,130 | 4,989 | 73.29% |
| **`PVC`** | `V`, `W`, `F` | Premature Ventricular Contraction / Ventricular Fusion | 1,931 | 1,809 | 26.58% |
| **`Other`** | `A`, `a`, `J`, `S`, `s`, `Q`, `P`, `!`, `+`, `~` | Supraventricular ectopy, atrial premature, paced, unknown | 91 | 9 | 0.13% |
| **Total** | — | — | **10,152** | **6,807** | **100.0%** |

---

## 5. Current Feature Engineering (28 Features)

The pipeline extracts 28 hand-crafted features per cardiac cycle:

1. **Time-Domain Statistics (12)**: `mean`, `std`, `min`, `max`, `range`, `median`, `energy`, `rms`, `mav` (mean absolute value), `snr`, `zero_crossing_rate`, `autocorr_first_peak`.
2. **ECG Morphology (6)**: `r_peak_amplitude`, `p_wave_amplitude`, `t_wave_amplitude`, `peak_to_peak_amplitude`, `max_slope`, `qrs_width_samples`.
3. **Spectral / Frequency-Domain (7)**: `total_power`, `lf_power` ($0.5$–$1.0$ Hz), `hf_power` ($1.0$–$10.0$ Hz), `vhf_power` ($>10$ Hz), `dominant_frequency`, `max_power`, `spectral_entropy`.
4. **R-R Interval Dynamics (3)**: `pre_rr` (coupling interval in s), `post_rr` (compensatory pause in s), `local_rr_ratio` ($\text{pre\_rr} / \text{mean\_rr}$).

**Top 5 Most Predictive Features (Gini Importance):**
1. `local_rr_ratio` ($21.69\%$) — Prematurity index
2. `pre_rr` ($13.39\%$) — Preceding coupling interval
3. `autocorr_first_peak` ($12.26\%$) — Periodicity disruption
4. `spectral_entropy` ($6.90\%$) — Spectral disorder of ventricular depolarization
5. `max_power` ($6.30\%$) — Low-frequency power peak

---

## 6. Current Model & Validated Performance

Evaluated strictly on held-out patient test records (`101`, `119`, `208`) representing $6,807$ beats:

| Metric | Primary Model (`RandomForestClassifier`, 100 Trees) | Baseline Model (`LogisticRegression`) | Performance Evaluation |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **97.86%** | 95.21% | High overall concordance |
| **Weighted F1-Score** | **97.92%** | 95.50% | Robust across dominant classes |
| **PVC Sensitivity (Recall)** | **99.89%** ($1,807 / 1,809$) | 95.80% ($1,733 / 1,809$) | Catches $99.9\%$ of ventricular ectopies |
| **PVC Specificity** | **97.52%** ($4,874 / 4,998$) | 95.10% ($4,745 / 4,998$) | Low false-alarm rate |
| **PVC Positive Predictive Value** | **93.58%** ($1,807 / 1,931$) | 89.61% ($1,733 / 1,934$) | High confidence when flagging PVC |
| **PVC Negative Predictive Value** | **99.96%** ($4,874 / 4,876$) | 98.42% ($4,745 / 4,821$) | Only 2 missed PVCs out of 1,809 |
| **AUROC (PVC)** | **0.9985** | 0.9810 | Near-ideal discrimination |
| **AUPRC (PVC)** | **0.9955** | 0.9620 | Resilient to class imbalance |
| **Class 'Other' Sensitivity** | **0.00%** ($0 / 9$) | 33.33% ($3 / 9$) | **CRITICAL FAILURE: Cannot detect SVEB/Other** |
| **Calibration Error (ECE)** | **0.1205** | 0.1650 | Moderate overconfidence in middle bins |

---

## 7. Data Leakage Assessment & Safeguards

- **Patient-Level Isolation**: Current splits in `training/splitting/patient_splitter.py` enforce `set(train_patients).isdisjoint(set(test_patients))`.
  - Records `100`, `106`, `200`, `213` are strictly separated from `101`, `119`, `208`.
  - Verified by unit tests in `tests/test_model_registry_and_training.py`.
- **Pre-processing Leakage**: The `StandardScaler` is fitted **strictly on training data** (`X_train`) and serialized (`models/scaler.pkl`). The test set is only transformed.
- **Historical Risk Identified**: In older scratch code (`training/prepare_dataset.py`), `sklearn.model_selection.train_test_split` on beat rows was present. It has been replaced by `training/train_test_split.py`, but any new multi-dataset loader must formally inherit the patient-isolation validator.

---

## 8. Current Limitations & Deficiencies

1. **Catastrophic Failure on Class 'Other'**:
   - The model has $0\%$ sensitivity for supraventricular ectopic beats (SVEB), atrial ectopy, and fusion beats.
   - **Root Cause**: Only 91 training samples and 9 test samples exist in the current 7-record subset.
2. **Single-Lead Constraint**:
   - The active model is strictly validated on Modified Lead II. It cannot ingest or analyze 12-lead standard recordings.
3. **Narrow Diagnostic Scope**:
   - The current model cannot detect Atrial Fibrillation (AFib), Acute Myocardial Infarction (STEMI), Conduction Blocks (LBBB, RBBB, AV Block), or Long-QT Syndrome.
4. **Lack of Deep Learning Architectures**:
   - No raw-waveform models (1D-CNN, ResNet-1D, BiLSTM) exist. All inference depends on handcrafted feature engineering.
5. **No Automated Multi-Dataset Ingestion**:
   - Datasets cannot be downloaded, checksummed, inspected, or converted autonomously.

---

## 9. Proposed Multi-Dataset to Multi-Task Mapping

| Task Identifier | Clinical Objective | Target Datasets | Modality & Leads | Target Output Schema |
| :--- | :--- | :--- | :--- | :--- |
| **TASK 1: Beat Arrhythmia** | Beat-by-beat cycle ectopy screening | MIT-BIH Arrhythmia (`mitdb`),<br>MIT-BIH Supraventricular (`svdb`),<br>Normal Sinus Rhythm (`nsrdb`) | Single-lead (Lead II), $360\text{ Hz}$ / $128\text{ Hz}$ resampled | Multi-class: `Normal`, `PVC`, `SVEB`, `Other` |
| **TASK 2: Atrial Fibrillation** | Episode and rhythm-level AFib detection | MIT-BIH Atrial Fibrillation (`afdb`),<br>MIT-BIH Arrhythmia (`mitdb`) | Single/Dual-lead, rhythm strips | Binary: `AFib` vs `Non-AFib` |
| **TASK 3: 12-Lead Diagnostic** | Global clinical 12-lead diagnostic interpretation | PTB-XL (`ptb-xl`),<br>PTB Diagnostic (`ptbdb`),<br>PTB-XL+ | 12 standard leads, $500\text{ Hz}$ / $100\text{ Hz}$ | Multi-label: `NORM`, `MI`, `STTC`, `CD`, `HYP` |
| **TASK 4: ST/T & Ischemia** | ST-segment elevation, depression, T-wave inversion | MIT-BIH ST Change (`stdb`),<br>European ST-T (`edb`),<br>PTB-XL | 2-lead to 12-lead, calibrated mV | Multi-class: `Normal`, `ST_Elevation`, `ST_Depression` |
| **TASK 5: Signal Quality Gate** | Hardware and electrode trust check | Ingested waveforms across all sources with artifact annotations | All configurations | Multi-class: `GOOD`, `ACCEPTABLE`, `POOR`, `UNUSABLE` |

---

## 10. Architectural Blueprint & Action Plan for Phase 1

To transition into the autonomous multi-dataset platform without disturbing existing functionality:

1. **Create `src/datasets/` Module**:
   - `registry.py`: Formal dataset catalog containing metadata, official PhysioNet/source URLs, licenses, sampling rates, lead configurations, and task suitability.
   - `downloader.py`: Resilient, checksum-verifying downloader using official `wfdb` and HTTPS endpoints with retry mechanisms and manual-fallback detection.
   - `inspector.py`: Automated dataset introspection generating inspection reports (`reports/datasets/<id>_inspection.json` and `.md`).
   - `patient_index.py`: Patient demographic registry extracting patient identifiers to prevent multi-record cross-contamination.
   - `label_mapper.py`: Deterministic mapping from dataset-specific diagnostic codes (SCP-ECG, SNOMED, AAMI) to canonical task categories.
   - `standardized_record.py`: Universal internal schema (`StandardizedECGRecord`).
2. **Directory Structure**:
   - Maintain `data/datasets/<dataset_id>/raw/`, `processed/`, `metadata/`, `annotations/`.
   - Maintain `configs/datasets/`, `configs/label_mappings/`, and `configs/preprocessing/`.
3. **Preservation Guarantee**:
   - Keep the existing `models/classifier.pkl` and `src/ml/inference/inference_engine.py` as the active production baseline until prospective multi-dataset models surpass all technical and clinical validation criteria.
