# Regulatory Model Card: ECG-RF-1.0.0
**Model Identifier:** `ECG-RF-1.0.0`  
**Model Architecture:** Random Forest Classifier (100 Estimators)  
**Task:** Beat-by-Beat Cardiac Cycle Abnormality Screening  
**Release Date:** September 2026  
**Status:** Approved for Technical Baseline / Research Decision Support  
**Regulatory Status:** Unvalidated Investigational Prototype. Not for autonomous clinical diagnosis.  

---

## 1. Model Overview & Purpose

- **Model ID**: `ECG-RF-1.0.0`
- **Algorithm**: Balanced Ensemble of 100 Decision Trees (`sklearn.ensemble.RandomForestClassifier`).
- **Feature Dimension**: 28 engineered numerical features (Time, Frequency, Morphology, R-R dynamics).
- **Intended Use**: Secondary algorithmic decision support assisting authorized healthcare professionals in identifying Normal Sinus Rhythm vs. Premature Ventricular Contractions (PVC) on single-lead Modified Lead II ECG recordings.
- **Not Intended Use**: Autonomous diagnostic decision-making, patient discharge, emergency triage without physician review, or detection of acute ischemic syndromes (STEMI).

---

## 2. Input Requirements

- **Signal Modality**: 1D continuous digital physiological voltage series.
- **Supported Lead**: **Modified Lead II (MLII)** or standard **Lead II**.
- **Sampling Frequency**: $360\text{ Hz}$ (other sampling rates must be resampled).
- **Signal Units**: Millivolts ($\text{mV}$) or standardized unit variance.
- **Minimum Duration**: $1.5\text{ seconds}$ (at least 2 consecutive cardiac cycles).
- **Optimal Duration**: $10.0\text{ seconds}$.

---

## 3. Supported Classes & Diagnostic Boundaries

The model outputs probabilities across three standardized classes:
1. **`Normal` (Normal Sinus Rhythm)**: Normal supraventricular conduction.
2. **`PVC` (Premature Ventricular Contraction)**: Ventricular ectopy / premature ventricular beats.
3. **`Other` (Other Ectopic / Paced Beats)**: Supraventricular, atrial premature, or fusion beats.

> [!CAUTION]
> **UNVALIDATED CLASS DISCLOSURE (CLASS 'OTHER'):**  
> On unseen test evaluations, class `Other` had a support of only 9 beats and achieved **0.0% sensitivity (recall)**. **The model must NOT be relied upon to detect supraventricular arrhythmias, atrial ectopy, or fusion beats.** It is clinically validated only for distinguishing Normal Sinus Rhythm from Ventricular Ectopy (PVC).

---

## 4. Training Data & Leakage Prevention

- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet).
- **Partitioning Method**: **Strict Patient / Record-Level Split** (enforced by `training/train_test_split.py`). Beats from the same patient never appear in both training and test sets.
- **Training Records**: Records `100`, `106`, `200`, `213` ($10,152$ beats).
- **Unseen Test Records**: Records `101`, `119`, `208` ($6,807$ beats).
- **Patient Count**: 4 training patients, 3 testing patients.

---

## 5. Performance Metrics (Unseen Patient Test Set)

| Metric | Measured Value | Standard / Target |
| :--- | :--- | :--- |
| **Overall Accuracy** | 97.86% | $\ge 90.0\%$ |
| **Weighted F1-Score** | 97.92% | $\ge 90.0\%$ |
| **Macro F1-Score** | 65.07% | Depressed by minority class `Other` |
| **Normal Precision / Recall / F1** | 99.88% / 97.29% / **98.57%** | High fidelity |
| **PVC Precision / Recall / F1** | 93.58% / 99.89% / **96.63%** | High sensitivity |
| **Other Precision / Recall / F1** | **0.00% / 0.00% / 0.00%** | **Unvalidated / Inadequate evidence** |

---

## 6. Top Feature Importances (Gini Weight)

1. `local_rr_ratio` ($0.2169$): Ratio of preceding R-R interval to local running average.
2. `pre_rr` ($0.1339$): Preceding coupling interval duration in seconds.
3. `autocorr_first_peak` ($0.1226$): Morphological periodicity and wave symmetry.
4. `spectral_entropy` ($0.0690$): Complexity of frequency distribution.
5. `max_power` ($0.0630$): Peak spectral power density.

---

## 7. Known Failure Modes & Limitations

1. **R-Peak Misdetection Sensitivity**: If motion artifacts or high-amplitude T-waves cause false R-peak detections, all 28 feature values degrade, leading to false-positive PVC classifications.
2. **Bundle Branch Block Confusion**: Wide QRS complexes caused by Left or Right Bundle Branch Block may mimic PVC morphology, creating false-positive ectopy alerts.
3. **Paced Rhythm Incompatibility**: Electronic pacemaker spikes create severe high-frequency derivative spikes (`max_slope`), corrupting feature space.
4. **Lead Dependency**: Applying this model to Lead V1, aVR, or precordial leads will generate invalid results because feature normalization assumes Lead II polarity and amplitude ratios.

---

## 8. Change Control & Governance

- **Weight Immutability**: Production weights in `models/production/classifier.pkl` are hash-locked.
- **Rollback Version**: `None` (Initial baseline release).
- **Retraining Authorization**: Retraining requires formal Protocol Approval under `regulatory/ALGORITHM_CHANGE_PROTOCOL.md`. Automated retraining from clinical user data is strictly prohibited.
