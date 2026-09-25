# Model Card: ECG Guardian Beat-Level Arrhythmia Classifier

## 1. Model Details
- **Model Name:** ECG-RF-2.0.0
- **Model Architecture:** Balanced Random Forest Classifier (`RandomForestClassifier`)
- **Ensemble Configuration:** 250 decision trees, maximum depth 20, `class_weight="balanced_subsample"`, random state 42.
- **Input Modality:** 28 morphometric and spectral beat-level features extracted from preprocessed ECG Lead II signals (bandpass filtered 0.5–40 Hz, baseline wander corrected, R-peak aligned).
- **Target Classes:**
  1. `Normal` (Sinus Rhythm)
  2. `PVC` (Premature Ventricular Contractions)
  3. `Other` (Supraventricular Ectopy / Nodal / Unclassified)
- **Deployment Platform:** Vercel Serverless Python Runtime & FastAPI Backend.

---

## 2. Dataset & Zero-Leakage Inter-Patient Partitioning

### 2.1 Dataset Provenance
- **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet).
- **Sampling Rate:** 360 Hz.
- **Partitioning Strategy:** **Strict Inter-Patient Splitting** (Beats from any given patient appear *exclusively* in either the training set or the test set; never in both).

| Partition | Patient Record IDs | Total Beats | Normal Beats | PVC Beats | Other Beats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | 100, 106, 200, 213 | 10,152 | 8,130 (80.08%) | 1,931 (19.02%) | 91 (0.89%) |
| **Test (Held-out)** | 101, 119, 208 | 6,807 | 4,989 (73.29%) | 1,809 (26.58%) | 9 (0.13%) |
| **Total** | 7 Distinct Patients | 16,959 | 13,119 | 3,740 | 100 |

### 2.2 Leakage Audit & Verification
- Record IDs in Train: `{"100", "106", "200", "213"}`
- Record IDs in Test: `{"101", "119", "208"}`
- **Intersection:** `Train ∩ Test = ∅` (Zero patient contamination).
- Standard Scaler was fitted strictly on `X_train` and applied forward to `X_test`.

---

## 3. Evaluation Metrics on Unseen Test Patients (6,807 Beats)

### 3.1 Primary Model: Random Forest (`n_estimators=250`, `max_depth=20`)
- **Overall Accuracy:** **98.74%** (6,721 / 6,807 correct predictions)
- **Macro Precision:** 65.30%
- **Macro Recall:** 66.06%
- **Macro F1-Score:** 65.67%
- **Weighted F1-Score:** **98.68%**

#### Detailed Per-Class Breakdown:
| Class | Precision | Recall | F1-Score | Support | Clinical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | **99.74%** | **98.62%** | **99.17%** | 4,989 | High specificity ensures low false alarms for healthy sinus rhythms |
| **PVC** | **96.16%** | **99.56%** | **97.83%** | 1,809 | **99.56% sensitivity** ensures dangerous ventricular ectopy is captured |
| **Other** | 0.00% | 0.00% | 0.00% | 9 | Rare class domain shift (detailed below) |

#### Confusion Matrix:
```
                Predicted Normal    Predicted PVC    Predicted Other
Actual Normal         4,920               68                1
Actual PVC                8            1,801                0
Actual Other              5                4                0
```

### 3.2 Baseline Model: Logistic Regression (`class_weight="balanced"`)
- **Overall Accuracy:** **95.21%**
- **Normal Recall:** 95.11% (Precision: 98.34%)
- **PVC Recall:** 95.80% (Precision: 89.61%)
- **Other Recall:** 33.33% (Precision: 6.25%, Support: 9)

---

## 4. In-Depth Analysis of the "Other" Class

1. **Extreme Sample Imbalance:**
   - In the training set, `Other` comprises only 0.89% of beats (91 / 10,152).
   - In the test set, `Other` comprises only 0.13% of beats (9 / 6,807).
2. **Sub-Type Domain Shift:**
   - In the training records (100, 106, 200, 213), all 91 `Other` beats are Atrial Premature Contractions (symbols `A` and `a`).
   - In the test records (101, 119, 208), the 9 `Other` beats include:
     - 4 Unclassifiable beats (`Q`)
     - 2 Supraventricular premature beats (`S`)
     - 3 Atrial premature beats (`A`)
   - The tree model has never seen `Q` or `S` beats during training, resulting in `Q` beats being classified as PVCs (due to bizarre morphology) and `S` beats being classified as Normal (due to narrow QRS).
3. **Clinical Mitigation:**
   - The Clinical Decision Support (CDS) engine flags beats with elevated uncertainty or borderline local RR ratios.
   - All AI predictions are categorized as AI-assisted decision support, requiring mandatory physician review.

---

## 5. Regulatory & Clinical Compliance
- **Intended Use:** Computer-assisted detection of ventricular arrhythmias and cardiac conduction abnormalities in single-lead and multi-lead ECGs.
- **Safety Gate:** Any signal with SNR < 3 dB, severe baseline drift, or non-physiological amplitude (> 10 mV or flatline) is rejected by the `SignalQualityGatekeeper` before reaching the ML classifier.
- **CDSCO MDR 2017 & IEC 62304 Compliance:** Class B Software as a Medical Device (SaMD). Requires qualified physician review before clinical action.
