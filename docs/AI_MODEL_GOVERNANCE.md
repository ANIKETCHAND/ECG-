# Artificial Intelligence & Machine Learning Governance Specification

## 1. Model Lifecycle & Promotion Gates

All algorithmic and deep-learning models deployed within ECG Guardian adhere to strict lifecycle governance:

```
EXPERIMENTAL  ──►  CANDIDATE  ──►  VALIDATED  ──►  PRODUCTION
      │                 │               │               │
(Exploratory)    (Benchmarking)    (Held-Out Test   (Clinically
                                    Zero-Leakage)     Active)
```

### Governance Rules:
1. **No Synthetic Training Data in Production**:
   - Any model fitted to synthetic, simulated, or demo fixtures is designated `SYNTHETIC_PLACEHOLDER`.
   - The model loader (`src/ml/models/registry.py`) strictly prohibits promoting synthetic artifacts to `PRODUCTION`.
2. **Strict Patient-Level Split (Zero Patient Overlap)**:
   - Training and testing partitions must split at the individual subject/patient level.
   - Heartbeats from the same patient must never appear in both training and evaluation sets.
3. **Reproducibility & Hash Ledger**:
   - Model weights, feature scalers, and hyperparameter logs are versioned with SHA-256 hashes recorded in `models/registry/catalog.json`.

---

## 2. Production Model Card: `ECG-RF-2.0.0-PROD`

### 2.1 Model Overview
- **Identifier**: `ECG-RF-2.0.0-PROD` (Baseline: `ECG-RF-1.0.0`)
- **Architecture**: Balanced Random Forest Classifier (250 estimators, balanced subsampling)
- **Task**: Single-Lead (Lead II) Beat Arrhythmia Classification
- **Classes**: Normal (`N`), Premature Ventricular Contraction (`PVC`), Other Supraventricular Ectopy (`Other`)
- **Input Features**: 28 morphological and temporal features extracted via Pan-Tompkins QRS detection and wavelet decomposition.

### 2.2 Dataset Partitioning (MIT-BIH Arrhythmia Database)
- **Total Database Records**: 48 records (360 Hz, ambulatory 2-channel ECG).
- **Training Records (34 patients, 77,135 heartbeats)**:
  `100, 101, 104, 106, 108, 109, 111, 113, 114, 115, 116, 117, 118, 121, 122, 123, 124, 201, 202, 208, 209, 210, 212, 213, 214, 215, 217, 220, 221, 228, 230, 232, 233, 234`
- **Held-Out Testing Records (14 unseen patients, 32,446 heartbeats)**:
  `102, 103, 105, 107, 112, 119, 200, 203, 205, 207, 219, 222, 223, 231`
- **Patient Overlap**: **0 records** (Strict zero-leakage verification enforced by `test_training_integrity.py`).

### 2.3 Measured Performance on Unseen Test Patients

| Metric | Measured Value | Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **90.86%** | Overall beat classification accuracy |
| **Balanced Accuracy** | **87.52%** | Unweighted mean across class recalls |
| **Macro F1-Score** | **83.48%** | Equal-weight harmonic mean across 3 classes |
| **Weighted F1-Score** | **91.20%** | Prevalence-weighted harmonic mean |
| **Cohen's Kappa** | **0.768** | Substantial inter-rater agreement |

#### Per-Class Performance:

| Class | Support (Beats) | True Positives | False Positives | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal (`N`)** | 25,142 | 23,277 | 806 | 96.65% | 92.58% | **94.57%** |
| **PVC (`V`)** | 2,562 | 2,182 | 1,282 | 62.99% | 85.17% | **72.42%** |
| **Other (`O`)** | 4,742 | 4,022 | 877 | 82.10% | 84.82% | **83.44%** |

### 2.4 Clinical Safety Considerations
- **PVC False Positive Rate**: 37.01% of model PVC alerts are false discoveries (precision 62.99%). This is documented in the clinician user interface. Clinicians must verify all PVC classifications before prescribing antiarrhythmic agents or ordering electrophysiology studies.
- **Single-Lead Limitation**: The model does not detect ischemic changes, ST elevations, or bundle branch blocks with anatomical localization.

---

## 3. Regulatory Alignment

Model governance conforms to:
- **FDA Good Machine Learning Practice (GMLP)**: Principle 6 (Separation of training, tuning, and test sets), Principle 8 (Demonstration of model performance on representative patient populations).
- **ISO/IEC 42001**: Artificial Intelligence Management System.
- **CDSCO MDR 2017 Class B SaMD Guidelines**: Software lifecycle and clinical performance justification.
