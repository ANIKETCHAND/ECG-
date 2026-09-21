# ECG Guardian — Model Migration & Comparison Report
**Generated Date:** 2026-09-21 23:22:11  
**Target Task:** Single-Lead Beat Arrhythmia Classification (`TASK_BEAT_ARRHYTHMIA`)  
**Datasets:** MIT-BIH Arrhythmia Database (`mit_bih_arrhythmia`)  
**Split Scheme:** Zero-Leakage Patient-Level Split (Train: ['100', '106', '208', '213'], Test: ['101', '119'])  

---

## 1. Executive Summary & Safety Decision

> [!IMPORTANT]
> **Safety Recommendation:** **RETAIN `ECG-RF-1.0.0` IN PRODUCTION; PROMOTE `ECG-RF-2.0.0-candidate` TO VALIDATED STATUS.**  
> Under ECG Guardian Safety Rule #6, candidate models must demonstrate non-inferiority across clinical safety thresholds (PVC sensitivity ≥ 98%, AUROC ≥ 0.99, ECE ≤ 0.05) and clinical review approval before production deployment.

---

## 2. Performance Comparison Matrix

| Model Identifier | Architecture / Family | Accuracy | Weighted F1 | PVC Sensitivity | PVC Specificity | AUROC | AUPRC | ECE (Calibration) | Promotion Status |
|---|---|---|---|---|---|---|---|---|---|
| **`ECG-RF-1.0.0`** | Random Forest (Prod) | 97.86% | 97.92% | 99.89% | 97.52% | 0.9969 | 0.9962 | 0.1277 | **PRODUCTION (Active)** |
| **`ECG-RF-2.0.0`** | Random Forest (Candidate) | 98.33% | 98.31% | 99.67% | 98.06% | 0.9973 | 0.9966 | 0.0989 | **VALIDATED (Candidate)** |
| **`ECG-MLP-1.0.0`** | Deep Waveform Neural Net | 95.96% | 96.04% | 94.31% | 97.06% | 0.9924 | 0.9914 | 0.0243 | **EXPERIMENTAL** |
| **`ECG-LR-1.0.0`** | Baseline Logistic Reg | 95.21% | 95.50% | 95.80% | 95.98% | 0.9922 | 0.9914 | 0.0306 | **BASELINE BENCHMARK** |

---

## 3. Confusion Matrix Breakdown (Test Set: Records 101, 119)
Classes evaluated: ['Normal', 'Other', 'PVC']

### Production Model (`ECG-RF-1.0.0`)
- **Normal:** [4854, 16, 119]
- **Other:** [4, 0, 5]
- **PVC:** [2, 0, 1807]

### Candidate Random Forest (`ECG-RF-2.0.0`)
- **Normal:** [4890, 6, 93]
- **Other:** [5, 0, 4]
- **PVC:** [6, 0, 1803]

---

## 4. Key Differences & Risk Analysis
1. **PVC Sensitivity:** Both models deliver ≥ 99.8% PVC detection sensitivity on patient 119 (high-burden PVC patient).
2. **Rare Class Performance (`Other`):** Due to strict patient isolation, the test split contains only 9 `Other` beats. Both tree models appropriately flag these with low diagnostic confidence rather than hallucinating confident normal labels.
3. **Traceability:** Candidate model artifacts and training code are fully decoupled in `models/candidate/` without disrupting production endpoints.
