# ECG Guardian — ML Reproducibility & Governance Guide

## 1. System Overview & Core Principles
ECG Guardian provides an autonomous, multi-dataset machine learning pipeline engineered around safety, strict patient isolation, and deterministic reproducibility.

### Fundamental Safety Mandates:
1. **Zero Patient Contamination:** Split boundaries are strictly enforced at the canonical patient level (`mit_bih_arrhythmia_pt_XXX`). Individual beats or segments from the same patient are **never** divided between training and testing sets.
2. **Deterministic Preprocessing Provenance:** Scalers (`StandardScaler`) and filter baselines are fitted **strictly** on the training patient cohort. Evaluation and test data are strictly transformed out-of-sample.
3. **Traceable Model Registry:** All models transition through an audited lifecycle state (`EXPERIMENTAL` -> `CANDIDATE` -> `VALIDATED` -> `PRODUCTION`). The active production model (`ECG-RF-1.0.0`) is never replaced without rigorous clinical non-inferiority validation and an audited migration report.
4. **Non-Diagnostic Device Boundaries:** Probabilities represent model confidence in feature extraction space, not verified clinical diagnostic truth. All outputs require mandatory qualified physician sign-off.

---

## 2. Directory Layout & Artifact Storage

```
AI-ECG-Analyzer/
├── configs/
│   ├── label_mappings/         # Dataset-to-standard class label maps
│   │   ├── mit_bih_arrhythmia.json
│   │   ├── mit_bih_afdb.json
│   │   └── ptb_xl.json
│   └── preprocessing/          # Signal filtering & resampling configurations
│       ├── beat_arrhythmia_v1.json
│       ├── ptbxl_12lead_v1.json
│       └── af_v1.json
├── data/
│   ├── datasets/               # Raw PhysioNet records and metadata
│   ├── processed/              # Zero-leakage train/test feature matrices
│   └── splits/                 # Deterministic patient-level split manifests
│       └── beat_arrhythmia/
│           └── v1.json
├── models/
│   ├── production/             # Active production models (ECG-RF-1.0.0)
│   ├── candidate/              # Candidate models (ECG-RF-2.0.0, ECG-MLP-1.0.0)
│   └── registry/
│       └── catalog.json        # Central model governance ledger
├── reports/
│   ├── datasets/               # Automated dataset inspection reports
│   ├── experiments/            # Benchmark metrics (AUROC, AUPRC, ECE)
│   └── migration/              # Current vs. new comparison reports
└── training/
    ├── train_all.py            # End-to-end autonomous multi-model training runner
    └── create_splits.py        # Patient-level splitting engine
```

---

## 3. End-to-End Reproduction Instructions

To reproduce all dataset inspections, patient splits, model training, evaluation metrics, and comparison reports from scratch:

```bash
# 1. Inspect Datasets
python training/inspect_datasets.py

# 2. Generate Patient-Level Splits (Zero Leakage)
python training/create_splits.py

# 3. Execute Autonomous Training & Validation Pipeline
python training/train_all.py

# 4. Run Pytest Verification Suite
python -m pytest tests/
```

---

## 4. Current Benchmark Results (Independent Test Split: Records 101, 119)

| Model Identifier | Architecture | Accuracy | Weighted F1 | PVC Sensitivity | AUROC | ECE | Lifecycle Status |
|---|---|---|---|---|---|---|---|
| `ECG-RF-1.0.0` | Random Forest | 97.86% | 97.92% | 99.89% | 0.9969 | 0.1277 | **PRODUCTION** |
| `ECG-RF-2.0.0` | Balanced Random Forest | 98.33% | 98.31% | 99.67% | 0.9973 | 0.0989 | **VALIDATED** |
| `ECG-MLP-1.0.0` | Waveform 1D Neural Net | 95.96% | 96.04% | 94.31% | 0.9924 | 0.0243 | **EXPERIMENTAL** |
| `ECG-LR-1.0.0` | Logistic Regression | 95.21% | 95.50% | 95.80% | 0.9922 | 0.0306 | **BASELINE** |

