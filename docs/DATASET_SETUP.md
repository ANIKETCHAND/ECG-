# ECG Dataset Setup & Ingestion Guide
**Platform:** ECG GUARDIAN  
**Document Ref:** `DOC-DATASET-SETUP-001`  

---

## 1. Directory Structure

Raw and processed datasets are maintained in strict isolation under `data/datasets/<dataset_id>/`:

```text
data/datasets/<dataset_id>/
├── raw/            # Untouched original recordings (.hea, .dat, .atr, .csv)
├── processed/      # Standardized NumPy / CSV feature arrays
├── metadata/       # Manifests, patient demographics, and checksums
└── annotations/    # Parsed ground-truth beat/rhythm labels
```

---

## 2. Approved Dataset Inventory & Access Instructions

### Group A: Beat & Rhythm Datasets

1. **MIT-BIH Arrhythmia Database (`mit_bih_arrhythmia`)**:
   - **Source:** [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
   - **License:** Open PhysioNet (ODC-BY 1.0).
   - **Command:** `python training/download_datasets.py --dataset mit_bih_arrhythmia`

2. **MIT-BIH Supraventricular Arrhythmia Database (`mit_bih_svdb`)**:
   - **Source:** [https://physionet.org/content/svdb/1.0.0/](https://physionet.org/content/svdb/1.0.0/)
   - **License:** Open PhysioNet (ODC-BY 1.0).
   - **Command:** `python training/download_datasets.py --dataset mit_bih_svdb`

3. **MIT-BIH Atrial Fibrillation Database (`mit_bih_afdb`)**:
   - **Source:** [https://physionet.org/content/afdb/1.0.0/](https://physionet.org/content/afdb/1.0.0/)
   - **Command:** `python training/download_datasets.py --dataset mit_bih_afdb`

4. **MIT-BIH Normal Sinus Rhythm Database (`mit_bih_nsrdb`)**:
   - **Source:** [https://physionet.org/content/nsrdb/1.0.0/](https://physionet.org/content/nsrdb/1.0.0/)
   - **Command:** `python training/download_datasets.py --dataset mit_bih_nsrdb`

---

### Group B: Diagnostic 12-Lead ECG Datasets

5. **PTB-XL: Clinical 12-Lead ECG Database (`ptb_xl`)**:
   - **Source:** [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)
   - **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
   - **Size:** 3.1 GB.
   - **Command:** `python training/download_datasets.py --dataset ptb_xl`
   - **Manual Download (Alternative):**
     1. Download `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip` from PhysioNet.
     2. Unpack into `data/datasets/ptb_xl/raw/`.

6. **PTB Diagnostic ECG Database (`ptbdb`)**:
   - **Source:** [https://physionet.org/content/ptbdb/1.0.0/](https://physionet.org/content/ptbdb/1.0.0/)
   - **Command:** `python training/download_datasets.py --dataset ptbdb`

---

### Group C: ST/T Myocardial Ischemia Datasets

7. **MIT-BIH ST Change Database (`mit_bih_stdb`)**:
   - **Source:** [https://physionet.org/content/stdb/1.0.0/](https://physionet.org/content/stdb/1.0.0/)
   - **Command:** `python training/download_datasets.py --dataset mit_bih_stdb`

8. **European ST-T Database (`european_st_t`)**:
   - **Source:** [https://physionet.org/content/edb/1.0.0/](https://physionet.org/content/edb/1.0.0/)
   - **Command:** `python training/download_datasets.py --dataset european_st_t`

---

## 3. Data Integrity & Checksum Verification

Every downloaded record is checked for non-empty binary files (`.dat`) and matching headers (`.hea`). Corrupted downloads are automatically flagged and quarantined before entering the preprocessing pipeline.
