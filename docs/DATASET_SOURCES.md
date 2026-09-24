# ECG GUARDIAN — Official Dataset Sources & License Register

**Document Version:** 1.0.0  
**Effective Date:** September 2026  
**Compliance Standard:** CDSCO MDR 2017 / PhysioNet Credentialed Data Use Agreement / Open Data Commons

---

## 1. Approved ECG Databases & Provenance Register

| Dataset Name | Official Source URL | Version | License Type | Access Requirements | Download Status | Leads | Sampling Rates | Intended ML Task |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MIT-BIH Arrhythmia Database** | `https://physionet.org/content/mitdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **AVAILABLE / VERIFIED** | 2 (`MLII`, `V1`) | 360 Hz | Beat Arrhythmia (`TASK_BEAT_ARRHYTHMIA`) |
| **MIT-BIH Supraventricular Arrhythmia** | `https://physionet.org/content/svdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 2 (`MLII`, `V1`) | 128 Hz | Supraventricular Ectopy (`TASK_BEAT_ARRHYTHMIA`) |
| **MIT-BIH Atrial Fibrillation Database** | `https://physionet.org/content/afdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 2 (`ECG1`, `ECG2`) | 250 Hz | Atrial Fibrillation Detection (`TASK_AF_DETECTION`) |
| **MIT-BIH Normal Sinus Rhythm Database** | `https://physionet.org/content/nsrdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 2 (`ECG1`, `ECG2`) | 128 Hz | Negative Baseline & Control Group |
| **MIT-BIH Long-Term ECG Database** | `https://physionet.org/content/ltdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 2 (`ECG1`, `ECG2`) | 128 Hz | Diurnal & Long-Term Stability |
| **PTB-XL Diagnostic ECG Database** | `https://physionet.org/content/ptb-xl/1.0.3/` | 1.0.3 | CC BY 4.0 | Open Public Access | **REGISTERED (On-Demand)** | 12 (Standard 12 Leads) | 100, 500 Hz | 12-Lead Clinical Superclasses (`TASK_12LEAD_DIAGNOSTIC`) |
| **PTB Diagnostic ECG Database** | `https://physionet.org/content/ptbdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 15 (12 Leads + Frank XYZ) | 1000 Hz | Diagnostic Resolution & MI Validation |
| **PTB-XL+ Companion Database** | `https://physionet.org/content/ptb-xl-plus/1.0.1/` | 1.0.1 | CC BY 4.0 | Open Public Access | **REGISTERED (On-Demand)** | 12 (Standard 12 Leads) | 500 Hz | Fiducial Points & Machine Statements |
| **MIT-BIH ST Change Database** | `https://physionet.org/content/stdb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 2 (`ECG1`, `ECG2`) | 360 Hz | Transient Ischemic ST Episodes (`TASK_ST_ANALYSIS`) |
| **European ST-T Database** | `https://physionet.org/content/edb/1.0.0/` | 1.0.0 | Open PhysioNet (ODC-BY 1.0) | Open Public Access | **REGISTERED (On-Demand)** | 2 (`ECG1`, `ECG2`) | 250 Hz | Myocardial Ischemia & Repolarization (`TASK_ST_ANALYSIS`) |
| **MIMIC-IV-ECG** | `https://physionet.org/content/mimic-iv-ecg/1.0/` | 1.0 | PhysioNet Credentialed 1.5.0 | **MANUAL_ACCESS_REQUIRED** | **RESTRICTED (Credentialed)** | 12 (Standard 12 Leads) | 500 Hz | Hospital EMR Linked 12-Lead Research |
| **MIMIC-IV Clinical Database** | `https://physionet.org/content/mimiciv/2.2/` | 2.2 | PhysioNet Credentialed 1.5.0 | **MANUAL_ACCESS_REQUIRED** | **RESTRICTED (Credentialed)** | N/A (Clinical Data) | N/A | Medication Correlates & ICU Outcomes |

---

## 2. Credentialed Access Instructions (MIMIC-IV Datasets)

Under Rule 0 (Zero Data Fabrication) and Rule 47 (Do Not Force Training If Inaccessible), MIMIC-IV and MIMIC-IV-ECG must never be downloaded through unauthorized mirrors or fabricated.

To ingest MIMIC-IV records:
1. Complete the CITI training course *Data or Specimens Only Research*.
2. Submit a formal application on PhysioNet (`https://physionet.org/settings/credentialing/`).
3. Sign the PhysioNet Data Use Agreement (DUA).
4. Export your credentialed session key into environment variable:
   ```bash
   export PHYSIONET_TOKEN="your_verified_token"
   ```
5. Run the downloader utility:
   ```bash
   python training/download_datasets.py --dataset mimic_iv_ecg
   ```
If the token is absent or pending verification, the system will explicitly report:
`STATUS: MANUAL_DOWNLOAD_REQUIRED`.
