# Statement of Intended Use & Clinical Indication
**Document ID:** REG-INT-2026-001  
**Regulatory Context:** Medical Devices Rules (MDR) 2017 (Central Drugs Standard Control Organisation - CDSCO, India), IMDRF SaMD Framework (N10/N12)  
**Software Name:** AI ECG Analysis & Clinical Review Platform  
**Version:** 2.0.0-dev  
**Status:** Draft for Technical & Clinical Governance Review  

---

## 1. Intended Purpose

The **AI ECG Analysis & Clinical Review Platform** is a Software as a Medical Device (SaMD) intended to process, measure, and analyze digital electrocardiogram (ECG) data acquired from compatible, legally marketed ECG acquisition hardware. 

The software applies signal processing algorithms and trained machine-learning classifiers to:
1. Assess the technical quality of the uploaded ECG recording (Signal-to-Noise Ratio, baseline drift, powerline interference, motion artifacts).
2. Detect cardiac R-peaks and segment individual cardiac cycles.
3. Quantify fundamental electrophysiological parameters (Heart Rate, R-R intervals, QRS duration when measurable).
4. Provide an automated, preliminary **AI-assisted classification** of supported cardiac rhythm patterns (Normal Sinus Rhythm versus Premature Ventricular Contractions / Ventricular Ectopy) on supported single-lead representations (Modified Lead II).
5. Present the preprocessed waveform, extracted parameters, model class probabilities, and diagnostic warnings to an appropriately qualified healthcare professional for mandatory clinical review, modification, and electronic sign-off.

> [!IMPORTANT]
> **PRIMARY CLINICAL SAFETY PRINCIPLE:**  
> The software is designed strictly as an **adjunctive clinical decision-support tool**. It provides preliminary analytical information to assist a qualified clinician. It **does NOT replace clinical judgment** and is **NOT an autonomous diagnostic device**.

---

## 2. Regulatory Status Disclosure

> [!CAUTION]
> **STATUTORY REGULATORY DISCLAIMER:**
> - This software is **NOT currently approved** by the Central Drugs Standard Control Organisation (CDSCO, India) under the Medical Devices Rules, 2017.
> - This software is **NOT cleared or approved** by the United States Food and Drug Administration (FDA) under Section 510(k) or De Novo pathways.
> - This software does **NOT carry a CE Mark** under the European Union Medical Device Regulation (EU MDR 2017/745).
> - This software has **NOT completed prospective clinical validation trials** in human hospital patient populations.
> - The software must **NOT be deployed for commercial clinical diagnostic use** until appropriate investigational device approvals, clinical evaluations, and statutory marketing authorizations have been formally granted by relevant regulatory authorities.

---

## 3. Intended User Population

The software is designed for use exclusively by individuals with appropriate medical training, electrophysiology education, or certified clinical roles:
1. **Cardiologists**: Specialists evaluating complex arrhythmia, confirming or overriding AI findings, and finalizing medical reports.
2. **Physicians & General Practitioners**: Registered Medical Practitioners (RMPs) conducting preliminary patient cardiovascular evaluations.
3. **Authorized ECG Technicians**: Hospital staff trained in ECG lead placement, recording protocols, and data export.
4. **Clinical Researchers & Biostatisticians**: Authorized academic personnel evaluating algorithm calibration, dataset performance, and model bias under institutional oversight.

**Layperson Use**: This software is **NOT intended for direct-to-consumer (OTC)**, home, or unmonitored use by patients or laypersons.

---

## 4. Intended Clinical Environment

The software is intended for deployment within controlled clinical and healthcare environments:
- Inpatient hospital wards and outpatient cardiology clinics.
- Cardiac diagnostic laboratories and telecardiology reading centers.
- Academic medical centers and clinical research organizations (CROs).
- Standard healthcare IT infrastructures adhering to ISO 27001 / HIPAA / DISHA cybersecurity standards.

**Environmental Exclusions**: The software is **NOT validated** for:
- Moving transport ambulances without certified telemetry hardware.
- Unmonitored consumer wearable fitness tracking.
- Home emergency monitoring without clinical oversight.

---

## 5. Intended Input Data

1. **Digital ECG Waveforms**: Single-lead digital ECG voltage series obtained from calibrated recording devices, sampled at supported rates ($\ge 125\text{ Hz}$, nominal $250 - 500\text{ Hz}$, resampled internally to $360\text{ Hz}$) with valid millivolt ($\text{mV}$) scaling.
2. **Supported Digital Formats**: Raw CSV, TSV, TXT (delimited numeric voltage data), NumPy arrays (`.npy`), and native WFDB format.
3. **Supported Lead Configuration**: **Modified Lead II (MLII)** or standard **Lead II**.
4. **Document Ingestion (Auxiliary Information)**: Clinical ECG PDF reports and scanned report images (`JPG`, `PNG`) may be ingested **strictly for metadata and printed measurement extraction**. If an extractable 1D waveform cannot be verified with $\ge 60\%$ confidence, the software will **NOT perform AI waveform classification**.

---

## 6. Intended Output

The software generates a structured, immutable analytical record (`ECGAnalysisResult`) containing:
1. **Signal Quality Assessment**: Categorical rating (`GOOD`, `ACCEPTABLE`, `POOR`, `UNUSABLE`) and quantitative SNR in decibels (dB).
2. **Computed Physiological Parameters**: Heart rate in beats per minute (BPM), mean R-R interval in milliseconds, and detected cardiac beat count.
3. **AI Rhythm Pattern Assessment**: Preliminary classification (`Normal Rhythm`, `Ventricular Ectopy (PVC)`, or `Uncertain / Unvalidated Other`).
4. **Model Class Probabilities**: Normalized probability distribution representing model statistical likelihood (explicitly labeled: *"Model probability is not equivalent to clinical diagnostic certainty"*).
5. **Traceability Metadata**: Active model ID, model version (`ECG-RF-1.0.0`), preprocessing pipeline version, execution timestamp, and data hash.
6. **Clinician Interface**: An interactive multi-scale waveform review portal enabling the physician to confirm, reject, or amend the AI output before generating a finalized signed clinical report.

---

## 7. Contraindications & Prohibitions (What the Software is NOT Intended For)

The software is strictly **NOT intended** for:
1. **Autonomous Clinical Diagnosis**: The software must never be configured to output a final medical diagnosis without an authorized physician's affirmative review and sign-off.
2. **Autonomous Treatment or Intervention**: The software must never directly trigger automated defibrillation, pace changes, medication infusion pumps, or autonomous clinical workflows.
3. **Immediate Life-Threatening Emergency Decisions**: The software is not a real-time bedside intensive care monitor (ICU monitor) and must not be used as the sole basis for acute resuscitation decisions during cardiac arrest.
4. **Unsupported Pathologies**: The current model is **NOT validated** to detect or rule out:
   - Acute Myocardial Infarction (STEMI / NSTEMI)
   - Atrial Fibrillation (AFib) or Atrial Flutter
   - Ventricular Fibrillation (VFib)
   - Third-Degree Atrioventricular Block
   - Brugada Syndrome or Long-QT Syndrome
   - Electrolyte Abnormalities (Hyperkalemia / Hypocalcemia)
5. **Unsupported Patient Populations**:
   - Pediatric patients under 18 years of age (unvalidated baseline).
   - Patients with active electronic cardiac pacemakers or implantable cardioverter-defibrillators (ICDs) where pacing spikes distort morphological feature extraction.
6. **Data Fabrication**: The software strictly forbids synthesizing, interpolating, or predicting values for missing leads, missing metadata, or corrupted waveforms.

---

## 8. Clinical Responsibility Boundary

In accordance with good clinical practice and regulatory jurisprudence:
- The treating physician retains sole legal and professional responsibility for the patient's diagnosis and therapeutic plan.
- The AI software provides secondary, algorithmic screening assistance. 
- In any instance of conflict between clinical judgment and AI output, **clinical judgment shall always supersede the AI output**.
