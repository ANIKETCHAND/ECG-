# ECG GUARDIAN: Statement of Intended Use & Clinical Indication
**Document ID:** DOC-INT-ECG-GUARDIAN-2026-001  
**Project Name:** ECG GUARDIAN  
**Regulatory Framework:** CDSCO Medical Devices Rules (MDR) 2017 (India), IMDRF SaMD Framework (N10/N12), IEC 62304:2006, ISO 14971:2019  
**Software Classification:** Software as a Medical Device (SaMD), Class B Clinical Decision Support  
**Version:** 1.0.0-dev  
**Status:** Approved Clinical Governance Specification  

---

## 1. Intended Purpose

**ECG GUARDIAN** is a Software as a Medical Device (SaMD) clinical decision support platform designed to assist qualified healthcare professionals in the technical verification, quality assessment, measurement quantification, rhythm abnormality analysis, and longitudinal tracking of electrocardiogram (ECG) recordings.

The platform performs the following sequential clinical decision support functions:
1. **Signal Ingestion & Integrity Verification:** Ingests digital ECG voltage recordings and clinical report documents from validated acquisition hardware, verifying sampling rate, file integrity, and channel metadata without synthesizing missing data.
2. **ECG Quality Copilot (Pre-Analysis Gatekeeper):** Objectively evaluates technical recording quality (Signal-to-Noise Ratio, baseline drift, 50/60 Hz powerline interference, clipping, flatlines, lead-level artifacts). Categorizes quality as `GOOD`, `ACCEPTABLE`, `POOR`, or `UNUSABLE`. Enforces the safety barrier: **UNUSABLE Signal $\rightarrow$ NO AI ANALYSIS**.
3. **Deterministic Electrophysiological Quantification:** Calculates ventricular heart rate (BPM), mean R-R interval (ms), QRS duration (ms), Bazett-corrected QTc (ms), and Fridericia-corrected QTc (ms) using established mathematical and signal processing algorithms.
4. **AI-Assisted Rhythm Pattern Analysis:** Applies a trained, locked machine-learning classifier (`ECG-RF-1.0.0`) to evaluate single-lead Modified Lead II (MLII) recordings for preliminary identification of **Normal Sinus Rhythm** versus **Premature Ventricular Contractions (PVC / Ventricular Ectopy)**.
5. **ECG Evidence Engine:** Attributes rhythm pattern predictions to specific detected cardiac cycles (e.g. Beat #37, Beat #84), presenting quantitative morphology, coupling intervals, and rhythm context to explain the algorithmic finding.
6. **ECG Machine vs AI Verification:** Extracts the machine-printed interpretation from source documents when available, performs semantic alignment, and flags discrepancies (`AGREE`, `MINOR_DIFFERENCE`, `SIGNIFICANT_DISAGREEMENT`) without unilaterally deciding which interpretation is correct.
7. **Longitudinal Trend & Delta Analysis:** Compares current ECG metrics against prior compatible ECGs for the same patient, presenting objective physiological deltas ($\Delta\text{HR}$, $\Delta\text{QRS}$, $\Delta\text{QTc}$) to track changes over time.
8. **Clinician Review & Cryptographic Sealing:** Mandates affirmative human review by a qualified physician (`CONFIRMED`, `MODIFIED`, or `REJECTED`) with recording of medical council registration credentials, cryptographically sealing the final medical report in an immutable, append-only audit trail.

---

## 2. Statutory Regulatory Disclosures

> [!CAUTION]
> **STATUTORY DISCLAIMER ON REGULATORY STATUS:**
> - **NOT CDSCO Approved:** This software is an investigational platform and has **NOT** received medical device registration or manufacturing licenses from the Central Drugs Standard Control Organisation (CDSCO, India) under the Medical Devices Rules (MDR), 2017.
> - **NOT FDA Cleared/Approved:** This software has **NOT** been cleared via 510(k) premarket notification or approved via PMA/De Novo pathways by the United States Food and Drug Administration (US FDA).
> - **NOT CE Marked:** This software does **NOT** bear a CE mark of conformity under the European Union Medical Device Regulation (EU MDR 2017/745).
> - **NOT Clinically Validated:** This software has **NOT** completed prospective clinical trials in clinical hospital patient cohorts.
> - **NOT Authorized for Autonomous Diagnosis:** The software is strictly forbidden from being deployed as an autonomous diagnostic device.
> - **NOT a Replacement for a Physician:** The software must never supersede or replace the clinical diagnostic judgment of an authorized medical practitioner.

---

## 3. Global Operating Rules (Safety Interlocks)

The software strictly enforces the following engineering safety rules:
- **RULE 1 — Never Fabricate Data:** Signals, demographics, measurements, and probabilities are strictly derived from real data. Zero data synthesis.
- **RULE 2 — Never Convert Failure into Normal:** Any calculation error, model exception, or unreadable trace outputs **NO RESULT**; errors never default to "Normal".
- **RULE 3 — Never Generate Synthetic ECG Data:** If waveform digitization or extraction is of insufficient confidence, the system halts with **NO AI ANALYSIS**.
- **RULE 4 — Never Assume Missing Metadata:** Sampling rate, duration, and lead configurations must be verified from file headers or explicit user confirmation.
- **RULE 5 — Never Call Model Probability "Diagnostic Confidence":** Output is explicitly designated **Model probability** to avoid misleading clinicians regarding statistical vs. clinical certainty.
- **RULE 6 — Do Not Invent Disease Classes:** The system only exposes the two classes validated on the active model (`Normal Sinus Rhythm` and `Premature Ventricular Contraction`).
- **RULE 7 — Clinician Responsibility:** The AI prediction and the reviewing clinician's interpretation are strictly separated in all user interfaces and reports.
- **RULE 8 — Clinical Validation Honesty:** No claims of clinical efficacy or multi-center validation are permitted until formal trials are conducted.
- **RULE 9 — Full Reproducibility:** Every generated report encodes ECG ID, model version, preprocessing version, software release, and cryptographic hash.
- **RULE 10 — Test Verification:** Every phase requires 100% automated test passing before advancement.

---

## 4. Intended User Population

ECG GUARDIAN is intended exclusively for use by:
1. **Consultant Cardiologists & Electrophysiologists:** Interpreting complex arrhythmias, verifying AI evidence beats, and signing final clinical reports.
2. **Registered Medical Practitioners (RMPs) & General Physicians:** Conducting clinical cardiovascular reviews and evaluating patient trends.
3. **Certified ECG Technicians:** Performing digital ingestion, verifying signal quality scores, and preparing records for physician sign-off.
4. **Clinical Investigators & Biostatisticians:** Auditing model calibration, quality metrics, and algorithm performance under Institutional Review Board (IRB) / Ethics Committee oversight.

**Exclusion:** Direct-to-consumer (OTC), home, or unmonitored use by patients or laypersons is strictly prohibited.

---

## 5. Intended Clinical Environment

- Inpatient cardiology wards, intensive care units (as an adjunctive offline review tool), and outpatient cardiology clinics.
- Hospital central telemetry reading rooms and diagnostic laboratories.
- Hospital environments equipped with enterprise IT infrastructure, secure role-based access, and certified electrical power supplies.

**Environmental Exclusions:**
- Unmonitored consumer wearable fitness tracking.
- Real-time emergency defibrillation decision loops.
- Pre-hospital moving transport ambulances without certified telemetry links.

---

## 6. Supported vs. Unsupported Pathologies & Indications

### 6.1 Supported Indications (Single-Lead Modified Lead II)
- Adjunctive identification of **Normal Sinus Rhythm** in resting single-lead recordings.
- Adjunctive identification of isolated or frequent **Premature Ventricular Contractions (PVC)** and ventricular ectopic beats.
- Deterministic calculation of ventricular heart rate and QTc intervals.

### 6.2 Explicitly Unsupported Pathologies (Contraindicated)
The active model (`ECG-RF-1.0.0`) **CANNOT** detect, rule out, or evaluate:
- **Acute Myocardial Infarction:** ST-Segment Elevation Myocardial Infarction (STEMI), Non-ST Elevation Myocardial Infarction (NSTEMI), or acute coronary syndromes.
- **Atrial Fibrillation (AFib) & Atrial Flutter.**
- **Lethal Ventricular Arrhythmias:** Ventricular Tachycardia (sustained VT), Ventricular Fibrillation (VFib), or Torsades de Pointes.
- **Conduction Blocks:** Left Bundle Branch Block (LBBB), Right Bundle Branch Block (RBBB), or Third-Degree (Complete) Heart Block.
- **Channelopathies:** Brugada Syndrome, Long QT Syndrome, or Short QT Syndrome.
- **Metabolic & Electrolyte Disorders:** Hyperkalemia, Hypokalemia, or Hypocalcemia.
- **Pediatric ECGs:** Patients under 18 years of age (pediatric electrophysiology differs fundamentally from adult benchmark databases).

---

## 7. Contraindications & Prohibitions

1. **PROHIBITION 1: Autonomous Emergency Decisions:** The software must never be used as the sole basis for resuscitation, cardioversion, or emergency medical procedures.
2. **PROHIBITION 2: Silent Metadata Assumption:** Processing an ECG without a verified sampling rate or lead identifier is prohibited.
3. **PROHIBITION 3: Overriding Signal Quality Gate:** Running AI inference on a signal flagged as `UNUSABLE` is strictly blocked by the system safety interlock.
4. **PROHIBITION 4: Unsigned Reports:** An AI analysis report must never be issued to a patient or medical record system without an authorized clinician's affirmative review and cryptographic signature.
