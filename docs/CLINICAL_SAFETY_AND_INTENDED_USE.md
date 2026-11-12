# Clinical Safety, Intended Use, and Boundary Specification

## 1. Intended Use Definition

**Product Identity**: ECG Guardian Platform  
**Regulatory Category**: Clinical Decision Support Software / Software as a Medical Device (SaMD)  
**Intended Use**:  
ECG Guardian is an investigational software platform designed to assist qualified healthcare professionals in reviewing, annotating, and interpreting single-lead electrocardiogram (ECG) rhythm strips (specifically modified Lead II recordings) obtained from adult individuals.

The system performs:
1. Automated QRS detection and heart rate determination.
2. Beat-level morphological classification into Normal, Premature Ventricular Contraction (PVC), or Other/Supraventricular ectopy.
3. Signal-quality index (SQI) assessment to alert clinicians of artifact, noise, or lead detachment.
4. Multimodal guideline-informed medication candidate suggestion subject to deterministic safety gating and drug-interaction verification.

---

## 2. Explicit Non-Intended Uses & Contraindications

The software is **NOT** cleared, certified, or intended for the following uses:
1. **Autonomous Diagnostic Device**: The software must never be used as an independent, autonomous diagnostic system without human clinician oversight.
2. **12-Lead Ischemia / STEMI Diagnosis**: Single-lead ECG analysis cannot localize ST-segment elevation myocardial infarction (STEMI), reciprocal depressions, or anatomical vascular territories. Diagnoses of acute coronary syndrome (ACS) or myocardial infarction (MI) based on single-lead input are strictly prohibited.
3. **Emergency Defibrillation or Resuscitation Triggers**: Not intended for real-time automated external defibrillator (AED) shock decisions or critical care alarm dispatch.
4. **Pediatric or Neonatal Patients**: The underlying models and algorithmic thresholds are validated only on adult cohorts (age >= 18).
5. **Autonomous Drug Prescribing**: The medication candidate module is an informational guideline lookup tool. Autonomous prescription generation is disabled by architectural policy.
6. **Definitive QT Interval Certification**: When ECG waveforms are derived from uncalibrated optical scans or photographs, interval measurements are estimates. QT prolongation warnings must be confirmed by manual caliper measurement on a calibrated standard 25 mm/s strip.

---

## 3. Clinical Risk Analysis & Known Performance Boundaries

### 3.1 Premature Ventricular Contraction (PVC) False Discovery
- **Measured Metric**: In held-out patient evaluation on 14 unseen MIT-BIH records (32,446 beats), the production Random Forest model achieved a PVC precision of 62.99% and recall of 85.17% (F1-score: 72.42%).
- **Clinical Implication**: The model exhibits a false positive discovery rate of approximately 37.0% for isolated PVC beats. Aberrant supraventricular beats, bundle branch blocks, or motion artifacts may trigger false PVC flags.
- **Safety Mitigation**:
  - The UI presents PVC classifications as preliminary suggestions.
  - Clinician confirmation is mandatory prior to generating diagnostic summaries.
  - Low-confidence beat detections are flagged with visual alerts.

### 3.2 Missing Laboratory Safety Withholding
- Prior synthetic medication engines silently substituted normal laboratory values (e.g., serum potassium = 4.2 mEq/L, creatinine = 0.95 mg/dL) when lab data was omitted from the patient context.
- In a clinical setting, this created an unacceptably high risk of lethal hyperkalemia (e.g., suggesting ACE inhibitors or spironolactone to an undiagnosed anuric patient).
- **Remediation**: The clinical pharmacology engine enforces **Mandatory Laboratory Withholding**. If serum potassium or renal parameters are unrecorded, the recommendation status transitions to `REVIEW_REQUIRED`, candidates are annotated with `MANDATORY LAB WITHHOLDING`, and a STAT laboratory order is required before therapy initiation.

---

## 4. Clinician-in-the-Loop Governance

1. **Role-Based Report Finalization**:
   - Only accounts with role `CARDIOLOGIST`, `PHYSICIAN`, or `MEDICAL_DIRECTOR` possess signing authority.
   - Patient accounts attempting to finalize or edit reports receive `HTTP 403 Forbidden`.
2. **Cryptographic Review Sealing**:
   - When a clinician confirms an analysis, the backend records:
     - Reviewer full name and medical registration ID.
     - Timestamp (ISO 8601 UTC).
     - Action status: `CONFIRMED`, `MODIFIED`, or `REJECTED`.
     - Clinician diagnostic note.
     - SHA-256 seal linking the review to the exact immutable report payload.
3. **Separation of Technical and Patient Reports**:
   - Patients have access only to finalized, clinician-approved summaries.
   - Preliminary unverified AI hypotheses are isolated from patient-facing views to prevent unnecessary patient distress or inappropriate self-medication.
