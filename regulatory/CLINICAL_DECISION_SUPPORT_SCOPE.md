# Clinical Decision Support (CDS) Scope & Regulatory Boundary

**Document ID:** REG-CDS-2026-01  
**Version:** 1.0.0  
**Effective Date:** September 22, 2026  
**Regulatory Frameworks:** FDA Clinical Decision Support Software Guidance (September 2022), CDSCO Medical Device Rules 2017, EU MDR 2017/745 Rule 11.

---

## 1. Scope and Intended Purpose

The ECG Guardian Clinical Decision Support (CDS) Engine (`src/clinical/recommendation_engine.py`) provides algorithmic pattern screening and evidence-backed clinical considerations to assist qualified healthcare professionals during electrocardiographic review.

### 1.1 Non-Autonomous Mandate (Absolute Safety Rule 1)
Under **no circumstances** does the CDS engine:
1. Prescribe medications, calculate pharmaceutical dosages, or modify patient treatment regimens autonomously.
2. Formulate an unassisted definitive medical diagnosis.
3. Overrule or replace the diagnostic judgment of a licensed physician.

---

## 2. Four FDA CDS Exclusion Criteria Assessment

Under Section 520(o)(1)(E) of the FD&C Act, software functions are classified as Non-Device CDS when meeting all four of the following statutory criteria:

| Criterion | Requirement | ECG Guardian Implementation & Evidence | Compliance Status |
|---|---|---|---|
| **Criterion 1** | Not intended to acquire, process, or analyze medical image / signal directly for autonomous diagnostics. | Ingestion gatekeeper analyzes signals strictly for research triage screening; requires manual physician interpretation. | Compliant |
| **Criterion 2** | Intended for the purpose of displaying, analyzing, or printing medical information. | Formats and organizes clinical parameters (HR, PR, QRS, QT/QTc, PVC burden) from validated extraction engines. | Compliant |
| **Criterion 3** | Intended for supporting or providing recommendations to a healthcare professional about prevention, diagnosis, or treatment. | Provides condition-specific clinical considerations and suggested secondary assessments based on consensus guidelines. | Compliant |
| **Criterion 4** | Intended to enable healthcare professionals to independently review the basis for the recommendations. | Discloses all underlying features, machine comparison, guideline citations, and source publications directly in the report UI. | Compliant |

---

## 3. Evidence-Based Clinical Guideline Provenance

All CDS considerations and contraindication alerts derive from authoritative, peer-reviewed clinical cardiology consensus guidelines:

1. **Ventricular Arrhythmias / PVCs:**
   - *2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure* (Circulation. 2022;145:e895–e1032).
   - *2019 HRS/EHRA/APHRS/LAHRS Expert Consensus Statement on Catheter Ablation of Ventricular Arrhythmias* (Heart Rhythm. 2019;16:e65–e130).
   - *CAST Trial Advisory:* Contraindication of Class IC antiarrhythmics in structural heart disease or previous myocardial infarction.

2. **Atrial Fibrillation:**
   - *2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation* (Circulation. 2024;149:e1–e156).
   - *2020 ESC Guidelines for the Diagnosis and Management of Atrial Fibrillation* (Eur Heart J. 2021;42:373–498).
   - *CHA2DS2-VASc and HAS-BLED clinical risk score protocols.*

3. **Resting Electrocardiography:**
   - *AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram* (Circulation. 2009;119:e235–e240).

---

## 4. Triage Urgency Definitions

| Triage Urgency Level | Clinical Criteria | Mandatory Action |
|---|---|---|
| **ROUTINE REVIEW** | Normal Sinus Rhythm or isolated unifocal PVCs with normal heart rate (50–100 bpm). | Review and sign off during standard clinical rounds. |
| **PROMPT REVIEW** | Frequent ectopy, tachy/bradycardia (HR 40–50 or 100–140 bpm), or stable Atrial Fibrillation. | Review within 2 hours; correlate with patient clinical status and laboratory electrolytes. |
| **URGENT CLINICIAN REVIEW** | Atrial Fibrillation with rapid ventricular response (HR > 140 bpm) or severe bradycardia (HR < 40 bpm). | Immediate attending physician bedside evaluation and vital sign assessment. |

---

## 5. Clinician Override & Audit Trail

Every clinical recommendation can be accepted, modified, or rejected by the attending physician in the Clinician Review Workspace (`app.py`). All override actions are cryptographically sealed and recorded in the append-only SQLite audit trail (`data/audit_trail.db`) with user ID, medical registration number, timestamp, and rationale.
