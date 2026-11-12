# Clinical Pharmacology & Medication Safety Specification

## 1. System Scope & Clinical Intended Role

The Multimodal Medication Candidate Engine (`src/clinical/medication_candidate_engine.py`) provides guideline-referenced clinical decision support for attending physicians managing patients with rhythm disorders.

**Non-Autonomous Mandate**:
The engine does **not** autonomously prescribe or adjust dosages. It generates candidate therapeutic classes, ranks association with guideline indications, and executes multi-tiered safety verification.

---

## 2. Multi-Tier Deterministic Safety Verification Architecture

Candidate recommendations must survive five sequential deterministic safety gates:

```
                    ┌───────────────────────────────────────────────┐
                    │ Multimodal Guideline Indication Association  │
                    │   (ACC/AHA, ESC, DailyMed Clinical Rules)     │
                    └───────────────────────┬───────────────────────┘
                                            │ Candidate List
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ GATE 1: Patient Allergy Cross-Matching        │
                    │ (Checks drug class, substance, cross-react.)  │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ GATE 2: Drug-Drug Interaction Checker         │
                    │ (Checks existing patient meds vs candidate)   │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ GATE 3: Physiological Contraindication Checks │
                    │ (Severe bradycardia, long QTc, hypotension)   │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ GATE 4: Mandatory Laboratory Gating (CLIN-01) │
                    │ (Serum K+, eGFR, Creatinine, Magnesium)       │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Final Candidate Status:                       │
                    │ SAFE / REVIEW_REQUIRED / BLOCKED              │
                    └───────────────────────────────────────────────┘
```

---

## 3. Mandatory Laboratory Withholding Protocol (CLIN-01 & CLIN-02 Fix)

### 3.1 Historical Defect Analysis
Legacy prototypes silently defaulted missing potassium to `4.2 mEq/L` and missing creatinine to `0.95 mg/dL` when patient lab panels were omitted. This masked potentially lethal contraindications:
- An anuric or renal failure patient could be recommended potassium-sparing diuretics or ACE inhibitors without warning.
- A severely hypokalemic patient (< 3.0 mEq/L) could be recommended antiarrhythmics that trigger Torsades de Pointes.

### 3.2 Implemented Policy & Code Remediation
1. **Zero Silent Substitution**:
   - In `src/clinical/medication_candidate_engine.py`, missing laboratory values are never replaced with normal constants.
2. **Explicit Withholding Triggers**:
   - **Potassium Withholding**:
     ```python
     if not has_k:
         if c_status != "BLOCKED":
             c_status = "REVIEW_REQUIRED"
         c_contraindications.append(
             "MANDATORY LAB WITHHOLDING: Serum potassium (K+) unrecorded. "
             "Obtain STAT electrolyte panel before initiating potassium-altering therapy."
         )
         c_lab_checks.append("Serum K+: NOT RECORDED (Required)")
     ```
   - **Renal Function Withholding**:
     ```python
     if not has_cr:
         if c_status != "BLOCKED":
             c_status = "REVIEW_REQUIRED"
         c_contraindications.append(
             "MANDATORY LAB WITHHOLDING: Renal function (eGFR / Serum Creatinine) unrecorded. "
             "Verify renal clearance before administering renally eliminated agents."
         )
         c_lab_checks.append("Renal Panel: NOT RECORDED (Required)")
     ```
3. **Emergency Escalation Alerts & Hypokalemia Gating**:
   - `k_val >= 5.5`: Severe hyperkalemia alert generated; recommends calcium gluconate and insulin-dextrose escalation.
   - `k_val < 3.5`: Hypokalemia contraindication gate triggered; blocks torsadogenic, QT-prolonging, and antiarrhythmic agents with critical alert to correct potassium prior to therapy.
   - `k_val < 3.0`: Severe hypokalemia emergency alert generated; warns of immediate ventricular fibrillation and Torsades de Pointes risk.

---

## 4. Allergy & Drug Interaction Rules

- **Allergy Engine**: Checks active substance names and therapeutic classes against the patient's documented allergy list. If an allergy match is detected, the candidate status transitions immediately to `BLOCKED`.
- **QTc Gating**: If measured QTc interval exceeds 480 ms in males or 500 ms in females, all Class IA, Class III, and QT-prolonging psychotropic/antimicrobial candidates are categorized as `BLOCKED`.
- **Hemodynamic Gating**: If heart rate is below 50 bpm, beta-blockers, non-dihydropyridine calcium channel blockers, and ivabradine are marked `BLOCKED`.
