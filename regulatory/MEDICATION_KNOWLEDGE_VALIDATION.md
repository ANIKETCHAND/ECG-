# Cardiovascular Medication Knowledge Base & Safety Validation

**Document ID:** REG-MED-2026-01  
**Version:** 1.0.0  
**Effective Date:** September 22, 2026  
**Regulatory References:** FDA 21 CFR Part 314, DailyMed / NLM RxNorm, CDSCO Drug Formularies, IEC 62304 / ISO 14971.

---

## 1. Safety Architecture Principles

The Cardiovascular Medication Knowledge Base (`src/medications/medication_database.py`) and Interaction Checker (`src/medications/interaction_checker.py`) are engineered to satisfy zero-harm clinical software principles:

1. **Zero Hallucination / Zero Speculation:**
   - No unverified or synthetic pharmacological data.
   - Every drug entry is indexed with its authoritative manufacturer prescribing information NDA number and DailyMed revision.
2. **Missing Context Guardrail:**
   - If a patient's age, renal function, known allergies, or clinical conditions are missing, the system outputs `INSUFFICIENT CLINICAL CONTEXT` and halts automated safety clearance.
3. **No Autonomous Prescribing:**
   - The platform strictly provides *safety alerts*, *drug interactions*, and *contraindications*. It never issues drug orders, quantities, or titrations.

---

## 2. Formulary Monograph Provenance

| Generic Name | Primary Class | Authoritative Source Document | Revision ID | Verification Date |
|---|---|---|---|---|
| **Metoprolol** | Beta-1 Selective Adrenergic Antagonist | FDA Prescribing Information / DailyMed | NDA-019962-Rev2024 | 2026-06-15 |
| **Amiodarone** | Class III Multi-channel Antiarrhythmic | FDA Prescribing Information / DailyMed | NDA-018972-Rev2025 | 2026-05-10 |
| **Apixaban** | Direct Oral Anticoagulant (DOAC) | FDA Prescribing Information / DailyMed | NDA-202155-Rev2025 | 2026-07-01 |

---

## 3. High-Risk Drug-Drug Interaction Safety Matrix

| Combination | Severity | Mechanism | Clinical Risk | Authoritative Reference |
|---|---|---|---|---|
| **Metoprolol + Amiodarone** | MAJOR | Synergistic SA and AV node inhibition | Severe sinus bradycardia, complete AV block, sinus arrest | FDA Prescribing Info / AHA Arrhythmia Advisory |
| **Amiodarone + Warfarin** | MAJOR | Amiodarone CYP2C9 inhibition decreases warfarin clearance | Extreme INR elevation, catastrophic hemorrhage | Chest Antithrombotic Guidelines / DailyMed |
| **Amiodarone + Digoxin** | MAJOR | P-glycoprotein efflux inhibition doubles serum digoxin | Fatal digitalis toxicity, life-threatening arrhythmias | DailyMed / FDA Prescribing Info |
| **Apixaban + Aspirin** | MODERATE | Dual anticoagulant / antiplatelet hemostatic impairment | Major GI or intracranial bleeding | AHA/ACC AFib Guidelines |

---

## 4. Allergy & Hypersensitivity Validation

The interaction engine continuously cross-checks active patient allergies:
1. **Direct Drug Name Match:** Exact or substring match against generic or brand names (e.g., Lopressor, Cordarone, Eliquis).
2. **Component & Hypersensitivity Match:** Semantic cross-referencing of chemical constituents (e.g., cross-checking documented "Iodine allergy" against Amiodarone's iodine content and contraindications).
3. **Clinical Escalation:** Any allergy match generates an immediate `CRITICAL` alert mandating explicit attending physician sign-off.
