# Usability Engineering File: ECG GUARDIAN
**Standard Reference:** IEC 62366-1:2015+AMD1:2020 (Application of usability engineering to medical devices)  
**System:** ECG GUARDIAN Clinical Decision Support Platform  
**Document Release:** September 2026  
**Status:** Pre-Market Formative Usability Baseline  

---

## 1. Intended User Profiles

1. **Attending Cardiologists**:
   - Secondary algorithmic review, rhythm verification, and final diagnostic sign-off.
   - High domain knowledge; requires rapid access to aberrant beat evidence and serial comparisons.
2. **Emergency Physicians & Critical Care Intensivists**:
   - Rapid rhythm triage, acute ectopy burden assessment.
   - Requires conspicuous alerts regarding limitations (e.g. AI cannot rule out STEMI or AV blocks).
3. **Cardiology Technicians & Nurses**:
   - Uploading ECG waveforms (CSV, DICOM, EDF, XML, Images).
   - Inspecting Signal Quality Copilot feedback to reposition electrodes if noise or wander is flagged.

---

## 2. Primary Operating Scenarios & Critical Tasks

| Task ID | Description | Potential Use Error | Risk Mitigation / UI Safeguard |
| :--- | :--- | :--- | :--- |
| **UT-01** | Review AI Rhythm Inference | Clinician mistaking AI output for autonomous diagnosis without reviewing tracing. | Unremovable banner: "FOR CLINICAL DECISION SUPPORT ONLY. PHYSICIAN INTERPRETATION MANDATORY." |
| **UT-02** | Inspection of Poor Quality ECG | Attempting to force AI analysis on flatline or clipped signal. | Automated Pre-Inference Quality Gatekeeper suppresses AI and outputs `NO RESULT_UNUSABLE`. |
| **UT-03** | Lead Selection | Uploading Lead V1/V5 expecting validated single-lead classification. | UI actively highlights lead name; non-Lead II inputs produce `NO_RESULT_UNSUPPORTED_LEAD`. |
| **UT-04** | Disagreement Reconciliation | Clinician ignoring conflict between machine header and AI finding. | High-visibility warning alert with reconciliation checklist on comparison card. |
| **UT-05** | Final Sign-off | Signing without valid credential registration. | Mandatory registration number validation field blocking signature generation. |

---

## 3. Human Factors & Formative Evaluation Plan

- **Formative Evaluations**: 2 rounds of simulated clinical reviews with 5 board-certified cardiologists and 5 emergency physicians.
- **Summative Usability Protocol**: 15 representative clinical users performing simulated acute and outpatient ECG review workflows under realistic time constraints. Zero critical safety-use errors required for pre-market clearance.
