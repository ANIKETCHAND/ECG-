# Clinical Validation Plan & Investigational Protocol

**Device Under Investigation:** AI-ECG Analyzer (SaMD)  
**Regulatory Framework:** CDSCO Medical Devices Rules, 2017 (Chapter VII) & IMDRF/SaMD WG/N41  
**Study Type:** Prospective, Multi-Center, Non-Interventional Clinical Performance Study  
**Protocol Ref:** CVP-2026-ECG-01  
**Status:** DRAFT / PENDING INSTITUTIONAL ETHICS COMMITTEE APPROVAL  

---

## 1. Clinical Purpose & Investigational Objective

The primary objective of this clinical evaluation protocol is to rigorously determine the diagnostic accuracy, sensitivity, specificity, and safety profile of the AI-ECG Analyzer compared against an adjudicated expert cardiologist consensus reference standard in hospital telemetry and outpatient cardiology settings.

### Primary Hypotheses
1. **$H_{01}$ (Sensitivity):** The sensitivity of the AI-ECG Analyzer for identifying Premature Ventricular Contractions (PVCs) is non-inferior to a pre-specified clinical threshold of 95.0% ($\alpha = 0.05, \beta = 0.20$).
2. **$H_{02}$ (Specificity):** The specificity for Normal Sinus Rhythm is non-inferior to 95.0%.
3. **$H_{03}$ (Safety & Quality Gate):** The automated Signal Quality Gate successfully identifies $\ge 98.0\%$ of diagnostically unusable or high-artifact ECG signals prior to inference.

---

## 2. Study Design & Sites

- **Design:** Prospective, multi-center, double-blinded observational study.
- **Participating Centers:**
  - Site 1: Tertiary Academic Cardiology Center (Urban, high-acuity inpatient/telemetry).
  - Site 2: Regional General Hospital (Semi-urban, outpatient screening).
  - Site 3: District Emergency Hospital (Triage intake).
- **Study Duration:** 12 months (enrollment through adjudicated reporting).

---

## 3. Patient Population & Sample Size Determination

### Target Sample Size: $N = 1,500$ patients ($\approx 15,000$ validated cardiac cycles)
- Stratification:
  - Normal Sinus Rhythm: 60%
  - Premature Ventricular Ectopy (unifocal, multifocal, couplets): 30%
  - Non-sinus rhythm / other morphologies (SVEB, bundle branch patterns): 10%

### Inclusion Criteria
1. Adult patients ($\ge 18$ years of age).
2. Clinical indication for resting 12-lead ECG or continuous bedside telemetry.
3. Digital ECG data available in standard formats (DICOM waveform, EDF+, CSV) or high-resolution clinical printout.
4. Written informed consent obtained (or Institutional Ethics Committee waiver for de-identified retrospective telemetry stream audits).

### Exclusion Criteria
1. Patients with active cardiac pacemakers or implantable cardioverter-defibrillators (ICDs) (as declared in Intended Use contraindications).
2. Paced ventricular rhythms.
3. Severe baseline noise exceeding $5\text{ mV}$ peak-to-peak where ground truth cannot be established by human experts.

---

## 4. Reference Standard Truthing Protocol

To avoid reference bias, all enrolled ECG recordings will undergo adjudicated review:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    REFERENCE TRUTHING PROTOCOL                  │
├───────────────────────────────┬─────────────────────────────────┤
│ Reviewer 1 (Cardiologist A)   │ Independent beat-by-beat label  │
│ Reviewer 2 (Cardiologist B)   │ Independent beat-by-beat label  │
├───────────────────────────────┴─────────────────────────────────┤
│                               │                                 │
│  ├── If Reviewer A == B ─────►│ Ground Truth Accepted           │
│  │                            │                                 │
│  └── If Reviewer A != B ─────►│ Reviewer 3 (Senior Adjudicator) │
│                               │ Final Consensus Binding         │
└───────────────────────────────┴─────────────────────────────────┘
```

All human reviewers are blinded to the AI model predictions, patient clinical history, and other reviewers' scores.

---

## 5. Statistical Endpoints & Acceptance Criteria

| Metric | Target Acceptance Threshold | Standard Formulation |
|---|---|---|
| **Sensitivity (PVC)** | $\ge 95.0\%$ (Lower 95% CI $\ge 92.0\%$) | $\frac{TP}{TP + FN}$ |
| **Specificity (Normal)** | $\ge 95.0\%$ (Lower 95% CI $\ge 92.0\%$) | $\frac{TN}{TN + FP}$ |
| **Positive Predictive Value (PPV)** | $\ge 90.0\%$ | $\frac{TP}{TP + FP}$ |
| **Overall Accuracy** | $\ge 96.0\%$ | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Inter-rater Agreement ($\kappa$)** | Cohen's $\kappa \ge 0.85$ | Degree of concordance above chance |
| **Heart Rate Mean Absolute Error** | $\le 2.0\text{ BPM}$ | $\frac{1}{N}\sum |HR_{est} - HR_{ref}|$ |

---

## 6. Safety Monitoring & Stopping Rules

The study will be temporarily paused and reviewed by the Data Safety Monitoring Board (DSMB) if:
1. The AI model experiences an uncaught software failure or crash rate $> 1.0\%$ across consecutive batches.
2. The Signal Quality Gate allows $> 2.0\%$ of genuinely corrupt signals to pass through to inference without warning.
3. Any clinical adverse event occurs attributable to physician misinterpretation of AI telemetry.

---

## 7. Regulatory Status Disclaimer
This Clinical Validation Plan represents a pre-market investigation protocol. **The AI-ECG Analyzer has not yet concluded human clinical trials, is not yet approved by the CDSCO or any international regulatory agency, and cannot be used for standalone clinical diagnosis.**
