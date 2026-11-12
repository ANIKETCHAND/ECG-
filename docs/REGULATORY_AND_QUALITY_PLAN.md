# Regulatory & Quality Management Plan

## 1. Regulatory Classification & Standards Harmonization

ECG Guardian is engineered to satisfy the rigorous technical, clinical, and quality requirements for healthcare software:

| Standard / Regulation | Classification / Scope | Status in Architecture |
| :--- | :--- | :--- |
| **CDSCO MDR 2017 (India)** | Class B SaMD (Low-Moderate Risk) | Technical Dossier aligned with rule 32; EP checklist complete |
| **IEC 62304:2006/AMD 1:2015** | Class B Medical Device Software Lifecycle | Architecture modularized; regression test suite (582 tests) automated |
| **ISO 14971:2019** | Application of Risk Management to Medical Devices | FMEA risk management file with mitigations established |
| **ISO 13485:2016** | Medical Devices — Quality Management Systems | Document version control, traceability, and sign-off governance |
| **IEC 81001-5-1:2021** | Health Software — Health IT Systems Security | Zero-trust authentication, RBAC, encrypted storage, audit trails |
| **DPDP Act 2025 (India)** | Digital Personal Data Protection Act | Explicit electronic consent endpoints (`/api/patients/{id}/consent`) |

---

## 2. ISO 14971 Failure Modes & Effects Analysis (FMEA)

| Failure Mode ID | Hazard / Event | Initial Risk | Implemented Engineering Control | Post-Mitigation Risk |
| :--- | :--- | :--- | :--- | :--- |
| **HAZ-01** | Anonymous user accesses private patient ECG records | **CRITICAL** | Deny-by-default JWT auth (`SEC-01`, `SEC-02`), RLS database policies | **NEGLIGIBLE** |
| **HAZ-02** | Patient approves/signs own diagnostic report | **HIGH** | Strict RBAC gate on `/api/review` prohibiting patient roles (`SEC-03`) | **NEGLIGIBLE** |
| **HAZ-03** | Silent assumption of normal potassium masks hyperkalemic cardiac arrest | **CRITICAL** | Mandatory laboratory withholding protocol (`CLIN-01`) | **ACCEPTABLE** |
| **HAZ-04** | Optical ECG extraction rescales amplitude to 1.5 mV, hiding ST elevation | **HIGH** | Removal of 1.5 mV scaling; uncalibrated optical flag disables ST metrics (`BIO-01`) | **ACCEPTABLE** |
| **HAZ-05** | Production model trained on synthetic data generates false clinical outputs | **CRITICAL** | Model Registry gate prohibits synthetic artifacts; strict patient split (`ML-01`) | **ACCEPTABLE** |

---

## 3. Technical Dossier Corrections (REG-01 Resolution)

### 3.1 Historical Defect in Essential Principles Checklist
In the preliminary draft of `regulatory/CDSCO_MDR_2017_CLASS_B_TECHNICAL_DOSSIER.md`, Essential Principle 3 incorrectly cited training records (`100` and `208`) as held-out test evidence, creating an audit vulnerability for data leakage.

### 3.2 Formal Correction
- Essential Principle 3 was formally amended:
  - Held-out evaluation evidence is verified exclusively against the 14 unseen MIT-BIH patient records:
    `102, 103, 105, 107, 112, 119, 200, 203, 205, 207, 219, 222, 223, 231`.
  - The measured test performance (32,446 beats) demonstrates 90.86% overall accuracy, 94.57% Normal F1, 72.42% PVC F1, and 83.44% Other F1.
  - Zero patient overlap between training and test sets is formally documented and validated by automated tests (`tests/test_training_integrity.py`).

---

## 4. Software Verification & Validation Strategy

1. **Automated Unit & Integration Testing**:
   - Continuous automated testing across biomedical, ML, pharmacology, and API tiers.
   - Zero test regressions allowed before merge.
2. **Clinical Evaluation Protocol**:
   - Multi-center prospective clinical study required prior to seeking Class B commercial clearance.
   - Benchmarking against 12-lead standard of care interpretation by certified cardiologists.
