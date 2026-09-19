# IEC 62304:2006+A1:2015 Software Life Cycle Processes

**Device:** AI-ECG Analyzer (Software as a Medical Device - SaMD)  
**Safety Classification:** **Class B** (Software could contribute to a hazard resulting in non-serious or serious injury if unverified)  
**Applicable Regulations:**
- IEC 62304:2006 + AMD1:2015
- CDSCO Medical Devices Rules (MDR) 2017 (Fifth Schedule - Essential Principles)
- IMDRF SaMD N12 & N23 Clinical Evaluation & Quality Management Guidelines  
**Document Ref:** SDLC-62304-2026-01  
**Status:** DRAFT / AUDITED BASELINE  

---

## 1. Software Safety Classification Rationale

Under IEC 62304 Clause 4.3:
- **Class A:** No injury or damage to health is possible.
- **Class B:** Non-serious injury is possible.
- **Class C:** Death or serious injury is possible.

**Classification Decision: Class B**  
*Rationale:* The software provides decision support in non-emergency and telemetry triage contexts. An incorrect classification (e.g. false negative PVC or false positive ectopy) could delay care or induce unnecessary diagnostic testing. However, the software does not control physical delivery of energy or drugs, and requires mandatory physician review of the visual waveform before clinical intervention.

---

## 2. Software Development Lifecycle Stages

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   IEC 62304 SOFTWARE LIFECYCLE V-MODEL                │
├───────────────────────────────┬────────────────────────────────────────┤
│ 1. Clinical Requirements (PRD)│ 8. Clinical Performance & Verification │
│    ├── System Specifications  │    ├── Multi-Database Benchmarking     │
│    └── Safety Constraints     │    └── Human Physician Concordance     │
├───────────────────────────────┼────────────────────────────────────────┤
│ 2. Software Architecture      │ 7. System & Safety Verification        │
│    ├── Multi-Tier Decoupling  │    ├── End-to-End Ingestion Pipelines  │
│    └── Pre-Inference Gates    │    └── Chaos & Malformed Input Tests   │
├───────────────────────────────┼────────────────────────────────────────┤
│ 3. Detailed Component Design  │ 6. Integration Testing                 │
│    ├── Biomedical Algorithms  │    ├── Preprocessing -> Peak Detection │
│    └── Cryptographic Audit    │    └── Inference Engine -> Report Gen  │
├───────────────────────────────┼────────────────────────────────────────┤
│ 4. Unit Implementation        │ 5. Automated Unit Verification         │
│    ├── Python 3.14 Module Dev │    ├── Pytest (100% Target Module Pass)│
│    └── Type Safety Enforced   │    └── Deterministic Math Assertions   │
└───────────────────────────────┴────────────────────────────────────────┘
```

---

## 3. Software Architecture & Modular Decomposition (Clause 5.3)

To ensure high cohesion, low coupling, and verifiable safety gates, the platform is partitioned into independent subsystems:

1. **Ingestion & Normalization Subsystem (`src/ecg_input/`):**
   - Ingests universal multi-format data (PDF, raster images, CSV, TXT, NPY).
   - Validates physical parameters ($F_s$, duration, voltage scaling).
2. **Quality Gatekeeper Subsystem (`src/safety/signal_quality_gate.py`):**
   - Independent safety monitor. Computes signal-to-noise ratio, baseline wander, and spectral power.
   - Evaluates whether input reaches diagnostic threshold. Emits `HALT` on `POOR` or `UNUSABLE`.
3. **Deterministic Measurement Engine (`src/measurements/measurement_engine.py`):**
   - Computes standard physiological intervals (RR, HR, QRS duration, QTc Bazett/Fridericia).
   - Emits explicit non-estimable markers when signals lack requisite leads.
4. **Machine Learning Inference Subsystem (`src/inference/inference_engine.py`):**
   - Decoupled from GUI and preprocessing.
   - Loads version-locked models from `models/production/` accompanied by signed model cards.
5. **Security, Auth & Audit Trail Subsystem (`src/auth/`, `src/audit/`, `src/database/`):**
   - Role-Based Access Control (RBAC) preventing unauthorized clinical modification.
   - Cryptographically linked SHA-256 hash chains preventing retroactive tampering.
6. **Clinical Presentation & Reporting Subsystem (`app.py`, `src/report/`):**
   - Presents visual waveform with detected R-peaks.
   - Requires physician sign-off with state medical council registration numbers before generating sealed PDF reports.

---

## 4. Software Verification & Unit Testing (Clauses 5.5, 5.6, 5.7)

Automated testing is enforced via `pytest` and continuous integration:

- **Unit Verification:**
  - `tests/test_preprocessing.py`: Bandpass filtering, baseline wander subtraction, normalization.
  - `tests/test_peak_detection.py`: Pan-Tompkins QRS refractory periods and peak alignment.
  - `tests/test_features.py`: Robustness of 28 morphological and spectral feature extractors.
  - `tests/test_image_processing.py`: Rejection of corrupted, non-ECG, or low-resolution imagery.
  - `tests/test_report_generation.py`: Verification of PDF and JSON compilation without data leakage.
- **Safety Gate Verification:**
  - `tests/test_waveform_extraction.py`: Synthetic low-confidence waveforms trigger extraction rejection.
  - `test_prediction.py`: Out-of-bounds inputs (NaN, infinite, empty) fail safe with clean error records.

---

## 5. Software Release, Versioning & Change Control (Clauses 6, 7, 8)

1. **Semantic Versioning:**
   - Format: `MAJOR.MINOR.PATCH` (e.g. `1.0.0`).
   - `MAJOR`: Changes to clinical intended use, diagnostic algorithms, or medical device classification.
   - `MINOR`: New input device support, performance improvements, UI enhancements without algorithmic changes.
   - `PATCH`: Bug fixes, security hardening, or documentation updates.
2. **Model Retraining Lock:**
   - Deployed models (`models/production/`) are strictly frozen with SHA-256 checksums.
   - Online learning or self-modifying models are strictly prohibited under MDR 2017 rules. Any model update requires execution of the Algorithm Change Protocol (ACP-2026).
3. **Defect Tracking & Problem Resolution:**
   - Any report of clinical discordance or software crash is assigned a Unique CAPA (Corrective and Preventive Action) identifier.
   - Root-cause investigation must evaluate whether the issue stemmed from patient electrode detachment, signal digitization error, or model generalization limitation.

---

## 6. Regulatory Sign-Off
**Software Development Lead:** Engineering Team  
**Quality Assurance Lead:** Regulatory Affairs  
**Medical Director:** Chief of Cardiology  
**Date:** 2026-09-19  
