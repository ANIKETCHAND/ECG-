# Software Requirements Traceability Matrix (RTM)

**Device:** AI-ECG Analyzer (SaMD)  
**Standard:** IEC 62304 Section 5.1.4 / CDSCO MDR 2017 Essential Principles  
**Document Ref:** RTM-2026-01  
**Status:** COMPLETE & VERIFIED  

---

## 1. Overview & Verification Method

The Traceability Matrix establishes complete bidirectional traceability between Clinical System Requirements (CSR), Software Architecture Components (SAC), Source Implementation Files (SRC), Automated Verification Tests (TEST), and Associated Clinical Hazards (HAZ).

---

## 2. Requirements Traceability Register

| Req ID | Clinical Requirement Description | Architecture Component | Source Code Implementation | Verification Test Case | Hazard ID | Status |
|---|---|---|---|---|---|---|
| **REQ-001** | Support universal ingestion of multi-format ECG documents (PDF, JPG/PNG, CSV, TXT, NPY). | Ingestion Subsystem | [`src/ecg_input/modality_detector.py`](file:///e:/CODE/AI-ECG-Analyzer/src/ecg_input/modality_detector.py)<br>[`src/ecg_input/digital_loader.py`](file:///e:/CODE/AI-ECG-Analyzer/src/ecg_input/digital_loader.py) | `tests/test_input_detection.py::test_detect_*` | H-04 | **VERIFIED** |
| **REQ-002** | Enforce pre-inference signal quality gatekeeper; block classification on `POOR` or `UNUSABLE` signals. | Safety Gatekeeper | [`src/safety/signal_quality_gate.py`](file:///e:/CODE/AI-ECG-Analyzer/src/safety/signal_quality_gate.py) | `tests/test_signal_quality.py`<br>`tests/test_waveform_extraction.py::test_validation_gate_*` | H-01 | **VERIFIED** |
| **REQ-003** | Deterministic baseline wander removal and bandpass filtering (0.5–45 Hz) preserving QRS morphology. | Signal Processing Pipeline | [`src/preprocessing.py`](file:///e:/CODE/AI-ECG-Analyzer/src/preprocessing.py) | `tests/test_preprocessing.py::test_remove_baseline_wander`<br>`test_bandpass_filter` | H-01 | **VERIFIED** |
| **REQ-004** | R-peak detection with refractory period lockout preventing double-counting during wide QRS complexes. | Segmentation Engine | [`src/peak_detection.py`](file:///e:/CODE/AI-ECG-Analyzer/src/peak_detection.py) | `tests/test_peak_detection.py::test_detect_r_peaks`<br>`test_validate_peaks_refractory_period` | H-01, H-02 | **VERIFIED** |
| **REQ-005** | Standard physiological parameter calculation (Heart Rate, RR intervals, QRS duration, QTc Bazett/Fridericia). | Measurement Engine | [`src/measurements/measurement_engine.py`](file:///e:/CODE/AI-ECG-Analyzer/src/measurements/measurement_engine.py) | `tests/test_segmentation.py::test_rr_intervals_and_heart_rate` | H-05 | **VERIFIED** |
| **REQ-006** | Decoupled machine learning inference engine evaluating individual heartbeats with calibrated probabilities. | Inference Engine | [`src/inference/inference_engine.py`](file:///e:/CODE/AI-ECG-Analyzer/src/inference/inference_engine.py) | `tests/test_prediction.py::test_predict_ecg_*` | H-02, H-06 | **VERIFIED** |
| **REQ-007** | Zero-hallucination separation between native printed report data and algorithmic predictions. | Ingestion & Extraction Engine | [`src/ecg_input/pdf_extractor.py`](file:///e:/CODE/AI-ECG-Analyzer/src/ecg_input/pdf_extractor.py)<br>[`src/ecg_input/image_extractor.py`](file:///e:/CODE/AI-ECG-Analyzer/src/ecg_input/image_extractor.py) | `tests/test_pdf_processing.py::test_pdf_measurement_extraction` | H-03, H-04 | **VERIFIED** |
| **REQ-008** | Multi-role clinical authentication and granular permission matrix (Doctor, Cardiologist, Tech, Admin, Researcher). | Auth & RBAC Subsystem | [`src/auth/auth_manager.py`](file:///e:/CODE/AI-ECG-Analyzer/src/auth/auth_manager.py) | `tests/test_auth_manager.py` | H-08 | **VERIFIED** |
| **REQ-009** | Relational transactional persistence for patients, ECG records, analysis runs, clinician reviews, and sealed reports. | Database Subsystem | [`src/database/db_manager.py`](file:///e:/CODE/AI-ECG-Analyzer/src/database/db_manager.py) | `tests/test_db_manager.py` | H-07 | **VERIFIED** |
| **REQ-010** | Cryptographically chained, append-only audit trail logging all clinical actions with SHA-256 integrity verification. | Audit Logger Subsystem | [`src/audit/audit_logger.py`](file:///e:/CODE/AI-ECG-Analyzer/src/audit/audit_logger.py) | `tests/test_audit_logger.py` | H-07 | **VERIFIED** |
| **REQ-011** | Mandatory clinician sign-off workflow supporting agreement status (`CONFIRM`, `MODIFY`, `REJECT`) and medical registration number. | Clinical Review Portal | [`app.py`](file:///e:/CODE/AI-ECG-Analyzer/app.py)<br>[`src/report/report_generator.py`](file:///e:/CODE/AI-ECG-Analyzer/src/report/report_generator.py) | `tests/test_report_generation.py::test_generate_structured_report` | H-03 | **VERIFIED** |
| **REQ-012** | Publication-grade clinical PDF and JSON reports embedding hospital letterhead, physician sign-off, waveform strip, and disclaimer. | Reporting Subsystem | [`src/report/pdf_generator.py`](file:///e:/CODE/AI-ECG-Analyzer/src/report/pdf_generator.py)<br>[`src/report/report_generator.py`](file:///e:/CODE/AI-ECG-Analyzer/src/report/report_generator.py) | `tests/test_report_generation.py::test_generate_pdf_report`<br>`test_export_report_to_json` | H-03, H-07 | **VERIFIED** |
| **REQ-013** | Clinical Decision Support (CDS) engine generating evidence-backed considerations, triage urgency, and guideline citations without autonomous prescribing. | CDS Engine | [`src/clinical/recommendation_engine.py`](file:///e:/CODE/AI-ECG-Analyzer/src/clinical/recommendation_engine.py) | `tests/test_clinical_decision_support.py` | H-02, H-03 | **VERIFIED** |
| **REQ-014** | Authoritative cardiovascular medication knowledge base, drug-drug interaction checker, allergy conflict detector, and missing-context guardrail. | Medication Safety Subsystem | [`src/medications/medication_database.py`](file:///e:/CODE/AI-ECG-Analyzer/src/medications/medication_database.py)<br>[`src/medications/interaction_checker.py`](file:///e:/CODE/AI-ECG-Analyzer/src/medications/interaction_checker.py) | `tests/test_medication_safety.py` | H-03, H-06 | **VERIFIED** |
| **REQ-015** | Alert and escalation engine categorizing AI-generated, technical, and critical hemodynamic alerts with physician acknowledgment auditing. | Alert Engine | [`src/alerts/alert_engine.py`](file:///e:/CODE/AI-ECG-Analyzer/src/alerts/alert_engine.py) | `tests/test_alert_engine.py` | H-01, H-05 | **VERIFIED** |

---

## 3. Verification Sign-Off
- **All 15 Core Requirements** have verified, executable test coverage in the automated test suite.
- **Traceability Coverage:** 100% of functional requirements link to architectural units, automated tests, and ISO 14971 hazards.

