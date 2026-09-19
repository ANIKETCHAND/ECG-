# ISO 14971:2019 Clinical Risk Management File

**Device:** AI-ECG Analyzer (Software as a Medical Device - SaMD)  
**Classification:** Class B (Moderate Risk under CDSCO MDR 2017 / IMDRF SaMD Category II)  
**Standard:** ISO 14971:2019 (Application of risk management to medical devices)  
**Document Ref:** RMF-2026-001  
**Status:** DRAFT / PENDING CLINICAL TRIAL  

---

## 1. Executive Summary & Policy

Patient safety is paramount. In accordance with ISO 14971:2019 and IEC 62304 Section 5.3, this Risk Management File identifies hazards associated with AI-assisted electrocardiogram analysis, estimates and evaluates associated risks, controls these risks to As Low As Reasonably Practicable (ALARP), and evaluates the overall residual risk.

### Core Safety Tenet
> **"No Reliable Input = No AI Result"**  
> Under no circumstances does the system fabricate, guess, or extrapolate cardiac rhythm predictions when the signal quality is unvalidated, noisy, corrupted, or unsupported. Furthermore, the software functions strictly as a Clinical Decision Support System (CDSS) requiring mandatory physician verification.

---

## 2. Risk Evaluation Scoring Criteria

### Severity Levels (S)
1. **Negligible (S1):** Inconvenience or temporary discomfort; no medical intervention required.
2. **Minor (S2):** Temporary injury or reversible impairment; standard medical care resolves without delay.
3. **Serious (S3):** Injury requiring medical intervention; temporary impairment of body function.
4. **Critical (S4):** Permanent impairment of body function or irreversible damage requiring surgical/intensive intervention.
5. **Catastrophic (S5):** Fatality or life-threatening event.

### Probability Levels (P)
1. **Improbable (P1):** $< 10^{-6}$ per analysis run.
2. **Remote (P2):** $10^{-6}$ to $10^{-4}$ per analysis run.
3. **Occasional (P3):** $10^{-4}$ to $10^{-2}$ per analysis run.
4. **Probable (P4):** $10^{-2}$ to $10^{-1}$ per analysis run.
5. **Frequent (P5):** $> 10^{-1}$ per analysis run.

### Risk Level Matrix
- **Unacceptable (Red):** Intolerable risk. Must be reduced before deployment.
- **ALARP (Yellow):** Tolerable only if further risk reduction is impractical and clinical benefit outweighs risk.
- **Broadly Acceptable (Green):** Acceptable without further reduction.

| Severity \ Probability | P1 (Improbable) | P2 (Remote) | P3 (Occasional) | P4 (Probable) | P5 (Frequent) |
|---|---|---|---|---|---|
| **S5 (Catastrophic)** | ALARP | Unacceptable | Unacceptable | Unacceptable | Unacceptable |
| **S4 (Critical)** | Acceptable | ALARP | Unacceptable | Unacceptable | Unacceptable |
| **S3 (Serious)** | Acceptable | Acceptable | ALARP | Unacceptable | Unacceptable |
| **S2 (Minor)** | Acceptable | Acceptable | Acceptable | ALARP | ALARP |
| **S1 (Negligible)** | Acceptable | Acceptable | Acceptable | Acceptable | Acceptable |

---

## 3. Hazard Analysis & Failure Modes (FMEA Register)

| ID | Hazard Scenario | Initial Risk (S / P) | Risk Control Measure (Mitigation) | Implementation Ref | Residual Risk (S / P) | Acceptability |
|---|---|---|---|---|---|---|
| **H-01** | **Degraded / High-Noise Signal:** Muscle tremor, baseline wander, or 50Hz mains noise causes spurious peak detection and false PVC diagnosis. | S4 / P4 (Unacceptable) | Automated pre-inference Signal Quality Gate (`SignalQualityGate`). Computes SNR, skewness, kurtosis, baseline wander. Quality `< 0.40` is classified `POOR`/`UNUSABLE` and blocks AI classification. | [`src/safety/signal_quality_gate.py`](file:///e:/CODE/AI-ECG-Analyzer/src/safety/signal_quality_gate.py) | S4 / P1 (Acceptable) | **Acceptable** |
| **H-02** | **False Negative Arrhythmia:** System fails to flag premature ventricular ectopy in an unstable patient, delaying intervention. | S4 / P3 (Unacceptable) | 1. Model outputs full calibrated probability distribution.<br>2. Prominent multi-beat PVC burden percentage.<br>3. Prominent disclaimer and mandatory qualified clinician sign-off workflow before patient discharge. | [`src/inference/inference_engine.py`](file:///e:/CODE/AI-ECG-Analyzer/src/inference/inference_engine.py)<br>[`src/report/report_generator.py`](file:///e:/CODE/AI-ECG-Analyzer/src/report/report_generator.py) | S4 / P2 (ALARP) | **ALARP** (Benefit outweighs risk) |
| **H-03** | **Automation Bias (Clinician Over-Reliance):** Attending physician blindly accepts AI recommendation without reviewing raw waveform. | S4 / P4 (Unacceptable) | 1. Visual side-by-side waveform strip with annotated R-peaks embedded directly in UI and PDF.<br>2. Explicit mandatory review agreement selection (`CONFIRM`, `MODIFY`, `REJECT`).<br>3. Physician registration number mandatory for report sealing. | [`app.py`](file:///e:/CODE/AI-ECG-Analyzer/app.py)<br>[`src/report/pdf_generator.py`](file:///e:/CODE/AI-ECG-Analyzer/src/report/pdf_generator.py) | S4 / P2 (ALARP) | **ALARP** |
| **H-04** | **Optical Distortion in Scanned ECGs:** Low-resolution camera capture or perspective distortion causes distorted waveform and erroneous measurements. | S3 / P4 (Unacceptable) | Multi-stage extraction validation gate (`validate_extracted_signal`). Verifies minimum duration, sample rate, peak amplitude, and confidence score. Returns structured extraction failure rather than guessing. | [`src/ecg_input/image_extractor.py`](file:///e:/CODE/AI-ECG-Analyzer/src/ecg_input/image_extractor.py) | S3 / P2 (Acceptable) | **Acceptable** |
| **H-05** | **Lead Geometry Mismatch:** User feeds single-lead ECG to an algorithm expecting 12-lead multi-lead orthogonal signals. | S3 / P3 (ALARP) | Deterministic lead compatibility verification. Electrical frontal axis (P, QRS, T) calculation rejects single-lead inputs with explicit error `Not measurable on single-lead ECG`. | [`src/measurements/measurement_engine.py`](file:///e:/CODE/AI-ECG-Analyzer/src/measurements/measurement_engine.py) | S3 / P1 (Acceptable) | **Acceptable** |
| **H-06** | **Unvalidated Minority Arrhythmias:** Clinician assumes model classifies AFib, VT, or bundle branch blocks because "AI analyzes ECG". | S4 / P3 (Unacceptable) | 1. Model Card transparently declares model was trained only for Normal vs PVC.<br>2. Minority class `Other` documented with low recall.<br>3. Explicit contraindication documentation against acute STEMI / VT / pacemaker analysis. | [`docs/models/ECG-RF-1.0.0/MODEL_CARD.md`](file:///e:/CODE/AI-ECG-Analyzer/docs/models/ECG-RF-1.0.0/MODEL_CARD.md)<br>[`docs/INTENDED_USE.md`](file:///e:/CODE/AI-ECG-Analyzer/docs/INTENDED_USE.md) | S4 / P1 (Acceptable) | **Acceptable** |
| **H-07** | **Tampering / Unauthorized Override:** Malicious actor modifies diagnosis or report after generation. | S4 / P2 (ALARP) | 1. Cryptographically chained append-only audit trail (`AuditLogger`) linking each event with previous entry hash.<br>2. SHA-256 data hash computed for raw signal and sealed PDF reports. | [`src/audit/audit_logger.py`](file:///e:/CODE/AI-ECG-Analyzer/src/audit/audit_logger.py)<br>[`src/database/db_manager.py`](file:///e:/CODE/AI-ECG-Analyzer/src/database/db_manager.py) | S4 / P1 (Acceptable) | **Acceptable** |
| **H-08** | **Patient PHI Leakage:** EMR integration or research download exposes patient identifiers. | S3 / P3 (ALARP) | Role-Based Access Control (RBAC). Researcher role automatically redacts patient names, phone numbers, and hospital MRNs into anonymized research IDs. | [`src/auth/auth_manager.py`](file:///e:/CODE/AI-ECG-Analyzer/src/auth/auth_manager.py) | S3 / P1 (Acceptable) | **Acceptable** |

---

## 4. Overall Residual Risk & Benefit-Risk Conclusion

All identified hazards have been mitigated through design controls, architectural separation, pre-inference safety gating, and mandatory clinician verification.
The residual risks are evaluated as ALARP or Broadly Acceptable. The clinical benefit of rapid, automated screening of premature ventricular ectopy in telemetry and triage environments significantly outweighs the controlled residual risks.

**Risk Management Lead:** Quality & Regulatory Affairs  
**Clinical Officer Review:** Attending Cardiologist  
**Approval Date:** 2026-09-19  
