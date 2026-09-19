# Predetermined Change Control Plan (PCCP) & Algorithm Change Protocol

**Device:** AI-ECG Analyzer (SaMD)  
**Standard:** FDA Guidance on PCCP / IMDRF SaMD Lifecycle / CDSCO MDR 2017  
**Document Ref:** ACP-2026-01  
**Status:** ACTIVE  

---

## 1. Purpose & Scope

Machine learning algorithms in medical devices require rigorous change management to ensure that continuous improvements do not introduce algorithmic drift, catastrophic forgetting, or unrecognized failure modes.

This Algorithm Change Protocol establishes:
- The permissible boundaries of algorithmic retraining.
- Validation protocols required prior to promoting any new model artifact to production.
- Regulatory notification thresholds under CDSCO MDR 2017.

> [!IMPORTANT]
> **Zero Online Adaptation Policy:** The AI-ECG Analyzer strictly prohibits online self-learning or continuous weight adaptation during live clinical operation. All models deployed in hospitals must be pre-trained, validated, deterministically frozen, and hashed.

---

## 2. Change Categorization Matrix

| Change Class | Description | Examples | Verification Required | Regulatory Action |
|---|---|---|---|---|
| **Class I (Administrative / Non-Algorithmic)** | Modifications to UI presentation, report templates, or logging format without altering model weights or feature extractors. | Updating PDF styling, adding telemetry gauges, improving database query speed. | Full unit and regression test suite pass (`pytest`). | Internal engineering sign-off; logged in audit trail. |
| **Class II (Performance Optimization)** | Retraining model on expanded multi-center datasets within the existing clinical intended use (e.g. improving sensitivity for PVCs). | Adding 5,000 new annotated lead II beats, adjusting Random Forest hyperparameter (e.g. tree count, max depth). | 1. Non-inferiority testing on Golden Benchmark Set.<br>2. Signal Quality Gate verification.<br>3. Regression test suite pass. | Documented in Model Card; reviewed by Institutional Safety Board. |
| **Class III (Expanded Intended Use / New Class)** | Introducing detection of previously unvalidated arrhythmias (e.g. Atrial Fibrillation, Left Bundle Branch Block, ST-elevation). | Adding new output classification heads or training on multi-lead geometries. | 1. Formal clinical trial execution.<br>2. Full re-validation protocol.<br>3. Independent cardiologist adjudication. | **Formal Prior Approval from CDSCO** under MDR 2017 rules. |

---

## 3. Retraining Data Governance & Quality Standards

Any dataset incorporated for retraining must adhere to the following data criteria:
1. **Provenance:** Sourced from ethically approved clinical trials, institutional data transfer agreements, or peer-reviewed public biobanks (e.g. PhysioNet).
2. **Quality Screening:** Pre-filtered using the Signal Quality Gate; uncurated or unverified recordings must not be used for supervised ground truth.
3. **Class Balance:** Deliberate balancing strategies (SMOTE, class-weighted loss, or downsampling) to prevent overwhelming majority bias.
4. **Demographic & Device Diversity:** Equal inclusion of diverse patient genders, age cohorts, and standard ECG acquisition hardware (GE, Philips, Welch Allyn, Contec).

---

## 4. Verification & Golden Dataset Benchmarking

Before any new model artifact can be designated for production release:
1. **Golden Dataset Evaluation:** The candidate model must run inference against the frozen Golden Benchmark Partition ($N = 10,000$ verified cycles).
2. **Acceptance Thresholds:**
   - Sensitivity for PVC must not decrease by $> 1.0\%$ relative to current production (`ECG-RF-1.0.0`).
   - Specificity for Normal rhythm must remain $\ge 98.0\%$.
   - False positive rate on clean sinus rhythm must not exceed $2.0\%$.
   - Inference latency must remain $< 250\text{ ms}$ for a 10-second strip.

---

## 5. Model Registry & Promotion Workflow

```text
┌────────────────────────────────────────────────────────┐
│               MODEL DEPLOYMENT PIPELINE                │
├────────────────────────────────────────────────────────┤
│ 1. Model Training & Offline Cross-Validation           │
│    └── Output: candidate.pkl                           │
├────────────────────────────────────────────────────────┤
│ 2. Automated Benchmark & Safety Gate Suite             │
│    ├── Run tests/test_prediction.py                    │
│    └── Verify against models/validation/golden_set/    │
├────────────────────────────────────────────────────────┤
│ 3. Model Registry Archival                             │
│    ├── Create models/registry/ECG-RF-<VERSION>/        │
│    ├── Generate SHA-256 Checksum                       │
│    └── Publish formal MODEL_CARD.md                    │
├────────────────────────────────────────────────────────┤
│ 4. Production Promotion                                │
│    ├── Copy artifact to models/production/             │
│    ├── Archive previous model to models/archived/      │
│    └── Update SYSTEM_ARCHITECTURE.md                   │
└────────────────────────────────────────────────────────┘
```

**Protocol Authorized By:** Machine Learning Director & Head of Regulatory Compliance  
**Date:** 2026-09-19  
