# Regulatory Clearance Roadmap: ECG GUARDIAN
**Document Ref:** `RR-ECG-GUARDIAN-001`  
**Target Classifications:**  
- **India (CDSCO)**: Class C SaMD (Form MD-14 / MD-15 Import or Form MD-7 / MD-9 Manufacture)  
- **United States (FDA)**: Class II (510(k) Premarket Notification with Predicate Device)  
- **European Union (CE / MDR)**: Class IIa (EU MDR 2017/745 Rule 11)  

---

## 1. Regulatory Clearance Strategy & Sequence

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Technical Foundation & IEC 62304 Baseline     │ <- [COMPLETED]
│ Architecture, Ingestion, Quality Copilot, Refactor ML  │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ Phase 2: Multi-Center Prospective Validation Study     │ <- [NEXT STEP]
│ 1,500 Hospital Patients (AI vs. Multi-Cardiologist consensus)
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ Phase 3: CDSCO Class C Medical Device Filing (India)   │
│ Form MD-22 / MD-24 Clinical Investigation Submission   │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ Phase 4: US FDA 510(k) Clearance Submission            │
│ Substantial Equivalence to GE Marquette 12SL / FDA SaMD│
└────────────────────────────────────────────────────────┘
```

---

## 2. Key Clearance Prerequisites

1. **IEC 62304 Software Life Cycle Documentation**: Architecture, Traceability Matrix, Unit & System Verification reports.
2. **ISO 14971 Risk Management File**: Hazard identification, FMEA, residual risk evaluation.
3. **Pivotal Multi-Center Clinical Trial**: Prospective evaluation across $\ge 3$ hospital centers proving non-inferiority to physician review without AI.
4. **Cybersecurity Pre-Market Submission**: Software Bill of Materials (SBOM), threat modeling, encryption, vulnerability testing.
