# Post-Market Surveillance (PMS) & Vigilance Plan
**Regulatory Standards:** ISO TR 20416:2020 / EU MDR Article 83–86 / CDSCO MDR 2017 Chapter VI  
**System:** ECG GUARDIAN Clinical Decision Support Platform  

---

## 1. Post-Market Data Collection Channels

1. **Clinician Feedback & Overrides**:
   - Every physician modification or rejection of an AI recommendation is logged in `clinician_reviews` with rationale.
   - Monthly aggregate analysis of disagreement rates to monitor real-world algorithm drift.
2. **Quality Failure Metrics**:
   - Tracking frequency of unparseable formats, high-noise rejections, and unsupported lead rejections.
3. **Formal Complaint Management**:
   - Hospital-reported adverse events, near-misses, or diagnostic discrepancies routed directly to Quality Management.

---

## 2. Vigilance & Incident Reporting Thresholds

- **Serious Adverse Event (SAE)**:
  - Definition: Any incident that directly or indirectly led to patient death, serious deterioration in health, or emergency intervention.
  - Notification Window: Within 48 hours to CDSCO / 15 days to FDA / competent authorities.
- **Root Cause Analysis (RCA)**:
  - Mandated within 14 days of any confirmed false-negative ventricular arrhythmia complaint.
