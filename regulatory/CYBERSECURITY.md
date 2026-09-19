# Medical Device Cybersecurity & Data Privacy Specification

**Device:** AI-ECG Analyzer  
**Standard:** IEC 81001-5-1 / ISO 27799 / CDSCO MDR 2017 Guidance on Cybersecurity  
**Document Ref:** SEC-2026-01  
**Status:** ACTIVE  

---

## 1. Cybersecurity Objective & Philosophy

In a connected hospital ecosystem, unauthorized alteration of patient ECG data or diagnostic algorithms represents a direct threat to patient safety. The AI-ECG Analyzer enforces defense-in-depth architecture:
- **Integrity First:** Physiological waveforms and model inferences are sealed with cryptographic digests.
- **Strict Least Privilege:** Clinical and administrative access is separated by granular Role-Based Access Control (RBAC).
- **Non-Repudiation:** Every action is chained into an append-only, tamper-evident cryptographic audit log.
- **Privacy by Design:** Patient identifiable information is decoupled and redacted in research contexts.

---

## 2. Threat Modeling (STRIDE Framework)

| Threat Category | Potential Attack Vector | Impact on Device | Mitigating Architectural Control |
|---|---|---|---|
| **Spoofing (Identity)** | Unauthorized user impersonating an attending cardiologist to sign off reports. | Compromise of clinical trust; invalid medical sign-off. | Strong credential hashing (`PBKDF2-HMAC-SHA256` with 100,000 rounds + unique salt), session expiration, and medical council registration number verification. |
| **Tampering (Data)** | Modification of ECG voltage array or AI probability predictions in transit or database. | Erroneous clinical decisions; false normal or false abnormal diagnosis. | SHA-256 cryptographic hashing of raw signal array upon ingestion; tamper-evident hash chaining in audit database; sealed PDF checksums. |
| **Repudiation** | Physician or technician denies having uploaded an ECG or modified a diagnosis. | Inability to trace clinical accountability during adverse event investigations. | Cryptographically chained audit trail (`AuditLogger`) linking user ID, role, client IP, action timestamp, and prior event SHA-256 hash. |
| **Information Disclosure** | Leakage of patient name, MRN, or diagnostic outcome to uncredentialed researchers. | Violation of DPDPA 2023 and hospital patient confidentiality rules. | Automated PHI redaction for `RESEARCHER` role (`research:view_deidentified`); patient names and MRNs masked with synthetic IDs. |
| **Denial of Service** | Flooding backend with corrupt image/PDF files or computationally heavy jobs. | Starvation of telemetry triage system during emergency screening. | Pre-inference file validation, rate limiting, and timeout caps on segmentation and inference loops. |
| **Elevation of Privilege** | Technician account attempting to alter system parameters or sign off clinical reviews. | Unauthorized practice of medicine and violation of MDR 2017 rules. | Strict server-side permission checks (`ROLE_PERMISSIONS`) denying `report:sign_off` and `system:configure` to non-physician roles. |

---

## 3. Cryptographic Controls & Data Integrity

1. **At-Rest Protection:**
   - Database files (`hospital_clinical.db`, `audit_trail.db`) stored with restricted filesystem permissions.
   - Production models (`models/production/`) validated against known SHA-256 digests on system startup.
2. **In-Transit Protection:**
   - Web application traffic must be terminated behind a reverse proxy (e.g. NGINX) enforcing **TLS 1.3** with modern cipher suites (`ECDHE-ECDSA-AES256-GCM-SHA384`).
3. **Audit Trail Cryptographic Hashing:**
   - Each audit event $E_i$ computes its hash as:
     $$\text{Hash}_i = \text{SHA256}(\text{seq}_i \parallel \text{timestamp}_i \parallel \text{event\_type}_i \parallel \text{user\_id}_i \parallel \text{action}_i \parallel \text{payload}_i \parallel \text{Hash}_{i-1})$$
   - Any retroactive mutation or row deletion produces a validation mismatch detectable via `AUDIT_LOGGER.verify_chain_integrity()`.

---

## 4. Software Bill of Materials (SBOM) & Dependency Management

The system maintains a deterministic dependency lock (`requirements.txt` / virtual environment):
- **Core Signal & Numerical Stack:** `numpy`, `scipy`, `pandas`
- **Machine Learning Runtime:** `scikit-learn`, `joblib`
- **Report & Document Processing:** `reportlab`, `PyPDF2`, `pdfplumber`, `opencv-python-headless`, `Pillow`
- **User Interface & Visualization:** `streamlit`, `plotly`, `matplotlib`
- **Testing & Quality Assurance:** `pytest`

### Vulnerability Monitoring
- Automated scanning using `safety` and GitHub Dependabot alerts.
- Critical vulnerabilities (CVSS $\ge 7.0$) require emergency patch triage within 72 hours under standard hospital SOP.

---

## 5. Hospital Network Deployment Security Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SECURE HOSPITAL NETWORK                         │
│                                                                        │
│  [ Hospital LAN / VLAN ]                                               │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────────────┐                                              │
│  │ TLS 1.3 NGINX Proxy  │ <── Port 443 (HTTPS)                         │
│  └──────────┬───────────┘                                              │
│             │                                                          │
│             ▼                                                          │
│  ┌──────────────────────┐      ┌─────────────────────────┐             │
│  │ AI-ECG Web Dashboard │ <──> │ RBAC & Session Manager  │             │
│  └──────────┬───────────┘      └─────────────────────────┘             │
│             │                                                          │
│             ▼                                                          │
│  ┌──────────────────────┐      ┌─────────────────────────┐             │
│  │ ML & Signal Pipeline │ <──> │ Production Model Card   │             │
│  └──────────┬───────────┘      └─────────────────────────┘             │
│             │                                                          │
│             ▼                                                          │
│  ┌──────────────────────┐      ┌─────────────────────────┐             │
│  │ Database & Storage   │ <──> │ Cryptographic Audit Log │             │
│  │ (SQLite / Data Dir)  │      │ (SHA-256 Hash Chaining) │             │
│  └──────────────────────┘      └─────────────────────────┘             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Incident Response & Breach Notification Policy

1. In the event of a detected data breach or cryptographic chain failure, the system automatically logs a high-priority alert.
2. The Medical IT Officer and Chief Data Officer are notified within 4 hours.
3. Relevant statutory notifications under India's Digital Personal Data Protection Act (DPDPA 2023) and CDSCO MDR 2017 are initiated in accordance with hospital institutional protocols.
