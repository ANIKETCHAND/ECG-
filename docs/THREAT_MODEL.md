# Comprehensive Threat Model (STRIDE Methodology)

## 1. Scope & System Assets

The threat model applies to the ECG Guardian healthcare platform, covering patient health records, diagnostic ECG data, machine learning pipelines, and clinical report delivery.

### Key System Assets:
1. **Protected Health Information (PHI) & PII**: Patient demographics, medical history, allergies, vitals, lab values.
2. **ECG Signal Telemetry**: Raw voltage waveforms, optical scans, EDF/DICOM-ECG files.
3. **Clinical Diagnostic Decisions**: AI inferences, interval measurements, clinician review sign-offs, and treatment plans.
4. **Authentication Secrets**: JWT signing secret, API keys, clinician passwords, Supabase service-role keys.
5. **System Audit Logs**: Immutable transaction logs proving regulatory compliance and data access history.

---

## 2. STRIDE Analysis & Security Mitigations

### 2.1 Spoofing (Identity Deception)
- **Threat S-1**: An attacker impersonates a licensed cardiologist to forge clinical report approvals or alter prescribed regimens.
  - *Mitigation*: Cryptographic JWT authentication (HS256 with 256-bit entropy secret). Session binding verifies that caller claims match the internal database user registry. Medical registration numbers are bound to authenticated sessions, not user-editable payload fields.
- **Threat S-2**: An attacker presents forged Google OAuth tokens to create elevated administrative accounts.
  - *Mitigation*: Cryptographic verification of Google ID tokens via JWKS public keys (`src/auth/jwks_validator.py`). Role auto-provisioning defaults to unprivileged roles unless explicitly pre-authorized in the hospital user directory.

### 2.2 Tampering (Data Manipulation)
- **Threat T-1**: Modification of ECG voltage samples in transit to induce misdiagnosis or suppress PVC detection.
  - *Mitigation*: Mandatory HTTPS / TLS 1.3 encryption in transit. Waveform extraction and inference pipeline generates SHA-256 content hashes recorded in the Authoritative Clinical Analysis Object (ACAO).
- **Threat T-2**: Tampering with finalized diagnostic reports.
  - *Mitigation*: Immutable report persistence. When a clinician seals a report, an SHA-256 cryptographic seal is computed over the findings, reviewer ID, timestamp, and signature, and stored in the database audit log.

### 2.3 Repudiation (Denial of Action)
- **Threat R-1**: A clinician denies having approved an incorrect medication candidate or missed an acute arrhythmia.
  - *Mitigation*: Non-repudiation audit logging. The `audit_events` ledger records the authenticated `user_id`, IP address, timestamp, exact action (`CLINICIAN_REVIEW_SEALED`), and cryptographic hash of the approved report.

### 2.4 Information Disclosure (Data Breach)
- **Threat I-1**: Cross-tenant data leakage where Hospital A staff access patient records from Hospital B.
  - *Mitigation*: Multi-tenant database partitioning. Every query is filtered by `hospital_id`. PostgreSQL Row Level Security (RLS) policies enforce tenant isolation at the database layer.
- **Threat I-2**: IDOR / BOLA attack where a patient queries another patient's ECG or longitudinal history.
  - *Mitigation*: Object-level ownership validation. In `/api/patients/{id}` and `/api/patients/{id}/longitudinal`, if the caller role is `PATIENT`, the endpoint enforces `caller.user_id == target_patient.patient_id` or raises `HTTP 403 Forbidden`.

### 2.5 Denial of Service (Service Disruption)
- **Threat D-1**: Flooding `/api/analyze` with oversized high-resolution images or malformed signals to exhaust serverless compute resources.
  - *Mitigation*: Ingress payload limits (max 10MB per file), optical extractor downscaling, decoupled asynchronous execution, and timeout caps (max 10s per request).

### 2.6 Elevation of Privilege (Unauthorized Capability)
- **Threat E-1**: A patient account promotes itself to `CARDIOLOGIST` or invokes `/api/review` to sign reports.
  - *Mitigation*: Strict RBAC dependency `require_roles(UserRole.CARDIOLOGIST, UserRole.PHYSICIAN)`. Role self-elevation via registration or update endpoints is explicitly blocked; role assignments require `HOSPITAL_ADMIN` authorization.

---

## 3. Residual Risks & Ongoing Monitoring

1. **Client-Side Storage**: In single-page app sessions, JWT tokens are cached in browser `localStorage`.
   - *Recommendation for Phase 2*: Transition to HttpOnly, SameSite=Strict cookies to eliminate cross-site scripting (XSS) token extraction risk.
2. **Third-Party Model Integrity**: Model pickle files (`.pkl`) are loaded via `joblib`.
   - *Mitigation*: Production models are stored in protected local directories with hash validation prior to deserialization (`models/registry/validator.py`).
