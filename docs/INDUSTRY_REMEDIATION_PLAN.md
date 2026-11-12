# ECG Guardian — Industry Remediation Plan

**Status:** Engineering remediation plan; not a certification, clinical validation, or deployment authorization.  
**Baseline:** Repository review of the default branch and selected source files.  
**Target:** A secure, testable hospital-workflow platform with a clearly bounded investigational ECG analysis function.

## 1. Product boundary

ECG Guardian is not approved or clinically validated for diagnosis, triage, treatment selection, or autonomous prescribing. Until the required regulatory and clinical evidence exists, all ECG outputs are investigational decision-support information for qualified clinician review. No patient-facing diagnosis or medication instruction may be generated from an unapproved AI result.

Keep the administrative hospital record workflow separate from the medical-device software functions. Define intended use, target population, users, care setting, supported ECG formats, supported leads, supported conditions, contraindications, and exclusions before expanding claims.

## 2. Release-blocking findings from the current review

The following issues must be treated as release blockers until resolved and verified:

1. Default clinical user accounts and passwords are created in application source.
2. Authentication/session state is held in process memory, which is not durable or consistent across serverless instances.
3. The API security module permits data endpoints to operate when the shared API key is unset.
4. SQLite data is directed to a temporary serverless filesystem path, which is not durable clinical persistence.
5. Model documentation reports strong aggregate accuracy but a 0 F1 result for the "Other" class in the described test evaluation.
6. Model-card and intended-use class scopes/model versions are inconsistent.
7. A model registry entry marked production has unverified legacy provenance.
8. Medication contraindication matching includes free-text keyword matching, which is not sufficient for clinical medication safety.
9. ECG image digitization documentation acknowledges limitations in absolute amplitude recovery.
10. The current evidence does not establish independent clinical validation, hospital acceptance, or regulatory authorization.

Do not silently downgrade, hide, or re-label any of these findings. Close each with a linked test or evidence artifact.

## 3. Workstream A — Identity, access control, and tenant isolation

### Required engineering
- Remove all hardcoded default user credentials and sample clinical identities from production code.
- Use a supported identity provider or a securely implemented persistent identity service. Do not keep users, passwords, or sessions in Python dictionaries.
- Use short-lived access tokens, secure refresh-token rotation, logout/revocation, session expiry, and account recovery.
- Enforce server-side authorization on every patient, ECG, report, message, appointment, and file operation.
- Establish tenant-scoped resource ownership. A user must not gain access by changing a patient ID, report ID, storage path, or URL.
- Separate hospital administrator, clinician, technician, nurse, researcher, and patient permissions using least privilege.
- Verify clinician credentials through a documented administrative workflow; do not trust a self-entered registration number.
- Require MFA for privileged hospital roles where supported.
- Add rate limiting, brute-force protection, security event logging, and safe error responses.

### Acceptance criteria
- No production default accounts or credentials exist.
- Unauthenticated requests are rejected on every protected route.
- Automated tests prove cross-hospital and cross-patient access is denied.
- Role-permission tests cover allow and deny cases for every protected resource.
- Sessions remain correct across multiple backend instances and restarts.
- No tokens, passwords, patient identifiers, or clinical payloads are written to ordinary application logs.

## 4. Workstream B — Durable clinical data and file storage

### Required engineering
- Make PostgreSQL the source of truth for hospital, user, patient, ECG study, analysis run, clinician review, report, message, appointment, consent, and audit records.
- Use schema migrations and enforce foreign keys, constraints, transactions, and tenant scoping.
- Use private object storage for ECG source files and generated reports. Use short-lived signed access URLs only after authorization.
- Store file hashes, media type, size, source metadata, uploader, patient association, and immutable provenance.
- Never use a serverless temporary filesystem as persistent clinical storage.
- Define backup, restore, retention, deletion, and disaster-recovery procedures.
- Treat missing values as unknown, not negative. Preserve source, timestamp, verification status, and author for clinical facts.

### Acceptance criteria
- A deployment restart does not lose committed records or files.
- Backup restoration is tested and documented.
- Concurrent writes do not create duplicate or orphaned clinical records.
- Storage authorization tests prove that one patient cannot retrieve another patient's files.
- Every clinical record has tenant and audit provenance.

## 5. Workstream C — ECG ingestion and biomedical validity

### Required engineering
- Separate digital waveform ingestion from PDF parsing, scanned ECGs, and camera photographs.
- Validate sampling frequency, units, lead names, calibration, duration, channel count, clipping, flatline, missing samples, and signal integrity.
- Preserve the original upload as immutable source evidence.
- Add a quality gate that can reject unusable or unsupported input without returning a diagnostic prediction.
- Make supported lead configurations explicit per model.
- Validate filtering, baseline correction, R-peak detection, beat segmentation, interval measurements, and waveform digitization against reference annotations.
- For images, record grid scale, calibration confidence, reconstruction error, and whether voltage amplitude is reliable.
- Never silently infer missing calibration or sampling metadata.

### Acceptance criteria
- Invalid and unsupported inputs return a clear failure state, not a normal/abnormal diagnosis.
- Measurement algorithms have reference-based test fixtures and documented error bounds.
- Image-digitization performance is measured separately from native digital waveform performance.
- Every analysis records input hash, parser version, preprocessing version, model version, and processing timestamp.

## 6. Workstream D — AI/ML model governance and validation

### Required engineering
- Establish one authoritative model registry and reconcile model IDs, versions, class labels, intended use, and deployment status.
- Block deployment of models with unverified provenance.
- Reproduce training from versioned datasets and scripts.
- Split data by patient, not by individual beats or image variants. Prevent source-record leakage across derived images.
- Report per-class sensitivity, specificity, precision, recall, macro F1, weighted F1, confusion matrices, AUROC/AUPRC where appropriate, calibration, and confidence intervals.
- Investigate minority-class failures and class imbalance. Do not use aggregate accuracy to conceal class failure.
- Conduct external validation on independent institutions/devices and representative populations before clinical claims.
- Add drift monitoring and controlled model change management.
- Implement abstention for low confidence, out-of-distribution input, unsupported lead configuration, and poor signal quality.

### Acceptance criteria
- No model can be marked production unless provenance, intended use, evaluation, and approval metadata are complete.
- Evaluation reports are reproducible from pinned data and code.
- Each supported class has pre-specified acceptance thresholds approved by qualified clinical reviewers.
- A model failing a class-specific threshold is blocked from that intended use.
- No unvalidated model output is described as a diagnosis.

## 7. Workstream E — Clinical workflow and medication safety

### Required engineering
- Treat AI output as preliminary and clearly distinguish it from clinician-confirmed interpretation.
- Require a clinician review/sign-off step before a report is released as a clinical report.
- Preserve original AI output, clinician edits, reviewer identity, timestamps, and report versions.
- Build medication checks as evidence-linked safety checks, not an autonomous prescribing engine.
- Verify ingredient identity, allergies and reaction, current medication list, interactions, contraindications, renal/hepatic context, age-specific cautions, and source currency.
- Use structured clinical terminology and curated knowledge sources; free-text keyword overlap must not be the sole safety mechanism.
- If required context is missing, show "insufficient information" and withhold a recommendation.
- Do not invent drug, dose, route, frequency, or duration.
- Keep patient-facing language educational and limited to clinician-approved content.

### Acceptance criteria
- No medication recommendation reaches a patient without clinician approval.
- Missing or unknown allergy status is never treated as "no known allergies."
- High-severity interaction and contraindication scenarios have validated test cases.
- Every safety alert records its source, version/date, matched patient facts, and reviewer disposition.
- The system never autonomously issues a prescription.

## 8. Workstream F — Hospital workflow and user experience

Implement in this order:
1. Hospital onboarding and verified staff accounts.
2. Patient registration, identity matching, and duplicate-record handling.
3. Secure doctor-patient assignment and revocation.
4. Clinical history, allergies, medication reconciliation, and vitals with provenance.
5. ECG upload, processing status, failure handling, and worklist.
6. Clinician review and sign-off.
7. Versioned technical and patient-friendly reports.
8. Patient access to approved reports.
9. Secure messaging and appointments.
10. Audit and hospital administration.

A screen is not complete because it renders. Every action must have a real API, authorization policy, validation, error state, loading state, and automated test.

## 9. Workstream G — Security, privacy, and operational resilience

- Threat-model patient, clinician, hospital, researcher, administrator, and service identities.
- Use TLS in transit and managed encryption at rest.
- Keep secrets in deployment secret management; rotate exposed credentials.
- Validate uploads by content and parser safety; enforce limits and malware scanning where feasible.
- Apply dependency scanning, static analysis, secret scanning, and container/build provenance.
- Maintain tamper-evident audit events and alert on suspicious access.
- Define incident response, breach handling, service recovery, and downtime procedures.
- Minimize collected data and define purpose, access, retention, and deletion rules.
- Review applicable DPDP Act/Rules obligations and contractual processor responsibilities with counsel.

### Acceptance criteria
- Security tests cover authentication, authorization, tenant isolation, upload validation, rate limits, and session revocation.
- Critical/high vulnerabilities block release.
- Backup restoration and service recovery are exercised.
- Audit trails can reconstruct who accessed or changed a record and when.

## 10. Workstream H — Quality system and regulatory evidence

Maintain controlled artifacts:
- Intended-use statement and product boundary
- Software requirements specification
- Architecture and interface specifications
- Hazard analysis and risk controls
- Software development and configuration-management plan
- Verification and validation plan/reports
- Traceability matrix from requirements to tests
- Clinical evaluation plan and report
- Usability engineering file
- Cybersecurity risk assessment and SBOM
- Model cards and dataset provenance
- Release notes and change-impact assessments
- Post-market monitoring and incident process, if/when marketed

Assess applicability of India's Medical Devices Rules, 2017 and CDSCO's Guidance Document on Medical Device Software (CDSCO/MD/GD/MDSW/01/2026), alongside applicable standards such as ISO 14971, IEC 62304, IEC 81001-5-1 and ISO 13485. Regulatory applicability and classification must be confirmed by qualified regulatory professionals.

## 11. Delivery sequence and gates

### Phase 0 — Baseline and containment
- Freeze diagnostic and medication claims.
- Remove production access to sample accounts.
- Ensure protected API routes fail closed.
- Identify and rotate any credentials that may have been exposed.
- Confirm the actual production persistence path.

**Gate:** No known critical exposure or unsafe default deployment configuration.

### Phase 1 — Secure platform foundation
- Persistent identity and sessions.
- PostgreSQL schema and migrations.
- Tenant-scoped authorization and private file storage.
- Audit events and backup/restore.

**Gate:** Automated security and persistence tests pass.

### Phase 2 — Reproducible ECG pipeline
- Input contracts and modality separation.
- Quality gates and validated measurements.
- Immutable analysis provenance.
- Reproducible model registry and evaluation.

**Gate:** Biomedical and ML acceptance criteria pass for the bounded research intended use.

### Phase 3 — Clinician-controlled workflow
- Worklist, review, sign-off, versioned reports.
- Medication safety alerts with evidence and clinician disposition.
- Patient portal only for approved reports.

**Gate:** End-to-end workflow tests and clinical usability review pass.

### Phase 4 — Interoperability and operational readiness
- FHIR mapping and integration tests.
- Monitoring, incident response, recovery, and release management.
- Hospital workflow usability validation.

**Gate:** Pilot readiness review by independent clinical, security, quality, and regulatory reviewers.

### Phase 5 — Clinical and regulatory pathway
- Prospective/appropriate clinical validation under approved governance.
- Regulatory classification and licensing strategy.
- Formal quality-system evidence and required submissions.

**Gate:** No commercial or clinical deployment claims until required approvals and evidence are obtained.

## 12. Engineering rules

- Do not perform broad rewrites without a migration plan.
- Do not silently change model labels, outputs, or clinical claims.
- Do not fabricate test results, accuracy, approvals, or clinical evidence.
- Do not store patient data in local files, browser storage, or temporary serverless disks.
- Do not use patient data in development fixtures.
- Do not log PHI or secrets.
- Do not deploy an unverified model as production.
- Do not let the AI autonomously diagnose, triage, or prescribe.
- Every code change must include tests, migration notes where relevant, and rollback guidance.
- Work in small reviewable pull requests with CI checks.

## 13. Definition of done

A feature is done only when:
1. Its requirement and intended users are documented.
2. Authorization is enforced server-side.
3. Data persistence and audit behavior are verified.
4. Positive, negative, failure, and boundary tests exist.
5. UI loading, empty, error, and success states work.
6. Security and privacy impact is reviewed.
7. Clinical features have qualified clinical review.
8. Deployment and rollback instructions exist.

**Final release rule:** "Industry-level" is a target, not a label. The product must earn release through traceable engineering, reproducible tests, independent validation, and the applicable regulatory pathway.
