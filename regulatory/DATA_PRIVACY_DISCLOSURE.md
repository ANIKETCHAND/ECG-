# Health Data Privacy & Security Disclosure
**Regulatory References:**
- India Digital Personal Data Protection (DPDP) Act, 2023
- US Health Insurance Portability and Accountability Act (HIPAA) 45 CFR Part 160/164
- EU General Data Protection Regulation (GDPR) (EU) 2016/679
- ISO 27799:2016 (Health informatics — Information security management in health)

---

## 1. Principles of Protected Health Information (PHI) Handling

1. **Zero-PHI Console & Error Logging**:
   - System error logs, diagnostic stack traces, and monitoring daemons NEVER write patient names, addresses, phone numbers, or unmasked MRNs.
2. **De-Identification by Default**:
   - Patient names are masked on public displays (`John Doe` $\rightarrow$ `J*** D**`).
   - Patient identifiers are stored as encrypted internal UUIDs.
3. **Data Ingestion Hygiene**:
   - ECG machine DICOM and XML headers are stripped of non-essential demographics prior to model inference caching.
4. **Data at Rest & in Transit**:
   - Transit: TLS 1.3 encryption mandatory for all web traffic and API invocations.
   - Rest: AES-256 encryption on database volumes, file storage, and audit logs.
5. **Access Control & Multi-Tenancy**:
   - Role-Based Access Control (RBAC): TECHNICIAN, CLINICIAN, AUDITOR, ADMIN.
   - Strict hospital boundary segregation; cross-tenant record leakage is prevented at the database schema query layer.
