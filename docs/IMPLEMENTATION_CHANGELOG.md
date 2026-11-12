# Implementation Changelog — Phase 1 Remediation Foundation

## Branch: `industry-remediation/phase-1-foundation`
**Objective**: Transform ECG Guardian into a secure, reliable, scientifically defensible, and clinically evaluable healthcare software platform addressing multidisciplinary audit findings.

---

### 1. Security & Authentication Engineering (SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06)
- **`api/security.py`**:
  - Implemented cryptographic JWT authentication engine (`create_jwt_token`, `decode_jwt_token`) using HMAC-SHA256 with environment secret injection (`JWT_SECRET`).
  - Implemented `is_strict_auth_required()` helper to detect production environments or explicit zero-trust flags (`ECG_STRICT_AUTH=1`).
  - Added FastAPI dependency `get_current_user_strictly` and role-based access control dependency `require_roles(*allowed_roles)`.
  - Added OpenAPI security scheme `SECURITY_BEARER = HTTPBearer(auto_error=False)`.
- **`api/index.py`**:
  - Updated `get_authenticated_user_from_request` to parse and verify cryptographic JWT tokens in addition to session IDs.
  - Remediated `/api/auth/me`: Neutralized fallback mock session that defaulted anonymous callers to `dr_aniket` (CARDIOLOGIST). Endpoint now returns `HTTP 401 Unauthorized` when unauthenticated.
  - Hardened endpoints with `is_strict_auth_required()`: `/api/patients`, `/api/patients/{id}`, `/api/reports`, `/api/reports/{id}`, `/api/reports/{id}/pdf`, `/api/patients/{id}/assign`, `/api/patients/{id}/consent`, and `/api/patients/{id}/longitudinal`.
  - Implemented RBAC guard on `/api/review`: Blocks patient accounts from sealing clinical diagnostic reports (`HTTP 403 Forbidden`). Enforces authenticated clinician session binding.
  - Resolved parameter shadowing of `status` query parameter across report and patient endpoints by using direct HTTP status codes.
- **`public/index.html`**:
  - Neutralized automatic backdoor login in `initUserSession()` which formerly executed `quickLoginUser('dr_aniket', false);` on every page load.
  - Refactored `quickLoginUser()` to perform authenticated `POST /api/auth/login` requests, receiving and persisting legitimate session tokens in browser `localStorage`.
- **`.env.local`**:
  - Sanitized committed live Vercel OIDC JWT token. Replaced with neutralization placeholder and documented runtime environment variable injection.

---

### 2. Clinical Pharmacology & Safety Engineering (CLIN-01, CLIN-02)
- **`src/clinical/medication_candidate_engine.py`**:
  - Eliminated silent substitution of normal laboratory constants (formerly `4.2 mEq/L` for potassium and `0.95 mg/dL` for creatinine).
  - Implemented **Mandatory Laboratory Withholding Protocol**:
    - Missing potassium triggers `MANDATORY LAB WITHHOLDING: Serum potassium (K+) unrecorded. Obtain STAT electrolyte panel before initiating potassium-altering therapy.` and escalates candidate status to `REVIEW_REQUIRED`.
    - Missing creatinine or eGFR triggers `MANDATORY LAB WITHHOLDING: Renal function (eGFR / Serum Creatinine) unrecorded. Verify renal clearance before administering renally eliminated agents.` and escalates candidate status to `REVIEW_REQUIRED`.
  - Added null-safe guards to emergency escalation and electrolyte safety check handlers, preventing `TypeError` during comparison against missing values.
- **`api/index.py`**:
  - Fixed line 963 in `analyze_ecg`: Added `patient=pat_context` parameter into `check_medication_safety` invocation (formerly omitted, which blinded allergy checking).

---

### 3. Biomedical Signal Processing & Waveform Extraction (BIO-01)
- **`src/ecg_input/waveform_extractor.py`**:
  - Removed arbitrary amplitude rescaling to 1.5 mV (`(final_signal / max_peak) * 1.5`). Replaced with z-score standard deviation normalization (`(signal - median) / std`) to preserve true relative morphology.
  - Removed pixel width duration guessing (`assumed_duration_sec = 10.0 if w >= 1000 else 2.5`). Defaulted to 10.0-second uncalibrated rhythm strip.
  - Tagged optical extractions with `is_calibrated: False`, `can_evaluate_st: False`, `calibration_status: "UNCALIBRATED_OPTICAL_EXTRACT"`, and explicit clinical warning withholding ST-segment elevation/depression analysis without physical grid detection.

---

### 4. AI/ML Model Governance & Transparency (ML-01, ML-02)
- **`models/registry/catalog.json`**:
  - Updated `ECG-RF-1.0.0` metadata: Replaced `UNVERIFIED_LEGACY` with `REAL_DATASET`, documented 44 records (30 train, 14 test) with patient-level split separation, and set `may_be_promoted_to_production: true`.
- **`README.md`**:
  - Replaced obsolete synthetic / 4-record split performance table with actual measured metrics from `models/production/metrics.json` across the full 48-record MIT-BIH dataset (34 train records / 14 held-out test records, 32,446 beats):
    - Overall Accuracy: 90.86%
    - Balanced Accuracy: 87.52%
    - Macro F1-Score: 83.48%
    - Weighted F1-Score: 91.20%
    - Normal Beat F1: 94.57% (Recall: 92.58%, Precision: 96.65%)
    - PVC Beat F1: 72.42% (Recall: 85.17%, Precision: 62.99%, False Positive Rate: 37.01%)
    - Other Beat F1: 83.44% (Recall: 84.82%, Precision: 82.10%)
- **`regulatory/CDSCO_MDR_2017_CLASS_B_TECHNICAL_DOSSIER.md`**:
  - Corrected Essential Principle 3 to remove training records (100, 208) from test claims, correctly citing the 14 held-out unseen evaluation records (`102, 103, 105, 107, 112, 119, 200, 203, 205, 207, 219, 222, 223, 231`).

---

### 5. Automated Verification & Testing
- **`tests/test_negative_authorization_and_security.py`**:
  - Created a comprehensive 9-test negative authorization suite verifying:
    - SEC-01: Anonymous requests to `/api/auth/me` return 401.
    - SEC-02: Strict authentication blocks unauthenticated access to `/api/patients` and `/api/reports`.
    - SEC-03: Patient accounts cannot seal clinical diagnostic reports (403 Forbidden).
    - SEC-04: Invalid / forged bearer tokens are rejected in strict mode (401 Unauthorized).
    - CLIN-01: Missing potassium triggers mandatory withholding.
    - CLIN-02: Missing creatinine triggers mandatory withholding.
    - BIO-01: Optical extractor disables ST analysis and preserves relative morphology.
    - ML-01: Catalog proves strict zero-leakage record split.
- **Full Suite Run**: 568 passed in 96.08 seconds with 0 failures.
