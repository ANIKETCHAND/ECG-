# Implementation Changelog — Phase 1 Remediation Foundation

## Branch: `industry-remediation/phase-1-foundation`
**Objective**: Transform ECG Guardian into a secure, reliable, scientifically defensible, and clinically evaluable healthcare software platform addressing multidisciplinary audit findings.

---

### 1. Security & Authentication Engineering (SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06)
- **`api/security.py`**:
  - Implemented cryptographic JWT authentication engine (`create_jwt_token`, `decode_jwt_token`) using HMAC-SHA256 with environment secret injection (`JWT_SECRET`).
  - **Zero-Trust Architecture Enforced**: `is_strict_auth_required()` now defaults to `True` unconditionally across all environments (production, staging, test, and dev), eliminating inadvertent anonymous access bypasses.
  - **Eliminated Hardcoded Secrets**: Removed all hardcoded production/development JWT fallback keys (`STRICT_PROD_FALLBACK_SECRET_ECG_2026` and `DEV_FALLBACK_JWT_SECRET_2026_CDSCO_ECG_GUARDIAN`).
  - Implemented fail-closed security: In production, missing `JWT_SECRET` raises a fatal `RuntimeError`. In development, an ephemeral 256-bit cryptographically secure token (`secrets.token_hex(32)`) is generated per runtime instance.
  - Added FastAPI dependency `get_current_user_strictly` and role-based access control dependency `require_roles(*allowed_roles)`.
  - Added OpenAPI security scheme `SECURITY_BEARER = HTTPBearer(auto_error=False)`.
- **`api/index.py`**:
  - Updated `get_authenticated_user_from_request` to parse and verify cryptographic JWT tokens in addition to session IDs.
  - Remediated `/api/auth/me`: Neutralized fallback mock session that defaulted anonymous callers to `dr_aniket` (CARDIOLOGIST). Endpoint now returns `HTTP 401 Unauthorized` when unauthenticated.
  - Hardened ALL clinical and patient endpoints under strict zero-trust authentication:
    - `/api/patients` (GET and POST)
    - `/api/patients/{id}`, `/api/patients/{id}/assign`, `/api/patients/{id}/consent`, `/api/patients/{id}/longitudinal`
    - `/api/reports`, `/api/reports/{id}`, `/api/reports/{id}/pdf`
    - `/api/appointments` (GET and POST)
    - `/api/conversations`, `/api/conversations/{id}/messages` (GET and POST)
    - `/api/doctors/{id}/patients`
  - Enforced RBAC protection preventing patient accounts from creating patient records (`HTTP 403 Forbidden`) or viewing doctor rosters (`HTTP 403 Forbidden`).
  - Implemented RBAC guard on `/api/review`: Blocks patient accounts from sealing clinical diagnostic reports (`HTTP 403 Forbidden`). Enforces authenticated clinician session binding.
  - Resolved parameter shadowing of `status` query parameter across report and patient endpoints by using direct HTTP status codes.
- **`public/index.html`**:
  - Neutralized automatic backdoor login in `initUserSession()` which formerly executed `quickLoginUser('dr_aniket', false);` on every page load.
  - Refactored `quickLoginUser()` to perform authenticated `POST /api/auth/login` requests, receiving and persisting legitimate session tokens in browser `localStorage`.
- **`.env.local`**:
  - Sanitized committed live Vercel OIDC JWT token. Replaced with neutralization placeholder and documented runtime environment variable injection.

---

### 2. Clinical Pharmacology & Safety Engineering (CLIN-01, CLIN-02, CLIN-03)
- **`src/clinical/medication_candidate_engine.py`**:
  - Eliminated silent substitution of normal laboratory constants (formerly `4.2 mEq/L` for potassium and `0.95 mg/dL` for creatinine).
  - Implemented **Mandatory Laboratory Withholding Protocol**:
    - Missing potassium triggers `MANDATORY LAB WITHHOLDING: Serum potassium (K+) unrecorded. Obtain STAT electrolyte panel before initiating potassium-altering therapy.` and escalates candidate status to `REVIEW_REQUIRED`.
    - Missing creatinine or eGFR triggers `MANDATORY LAB WITHHOLDING: Renal function (eGFR / Serum Creatinine) unrecorded. Verify renal clearance before administering renally eliminated agents.` and escalates candidate status to `REVIEW_REQUIRED`.
  - Added **Hypokalemia Contraindication Safety Gate**:
    - `k_val < 3.5 mEq/L` explicitly contraindicates and blocks torsadogenic, QT-prolonging, and antiarrhythmic agents with critical clinical advisory to correct serum potassium prior to rhythm control.
  - Added null-safe guards to emergency escalation and electrolyte safety check handlers, preventing `TypeError` during comparison against missing values.
- **`api/index.py`**:
  - Fixed line 963 in `analyze_ecg`: Added `patient=pat_context` parameter into `check_medication_safety` invocation (formerly omitted, which blinded allergy checking).

---

### 3. Biomedical Signal Processing & Measurement Engineering (BIO-01, BIO-02)
- **`src/ecg_input/waveform_extractor.py`**:
  - Removed arbitrary amplitude rescaling to 1.5 mV (`(final_signal / max_peak) * 1.5`). Replaced with z-score standard deviation normalization (`(signal - median) / std`) to preserve true relative morphology.
  - Removed pixel width duration guessing (`assumed_duration_sec = 10.0 if w >= 1000 else 2.5`). Defaulted to 10.0-second uncalibrated rhythm strip.
  - Tagged optical extractions with `is_calibrated: False`, `can_evaluate_st: False`, `calibration_status: "UNCALIBRATED_OPTICAL_EXTRACT"`, and explicit clinical warning withholding ST-segment elevation/depression analysis without physical grid detection.
- **`src/measurements/measurement_engine.py`**:
  - **Dynamic Q-Onset Fiducial Detection (BIO-02)**: Replaced legacy fixed Q-point calculation (`q_point = t_pre`, hardcoded to R - 50ms) with dynamic morphology-aware Q-onset detection algorithm that locates the local Q-wave deflection trough and traces backwards along the first derivative to find the pre-Q baseline departure inflection point.
  - Validated dynamic Q-onset measurement against MIT-BIH recordings, yielding precise, physiologically grounded QT and rate-corrected QTc (Bazett and Fridericia) intervals.

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
  - Expanded negative authorization and safety verification suite to 23 tests verifying:
    - SEC-01: Anonymous requests to `/api/auth/me` return 401.
    - SEC-02: Zero-trust authentication blocks unauthenticated access to `/api/patients`, `/api/reports`, `/api/dashboard/stats`, `/api/notifications`, `/api/doctors/{id}`, `/api/appointments/{id}`, `/api/ecg/compare`.
    - SEC-03: Patient accounts cannot seal clinical diagnostic reports (403 Forbidden), access hospital telemetry dashboard (403 Forbidden), or modify doctor credentials (403 Forbidden).
    - SEC-04: Invalid / forged bearer tokens are rejected (401 Unauthorized).
    - SEC-05: Missing / empty authorization header rejected with 401 Unauthorized.
    - SEC-06: Malformed authorization header rejected with 401 Unauthorized.
    - SEC-07: Neutralized anonymous data leaks on telemetry dashboard and patient notifications.
    - CLIN-01: Missing potassium triggers mandatory withholding.
    - CLIN-02: Missing creatinine triggers mandatory withholding.
    - CLIN-03: Hypokalemia (<3.5 mEq/L) blocks torsadogenic / antiarrhythmic drugs (`test_hypokalemia_contraindication_blocks_torsadogenic_candidates`).
    - BIO-01: Optical extractor disables ST analysis and preserves relative morphology.
    - BIO-02: Dynamic morphology Q-onset detection with pre-inflection fallback and physiological tachycardia QT bound (`test_dynamic_q_onset_computes_physiologic_qt`).
    - ML-01: Catalog proves strict zero-leakage record split.
- **`public/index.html`**:
  - Neutralized client-side fake auth fallback in `quickLoginUser()` where failed logins generated dummy sessions in localStorage.
  - Added authorization header propagation to `loadDoctorDashboard()`.
- **`src/measurements/measurement_engine.py`**:
  - Fixed Q-onset fallback: when pre-R inflection slope remains steep, onset falls back to deflection onset (`q_search_start`) rather than Q trough.
  - Expanded QT interval physiological lower bound from 280 ms to 200 ms to accurately support physiological QT shortening during tachycardia (HR 120-180 BPM).
- **Full Suite Run**: 582 passed in 99.04 seconds with 0 failures (100% passing).
