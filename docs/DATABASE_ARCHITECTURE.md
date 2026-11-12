# Production Database Architecture & Multi-Tenant Relational Schema

## 1. Relational Persistence Strategy

ECG Guardian utilizes a dual-tier persistence architecture designed for high availability, zero clinical data loss, and multi-tenant security:

1. **Primary Persistence**: Supabase Managed PostgreSQL.
   - Hosts all canonical tables, transactions, foreign keys, and Row Level Security (RLS) policies.
   - Backed by daily automated snapshots, WAL archiving, and point-in-time recovery (PITR).
2. **Local Dual Persistence & Resiliency Layer**:
   - High-performance local relational SQLite layer (`src/database/db_manager.py`).
   - Ensures continuous hospital floor and emergency room availability if external cloud connectivity experiences latency or outages.
   - When configured with Supabase credentials (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`), data is automatically synchronized to cloud PostgreSQL.

---

## 2. Normalized Entity Schema (18 Core Tables)

The schema satisfies third normal form (3NF) and enforces multi-tenant boundary integrity:

```
                                  ┌───────────────────────────┐
                                  │         hospitals         │
                                  └─────────────┬─────────────┘
                                                │ 1:N
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
          ┌───────────────────────────┐                   ┌───────────────────────────┐
          │           users           │                   │         patients          │
          └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                        │ 1:N                                           │ 1:N
            ┌───────────┴───────────┐                       ┌───────────┴───────────┐
            ▼                       ▼                       ▼                       ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────┐   ┌───────────────────┐
│  hospital_memberships │ │clinician_profiles │ │doctor_assignments │   │    ecg_studies    │
└───────────────────────┘ └───────────────────┘ └───────────────────┘   └─────────┬─────────┘
                                                                                  │ 1:N
                                                                        ┌─────────┴─────────┐
                                                                        ▼                   ▼
                                                              ┌───────────────────┐ ┌───────────────┐
                                                              │ ecg_analysis_runs │ │  ecg_reports  │
                                                              └─────────┬─────────┘ └───────┬───────┘
                                                                        │                   │
                                                                        ▼                   ▼
                                                              ┌───────────────────┐ ┌───────────────┐
                                                              │ clinician_reviews │ │ audit_events  │
                                                              └───────────────────┘ └───────────────┘
```

### Table Definitions & Key Constraints:

1. **`hospitals`**:
   - `hospital_id` (PK, TEXT), `name`, `code`, `tier`, `country`, `is_active`, `created_at`.
2. **`users`**:
   - `user_id` (PK, TEXT), `email` (UNIQUE), `username` (UNIQUE), `password_hash`, `role` (CHECK: ADMIN, CARDIOLOGIST, PHYSICIAN, NURSE, PATIENT), `hospital_id` (FK), `is_active`, `created_at`.
3. **`hospital_memberships`**:
   - `membership_id` (PK), `hospital_id` (FK), `user_id` (FK), `role`, `department`, `granted_at`.
4. **`clinician_profiles`**:
   - `profile_id` (PK), `user_id` (FK), `registration_number` (NOT NULL), `specialty`, `qualifications`, `signature_hash`.
5. **`patients`**:
   - `patient_id` (PK), `hospital_id` (FK), `hospital_mrn` (UNIQUE within hospital), `name`, `age`, `sex`, `phone`, `email`, `created_at`.
6. **`doctor_patient_assignments`**:
   - `assignment_id` (PK), `hospital_id` (FK), `patient_id` (FK), `doctor_id` (FK), `is_active`, `assigned_at`.
7. **`patient_conditions`**:
   - `condition_id` (PK), `patient_id` (FK), `icd10_code`, `condition_name`, `status`, `diagnosed_date`.
8. **`patient_allergies`**:
   - `allergy_id` (PK), `patient_id` (FK), `substance`, `reaction`, `severity`, `identified_at`.
9. **`patient_medications`**:
   - `medication_id` (PK), `patient_id` (FK), `drug_name`, `dosage`, `frequency`, `is_active`, `start_date`.
10. **`patient_vital_signs`**:
    - `vital_id` (PK), `patient_id` (FK), `systolic_bp`, `diastolic_bp`, `heart_rate`, `spo2`, `recorded_at`.
11. **`patient_lab_results`**:
    - `lab_id` (PK), `patient_id` (FK), `potassium`, `creatinine`, `egfr`, `magnesium`, `troponin_i`, `recorded_at`.
12. **`ecg_studies`**:
    - `study_id` (PK), `hospital_id` (FK), `patient_id` (FK), `source_file_id`, `acquisition_device`, `lead_config`, `sampling_rate_hz`, `recorded_at`.
13. **`ecg_source_files`**:
    - `file_id` (PK), `study_id` (FK), `storage_path`, `storage_bucket`, `sha256_hash`, `content_type`, `file_size_bytes`, `uploaded_at`.
14. **`ecg_analysis_runs`**:
    - `run_id` (PK), `study_id` (FK), `model_id`, `heart_rate_bpm`, `pr_interval_ms`, `qrs_duration_ms`, `qt_interval_ms`, `qtc_interval_ms`, `primary_finding`, `signal_quality_score`, `analysis_payload_json`, `created_at`.
15. **`model_registry`**:
    - `model_id` (PK), `version`, `task`, `status`, `artifact_path`, `accuracy`, `macro_f1`, `split_protocol`, `is_production_ready`.
16. **`clinician_reviews`**:
    - `review_id` (PK), `run_id` (FK), `clinician_id` (FK), `review_status` (CONFIRMED, MODIFIED, REJECTED), `clinician_notes`, `signature_seal`, `reviewed_at`.
17. **`ecg_reports`**:
    - `report_id` (PK), `study_id` (FK), `patient_id` (FK), `doctor_id` (FK), `hospital_id` (FK), `report_status` (PRELIMINARY, FINAL, AMENDED), `pdf_storage_path`, `sealed_at`.
18. **`audit_events`**:
    - `event_id` (PK), `hospital_id` (FK), `user_id`, `event_type`, `action`, `patient_id`, `record_id`, `ip_address`, `details_json`, `created_at`.

---

## 3. Row Level Security (RLS) & Multi-Tenant Protection

- **Tenant Isolation**: Every query from an authenticated clinician or staff member is scoped to their active `hospital_id`:
  ```sql
  CREATE POLICY hospital_tenant_isolation ON patients
      FOR ALL
      USING (hospital_id = current_setting('request.jwt.claim.hospital_id', true));
  ```
- **Patient Isolation**: Patients can access only records matching their verified `patient_id`:
  ```sql
  CREATE POLICY patient_own_records ON ecg_reports
      FOR SELECT
      USING (patient_id = current_setting('request.jwt.claim.patient_id', true) AND report_status = 'FINAL');
  ```
- **Audit Defense**: The `audit_events` table is append-only. UPDATE and DELETE actions are structurally revoked for all database roles.
