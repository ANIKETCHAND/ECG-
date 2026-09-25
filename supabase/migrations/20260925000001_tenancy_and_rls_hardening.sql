-- ==============================================================================
-- ECG Guardian — Tenancy & RLS Hardening
-- ==============================================================================
-- Follow-up to 20260924000001_initial_schema.sql.
--
-- That migration enabled RLS on nine tables and added policies, but three gaps
-- remained:
--
--   1. SEVEN patient-data tables had no RLS at all, so they defaulted to
--      readable by any authenticated caller: patient_vitals,
--      patient_medications, laboratory_results, machine_interpretations,
--      interpretation_comparisons, clinical_recommendations,
--      medication_safety_checks.
--
--   2. Self-registration privilege escalation. The profiles INSERT policy was
--      `WITH CHECK (user_id = auth.uid() OR public.current_user_role() = 'ADMIN')`.
--      A brand-new authenticated user satisfies the first clause, and the role
--      column was unchecked — so anyone could insert their own profile with
--      role = 'ADMIN'.
--
--   3. No hospital scoping. `is_clinical_staff()` asked only "is this person
--      clinical staff somewhere?", so a clinician at one hospital could read
--      patients belonging to another. The multi-tenant isolation the platform
--      claims was not actually enforced by the database.
--
-- This migration is written to be safe to re-run (IF NOT EXISTS / DROP POLICY
-- IF EXISTS guards throughout).
-- ==============================================================================


-- ------------------------------------------------------------------------------
-- 1. Give patients an explicit owning hospital
-- ------------------------------------------------------------------------------
-- Tenancy previously had to be inferred from `created_by`, which is fragile if a
-- record is created by a rotating account or imported in bulk.
ALTER TABLE public.patients
    ADD COLUMN IF NOT EXISTS hospital_id TEXT;

-- Backfill from the creating user's profile where possible. Rows that cannot be
-- attributed are left NULL and are deliberately NOT visible to tenant-scoped
-- staff policies below; they remain reachable by ADMIN for remediation.
UPDATE public.patients p
SET hospital_id = pr.hospital_id
FROM public.profiles pr
WHERE p.hospital_id IS NULL
  AND p.created_by = pr.user_id;

CREATE INDEX IF NOT EXISTS idx_patients_hospital ON public.patients(hospital_id);


-- ------------------------------------------------------------------------------
-- 2. Identity helpers (hospital aware)
-- ------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.current_user_hospital_id()
RETURNS TEXT AS $$
    SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid() LIMIT 1;
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- Resolve a patient's owning hospital, preferring the explicit column and
-- falling back to the creator's profile for legacy rows.
CREATE OR REPLACE FUNCTION public.patient_hospital(p_patient_id UUID)
RETURNS TEXT AS $$
    SELECT COALESCE(
        (SELECT hospital_id FROM public.patients WHERE id = p_patient_id),
        (SELECT pr.hospital_id
           FROM public.profiles pr
           JOIN public.patients pt ON pt.created_by = pr.user_id
          WHERE pt.id = p_patient_id
          LIMIT 1)
    );
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- True when the caller is clinical staff AND the patient belongs to the caller's
-- hospital. A NULL hospital on either side fails closed.
CREATE OR REPLACE FUNCTION public.can_access_patient(p_patient_id UUID)
RETURNS BOOLEAN AS $$
    SELECT public.is_clinical_staff()
       AND public.current_user_hospital_id() IS NOT NULL
       AND public.patient_hospital(p_patient_id) IS NOT NULL
       AND public.current_user_hospital_id() = public.patient_hospital(p_patient_id);
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- Same check reached through an ECG row.
CREATE OR REPLACE FUNCTION public.can_access_ecg(p_ecg_id UUID)
RETURNS BOOLEAN AS $$
    SELECT public.can_access_patient(
        (SELECT patient_id FROM public.ecg_recordings WHERE id = p_ecg_id)
    );
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- Patients may read their own record. A patient profile links to a patient
-- record through medical_registration_number -> patient_code.
CREATE OR REPLACE FUNCTION public.is_own_patient(p_patient_id UUID)
RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.profiles pr
        JOIN public.patients pt ON pt.patient_code = pr.medical_registration_number
        WHERE pr.user_id = auth.uid()
          AND pr.role = 'PATIENT'
          AND pt.id = p_patient_id
    );
$$ LANGUAGE sql SECURITY DEFINER STABLE;


-- ------------------------------------------------------------------------------
-- 3. Fix self-registration privilege escalation on profiles
-- ------------------------------------------------------------------------------
-- A new user may create their own profile row, but never with a privileged
-- role. Role assignment beyond the self-service set is an ADMIN action.
DROP POLICY IF EXISTS "profiles_insert_admin_or_own" ON public.profiles;

CREATE POLICY "profiles_insert_self_service_or_admin"
    ON public.profiles FOR INSERT
    TO authenticated
    WITH CHECK (
        (
            -- Self-service registration: identity must match and the role must
            -- be one a new user is allowed to claim.
            user_id = auth.uid()
            AND role IN ('NURSE', 'RESEARCHER', 'ECG_TECHNICIAN', 'PATIENT')
        )
        OR public.current_user_role() = 'ADMIN'
    );

-- Role changes are an ADMIN action; a user cannot promote themselves.
CREATE OR REPLACE FUNCTION public.prevent_self_role_change()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.role IS DISTINCT FROM OLD.role
       AND public.current_user_role() IS DISTINCT FROM 'ADMIN' THEN
        RAISE EXCEPTION 'Only an ADMIN may change a profile role (attempted % -> %)',
            OLD.role, NEW.role;
    END IF;
    IF NEW.hospital_id IS DISTINCT FROM OLD.hospital_id
       AND public.current_user_role() IS DISTINCT FROM 'ADMIN' THEN
        RAISE EXCEPTION 'Only an ADMIN may move a profile between hospitals';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS trg_profiles_prevent_self_role_change ON public.profiles;
CREATE TRIGGER trg_profiles_prevent_self_role_change
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_self_role_change();


-- ------------------------------------------------------------------------------
-- 4. Tenant-scope the existing clinical policies
-- ------------------------------------------------------------------------------
-- Replace broad staff checks with hospital-scoped equivalents, so "clinical
-- staff" means "clinical staff at this patient's hospital".

DROP POLICY IF EXISTS "patients_select_staff_or_self" ON public.patients;
CREATE POLICY "patients_select_tenant_or_self"
    ON public.patients FOR SELECT
    TO authenticated
    USING (
        public.can_access_patient(id)
        OR public.is_own_patient(id)
        OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "ecg_select_staff_or_patient" ON public.ecg_recordings;
CREATE POLICY "ecg_select_tenant_or_self"
    ON public.ecg_recordings FOR SELECT
    TO authenticated
    USING (
        public.can_access_ecg(id)
        OR public.is_own_patient(patient_id)
        OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "measurements_select_staff_or_patient" ON public.ecg_measurements;
CREATE POLICY "measurements_select_tenant_or_self"
    ON public.ecg_measurements FOR SELECT
    TO authenticated
    USING (
        public.can_access_ecg(ecg_id)
        OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "ai_analyses_select" ON public.ai_analyses;
CREATE POLICY "ai_analyses_select_tenant"
    ON public.ai_analyses FOR SELECT
    TO authenticated
    USING (
        public.can_access_patient(patient_id)
        OR public.can_access_ecg(ecg_id)
        OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "ai_evidence_select" ON public.ai_evidence;
CREATE POLICY "ai_evidence_select_tenant"
    ON public.ai_evidence FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.ai_analyses a
            WHERE a.id = ai_evidence.analysis_id
              AND (public.can_access_ecg(a.ecg_id) OR public.current_user_role() = 'ADMIN')
        )
    );

DROP POLICY IF EXISTS "reports_select_clinical_or_own_signed" ON public.reports;
CREATE POLICY "reports_select_tenant_or_own_signed"
    ON public.reports FOR SELECT
    TO authenticated
    USING (
        public.can_access_patient(patient_id)
        OR public.current_user_role() = 'ADMIN'
        OR (report_status = 'SIGNED' AND public.is_own_patient(patient_id))
    );

DROP POLICY IF EXISTS "clinical_reviews_select" ON public.clinical_reviews;
CREATE POLICY "clinical_reviews_select_tenant"
    ON public.clinical_reviews FOR SELECT
    TO authenticated
    USING (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');


-- ------------------------------------------------------------------------------
-- 5. RLS for the seven previously unprotected patient-data tables
-- ------------------------------------------------------------------------------

-- patient_vitals
ALTER TABLE public.patient_vitals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "patient_vitals_select_tenant" ON public.patient_vitals;
CREATE POLICY "patient_vitals_select_tenant"
    ON public.patient_vitals FOR SELECT TO authenticated
    USING (public.can_access_patient(patient_id) OR public.is_own_patient(patient_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "patient_vitals_write_tenant" ON public.patient_vitals;
CREATE POLICY "patient_vitals_write_tenant"
    ON public.patient_vitals FOR INSERT TO authenticated
    WITH CHECK (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "patient_vitals_update_tenant" ON public.patient_vitals;
CREATE POLICY "patient_vitals_update_tenant"
    ON public.patient_vitals FOR UPDATE TO authenticated
    USING (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');

-- patient_medications
ALTER TABLE public.patient_medications ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "patient_medications_select_tenant" ON public.patient_medications;
CREATE POLICY "patient_medications_select_tenant"
    ON public.patient_medications FOR SELECT TO authenticated
    USING (public.can_access_patient(patient_id) OR public.is_own_patient(patient_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "patient_medications_write_tenant" ON public.patient_medications;
CREATE POLICY "patient_medications_write_tenant"
    ON public.patient_medications FOR INSERT TO authenticated
    WITH CHECK (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "patient_medications_update_tenant" ON public.patient_medications;
CREATE POLICY "patient_medications_update_tenant"
    ON public.patient_medications FOR UPDATE TO authenticated
    USING (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');

-- laboratory_results
ALTER TABLE public.laboratory_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "laboratory_results_select_tenant" ON public.laboratory_results;
CREATE POLICY "laboratory_results_select_tenant"
    ON public.laboratory_results FOR SELECT TO authenticated
    USING (public.can_access_patient(patient_id) OR public.is_own_patient(patient_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "laboratory_results_write_tenant" ON public.laboratory_results;
CREATE POLICY "laboratory_results_write_tenant"
    ON public.laboratory_results FOR INSERT TO authenticated
    WITH CHECK (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');

-- machine_interpretations
ALTER TABLE public.machine_interpretations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "machine_interpretations_select_tenant" ON public.machine_interpretations;
CREATE POLICY "machine_interpretations_select_tenant"
    ON public.machine_interpretations FOR SELECT TO authenticated
    USING (public.can_access_ecg(ecg_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "machine_interpretations_write_tenant" ON public.machine_interpretations;
CREATE POLICY "machine_interpretations_write_tenant"
    ON public.machine_interpretations FOR INSERT TO authenticated
    WITH CHECK (public.can_access_ecg(ecg_id) OR public.current_user_role() = 'ADMIN');

-- interpretation_comparisons
ALTER TABLE public.interpretation_comparisons ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "interpretation_comparisons_select_tenant" ON public.interpretation_comparisons;
CREATE POLICY "interpretation_comparisons_select_tenant"
    ON public.interpretation_comparisons FOR SELECT TO authenticated
    USING (public.can_access_ecg(ecg_id) OR public.current_user_role() = 'ADMIN');
DROP POLICY IF EXISTS "interpretation_comparisons_write_tenant" ON public.interpretation_comparisons;
CREATE POLICY "interpretation_comparisons_write_tenant"
    ON public.interpretation_comparisons FOR INSERT TO authenticated
    WITH CHECK (public.can_access_ecg(ecg_id) OR public.current_user_role() = 'ADMIN');

-- clinical_recommendations
ALTER TABLE public.clinical_recommendations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "clinical_recommendations_select_tenant" ON public.clinical_recommendations;
CREATE POLICY "clinical_recommendations_select_tenant"
    ON public.clinical_recommendations FOR SELECT TO authenticated
    USING (
        public.can_access_patient(patient_id)
        OR public.can_access_ecg(ecg_id)
        OR public.current_user_role() = 'ADMIN'
    );
DROP POLICY IF EXISTS "clinical_recommendations_write_tenant" ON public.clinical_recommendations;
CREATE POLICY "clinical_recommendations_write_tenant"
    ON public.clinical_recommendations FOR INSERT TO authenticated
    WITH CHECK (public.can_access_ecg(ecg_id) OR public.current_user_role() = 'ADMIN');
-- Recommendation status changes are a clinician act, not a patient act.
DROP POLICY IF EXISTS "clinical_recommendations_update_clinician" ON public.clinical_recommendations;
CREATE POLICY "clinical_recommendations_update_clinician"
    ON public.clinical_recommendations FOR UPDATE TO authenticated
    USING (public.is_doctor_or_admin() OR public.can_access_patient(patient_id));

-- medication_safety_checks
ALTER TABLE public.medication_safety_checks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "medication_safety_select_tenant" ON public.medication_safety_checks;
CREATE POLICY "medication_safety_select_tenant"
    ON public.medication_safety_checks FOR SELECT TO authenticated
    USING (
        public.can_access_patient(patient_id)
        OR public.can_access_ecg(ecg_id)
        OR public.current_user_role() = 'ADMIN'
    );
DROP POLICY IF EXISTS "medication_safety_write_tenant" ON public.medication_safety_checks;
CREATE POLICY "medication_safety_write_tenant"
    ON public.medication_safety_checks FOR INSERT TO authenticated
    WITH CHECK (public.can_access_patient(patient_id) OR public.current_user_role() = 'ADMIN');


-- ------------------------------------------------------------------------------
-- 6. Supporting indexes for the new tenant predicates
-- ------------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_profiles_hospital ON public.profiles(hospital_id);
CREATE INDEX IF NOT EXISTS idx_profiles_user_role ON public.profiles(user_id, role);
CREATE INDEX IF NOT EXISTS idx_vitals_patient ON public.patient_vitals(patient_id);
CREATE INDEX IF NOT EXISTS idx_medications_patient ON public.patient_medications(patient_id);
CREATE INDEX IF NOT EXISTS idx_labs_patient ON public.laboratory_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_machine_interp_ecg ON public.machine_interpretations(ecg_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_ecg ON public.interpretation_comparisons(ecg_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_ecg ON public.clinical_recommendations(ecg_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_patient ON public.clinical_recommendations(patient_id);
CREATE INDEX IF NOT EXISTS idx_med_safety_patient ON public.medication_safety_checks(patient_id);
CREATE INDEX IF NOT EXISTS idx_med_safety_ecg ON public.medication_safety_checks(ecg_id);
