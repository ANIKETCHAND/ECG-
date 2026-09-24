-- ==============================================================================
-- ECG Guardian — Production Database Migration (Phases 2 through 22, 35, 38, 39)
-- Regulatory References: CDSCO Medical Devices Rules 2017 & IEC 62304 Class B
-- ==============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ------------------------------------------------------------------------------
-- Phase 3: Profiles Table (Links to Supabase auth.users)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN (
        'ADMIN', 'DOCTOR', 'CARDIOLOGIST', 'ECG_TECHNICIAN', 'NURSE', 'RESEARCHER', 'PATIENT'
    )),
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
    medical_registration_number TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 4: Patients Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_code TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    date_of_birth DATE,
    sex TEXT CHECK (sex IN ('M', 'F', 'O', 'Male', 'Female', 'Other')),
    blood_group TEXT,
    known_allergies JSONB DEFAULT '[]'::jsonb,
    existing_conditions JSONB DEFAULT '[]'::jsonb,
    cardiac_history JSONB DEFAULT '[]'::jsonb,
    family_history JSONB DEFAULT '[]'::jsonb,
    smoking_status TEXT,
    created_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 5: Patient Vitals Table (Historical, non-overwriting)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.patient_vitals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    systolic_bp NUMERIC,
    diastolic_bp NUMERIC,
    heart_rate NUMERIC,
    spo2 NUMERIC,
    temperature NUMERIC,
    respiratory_rate NUMERIC,
    source TEXT DEFAULT 'VITAL_MONITOR',
    recorded_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 6: Patient Medications Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.patient_medications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    medication_name TEXT NOT NULL,
    generic_name TEXT,
    dose TEXT,
    route TEXT,
    frequency TEXT,
    start_date DATE,
    end_date DATE,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DISCONTINUED', 'HISTORICAL', 'UNKNOWN')),
    source TEXT,
    created_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 7: Laboratory Results Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.laboratory_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    test_name TEXT NOT NULL,
    value NUMERIC,
    value_text TEXT,
    unit TEXT,
    reference_range TEXT,
    abnormal_flag TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT DEFAULT 'LABORATORY',
    created_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 8: ECG Recordings Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.ecg_recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    recording_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_format TEXT NOT NULL DEFAULT 'DIGITAL',
    sampling_rate NUMERIC NOT NULL DEFAULT 360.0,
    lead_names JSONB NOT NULL DEFAULT '["II"]'::jsonb,
    number_of_leads INTEGER NOT NULL DEFAULT 1,
    duration_seconds NUMERIC NOT NULL DEFAULT 10.0,
    device_name TEXT,
    device_manufacturer TEXT,
    signal_quality TEXT,
    signal_quality_score NUMERIC,
    snr NUMERIC,
    storage_path TEXT,
    status TEXT NOT NULL DEFAULT 'UPLOADED' CHECK (status IN (
        'UPLOADED', 'VALIDATING', 'QUALITY_CHECK', 'ANALYZING', 'ANALYSIS_COMPLETE', 'REVIEW_REQUIRED', 'SIGNED', 'FAILED'
    )),
    created_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 9: ECG Measurements Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.ecg_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    heart_rate NUMERIC,
    mean_rr NUMERIC,
    pr_interval NUMERIC,
    qrs_duration NUMERIC,
    qt_interval NUMERIC,
    qtc_interval NUMERIC,
    detected_beats INTEGER,
    baseline_drift BOOLEAN DEFAULT false,
    powerline_interference BOOLEAN DEFAULT false,
    motion_artifacts BOOLEAN DEFAULT false,
    measurement_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 10: AI Analyses Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.ai_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL DEFAULT 'RandomForestClassifier',
    model_version TEXT NOT NULL DEFAULT '1.0.0',
    task TEXT NOT NULL DEFAULT 'ARRHYTHMIA_CLASSIFICATION',
    primary_prediction TEXT NOT NULL,
    model_probabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    beat_counts JSONB DEFAULT '{}'::jsonb,
    warnings JSONB DEFAULT '[]'::jsonb,
    limitations JSONB DEFAULT '[]'::jsonb,
    processing_time_ms NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 11: AI Evidence Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.ai_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES public.ai_analyses(id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    lead TEXT DEFAULT 'II',
    start_time NUMERIC,
    end_time NUMERIC,
    feature_name TEXT,
    feature_value NUMERIC,
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 12: Machine Interpretations Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.machine_interpretations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'ECG Machine Firmware',
    interpretation_text TEXT,
    structured_findings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 13: Interpretation Comparisons Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.interpretation_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'AGREEMENT' CHECK (status IN (
        'AGREEMENT', 'MINOR_DIFFERENCE', 'SIGNIFICANT_DISAGREEMENT', 'UNABLE_TO_COMPARE'
    )),
    ai_findings JSONB DEFAULT '{}'::jsonb,
    machine_findings JSONB DEFAULT '{}'::jsonb,
    differences JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 14: Clinical Recommendations (CDS) Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.clinical_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
    recommendation_type TEXT NOT NULL,
    recommendation_text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'AHA/ACC/ESC Guidelines',
    source_version TEXT,
    evidence JSONB DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'PENDING_REVIEW' CHECK (status IN (
        'PENDING_REVIEW', 'REVIEWED', 'ACCEPTED', 'REJECTED', 'MODIFIED'
    )),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 15: Medication Safety Checks Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.medication_safety_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES public.patients(id) ON DELETE CASCADE,
    ecg_id UUID REFERENCES public.ecg_recordings(id) ON DELETE SET NULL,
    active_medications JSONB DEFAULT '[]'::jsonb,
    interaction_alerts JSONB DEFAULT '[]'::jsonb,
    allergy_alerts JSONB DEFAULT '[]'::jsonb,
    contraindication_alerts JSONB DEFAULT '[]'::jsonb,
    source_versions JSONB DEFAULT '{}'::jsonb,
    overall_status TEXT NOT NULL DEFAULT 'NO_ALERT' CHECK (overall_status IN (
        'NO_ALERT', 'REVIEW_REQUIRED', 'INCOMPLETE_INFORMATION'
    )),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 16: Clinical Reviews Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.clinical_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
    review_status TEXT NOT NULL DEFAULT 'CONFIRMED' CHECK (review_status IN ('CONFIRMED', 'MODIFIED', 'REJECTED')),
    clinician_id UUID REFERENCES auth.users(id),
    clinician_interpretation TEXT NOT NULL,
    clinical_diagnosis TEXT,
    clinical_directives TEXT,
    treatment_plan TEXT,
    medication_plan TEXT,
    follow_up_plan TEXT,
    signed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 17 & 18: Reports Table (IMMUTABLE SNAPSHOT STORAGE)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_number TEXT UNIQUE NOT NULL,
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    ecg_id UUID NOT NULL REFERENCES public.ecg_recordings(id) ON DELETE CASCADE,
    ai_analysis_id UUID REFERENCES public.ai_analyses(id) ON DELETE SET NULL,
    report_status TEXT NOT NULL DEFAULT 'DRAFT' CHECK (report_status IN (
        'DRAFT', 'PENDING_REVIEW', 'REVIEWED', 'SIGNED', 'AMENDED', 'ARCHIVED'
    )),
    report_title TEXT NOT NULL DEFAULT 'Clinical ECG Analysis Report',
    report_data JSONB NOT NULL,
    report_version INTEGER NOT NULL DEFAULT 1,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    signed_at TIMESTAMPTZ,
    created_by UUID REFERENCES auth.users(id),
    reviewed_by UUID REFERENCES auth.users(id),
    pdf_storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- Phase 35: Audit Logs Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    action TEXT NOT NULL CHECK (action IN (
        'REPORT_CREATED', 'REPORT_VIEWED', 'REPORT_DOWNLOADED', 'REPORT_EDITED',
        'REPORT_SIGNED', 'REPORT_VERSION_CREATED', 'ECG_UPLOADED', 'ECG_ANALYZED',
        'PATIENT_CREATED', 'PATIENT_UPDATED'
    )),
    entity_type TEXT NOT NULL,
    entity_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ------------------------------------------------------------------------------
-- Phase 38 & 39: Performance Indexes and Integrity Constraints
-- ------------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_patients_code ON public.patients(patient_code);
CREATE INDEX IF NOT EXISTS idx_patients_name ON public.patients(full_name);

CREATE INDEX IF NOT EXISTS idx_ecg_patient ON public.ecg_recordings(patient_id);
CREATE INDEX IF NOT EXISTS idx_ecg_timestamp ON public.ecg_recordings(recording_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ecg_status ON public.ecg_recordings(status);

CREATE INDEX IF NOT EXISTS idx_measurements_ecg ON public.ecg_measurements(ecg_id);

CREATE INDEX IF NOT EXISTS idx_ai_analyses_ecg ON public.ai_analyses(ecg_id);
CREATE INDEX IF NOT EXISTS idx_ai_analyses_patient ON public.ai_analyses(patient_id);

CREATE INDEX IF NOT EXISTS idx_reports_patient ON public.reports(patient_id);
CREATE INDEX IF NOT EXISTS idx_reports_ecg ON public.reports(ecg_id);
CREATE INDEX IF NOT EXISTS idx_reports_number ON public.reports(report_number);
CREATE INDEX IF NOT EXISTS idx_reports_status ON public.reports(report_status);
CREATE INDEX IF NOT EXISTS idx_reports_generated ON public.reports(generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_clinical_reviews_ecg ON public.clinical_reviews(ecg_id);
CREATE INDEX IF NOT EXISTS idx_clinical_reviews_patient ON public.clinical_reviews(patient_id);

CREATE INDEX IF NOT EXISTS idx_audit_user ON public.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON public.audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON public.audit_logs(action);

-- ------------------------------------------------------------------------------
-- Phase 21 & 22: Row Level Security (RLS) Policies
-- ------------------------------------------------------------------------------

-- Helper functions to identify user roles and patient bindings securely
CREATE OR REPLACE FUNCTION public.current_user_role()
RETURNS TEXT AS $$
    SELECT role FROM public.profiles WHERE user_id = auth.uid() LIMIT 1;
$$ LANGUAGE sql SECURITY DEFINER STABLE;

CREATE OR REPLACE FUNCTION public.is_clinical_staff()
RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.profiles
        WHERE user_id = auth.uid()
          AND role IN ('ADMIN', 'DOCTOR', 'CARDIOLOGIST', 'ECG_TECHNICIAN', 'NURSE')
    );
$$ LANGUAGE sql SECURITY DEFINER STABLE;

CREATE OR REPLACE FUNCTION public.is_doctor_or_admin()
RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.profiles
        WHERE user_id = auth.uid()
          AND role IN ('ADMIN', 'DOCTOR', 'CARDIOLOGIST')
    );
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- 1. Profiles Table RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "profiles_select_own_or_staff"
    ON public.profiles FOR SELECT
    TO authenticated
    USING (user_id = auth.uid() OR public.is_clinical_staff());

CREATE POLICY "profiles_insert_admin_or_own"
    ON public.profiles FOR INSERT
    TO authenticated
    WITH CHECK (user_id = auth.uid() OR public.current_user_role() = 'ADMIN');

CREATE POLICY "profiles_update_own_or_admin"
    ON public.profiles FOR UPDATE
    TO authenticated
    USING (user_id = auth.uid() OR public.current_user_role() = 'ADMIN');

-- 2. Patients Table RLS
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

CREATE POLICY "patients_select_staff_or_self"
    ON public.patients FOR SELECT
    TO authenticated
    USING (
        public.is_clinical_staff()
        OR id = (SELECT id FROM public.patients WHERE patient_code = (SELECT medical_registration_number FROM public.profiles WHERE user_id = auth.uid()))
    );

CREATE POLICY "patients_insert_staff"
    ON public.patients FOR INSERT
    TO authenticated
    WITH CHECK (public.is_clinical_staff());

CREATE POLICY "patients_update_staff"
    ON public.patients FOR UPDATE
    TO authenticated
    USING (public.is_clinical_staff());

-- 3. ECG Recordings Table RLS
ALTER TABLE public.ecg_recordings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "ecg_select_staff_or_patient"
    ON public.ecg_recordings FOR SELECT
    TO authenticated
    USING (
        public.is_clinical_staff()
        OR patient_id = (SELECT id FROM public.patients WHERE patient_code = (SELECT medical_registration_number FROM public.profiles WHERE user_id = auth.uid()))
    );

CREATE POLICY "ecg_insert_staff"
    ON public.ecg_recordings FOR INSERT
    TO authenticated
    WITH CHECK (public.is_clinical_staff());

CREATE POLICY "ecg_update_staff"
    ON public.ecg_recordings FOR UPDATE
    TO authenticated
    USING (public.is_clinical_staff());

-- 4. ECG Measurements Table RLS
ALTER TABLE public.ecg_measurements ENABLE ROW LEVEL SECURITY;

CREATE POLICY "measurements_select_staff_or_patient"
    ON public.ecg_measurements FOR SELECT
    TO authenticated
    USING (
        public.is_clinical_staff()
        OR ecg_id IN (
            SELECT id FROM public.ecg_recordings WHERE patient_id = (
                SELECT id FROM public.patients WHERE patient_code = (SELECT medical_registration_number FROM public.profiles WHERE user_id = auth.uid())
            )
        )
    );

CREATE POLICY "measurements_insert_staff"
    ON public.ecg_measurements FOR INSERT
    TO authenticated
    WITH CHECK (public.is_clinical_staff());

-- 5. AI Analyses & Evidence RLS
ALTER TABLE public.ai_analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "ai_analyses_select"
    ON public.ai_analyses FOR SELECT
    TO authenticated
    USING (
        public.is_clinical_staff()
        OR patient_id = (SELECT id FROM public.patients WHERE patient_code = (SELECT medical_registration_number FROM public.profiles WHERE user_id = auth.uid()))
    );

CREATE POLICY "ai_analyses_insert_staff"
    ON public.ai_analyses FOR INSERT
    TO authenticated
    WITH CHECK (public.is_clinical_staff());

ALTER TABLE public.ai_evidence ENABLE ROW LEVEL SECURITY;

CREATE POLICY "ai_evidence_select"
    ON public.ai_evidence FOR SELECT
    TO authenticated
    USING (public.is_clinical_staff());

CREATE POLICY "ai_evidence_insert_staff"
    ON public.ai_evidence FOR INSERT
    TO authenticated
    WITH CHECK (public.is_clinical_staff());

-- 6. Reports Table RLS (THE PRIMARY TABLE)
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "reports_select_clinical_or_own_signed"
    ON public.reports FOR SELECT
    TO authenticated
    USING (
        public.is_clinical_staff()
        OR (
            -- Patients can ONLY see their own SIGNED reports
            report_status = 'SIGNED'
            AND patient_id = (SELECT id FROM public.patients WHERE patient_code = (SELECT medical_registration_number FROM public.profiles WHERE user_id = auth.uid()))
        )
    );

CREATE POLICY "reports_insert_staff"
    ON public.reports FOR INSERT
    TO authenticated
    WITH CHECK (public.is_clinical_staff());

CREATE POLICY "reports_update_doctor_or_admin"
    ON public.reports FOR UPDATE
    TO authenticated
    USING (public.is_doctor_or_admin());

-- 7. Clinical Reviews Table RLS
ALTER TABLE public.clinical_reviews ENABLE ROW LEVEL SECURITY;

CREATE POLICY "clinical_reviews_select"
    ON public.clinical_reviews FOR SELECT
    TO authenticated
    USING (public.is_clinical_staff());

CREATE POLICY "clinical_reviews_insert_doctor"
    ON public.clinical_reviews FOR INSERT
    TO authenticated
    WITH CHECK (public.is_doctor_or_admin());

-- 8. Audit Logs Table RLS
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "audit_logs_select_admin"
    ON public.audit_logs FOR SELECT
    TO authenticated
    USING (public.current_user_role() = 'ADMIN');

CREATE POLICY "audit_logs_insert_authenticated"
    ON public.audit_logs FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() IS NOT NULL);

-- ------------------------------------------------------------------------------
-- Phases 19 & 20: Storage Bucket Setup & Access Policies
-- ------------------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public)
VALUES ('ecg-recordings', 'ecg-recordings', false)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public)
VALUES ('ecg-reports', 'ecg-reports', false)
ON CONFLICT (id) DO NOTHING;

-- Storage RLS: ecg-reports (Private)
CREATE POLICY "ecg_reports_storage_select"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (
        bucket_id = 'ecg-reports'
        AND (
            public.is_clinical_staff()
            OR (storage.foldername(name))[2] = (SELECT id::text FROM public.patients WHERE patient_code = (SELECT medical_registration_number FROM public.profiles WHERE user_id = auth.uid()))
        )
    );

CREATE POLICY "ecg_reports_storage_insert"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (
        bucket_id = 'ecg-reports'
        AND public.is_clinical_staff()
    );

-- Storage RLS: ecg-recordings (Private)
CREATE POLICY "ecg_recordings_storage_select"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (
        bucket_id = 'ecg-recordings'
        AND public.is_clinical_staff()
    );

CREATE POLICY "ecg_recordings_storage_insert"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (
        bucket_id = 'ecg-recordings'
        AND public.is_clinical_staff()
    );
