-- ==============================================================================
-- ECG Guardian — Patient-Specific Intelligence & Medication Safety Schema Extensions
-- Migration: 20261112000002_patient_specific_ecg_and_medication_intelligence.sql
-- ==============================================================================
-- Idempotent schema migration supporting Sections 1-9:
--   1. Complete patient profile: Rh factor, physical anthropometry (height, weight, BMI),
--      emergency contacts, food/other allergies, surgical and hospitalization history,
--      discrete diagnostic condition flags (diabetes, HTN, CAD, prior MI, HF, arrhythmia, CKD, LFT).
--   2. Extended vitals (height, weight, BMI, temperature_celsius).
--   3. Extended medication history (brand name, discontinuation metadata, adverse reactions).
--   4. Multimodal laboratory panel (electrolytes Na+/K+, renal function Cr/eGFR, troponin, hepatic panel).
--   5. Clinician review & prescribing workflow (clinical diagnosis, structured prescriptions, selections/rejections).
-- ==============================================================================

DO $$
BEGIN
    -- 1. Patients Table Extensions
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patients') THEN
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS rh_factor TEXT DEFAULT 'Unknown';
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS height_cm NUMERIC;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS weight_kg NUMERIC;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS bmi NUMERIC;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS contact TEXT;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS emergency_contact TEXT;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS food_allergies JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS other_allergies JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS past_medications JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS discontinued_medications JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS previous_surgeries JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS previous_hospitalizations JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS diabetes BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS hypertension BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS coronary_artery_disease BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS previous_heart_attack BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS heart_failure BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS arrhythmia_history BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS kidney_disease BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS liver_disease BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS thyroid_disorders BOOLEAN DEFAULT FALSE;
        ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS other_diagnosed_diseases TEXT;
    END IF;

    -- 2. Patient Vitals Table Extensions
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patient_vitals') THEN
        ALTER TABLE public.patient_vitals ADD COLUMN IF NOT EXISTS height_cm NUMERIC;
        ALTER TABLE public.patient_vitals ADD COLUMN IF NOT EXISTS weight_kg NUMERIC;
        ALTER TABLE public.patient_vitals ADD COLUMN IF NOT EXISTS bmi NUMERIC;
        ALTER TABLE public.patient_vitals ADD COLUMN IF NOT EXISTS temperature_celsius NUMERIC;
    END IF;

    -- 3. Patient Medications Table Extensions
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patient_medications') THEN
        ALTER TABLE public.patient_medications ADD COLUMN IF NOT EXISTS brand_name TEXT;
        ALTER TABLE public.patient_medications ADD COLUMN IF NOT EXISTS discontinued_date DATE;
        ALTER TABLE public.patient_medications ADD COLUMN IF NOT EXISTS discontinue_reason TEXT;
        ALTER TABLE public.patient_medications ADD COLUMN IF NOT EXISTS adverse_reactions TEXT;
    END IF;

    -- 4. Laboratory Results Table Extensions
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'laboratory_results') THEN
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS potassium_mmol_l NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS sodium_mmol_l NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS creatinine_mg_dl NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS egfr_ml_min NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS troponin_i_ng_ml NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS alt_u_l NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS ast_u_l NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS bilirubin_mg_dl NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS alk_phos_u_l NUMERIC;
        ALTER TABLE public.laboratory_results ADD COLUMN IF NOT EXISTS lft_summary TEXT;
    END IF;

    -- 5. Clinical Reviews Table Extensions
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'clinical_reviews') THEN
        ALTER TABLE public.clinical_reviews ADD COLUMN IF NOT EXISTS clinician_diagnosis TEXT;
        ALTER TABLE public.clinical_reviews ADD COLUMN IF NOT EXISTS prescriptions JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.clinical_reviews ADD COLUMN IF NOT EXISTS selected_medications JSONB DEFAULT '[]'::jsonb;
        ALTER TABLE public.clinical_reviews ADD COLUMN IF NOT EXISTS rejected_medications JSONB DEFAULT '[]'::jsonb;
    END IF;
END $$;

-- Indexes for efficient patient retrieval
CREATE INDEX IF NOT EXISTS idx_patients_rh ON public.patients(rh_factor);
CREATE INDEX IF NOT EXISTS idx_patients_conditions ON public.patients USING gin(existing_conditions);
