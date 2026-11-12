-- ==============================================================================
-- ECG Guardian — Appointments, Messaging, Notifications & DPDP Compliance
-- ==============================================================================
-- Follow-up migration establishing Row Level Security (RLS) for:
--   1. Clinical Appointments & Consultations
--   2. Doctor-Patient Messaging (Conversations & Messages)
--   3. In-App Notifications
--   4. Patient Care Team Assignments
--   5. DPDP Act 2025 Informed Consent Registry
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Appointments Table & RLS
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.appointments (
    appointment_id TEXT PRIMARY KEY,
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
    patient_id TEXT NOT NULL,
    patient_mrn TEXT,
    patient_name TEXT NOT NULL,
    doctor_id TEXT,
    doctor_name TEXT NOT NULL,
    appointment_date TEXT NOT NULL,
    appointment_time TEXT NOT NULL,
    consultation_type TEXT DEFAULT 'In-Person Hospital Visit',
    location TEXT,
    reason TEXT,
    notes TEXT,
    status TEXT DEFAULT 'CONFIRMED',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.appointments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Staff view hospital appointments" ON public.appointments;
CREATE POLICY "Staff view hospital appointments" ON public.appointments
    FOR SELECT USING (
        public.is_clinical_staff() AND hospital_id = public.current_user_hospital_id()
    );

DROP POLICY IF EXISTS "Patients view own appointments" ON public.appointments;
CREATE POLICY "Patients view own appointments" ON public.appointments
    FOR SELECT USING (
        auth.uid()::text = patient_id OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "Staff manage hospital appointments" ON public.appointments;
CREATE POLICY "Staff manage hospital appointments" ON public.appointments
    FOR ALL USING (
        (public.is_clinical_staff() AND hospital_id = public.current_user_hospital_id())
        OR public.current_user_role() = 'ADMIN'
    );

CREATE INDEX IF NOT EXISTS idx_appointments_hosp_date ON public.appointments(hospital_id, appointment_date);
CREATE INDEX IF NOT EXISTS idx_appointments_pat ON public.appointments(patient_id);


-- ------------------------------------------------------------------------------
-- 2. Conversations & Messaging Tables & RLS
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.conversations (
    conversation_id TEXT PRIMARY KEY,
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
    patient_id TEXT NOT NULL,
    doctor_id TEXT NOT NULL,
    patient_name TEXT,
    doctor_name TEXT,
    subject TEXT,
    status TEXT DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES public.conversations(conversation_id) ON DELETE CASCADE,
    sender_id TEXT NOT NULL,
    sender_name TEXT NOT NULL,
    sender_role TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    content TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Participants view conversations" ON public.conversations;
CREATE POLICY "Participants view conversations" ON public.conversations
    FOR SELECT USING (
        auth.uid()::text = patient_id
        OR auth.uid()::text = doctor_id
        OR (public.is_clinical_staff() AND hospital_id = public.current_user_hospital_id())
        OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "Participants view messages" ON public.messages;
CREATE POLICY "Participants view messages" ON public.messages
    FOR SELECT USING (
        auth.uid()::text = sender_id
        OR auth.uid()::text = recipient_id
        OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "Participants send messages" ON public.messages;
CREATE POLICY "Participants send messages" ON public.messages
    FOR INSERT WITH CHECK (
        auth.uid()::text = sender_id OR public.is_clinical_staff() OR public.current_user_role() = 'ADMIN'
    );


-- ------------------------------------------------------------------------------
-- 3. Notifications Table & RLS
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.notifications (
    notification_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    notification_type TEXT DEFAULT 'SYSTEM',
    related_id TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users view own notifications" ON public.notifications;
CREATE POLICY "Users view own notifications" ON public.notifications
    FOR SELECT USING (
        auth.uid()::text = user_id OR user_id = 'ALL' OR public.current_user_role() = 'ADMIN'
    );

DROP POLICY IF EXISTS "Users update own notifications" ON public.notifications;
CREATE POLICY "Users update own notifications" ON public.notifications
    FOR UPDATE USING (
        auth.uid()::text = user_id OR public.current_user_role() = 'ADMIN'
    );


-- ------------------------------------------------------------------------------
-- 4. DPDP Act 2025 Patient Consent Registry
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.patient_consents (
    consent_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-APEX',
    consent_type TEXT NOT NULL DEFAULT 'ECG_DATA_PROCESSING',
    status TEXT NOT NULL DEFAULT 'GRANTED',
    purposes TEXT[] NOT NULL DEFAULT '{"AI_DIAGNOSTIC_ASSISTANCE", "TELEMETRY_MONITORING"}',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    notes TEXT
);

ALTER TABLE public.patient_consents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Staff and Patients view consent" ON public.patient_consents;
CREATE POLICY "Staff and Patients view consent" ON public.patient_consents
    FOR SELECT USING (
        (public.is_clinical_staff() AND hospital_id = public.current_user_hospital_id())
        OR auth.uid()::text = patient_id
        OR public.current_user_role() = 'ADMIN'
    );
