"""ECG Guardian — Hospital Clinical Decision Support Platform."""

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Setup system path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import get_available_records, load_annotations, load_record
from label_mapping import map_symbol_to_class
from prediction import get_trained_artifacts, predict_ecg
from visualization import (
    plot_beats_overlay,
    plot_ecg_signal,
    plot_feature_importance,
    plot_prediction_probabilities,
)
from ecg_input import (
    InputModality,
    detect_input_modality,
    load_digital_signal,
    STANDARD_SAMPLING_RATES,
    extract_report_measurements,
    process_pdf_report,
    process_ecg_image,
    extract_waveform_from_image,
    validate_extracted_signal,
)
from report import (
    generate_structured_report,
    export_report_to_text,
    export_report_to_json,
    generate_pdf_report,
    generate_doctor_report,
    generate_patient_report,
)
from auth.auth_manager import AUTH_MANAGER, UserRole
from database.db_manager import DB_MANAGER, PatientRecord
from audit.audit_logger import AUDIT_LOGGER
from clinical import GLOBAL_CDS_ENGINE, ClinicalRecommendation
from medications import GLOBAL_MEDICATION_DB, check_medication_safety
from alerts import GLOBAL_ALERT_ENGINE, AlertType, AlertSeverity
from components.analysis import (
    render_criteria_footer,
    get_waveform_criteria,
    get_segmentation_criteria,
    get_ai_classification_criteria,
    get_signal_quality_criteria,
    get_feature_table_criteria,
    get_patient_context_criteria,
    get_medication_safety_criteria,
    get_cds_criteria,
    get_feature_importance_criteria,
    CriteriaDescriptor,
)
from services import REPORT_PERSISTENCE_SERVICE, PersistenceStatus
from database.supabase_client import is_supabase_configured
from components.history_view import render_history_view
import secrets
from datetime import datetime

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ECG Guardian",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# Global CSS — Minimal Dark Clinical Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Top nav brand bar ── */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0 16px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 18px;
}
.brand-icon { font-size: 1.6rem; }
.brand-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.brand-sub {
    font-size: 0.7rem;
    color: #8b949e;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section headers ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 18px 0 8px 0;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 16px;
}
.metric-card-label {
    font-size: 0.68rem;
    color: #8b949e;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.metric-card-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1.1;
}
.metric-card-delta {
    font-size: 0.72rem;
    margin-top: 3px;
    font-weight: 500;
}
.delta-good { color: #3fb950; }
.delta-warn { color: #d29922; }
.delta-bad  { color: #f85149; }
.delta-off  { color: #8b949e; }

/* ── Patient identity strip ── */
.patient-strip {
    background: #161b22;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.patient-name { font-size: 0.95rem; font-weight: 700; color: #79c0ff; }
.patient-mrn  {
    background: #1c2d3f;
    color: #58a6ff;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: monospace;
}
.patient-meta { font-size: 0.78rem; color: #8b949e; }

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-green  { background: #1a3a2a; color: #3fb950; border: 1px solid #2ea043; }
.badge-yellow { background: #2d2200; color: #d29922; border: 1px solid #9e6a03; }
.badge-red    { background: #3d0a0a; color: #f85149; border: 1px solid #b91c1c; }
.badge-blue   { background: #1c2d3f; color: #79c0ff; border: 1px solid #1f6feb; }
.badge-gray   { background: #161b22; color: #8b949e; border: 1px solid #30363d; }

/* ── Glass panels ── */
.glass-panel {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.glass-panel-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 18px 0 !important; }

/* ── Info / warning / error override ── */
.stAlert {
    background: #161b22!important;
    border-radius: 8px !important;
    border-left-width: 3px !important;
}

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
.stButton button[kind="primary"] {
    background: #238636 !important;
    border: 1px solid #2ea043 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stDownloadButton button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    width: 100% !important;
    font-size: 0.82rem !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px !important; }

/* ── Remove Streamlit default branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #c9d1d9 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
DATA_RAW_DIR   = Path(__file__).resolve().parent / "data" / "raw"
MODELS_DIR     = Path(__file__).resolve().parent / "models"
REPORTS_DIR    = Path(__file__).resolve().parent / "reports"
SAMPLE_ECGS_DIR = Path(__file__).resolve().parent / "sample_ecgs"

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div class="brand-bar">
  <span class="brand-icon">🫀</span>
  <div>
    <div class="brand-title">ECG Guardian</div>
    <div class="brand-sub">Clinical Decision Support</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Hospital
all_hospitals = DB_MANAGER.list_hospitals()
hosp_map = {h["hospital_id"]: h["hospital_name"] for h in all_hospitals}
if not hosp_map:
    hosp_map = {"HOSP-APEX": "Apex Heart Hospital"}

if "active_hospital_id" not in st.session_state or st.session_state.active_hospital_id not in hosp_map:
    st.session_state.active_hospital_id = list(hosp_map.keys())[0]

st.sidebar.markdown('<div class="section-label">Facility</div>', unsafe_allow_html=True)
chosen_hospital_id = st.sidebar.selectbox(
    "Hospital", options=list(hosp_map.keys()),
    format_func=lambda x: hosp_map[x],
    index=list(hosp_map.keys()).index(st.session_state.active_hospital_id),
    label_visibility="collapsed",
)
st.session_state.active_hospital_id = chosen_hospital_id

# Clinician
st.sidebar.markdown('<div class="section-label">Clinician</div>', unsafe_allow_html=True)
all_users = AUTH_MANAGER.list_users()
user_map = {u.username: f"{u.full_name} · {u.role.value}" for u in all_users}
if "active_user_name" not in st.session_state:
    st.session_state.active_user_name = "cardiologist"

chosen_username = st.sidebar.selectbox(
    "Clinician", options=list(user_map.keys()),
    format_func=lambda x: user_map[x],
    index=list(user_map.keys()).index(st.session_state.active_user_name),
    label_visibility="collapsed",
)
st.session_state.active_user_name = chosen_username
current_user = AUTH_MANAGER.get_user_by_username(chosen_username)

st.sidebar.divider()

# View Mode
st.sidebar.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
view_mode = st.sidebar.radio(
    "View",
    ["🏥 Clinical Workspace", "📜 ECG History", "🧑‍💼 Patient Portal"],
    index=0, label_visibility="collapsed",
)

if view_mode in ["🧑‍💼 Patient Portal", "📜 ECG History"]:
    input_source_mode = view_mode
    uploaded_file = None
    selected_record = None
    sampling_rate_setting = 360
    analysis_duration_sec = 10
    start_offset_sec = 0.0
else:
    st.sidebar.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    input_source_mode = st.sidebar.radio(
        "Source",
        ["Upload ECG File", "MIT-BIH Demo"],
        index=0, label_visibility="collapsed",
    )

    available_records = get_available_records(DATA_RAW_DIR)
    if not available_records:
        available_records = ["100", "101", "106", "119", "200", "208", "213"]

    selected_record = None
    uploaded_file = None

    if input_source_mode == "Upload ECG File":
        # Patient Registration
        if "registered_patient_data" not in st.session_state:
            st.session_state["registered_patient_data"] = None
        if "sidebar_mrn" not in st.session_state:
            import secrets as _sec_init
            st.session_state["sidebar_mrn"] = f"MRN-{_sec_init.token_hex(3).upper()}"

        if st.session_state["registered_patient_data"] is None:
            st.sidebar.markdown('<div class="section-label">Patient</div>', unsafe_allow_html=True)
            import secrets as _sec
            with st.sidebar.form("patient_registration_form", clear_on_submit=False):
                _mrn  = st.text_input("MRN", value=st.session_state["sidebar_mrn"], key="form_mrn")
                _name = st.text_input("Full Name *", placeholder="e.g. Rajesh Kumar", key="form_name")
                _c1, _c2 = st.columns(2)
                with _c1:
                    _age = st.number_input("Age", 1, 120, 55, key="form_age")
                    _sex = st.selectbox("Sex", ["M", "F", "Other"], key="form_sex")
                with _c2:
                    _blood = st.selectbox("Blood Group", ["A+","A-","B+","B-","AB+","AB-","O+","O-","Unknown"], key="form_blood")
                    _smoke = st.selectbox("Smoking", ["Non-Smoker","Former","Current","Unknown"], key="form_smoke")
                _contact  = st.text_input("Contact", value="+91-", key="form_contact")
                _allergies = st.text_input("Allergies", placeholder="e.g. Penicillin", key="form_allergies")
                _conds    = st.text_input("Conditions", placeholder="e.g. Hypertension", key="form_conds")
                _cardiac  = st.text_input("Cardiac Hx", placeholder="e.g. Prior MI 2021", key="form_cardiac")
                _meds     = st.text_input("Medications", placeholder="e.g. Metoprolol 50mg, Amiodarone", key="form_meds")
                st.caption("🩺 Vital Signs & Labs (Optional)")
                _vc1, _vc2 = st.columns(2)
                with _vc1:
                    _sbp = st.number_input("Systolic BP", 60, 250, 120, key="form_sbp")
                    _spo2 = st.number_input("SpO2 (%)", 50, 100, 98, key="form_spo2")
                    _k_val = st.number_input("Potassium (K+)", 1.0, 9.0, 4.2, step=0.1, key="form_k")
                with _vc2:
                    _dbp = st.number_input("Diastolic BP", 30, 150, 80, key="form_dbp")
                    _temp_c = st.number_input("Temp (°C)", 30.0, 43.0, 36.8, step=0.1, key="form_temp")
                    _cr_val = st.number_input("Creatinine", 0.2, 10.0, 1.0, step=0.1, key="form_cr")

                _submitted = st.form_submit_button("Register & Continue →", type="primary", use_container_width=True)

            if _submitted:
                if not st.session_state.get("form_name", "").strip():
                    st.sidebar.error("Patient name is required.")
                else:
                    _p_id = f"PAT-{_sec.token_hex(4).upper()}"
                    try:
                        DB_MANAGER.create_patient(
                            patient_id=_p_id,
                            hospital_mrn=st.session_state["form_mrn"],
                            name=st.session_state["form_name"],
                            hospital_id=st.session_state.active_hospital_id,
                            age=int(st.session_state["form_age"]),
                            sex=st.session_state["form_sex"],
                            contact=st.session_state["form_contact"],
                            blood_group=st.session_state["form_blood"],
                            known_allergies=st.session_state["form_allergies"],
                            existing_conditions=st.session_state["form_conds"],
                            current_medications=st.session_state["form_meds"],
                            previous_cardiac_history=st.session_state["form_cardiac"],
                            smoking_status=st.session_state["form_smoke"],
                        )
                        AUDIT_LOGGER.log_event(
                            event_type="PATIENT_CREATED",
                            user_id=current_user.user_id,
                            username=current_user.username,
                            user_role=current_user.role.value,
                            action=f"Registered {st.session_state['form_name']} ({st.session_state['form_mrn']})",
                            patient_id=_p_id,
                        )
                    except Exception:
                        pass
                    st.session_state["registered_patient_data"] = {
                        "patient_id": _p_id,
                        "hospital_mrn": st.session_state["form_mrn"],
                        "name": st.session_state["form_name"],
                        "age": int(st.session_state["form_age"]),
                        "sex": st.session_state["form_sex"],
                        "blood_group": st.session_state["form_blood"],
                        "contact": st.session_state["form_contact"],
                        "known_allergies": st.session_state["form_allergies"] or "None",
                        "existing_conditions": st.session_state["form_conds"] or "None",
                        "previous_cardiac_history": st.session_state["form_cardiac"] or "None",
                        "current_medications": st.session_state["form_meds"] or "None",
                        "smoking_status": st.session_state["form_smoke"],
                    }
                    st.session_state["registered_patient_vitals"] = {
                        "systolic_bp": int(st.session_state["form_sbp"]),
                        "diastolic_bp": int(st.session_state["form_dbp"]),
                        "spo2": int(st.session_state["form_spo2"]),
                        "temperature_c": float(st.session_state["form_temp"]),
                    }
                    st.session_state["registered_patient_labs"] = {
                        "potassium": float(st.session_state["form_k"]),
                        "creatinine": float(st.session_state["form_cr"]),
                    }
                    st.rerun()

            uploaded_file = None
            sampling_rate_setting = 360
            analysis_duration_sec = 10
            start_offset_sec = 0.0

        else:
            _pat = st.session_state["registered_patient_data"]
            st.sidebar.markdown(f"""
<div style="background:#1a3a2a;border:1px solid #2ea043;border-radius:8px;padding:10px 12px;margin:8px 0;">
  <div style="font-size:0.85rem;font-weight:700;color:#3fb950;">{_pat['name']}</div>
  <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
    <code style="color:#58a6ff;">{_pat['hospital_mrn']}</code> &nbsp;·&nbsp; {_pat['age']}y / {_pat['sex']}
  </div>
</div>
""", unsafe_allow_html=True)
            if st.sidebar.button("↩ Change Patient", key="btn_change_patient"):
                st.session_state["registered_patient_data"] = None
                import secrets as _sec_r
                st.session_state["sidebar_mrn"] = f"MRN-{_sec_r.token_hex(3).upper()}"
                st.rerun()

            st.sidebar.markdown('<div class="section-label">ECG File</div>', unsafe_allow_html=True)
            uploaded_file = st.sidebar.file_uploader(
                "Upload ECG",
                type=["pdf","jpg","jpeg","png","bmp","tiff","csv","txt","npy"],
                label_visibility="collapsed",
            )
            sampling_rate_setting = st.sidebar.selectbox(
                "Sampling Rate (Hz)", options=STANDARD_SAMPLING_RATES, index=2,
            )
            analysis_duration_sec = st.sidebar.slider("Duration (s)", 3, 30, 10, 1)
            start_offset_sec = 0.0

    else:
        # MIT-BIH Demo
        record_descriptions = {
            "100": "Record 100 — Normal Sinus",
            "101": "Record 101 — Normal Baseline",
            "106": "Record 106 — Frequent PVCs",
            "119": "Record 119 — High-Freq PVCs",
            "200": "Record 200 — Ventricular Tach",
            "208": "Record 208 — Fusion Beats",
            "213": "Record 213 — Mixed Ectopy",
        }
        record_options = [r for r in available_records if r in record_descriptions] or available_records
        selected_record = st.sidebar.selectbox(
            "Record", options=record_options,
            format_func=lambda x: record_descriptions.get(x, f"Record {x}"),
        )
        sampling_rate_setting = 360
        start_offset_sec = st.sidebar.slider("Start Offset (s)", 0, 120, 0, 1)
        analysis_duration_sec = st.sidebar.slider("Duration (s)", 2, 30, 10, 1)

st.sidebar.divider()
_sb_active = is_supabase_configured()
_sb_icon = "🟢" if _sb_active else "🟡"
_sb_label = "Supabase Cloud Connected" if _sb_active else "Local Storage (Offline Fallback)"
st.sidebar.markdown(f"""
<div style="font-size:0.75rem;color:#8b949e;padding:2px 0 6px 0;">
  <span>{_sb_icon}</span> <span style="color:#c9d1d9;">{_sb_label}</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("⚠️ Research & educational use only. Not a certified diagnostic device.")

# ═══════════════════════════════════════════════════════════════
# ECG HISTORY VIEW
# ═══════════════════════════════════════════════════════════════
if view_mode == "📜 ECG History":
    render_history_view(current_user=current_user)
    st.stop()

# ═══════════════════════════════════════════════════════════════
# PATIENT PORTAL VIEW
# ═══════════════════════════════════════════════════════════════
if view_mode == "🧑‍💼 Patient Portal":
    st.markdown("""
<div style="margin-bottom:20px;">
  <div style="font-size:1.4rem;font-weight:700;color:#f0f6fc;">Patient Health Portal</div>
  <div style="font-size:0.82rem;color:#8b949e;margin-top:2px;">Signed ECG evaluations & physician directives</div>
</div>
""", unsafe_allow_html=True)

    enrolled_patients = DB_MANAGER.list_patients()
    if enrolled_patients:
        p_options = {p.hospital_mrn: f"{p.name} ({p.hospital_mrn})" for p in enrolled_patients}
        chosen_mrn = st.selectbox("Patient", options=list(p_options.keys()), format_func=lambda x: p_options[x], key="portal_pat_sel")
        curr_patient = DB_MANAGER.get_patient_by_mrn(chosen_mrn)
    else:
        curr_patient = PatientRecord(
            patient_id="PT-DEMO-001", hospital_mrn="MRN-DEMO-101", name="Ramesh Sharma",
            age=58, sex="M", contact="+91-9876543210", blood_group="B+",
            known_allergies="None", existing_conditions="Mild Hypertension",
            current_medications="Metoprolol 25mg", created_at=datetime.now().isoformat(),
        )

    st.markdown(f"""
<div class="patient-strip">
  <span class="patient-name">👤 {curr_patient.name}</span>
  <span class="patient-mrn">MRN: {curr_patient.hospital_mrn}</span>
  <span class="patient-meta">{curr_patient.age or '—'}y / {curr_patient.sex or '—'} &nbsp;·&nbsp; {curr_patient.blood_group or '—'}</span>
  <span class="patient-meta">Allergies: {curr_patient.known_allergies or 'None'}</span>
</div>
""", unsafe_allow_html=True)

    # Query persistent historical reports from Supabase/SQLite
    pat_historical_reports = REPORT_PERSISTENCE_SERVICE.get_patient_timeline(curr_patient.hospital_mrn)
    if not pat_historical_reports and curr_patient.name:
        pat_historical_reports = REPORT_PERSISTENCE_SERVICE.get_patient_timeline(curr_patient.name)

    if pat_historical_reports:
        st.markdown(f'<div class="section-label">Permanent ECG Reports ({len(pat_historical_reports)})</div>', unsafe_allow_html=True)
        for h_rep in pat_historical_reports:
            h_rep_id = h_rep.get("report_id")
            h_rep_num = h_rep.get("report_number") or f"ECG-REP-{h_rep_id[:6].upper()}"
            h_stat = h_rep.get("status", "DRAFT")
            h_is_signed = h_stat in ["SIGNED", "SEALED"]
            h_date = (h_rep.get("generated_at") or "")[:10]
            h_finding = h_rep.get("primary_prediction") or "Normal Rhythm"

            col_p1, col_p2, col_p3 = st.columns([3, 1, 1.2])
            with col_p1:
                h_badge = '<span class="badge badge-green">✓ SIGNED</span>' if h_is_signed else '<span class="badge badge-yellow">⏳ PENDING</span>'
                st.markdown(f"""
<div class="glass-panel" style="margin-bottom:8px;padding:12px 16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:monospace;font-weight:700;color:#58a6ff;font-size:0.88rem;">{h_rep_num}</span>
    {h_badge}
  </div>
  <div style="margin-top:4px;font-size:0.84rem;color:#e6edf3;">
    Finding: <b>{h_finding}</b> &nbsp;·&nbsp; <span style="color:#8b949e;font-size:0.75rem;">Quality: {h_rep.get('signal_quality','GOOD')}</span>
  </div>
</div>
""", unsafe_allow_html=True)
            with col_p2:
                st.metric("Recorded", h_date)
            with col_p3:
                pdf_data = REPORT_PERSISTENCE_SERVICE.get_report_pdf(h_rep_id, report_type="patient")
                if pdf_data:
                    st.download_button(
                        "⬇ Patient Summary",
                        data=pdf_data,
                        file_name=f"{h_rep_num}_patient_summary.pdf",
                        mime="application/pdf",
                        key=f"portal_dl_{h_rep_id}",
                    )
                else:
                    st.caption("PDF generating…")

    elif patient_records:
        st.markdown('<div class="section-label">ECG Recordings</div>', unsafe_allow_html=True)
        for rec in patient_records:
            rec_id = rec["record_id"]
            review = DB_MANAGER.get_clinician_review(rec_id) or st.session_state.get(f"review_{rec_id}")
            is_signed = (rec.get("workflow_status") == "SIGNED") or (review and review.get("agreement_status") in ["CONFIRMED", "MODIFIED"])
            hr_bpm = rec.get("heart_rate_bpm") or 72.0

            with st.container():
                col_r1, col_r2, col_r3 = st.columns([3, 1, 1])
                with col_r1:
                    status_html = '<span class="badge badge-green">✓ SIGNED</span>' if is_signed else '<span class="badge badge-yellow">⏳ PENDING</span>'
                    _doc_interp = f'<div style="font-size:0.85rem;color:#e6edf3;">📋 {review.get("clinician_interpretation","")}</div>' if is_signed and review else '<div style="font-size:0.82rem;color:#8b949e;">Awaiting physician review</div>'
                    st.markdown(
                        f'<div class="glass-panel" style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                        f'<span style="font-size:0.8rem;color:#8b949e;font-family:monospace;">{rec_id}</span>'
                        f'{status_html}'
                        f'</div>'
                        f'{_doc_interp}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_r2:
                    st.metric("Heart Rate", f"{hr_bpm:.0f} BPM")
                with col_r3:
                    if is_signed and review:
                        pat_rep = generate_structured_report(
                            input_info={"file_name": f"{rec_id}.csv","modality":"DIGITAL_SIGNAL","sampling_rate":rec["sampling_rate"],"duration_sec":rec["duration_sec"],"lead":"Lead II"},
                            ai_results={"predicted_class":rec.get("prediction","Normal Rhythm"),"signal_quality":rec.get("signal_quality","GOOD"),"quality_score":0.95,"heart_rate_bpm":hr_bpm,"quality_indicators":{"snr_db":22.0}},
                            clinician_review=review,
                        )
                        st.download_button("⬇ Download Report", data=generate_patient_report(pat_rep),
                            file_name=f"ecg_{curr_patient.name.replace(' ','_')}_{rec_id}.pdf",
                            mime="application/pdf", key=f"btn_dl_p_{rec_id}")
    else:
        st.info("No ECG records on file. Records appear after clinic upload and physician sign-off.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# STATE VARIABLES
# ═══════════════════════════════════════════════════════════════
signal_data: Optional[np.ndarray] = None
sampling_rate: float = float(sampling_rate_setting)
reference_annotations = None
input_info: Dict[str, Any] = {}
extracted_measurements = None
waveform_status: Dict[str, Any] = {"is_extracted": False, "message": ""}
ai_results = None


# ─────────────────────────────────────────
# Helper: Worklist & Audit (collapsed panels)
# ─────────────────────────────────────────
def render_hospital_worklist(expanded: bool = False):
    with st.expander("🏥 Patient Worklist & Directory", expanded=expanded):
        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            status_sel = st.selectbox("Filter", ["ALL","UPLOADED","ANALYZED","SIGNED","REJECTED"], key=f"sel_wk_{expanded}")
            records_list = DB_MANAGER.list_ecg_records(
                limit=25,
                status_filter=None if status_sel == "ALL" else status_sel,
                hospital_id=st.session_state.get("active_hospital_id"),
            )
            if records_list:
                st.dataframe(pd.DataFrame([{
                    "Record": r["record_id"], "Patient": r.get("patient_name","—"),
                    "Status": r.get("workflow_status","UPLOADED"), "AI": r.get("prediction","—"),
                    "Quality": r.get("signal_quality","—"), "Time": r["uploaded_at"][:16].replace("T"," "),
                } for r in records_list]), use_container_width=True)
            else:
                st.caption("No records found.")
            st.caption("**Enrolled Patients**")
            p_list = DB_MANAGER.list_patients(limit=10, hospital_id=st.session_state.get("active_hospital_id"))
            if p_list:
                st.dataframe(pd.DataFrame([{
                    "MRN": p.hospital_mrn, "Name": p.name,
                    "Age/Sex": f"{p.age or '—'}/{p.sex or '—'}", "Blood": p.blood_group or "—",
                } for p in p_list]), use_container_width=True)

        with col_p2:
            st.caption("**Register New Patient**")
            new_mrn  = st.text_input("MRN", value=f"MRN-{secrets.token_hex(3).upper()}", key=f"inp_mrn_{expanded}")
            new_name = st.text_input("Name", key=f"inp_name_{expanded}")
            c_a, c_b = st.columns(2)
            with c_a:
                new_age  = st.number_input("Age", 1, 120, 55, key=f"inp_age_{expanded}")
                new_sex  = st.selectbox("Sex", ["M","F","Other"], key=f"inp_sex_{expanded}")
            with c_b:
                new_blood   = st.selectbox("Blood", ["A+","A-","B+","B-","AB+","AB-","O+","O-","Unknown"], key=f"inp_blood_{expanded}")
                new_contact = st.text_input("Contact", value="+91-", key=f"inp_contact_{expanded}")
            new_meds = st.text_input("Medications", key=f"inp_meds_{expanded}")
            if st.button("Enroll Patient", key=f"btn_reg_{expanded}", type="primary"):
                if new_name.strip():
                    p_id = f"PAT-{secrets.token_hex(4).upper()}"
                    DB_MANAGER.create_patient(
                        patient_id=p_id, hospital_mrn=new_mrn, name=new_name,
                        hospital_id=st.session_state.get("active_hospital_id","HOSP-APEX"),
                        age=int(new_age), sex=new_sex, contact=new_contact, blood_group=new_blood,
                        current_medications=new_meds,
                    )
                    st.success(f"Enrolled: {new_name}")
                    st.rerun()
                else:
                    st.warning("Name is required.")


def render_audit_trail(expanded: bool = False):
    with st.expander("🔐 Audit Trail", expanded=expanded):
        if st.button("Verify Chain Integrity", key=f"btn_verify_{expanded}"):
            is_valid, issues = AUDIT_LOGGER.verify_chain_integrity()
            if is_valid:
                st.success("✓ Hash chain valid — no tampering detected.")
            else:
                st.error(f"Chain broken: {issues}")
        logs = AUDIT_LOGGER.get_logs(limit=20)
        if logs:
            st.dataframe(pd.DataFrame([{
                "#": l["sequence_id"],
                "Time": l["timestamp"][:19].replace("T"," "),
                "Event": l["event_type"],
                "User": l["username"],
                "Action": l["action"][:60],
                "Hash": l["entry_hash"][:12]+"…",
            } for l in logs]), use_container_width=True)
        else:
            st.caption("No audit events yet.")


# ═══════════════════════════════════════════════════════════════
# LOADING SCREEN — No File Yet
# ═══════════════════════════════════════════════════════════════
if input_source_mode == "Upload ECG File":
    if uploaded_file is None:
        # Hero
        st.markdown("""
<div style="margin-bottom:24px;">
  <div style="font-size:1.5rem;font-weight:700;color:#f0f6fc;letter-spacing:-0.02em;">Clinical ECG Analysis</div>
  <div style="font-size:0.85rem;color:#8b949e;margin-top:4px;">Upload an ECG file to begin AI-assisted interpretation</div>
</div>
""", unsafe_allow_html=True)

        # Upload format cards
        col_f1, col_f2, col_f3 = st.columns(3)
        cards = [
            ("📄", "PDF Report", "Extracts machine measurements, printed interpretations & embedded rhythm strips"),
            ("📷", "Scanned Image", "Grid-removal color isolation, trace extraction & signal validation"),
            ("📊", "Digital Signal", "CSV / TXT / NPY — full 28-feature AI classification pipeline"),
        ]
        for col, (icon, title, desc) in zip([col_f1, col_f2, col_f3], cards):
            col.markdown(f"""
<div class="glass-panel" style="text-align:center;min-height:110px;">
  <div style="font-size:1.8rem;margin-bottom:6px;">{icon}</div>
  <div style="font-weight:600;font-size:0.9rem;color:#f0f6fc;margin-bottom:4px;">{title}</div>
  <div style="font-size:0.76rem;color:#8b949e;line-height:1.5;">{desc}</div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        recent_reps = REPORT_PERSISTENCE_SERVICE.list_reports(limit=3)
        if recent_reps:
            st.markdown('<div class="section-label">Recent Persistent Reports</div>', unsafe_allow_html=True)
            r_cols = st.columns(len(recent_reps))
            for i, r_item in enumerate(recent_reps):
                with r_cols[i]:
                    r_stat = r_item.get("status", "DRAFT")
                    r_badge = '<span class="badge badge-green">✓ SIGNED</span>' if r_stat in ["SIGNED","SEALED"] else '<span class="badge badge-yellow">⏳ DRAFT</span>'
                    st.markdown(f"""
<div class="glass-panel" style="padding:12px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-family:monospace;font-size:0.8rem;color:#58a6ff;font-weight:700;">{r_item.get('report_number')}</span>
    {r_badge}
  </div>
  <div style="font-size:0.82rem;font-weight:600;color:#f0f6fc;">{r_item.get('patient_name','Anonymous')}</div>
  <div style="font-size:0.75rem;color:#8b949e;margin-top:2px;">{r_item.get('primary_prediction','Normal Rhythm')}</div>
</div>
""", unsafe_allow_html=True)

        render_hospital_worklist(expanded=True)
        st.divider()
        render_audit_trail(expanded=False)
        st.stop()

    # ── Process uploaded file ──
    file_name  = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    modality, ext = detect_input_modality(file_name, file_bytes=file_bytes)

    _prog = st.progress(0, text="Detecting file format…")

    input_info = {
        "file_name": file_name, "file_modality": modality.value,
        "format": ext, "sampling_rate": sampling_rate,
        "duration_sec": analysis_duration_sec, "lead": "Lead II",
    }

    if modality == InputModality.DIGITAL_SIGNAL:
        _prog.progress(30, text="Loading digital signal…")
        sig_res = load_digital_signal(uploaded_file, sampling_rate_hint=sampling_rate, max_duration_sec=analysis_duration_sec)
        if not sig_res["success"]:
            _prog.empty()
            st.error(f"Could not parse signal: {sig_res['message']}")
            st.stop()
        signal_data  = sig_res["signal"]
        sampling_rate = float(sig_res["sampling_rate"])
        input_info["sampling_rate"] = sampling_rate
        input_info["total_samples"] = len(signal_data)
        input_info["duration_sec"]  = round(len(signal_data) / sampling_rate, 2)
        waveform_status = {"is_extracted": False, "message": "Direct digital signal."}

    elif modality == InputModality.REPORT_PDF:
        _prog.progress(30, text="Parsing PDF…")
        pdf_res = process_pdf_report(io.BytesIO(file_bytes))
        extracted_measurements = pdf_res.get("measurements", {})
        if pdf_res.get("has_embedded_images") and pdf_res.get("images"):
            _prog.progress(50, text="Extracting waveform from PDF image…")
            img_res = process_ecg_image(pdf_res["images"][0])
            if img_res["is_ecg"]:
                wf_res = extract_waveform_from_image(img_res["cv_image"], target_fs=360.0)
                if wf_res["success"] and wf_res["signal"] is not None:
                    is_valid, v_msg = validate_extracted_signal(wf_res["signal"], fs=wf_res["sampling_rate"], confidence_score=wf_res["confidence_score"])
                    if is_valid:
                        signal_data   = wf_res["signal"]
                        sampling_rate = wf_res["sampling_rate"]
                        waveform_status = {"is_extracted": True, "message": "Extracted from PDF strip."}
                    else:
                        waveform_status = {"is_extracted": False, "message": v_msg}
                else:
                    waveform_status = {"is_extracted": False, "message": wf_res.get("message", "Extraction failed.")}
        else:
            waveform_status = {"is_extracted": False, "message": "PDF contains printed text only — no raw waveform strip."}

    elif modality == InputModality.REPORT_IMAGE:
        _prog.progress(30, text="Preprocessing image…")
        img_res = process_ecg_image(io.BytesIO(file_bytes))
        if not img_res["is_ecg"]:
            _prog.empty()
            st.error(f"Image error: {img_res['status_message']}")
            st.stop()
        _prog.progress(55, text="Extracting waveform trace…")
        wf_res = extract_waveform_from_image(img_res["cv_image"], target_fs=360.0)
        if wf_res["success"] and wf_res["signal"] is not None:
            is_valid, v_msg = validate_extracted_signal(wf_res["signal"], fs=wf_res["sampling_rate"], confidence_score=wf_res["confidence_score"])
            if is_valid:
                signal_data   = wf_res["signal"]
                sampling_rate = wf_res["sampling_rate"]
                waveform_status = {"is_extracted": True, "confidence": wf_res["confidence_score"], "message": "Waveform extracted."}
            else:
                waveform_status = {"is_extracted": False, "message": v_msg}
        else:
            waveform_status = {"is_extracted": False, "message": wf_res.get("message", "Extraction failed.")}
    else:
        _prog.empty()
        st.error(f"Unsupported format: `{ext}`")
        st.stop()

    if signal_data is not None and len(signal_data) > 0:
        _prog.progress(80, text="Running AI classification…")
        try:
            ai_results = predict_ecg(signal_data, fs=sampling_rate, models_dir=MODELS_DIR)
        except Exception as exc:
            st.warning(f"AI inference: {exc}")

    _prog.progress(100, text="Complete")
    time.sleep(0.15)
    _prog.empty()

else:
    # MIT-BIH Demo
    try:
        raw_full, fs_rec, total_samples = load_record(selected_record, DATA_RAW_DIR)
        start_samp = int(start_offset_sec * fs_rec)
        end_samp   = min(len(raw_full), int((start_offset_sec + analysis_duration_sec) * fs_rec))
        signal_data   = raw_full[start_samp:end_samp]
        sampling_rate = float(fs_rec)

        ann_df = load_annotations(selected_record, DATA_RAW_DIR)
        ref_mask = (ann_df["sample_index"] >= start_samp) & (ann_df["sample_index"] < end_samp)
        reference_annotations = ann_df[ref_mask].copy()
        reference_annotations["sample_relative"] = reference_annotations["sample_index"] - start_samp
        reference_annotations["time_sec"] = reference_annotations["sample_relative"] / fs_rec

        input_info = {
            "file_name": f"MIT-BIH #{selected_record}", "file_modality": "BENCHMARK_DEMO",
            "format": "wfdb", "sampling_rate": sampling_rate,
            "duration_sec": analysis_duration_sec, "total_samples": len(signal_data), "lead": "MLII",
        }
        waveform_status = {"is_extracted": False, "message": "MIT-BIH Benchmark"}

        with st.spinner("Analysing benchmark signal…"):
            ai_results = predict_ecg(signal_data, fs=sampling_rate, models_dir=MODELS_DIR)
    except Exception as exc:
        st.error(f"Error loading record {selected_record}: {exc}")
        st.stop()


# ═══════════════════════════════════════════════════════════════
# COMPILE REPORT
# ═══════════════════════════════════════════════════════════════
file_stem   = input_info.get("file_name", "ecg_file")
rec_id      = f"REC-{abs(hash(file_stem)) % 1000000:06d}"
active_review = st.session_state.get(f"review_{rec_id}")

primary_finding_str = ai_results["predicted_class"] if ai_results and "predicted_class" in ai_results else "Normal Sinus Rhythm"
cds_rec_obj = GLOBAL_CDS_ENGINE.evaluate_finding(ecg_finding=primary_finding_str, heart_rate=ai_results.get("heart_rate_bpm") if ai_results else None)
cds_data_dict = {
    "primary_finding": cds_rec_obj.finding, "urgency": cds_rec_obj.urgency,
    "summary": cds_rec_obj.clinician_action_required,
    "guideline_citations": cds_rec_obj.relevant_guidelines,
    "considerations": cds_rec_obj.clinical_considerations,
    "contraindications": cds_rec_obj.contraindication_warnings,
    "medication_recommendations": cds_rec_obj.medication_recommendations,
}


_reg_pat = st.session_state.get("registered_patient_data")
temp_eval_patient = PatientRecord(
    patient_id=_reg_pat.get("patient_id","PAT-ACTIVE-01") if _reg_pat else "PAT-ACTIVE-01",
    hospital_mrn=_reg_pat.get("hospital_mrn","MRN-ACTIVE") if _reg_pat else "MRN-ACTIVE",
    name=_reg_pat.get("name","Active Patient") if _reg_pat else "Active Patient",
    age=_reg_pat.get("age",65) if _reg_pat else 65,
    sex=_reg_pat.get("sex","M") if _reg_pat else "M",
    known_allergies=_reg_pat.get("known_allergies","None") if _reg_pat else "None",
    existing_conditions=_reg_pat.get("existing_conditions","Hypertension") if _reg_pat else "Hypertension",
    current_medications=_reg_pat.get("current_medications","Metoprolol") if _reg_pat else "Metoprolol",
)
_curr_meds_list = [m.strip() for m in ((_reg_pat or {}).get("current_medications") or "").split(",") if m.strip()] or ["Metoprolol", "Amiodarone"]
_vitals = st.session_state.get("registered_patient_vitals")
_labs = st.session_state.get("registered_patient_labs")

med_safety_eval = check_medication_safety(
    _curr_meds_list,
    patient=temp_eval_patient,
    ecg_finding=primary_finding_str,
    ecg_measurements={
        "heart_rate": ai_results.get("heart_rate_bpm") if ai_results else None,
        "qtc_ms": (extracted_measurements or {}).get("qtc_interval_ms"),
    },
    vital_signs=_vitals,
    laboratory_results=_labs,
)

report_data = generate_structured_report(
    input_info=input_info,
    ai_results=ai_results,
    extracted_measurements=extracted_measurements,
    waveform_status=waveform_status,
    clinician_review=active_review,
    cds_report=cds_data_dict,
    medication_safety=med_safety_eval.to_dict(),
    patient_profile=_reg_pat,
    vital_signs=_vitals,
    laboratory_results=_labs,
)
if _reg_pat:
    report_data["patient_info"].update({
        "patient_name": _reg_pat.get("name",""),
        "patient_age":  _reg_pat.get("age",""),
        "patient_sex":  _reg_pat.get("sex",""),
        "hospital_mrn": _reg_pat.get("hospital_mrn",""),
        "blood_group":  _reg_pat.get("blood_group",""),
        "known_allergies":   _reg_pat.get("known_allergies","None"),
        "existing_conditions": _reg_pat.get("existing_conditions","None"),
        "current_medications": _reg_pat.get("current_medications","None"),
        "previous_cardiac_history": _reg_pat.get("previous_cardiac_history","None"),
        "smoking_status": _reg_pat.get("smoking_status",""),
        "contact": _reg_pat.get("contact",""),
    })

if ai_results and "predicted_class" in ai_results:
    try:
        DB_MANAGER.save_ecg_record(
            record_id=rec_id, sampling_rate=sampling_rate, lead_names=["II"],
            duration_sec=float(input_info.get("duration_sec",10.0)),
            file_hash=f"{abs(hash(str(ai_results.get('processed_signal','')))):x}",
            source_format=input_info.get("modality","DIGITAL"),
            signal_quality=ai_results.get("signal_quality","GOOD"),
            quality_score=ai_results.get("quality_score",1.0),
            uploaded_by=current_user.username,
        )
        DB_MANAGER.save_analysis_result(
            analysis_id=f"ANL-{rec_id[4:]}",
            record_id=rec_id, model_id="ECG-RF-1.0.0", model_version="1.0.0",
            prediction=ai_results["predicted_class"],
            probabilities=ai_results.get("probabilities",{}),
            signal_quality=ai_results.get("signal_quality","GOOD"),
            quality_score=ai_results.get("quality_score",1.0),
            heart_rate_bpm=ai_results.get("heart_rate_bpm"),
            mean_rr_ms=ai_results.get("mean_rr_sec",0.8)*1000.0 if ai_results.get("mean_rr_sec") else None,
            detected_beats_count=ai_results.get("beat_count",0),
        )
    except Exception:
        pass

waveform_for_pdf = ai_results["processed_signal"] if ai_results else None
peaks_for_pdf    = np.array(ai_results["detected_peaks"]) if ai_results else None

# ── Auto-persist report snapshot (Idempotent Dual-Store) ──
try:
    _doc_pdf_bytes_auto = generate_doctor_report(
        report_data=report_data,
        waveform=waveform_for_pdf,
        fs=sampling_rate,
        r_peaks=peaks_for_pdf,
    )
    _p_res = REPORT_PERSISTENCE_SERVICE.save_report_snapshot(
        report_data=report_data,
        pdf_bytes=_doc_pdf_bytes_auto,
        patient_id=(_disp_pat.get("hospital_mrn") if "_disp_pat" in locals() else None) or (_reg_pat.get("hospital_mrn") if _reg_pat else "PT-DEMO-001"),
        ecg_id=rec_id,
        analysis_id=f"ANL-{rec_id[4:]}",
        user_id=current_user.user_id,
        status="SIGNED" if active_review else "DRAFT",
    )
    st.session_state[f"persisted_meta_{rec_id}"] = _p_res
except Exception as p_err:
    st.session_state[f"persisted_meta_{rec_id}"] = {"status": "SAVE_ERROR", "error": str(p_err)}


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT — Results
# ═══════════════════════════════════════════════════════════════

# ── Page title & Permanent Report ID Banner ──
_disp_pat = _reg_pat or report_data.get("patient_info", {})
_pat_name = _disp_pat.get("name") or _disp_pat.get("patient_name", "")

_p_meta = st.session_state.get(f"persisted_meta_{rec_id}", {})
_p_rep_num = _p_meta.get("report_number") or f"ECG-2026-{rec_id[:6].upper()}"
_p_is_sb = is_supabase_configured()
_p_storage_desc = "Supabase Storage Bucket ('ecg-reports')" if _p_is_sb else "Local Verified Dual Storage"

st.markdown(f"""
<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:8px 14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.06em;">Archived Report ID:</span>
    <span style="font-family:monospace;font-weight:700;color:#58a6ff;font-size:0.9rem;">{_p_rep_num}</span>
    <span class="badge badge-green">✓ PERSISTED</span>
  </div>
  <div style="font-size:0.75rem;color:#8b949e;">
    Storage: <span style="color:#79c0ff;">{_p_storage_desc}</span> &nbsp;·&nbsp;
    Snapshot: <span style="color:#3fb950;">Permanent & Queryable</span>
  </div>
</div>
""", unsafe_allow_html=True)

if _pat_name:
    st.markdown(f"""
<div class="patient-strip">
  <span class="patient-name">👤 {_pat_name}</span>
  <span class="patient-mrn">MRN: {_disp_pat.get('hospital_mrn') or '—'}</span>
  <span class="patient-meta">{_disp_pat.get('age') or _disp_pat.get('patient_age','—')}y / {_disp_pat.get('sex') or _disp_pat.get('patient_sex','—')}</span>
  <span class="patient-meta">Blood: {_disp_pat.get('blood_group','—')}</span>
  <span class="patient-meta">Allergies: {_disp_pat.get('known_allergies','None')}</span>
  <span class="patient-meta">Meds: {_disp_pat.get('current_medications','None')}</span>
</div>
""", unsafe_allow_html=True)
    crit_pat = get_patient_context_criteria(
        patient_profile=_disp_pat,
        vital_signs=_vitals,
        laboratory_results=_labs,
    )
    render_criteria_footer(crit_pat, key_suffix="patient_banner", show_details=False)
else:
    st.markdown("""
<div style="margin-bottom:18px;">
  <div style="font-size:1.4rem;font-weight:700;color:#f0f6fc;letter-spacing:-0.02em;">Analysis Results</div>
  <div style="font-size:0.8rem;color:#8b949e;margin-top:2px;">ECG Guardian · Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────
c_params = report_data["cardiac_parameters"]
c_quality = report_data["signal_quality"]

# Build 5 metrics
def _metric(label: str, value: str, delta: str = "", delta_class: str = "delta-off") -> str:
    delta_html = f'<div class="metric-card-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
<div class="metric-card">
  <div class="metric-card-label">{label}</div>
  <div class="metric-card-value">{value}</div>
  {delta_html}
</div>"""

# 1. Pattern
if ai_results and "predicted_class" in ai_results:
    _pred = ai_results["predicted_class"].split("(")[0].strip()
    _is_abn = "PVC" in _pred or "Other" in _pred
    m1 = _metric("AI Pattern", _pred, "Abnormal" if _is_abn else "Normal", "delta-bad" if _is_abn else "delta-good")
elif extracted_measurements and extracted_measurements.get("machine_interpretation"):
    _first = extracted_measurements["machine_interpretation"][0]
    m1 = _metric("Printed Dx", _first[:20]+"…" if len(_first)>20 else _first, "Machine Interp", "delta-off")
else:
    m1 = _metric("Pattern", "N/A", "", "delta-off")

# 2. Heart Rate
_hr = c_params.get("heart_rate_bpm")
if _hr:
    _hr_cat = c_params.get("heart_rate_category","")
    _hr_cls = "delta-bad" if "Brady" in _hr_cat or "Tachy" in _hr_cat else "delta-good"
    m2 = _metric("Heart Rate", f"{_hr:.0f} BPM", _hr_cat, _hr_cls)
else:
    m2 = _metric("Heart Rate", "N/A")

# 3. Signal Quality
_sq  = c_quality.get("category","UNKNOWN")
_sqs = c_quality.get("quality_score")
_sq_cls = "delta-good" if _sq=="GOOD" else ("delta-warn" if _sq=="ACCEPTABLE" else "delta-bad")
m3 = _metric("Signal Quality", _sq, f"{_sqs:.2f}" if _sqs else "", _sq_cls)

# 4. Detected Beats
_beats = c_params.get("detected_beats")
m4 = _metric("Beats Detected", f"{_beats} cycles" if _beats else "N/A", "", "delta-off")

# 5. Mean R-R
_rr = c_params.get("mean_rr_ms")
_qrs = c_params.get("qrs_duration_ms")
if _rr:
    m5 = _metric("Mean R-R", f"{_rr:.0f} ms", "", "delta-off")
elif _qrs:
    m5 = _metric("QRS Duration", f"{_qrs:.0f} ms", "", "delta-off")
else:
    m5 = _metric("Interval", "N/A", "", "delta-off")

st.markdown(f"""
<div class="metric-row">{m1}{m2}{m3}{m4}{m5}</div>
""", unsafe_allow_html=True)

crit_metrics = CriteriaDescriptor(
    label="Criteria used",
    criteria=["Pan-Tompkins R-peak detector", "Instantaneous R-R intervals", "Signal quality gatekeeper (SNR / noise harmonics)", "QRS duration / morphology"],
    chart_id="primary_metrics",
)
render_criteria_footer(crit_metrics, key_suffix="primary_metrics", show_details=False)

# ─────────────────────────────────────────
# PRINTED MACHINE MEASUREMENTS (PDF / Image)
# ─────────────────────────────────────────
if extracted_measurements and extracted_measurements.get("has_extracted_data"):
    with st.expander("📋 Machine Measurements (Source Document)", expanded=True):
        em = extracted_measurements
        mc1, mc2 = st.columns(2)
        with mc1:
            st.caption("Patient & Recording")
            st.table(pd.DataFrame([
                ("Name", em.get("patient_name") or "—"),
                ("Age/Sex", f"{em.get('patient_age','—')}y / {em.get('patient_sex','—')}"),
                ("Date", em.get("recording_date") or "—"),
                ("Vent. Rate", f"{em.get('heart_rate_printed','—')} BPM"),
            ], columns=["Parameter","Value"]))
        with mc2:
            st.caption("Intervals & Axes")
            st.table(pd.DataFrame([
                ("PR", f"{em.get('pr_interval_ms','—')} ms"),
                ("QRS", f"{em.get('qrs_duration_ms','—')} ms"),
                ("QT/QTc", f"{em.get('qt_interval_ms','—')} / {em.get('qtc_interval_ms','—')} ms"),
                ("P·QRS·T Axes", f"{em.get('p_axis_deg','—')}° / {em.get('qrs_axis_deg','—')}° / {em.get('t_axis_deg','—')}°"),
            ], columns=["Interval","Value"]))
        if em.get("machine_interpretation"):
            st.caption("Printed Findings")
            for interp in em["machine_interpretation"]:
                st.markdown(f"- `{interp}`")
        crit_mach = CriteriaDescriptor(
            label="Criteria extracted",
            criteria=["Printed ECG document / report", "OCR text extraction", "Ventricular rate", "Intervals (PR, QRS, QT/QTc)", "Axes (P, QRS, T)"],
            chart_id="machine_measurements",
        )
        render_criteria_footer(crit_mach, key_suffix="machine_measurements", show_details=False)

# ─────────────────────────────────────────
# WAVEFORM VISUALIZATION
# ─────────────────────────────────────────
if signal_data is not None and len(signal_data) > 0 and ai_results is not None:
    st.markdown('<div class="section-label">ECG Waveform</div>', unsafe_allow_html=True)
    show_raw = st.checkbox("Overlay raw signal", value=False)
    fig_waveform = plot_ecg_signal(
        signal=ai_results["processed_signal"], fs=sampling_rate,
        r_peaks=np.array(ai_results["detected_peaks"]),
        raw_signal=ai_results["raw_signal"] if show_raw else None,
        title=f"ECG · {len(signal_data)/sampling_rate:.1f}s window",
        max_duration_sec=len(signal_data)/sampling_rate, start_sec=0.0,
    )
    # Dark chart theme
    fig_waveform.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#c9d1d9",
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    )
    st.plotly_chart(fig_waveform, use_container_width=True)
    crit_wf = get_waveform_criteria(
        input_info=input_info,
        sampling_rate=sampling_rate,
        lead="II",
        duration_sec=len(signal_data)/sampling_rate,
        has_raw=show_raw,
    )
    render_criteria_footer(crit_wf, key_suffix="waveform", show_details=False)

    # ── AI Classification + Signal Quality ──
    st.markdown('<div class="section-label">AI Classification</div>', unsafe_allow_html=True)
    cl, cr = st.columns([1, 1])
    with cl:
        fig_prob = plot_prediction_probabilities(
            ai_results["probabilities"],
            predicted_class="PVC" if "PVC" in ai_results["predicted_class"] else ("Normal" if "Normal" in ai_results["predicted_class"] else "Other"),
        )
        fig_prob.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
        st.plotly_chart(fig_prob, use_container_width=True)
        crit_ai = get_ai_classification_criteria(
            ai_results=ai_results,
            model_id="ECG-RF-1.0.0",
            is_multimodal=False,
        )
        render_criteria_footer(crit_ai, key_suffix="ai_prob", show_details=True)
        counts = ai_results.get("class_counts", {})
        total_beats = max(1, ai_results["beat_count"])
        st.markdown(
            f"🟢 Normal: **{counts.get('Normal',0)}** ({counts.get('Normal',0)/total_beats*100:.1f}%)  &nbsp;·&nbsp;  "
            f"🔴 PVC: **{counts.get('PVC',0)}** ({counts.get('PVC',0)/total_beats*100:.1f}%)  &nbsp;·&nbsp;  "
            f"🟡 Other: **{counts.get('Other',0)}** ({counts.get('Other',0)/total_beats*100:.1f}%)"
        )

    with cr:
        q_ind = ai_results.get("quality_indicators", {})
        st.table(pd.DataFrame([
            {"Indicator": "SNR",              "Value": f"{q_ind.get('snr_db',0):.1f} dB",   "Status": "✓ Good" if q_ind.get('snr_db',0)>15 else "⚠ Low"},
            {"Indicator": "Baseline Drift",   "Value": str(q_ind.get('baseline_wander',False)), "Status": "⚠ Present" if q_ind.get('baseline_wander') else "✓ Clean"},
            {"Indicator": "Powerline (50Hz)", "Value": str(q_ind.get('has_powerline_interference',False)), "Status": "⚠ Present" if q_ind.get('has_powerline_interference') else "✓ Suppressed"},
            {"Indicator": "Motion Artifacts", "Value": str(q_ind.get('has_motion_artifacts',False)), "Status": "⚠ Elevated" if q_ind.get('has_motion_artifacts') else "✓ Minimal"},
        ]))
        crit_sq = get_signal_quality_criteria(q_ind)
        render_criteria_footer(crit_sq, key_suffix="signal_quality", show_details=False)

    # ── Beat Segmentation ──
    st.markdown('<div class="section-label">Beat Segmentation</div>', unsafe_allow_html=True)
    cs, cf = st.columns([1, 1])
    with cs:
        if ai_results["beats"] is not None and len(ai_results["beats"]) > 0:
            fig_beats = plot_beats_overlay(beats=ai_results["beats"], fs=sampling_rate, labels=ai_results.get("beat_predictions"))
            fig_beats.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
            st.plotly_chart(fig_beats, use_container_width=True)
            crit_beats = get_segmentation_criteria(
                beats=ai_results["beats"],
                fs=sampling_rate,
                detected_peaks_count=len(ai_results.get("detected_peaks", [])),
            )
            render_criteria_footer(crit_beats, key_suffix="beats_overlay", show_details=False)
        else:
            st.caption("No beats segmented.")
    with cf:
        feats_df = ai_results.get("features_df")
        if feats_df is not None and not feats_df.empty:
            st.table(pd.DataFrame([
                {"Feature": "R-Peak Amplitude",    "Value": f"{feats_df['r_peak_amplitude'].mean():.3f} mV"},
                {"Feature": "QRS Width",           "Value": f"{feats_df['qrs_width_samples'].mean():.3f} s"},
                {"Feature": "Peak-to-Peak",        "Value": f"{feats_df['peak_to_peak_amplitude'].mean():.3f} mV"},
                {"Feature": "Signal Energy",       "Value": f"{feats_df['energy'].mean():.2f}"},
                {"Feature": "Spectral Entropy",    "Value": f"{feats_df['spectral_entropy'].mean():.3f}"},
                {"Feature": "Dominant Frequency",  "Value": f"{feats_df['dominant_frequency'].mean():.2f} Hz"},
                {"Feature": "Local RR Ratio",      "Value": f"{feats_df['local_rr_ratio'].mean():.3f}"},
            ]))
            crit_feats = get_feature_table_criteria(feats_df)
            render_criteria_footer(crit_feats, key_suffix="feature_table", show_details=False)
        else:
            st.caption("Feature table unavailable.")

# ─────────────────────────────────────────
# MIT-BIH Ground Truth
# ─────────────────────────────────────────
if reference_annotations is not None and not reference_annotations.empty:
    with st.expander("📚 MIT-BIH Ground Truth Comparison"):
        ref_symbols = reference_annotations["symbol"].values
        ref_labels  = [map_symbol_to_class(s, mode="3class") for s in ref_symbols]
        st.dataframe(pd.DataFrame({
            "Time (s)": [f"{t:.2f}" for t in reference_annotations["time_sec"].values],
            "Symbol": ref_symbols, "Class": ref_labels,
            "Annotation": reference_annotations["description"].values,
        }), use_container_width=True)
        crit_gt = CriteriaDescriptor(
            label="Criteria matched",
            criteria=["MIT-BIH PhysioNet reference annotations", "AAMI 3-class mapping (Normal/PVC/Other)", "R-peak temporal alignment"],
            chart_id="ground_truth",
        )
        render_criteria_footer(crit_gt, key_suffix="ground_truth", show_details=False)

# ─────────────────────────────────────────
# CLINICAL DECISION SUPPORT
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Clinical Decision Support</div>', unsafe_allow_html=True)

urgency_badge = {
    "ROUTINE REVIEW": '<span class="badge badge-blue">ROUTINE</span>',
    "PROMPT REVIEW":  '<span class="badge badge-yellow">PROMPT REVIEW</span>',
}.get(cds_rec_obj.urgency, '<span class="badge badge-red">URGENT</span>')

st.markdown(
    f'<div class="glass-panel">'
    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
    f'<span style="font-weight:700;color:#f0f6fc;font-size:0.95rem;">{cds_rec_obj.finding}</span>'
    f'{urgency_badge}'
    f'</div>'
    f'<div style="font-size:0.8rem;color:#8b949e;margin-bottom:6px;">'
    f'<b style="color:#c9d1d9;">Guideline:</b> {cds_rec_obj.guideline_source} · {cds_rec_obj.last_verified}'
    f'</div>'
    f'<div style="font-size:0.83rem;color:#3fb950;">{cds_rec_obj.clinician_action_required}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

cds_c1, cds_c2 = st.columns(2)
with cds_c1:
    if cds_rec_obj.clinical_considerations:
        st.caption("Considerations")
        for c in cds_rec_obj.clinical_considerations:
            st.markdown(f"- {c}")
    if cds_rec_obj.contraindication_warnings:
        for cw in cds_rec_obj.contraindication_warnings:
            st.error(f"⚠ {cw}")
with cds_c2:
    if cds_rec_obj.suggested_assessments:
        st.caption("Next Assessments")
        for a in cds_rec_obj.suggested_assessments:
            st.markdown(f"- {a}")
    if cds_rec_obj.relevant_guidelines:
        st.caption("Guidelines")
        for g in cds_rec_obj.relevant_guidelines:
            st.markdown(f"- *{g}*")

crit_cds = get_cds_criteria(
    cds_rec=cds_rec_obj,
    has_patient_history=bool(_reg_pat and _reg_pat.get("existing_conditions")),
    has_vitals=bool(_vitals),
    has_labs=bool(_labs),
    has_meds=bool(_curr_meds_list),
)
render_criteria_footer(crit_cds, key_suffix="cds_panel", show_details=False)

# ─────────────────────────────────────────
# MEDICATION SUGGESTIONS (from CDS Engine)
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Guideline-Based Medication Suggestions</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:#1a2332;border:1px solid #1f6feb;border-radius:8px;padding:8px 14px;margin-bottom:12px;font-size:0.75rem;color:#79c0ff;">
  ⚕ <b>For Physician Reference Only</b> — These are evidence-based drug class considerations from AHA/ACC/ESC guidelines.
  No medication is prescribed by this system. All prescribing decisions must be made by a licensed clinician.
</div>
""", unsafe_allow_html=True)

med_recs = cds_rec_obj.medication_recommendations
if med_recs:
    for i, rec in enumerate(med_recs):
        _is_na = rec.get("example_agents","").strip().upper() == "N/A"
        _border_col = "#21262d" if _is_na else "#30363d"
        _accent = "#8b949e" if _is_na else "#58a6ff"
        st.markdown(f"""<div style="background:#161b22;border:1px solid {_border_col};border-left:3px solid {_accent};border-radius:8px;padding:12px 16px;margin-bottom:8px;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
<div style="flex:2;min-width:220px;">
<div style="font-size:0.78rem;font-weight:700;color:{_accent};margin-bottom:3px;">💊 {rec.get('drug_class','—')}</div>
<div style="font-size:0.82rem;color:#e6edf3;margin-bottom:2px;"><b style="color:#c9d1d9;">Agents:</b> {rec.get('example_agents','—')}</div>
<div style="font-size:0.78rem;color:#8b949e;"><b style="color:#c9d1d9;">Indication:</b> {rec.get('indication','—')}</div>
</div>
<div style="flex:1.5;min-width:200px;">
<div style="font-size:0.73rem;color:#3fb950;margin-bottom:3px;">📚 {rec.get('guideline','—')}</div>
<div style="font-size:0.72rem;color:#8b949e;font-style:italic;">ℹ {rec.get('note','—')}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
else:
    st.caption("No medication recommendations available for this finding.")

crit_med_sug = CriteriaDescriptor(
    label="Criteria considered",
    criteria=["Primary rhythm finding", "AHA/ACC/ESC Class I/IIa guidelines", "Heart rate category", "Contraindication screening"],
    chart_id="med_suggestions",
)
render_criteria_footer(crit_med_sug, key_suffix="med_suggestions", show_details=False)

# ─────────────────────────────────────────
# MEDICATION SAFETY
# ─────────────────────────────────────────
with st.expander("💊 Medication Safety Check"):
    med_c1, med_c2 = st.columns([1,1])
    with med_c1:
        patient_meds_input    = st.text_input("Medications (comma-separated)", value="Metoprolol, Amiodarone", key="inp_patient_meds")
        patient_allergies_input = st.text_input("Known Allergies", value="Penicillin", key="inp_patient_allergies")
        patient_conditions_input = st.text_input("Conditions", value="Hypertension, PVC Arrhythmia", key="inp_patient_conditions")
    with med_c2:
        all_meds = GLOBAL_MEDICATION_DB.list_all()
        st.caption(f"Formulary: {len(all_meds)} cardiovascular agents indexed")

    med_list = [m.strip() for m in patient_meds_input.split(",") if m.strip()]
    med_safety_report = check_medication_safety(med_list, patient=PatientRecord(
        patient_id="PAT-TMP", hospital_mrn="MRN-EVAL", name="Eval",
        age=65, sex="M", known_allergies=patient_allergies_input,
        existing_conditions=patient_conditions_input, current_medications=patient_meds_input,
    ))
    if med_safety_report.alerts:
        for alert in med_safety_report.alerts:
            if alert.severity == "CRITICAL":
                st.error(f"🚨 **{alert.title}** — {alert.description}  \n*{alert.clinical_recommendation}*")
            elif alert.severity == "MAJOR":
                st.warning(f"⚠ **{alert.title}** — {alert.description}  \n*{alert.clinical_recommendation}*")
            else:
                st.info(f"**{alert.title}** — {alert.description}")
    else:
        st.success("✓ No major drug-drug interactions or allergy conflicts detected.")

    crit_med = get_medication_safety_criteria(
        med_safety_eval=med_safety_report,
        patient_medications=med_list,
        allergies=patient_allergies_input,
        vital_signs=_vitals,
        laboratory_results=_labs,
    )
    render_criteria_footer(crit_med, key_suffix="med_safety_check", show_details=False)

# ─────────────────────────────────────────
# CLINICIAN SIGN-OFF
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Physician Sign-Off</div>', unsafe_allow_html=True)

if active_review:
    _rs = active_review.get("status","REVIEWED")
    _badge = f'<span class="badge badge-green">✓ {_rs}</span>' if _rs == "CONFIRMED" else f'<span class="badge badge-yellow">{_rs}</span>'
    _notes_html = f'<div style="font-size:0.8rem;color:#8b949e;margin-top:4px;">{active_review.get("clinical_notes","")}</div>' if active_review.get("clinical_notes") else ""
    st.markdown(
        f'<div class="glass-panel">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'{_badge}'
        f'<span style="font-size:0.72rem;color:#8b949e;">{active_review.get("reviewed_at","")}</span>'
        f'</div>'
        f'<div style="font-size:0.83rem;color:#c9d1d9;">'
        f'<b>{active_review.get("clinician_name","")}</b> ({active_review.get("clinician_role","")}) · Reg {active_review.get("registration_number","—")}'
        f'</div>'
        f'<div style="margin-top:6px;font-size:0.85rem;color:#e6edf3;">{active_review.get("clinician_interpretation","")}</div>'
        f'{_notes_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.caption("⏳ Pending physician review — analysis not yet sealed.")

crit_sign = CriteriaDescriptor(
    label="Criteria verified",
    criteria=["Attending physician clinical evaluation", "Bedside correlation", "AI findings agreement", "Digital cryptographic sealing"],
    chart_id="physician_sign_off",
)
render_criteria_footer(crit_sign, key_suffix="physician_sign_off", show_details=False)

if AUTH_MANAGER.has_permission(current_user, "report:sign_off"):
    with st.expander("✍ Sign & Seal Report", expanded=(active_review is None)):
        col_ag1, col_ag2 = st.columns([1, 1])
        with col_ag1:
            agreement_choice = st.radio(
                "Agreement",
                options=["CONFIRMED","MODIFIED","REJECTED"],
                format_func=lambda x: {"CONFIRMED":"✓ Confirmed","MODIFIED":"⚠ Modified","REJECTED":"✗ Rejected"}[x],
                index=0 if not active_review else (["CONFIRMED","MODIFIED","REJECTED"].index(active_review.get("agreement_status","CONFIRMED")) if active_review and active_review.get("agreement_status") in ["CONFIRMED","MODIFIED","REJECTED"] else 0),
            )
        with col_ag2:
            st.caption(f"**{current_user.full_name}**  \n`{current_user.role.value}` · `{current_user.registration_number or 'MCI-PENDING'}`")

        _default_dx = "Normal Sinus Rhythm. No acute ischemic ST-T changes."
        if ai_results and "PVC" in ai_results.get("predicted_class",""):
            _default_dx = "Sinus rhythm with PVCs. Recommend 24-hr Holter."
        elif ai_results and "Other" in ai_results.get("predicted_class",""):
            _default_dx = "Atypical complexes. Recommend 12-lead ECG + cardiology consult."

        custom_diag = st.text_area("Clinical Finding", value=active_review.get("clinician_interpretation",_default_dx) if active_review else _default_dx, height=70)
        custom_directives = st.text_area("Directives", value=active_review.get("clinical_notes","Routine outpatient follow-up. Repeat ECG if symptomatic.") if active_review else "Routine outpatient follow-up. Repeat ECG if symptomatic.", height=60)

        if st.button("✍ Sign & Seal Report", type="primary"):
            rev_id = f"REV-{secrets.token_hex(4).upper()}"
            DB_MANAGER.save_clinician_review(
                review_id=rev_id, analysis_id=f"ANL-{rec_id[4:]}", record_id=rec_id,
                clinician_id=current_user.user_id, clinician_name=current_user.full_name,
                clinician_role=current_user.role.value, agreement_status=agreement_choice,
                clinician_interpretation=custom_diag, clinical_notes=custom_directives,
                registration_number=current_user.registration_number,
            )
            DB_MANAGER.save_report(report_id=f"REP-{secrets.token_hex(4).upper()}", record_id=rec_id,
                analysis_id=f"ANL-{rec_id[4:]}", review_id=rev_id, report_type="CLINICAL_PDF",
                status="SEALED", report_sha256=f"{abs(hash(custom_diag)):x}")
            DB_MANAGER.update_ecg_workflow_status(rec_id, "SIGNED", assigned_doctor=current_user.full_name)

            # Persist sealed snapshot and signed PDF in Supabase & SQLite
            try:
                _signed_pdf = generate_doctor_report(
                    report_data=report_data,
                    waveform=waveform_for_pdf,
                    fs=sampling_rate,
                    r_peaks=peaks_for_pdf,
                )
                REPORT_PERSISTENCE_SERVICE.sign_and_seal_report(
                    report_id=_p_meta.get("report_id") or rec_id,
                    clinician_user_id=current_user.user_id,
                    clinician_name=current_user.full_name,
                    clinician_role=current_user.role.value,
                    registration_number=current_user.registration_number or "MCI-VERIFIED",
                    agreement_status=agreement_choice,
                    clinician_interpretation=custom_diag,
                    clinical_notes=custom_directives,
                    signed_pdf_bytes=_signed_pdf,
                )
            except Exception:
                pass

            AUDIT_LOGGER.log_event(
                event_type="CLINICIAN_SIGN_OFF", user_id=current_user.user_id,
                username=current_user.username, user_role=current_user.role.value,
                action=f"Sealed report: {agreement_choice}", record_id=rec_id,
                details={"record_id":rec_id,"status":agreement_choice,"interpretation":custom_diag},
            )
            st.session_state[f"review_{rec_id}"] = {
                "status": agreement_choice, "agreement_status": agreement_choice,
                "clinician_name": current_user.full_name, "clinician_role": current_user.role.value,
                "registration_number": current_user.registration_number,
                "clinician_interpretation": custom_diag, "clinical_notes": custom_directives,
                "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.success("Report sealed with physician digital signature.")
            st.rerun()
else:
    st.caption(f"🔒 Sign-off restricted for role `{current_user.role.value}`. Switch to a Doctor/Cardiologist profile.")

# ─────────────────────────────────────────
# EXPORT REPORTS
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)

doctor_pdf_bytes  = generate_doctor_report(report_data=report_data, waveform=waveform_for_pdf, fs=sampling_rate, r_peaks=peaks_for_pdf)
patient_pdf_bytes = generate_patient_report(report_data=report_data, waveform=waveform_for_pdf, fs=sampling_rate, r_peaks=peaks_for_pdf)
json_str = export_report_to_json(report_data)
text_str = export_report_to_text(report_data)
stem_name = Path(input_info.get("file_name","ecg_analysis")).stem.replace(" ","_").lower()

dc1, dc2, dc3, dc4 = st.columns(4)
with dc1:
    st.download_button("🩺 Doctor Report (PDF)", data=doctor_pdf_bytes, file_name=f"doctor_{stem_name}.pdf", mime="application/pdf")
with dc2:
    st.download_button("🧑‍💼 Patient Summary (PDF)", data=patient_pdf_bytes, file_name=f"patient_{stem_name}.pdf", mime="application/pdf")
with dc3:
    st.download_button("📊 Data (JSON)", data=json_str, file_name=f"ecg_{stem_name}.json", mime="application/json")
with dc4:
    st.download_button("📝 Summary (TXT)", data=text_str, file_name=f"ecg_{stem_name}.txt", mime="text/plain")

# ─────────────────────────────────────────
# MODEL TRANSPARENCY
# ─────────────────────────────────────────
with st.expander("🔍 Model Architecture & Validation"):
    mcol_l, mcol_r = st.columns(2)
    with mcol_l:
        try:
            _, _, meta = get_trained_artifacts(MODELS_DIR)
            with open(REPORTS_DIR / "evaluation_report.json","r") as f:
                eval_data = json.load(f)
            rf_m = eval_data.get("random_forest",{})
            st.markdown(f"""
- **Architecture:** `{meta.get('model_name')}` — 100 Trees, Gini
- **Features:** 28 morphological, spectral, rhythm
- **Train Set:** MIT-BIH ({meta.get('train_samples')} beats)
- **Test Set:** {meta.get('test_samples')} beats
- **Macro F1:** `{rf_m.get('f1_macro',0)*100:.2f}%`
- **Accuracy:** `{rf_m.get('accuracy',0)*100:.2f}%`
""")
        except Exception as exc:
            st.caption(f"Model config: {exc}")
    with mcol_r:
        try:
            _, _, meta = get_trained_artifacts(MODELS_DIR)
            importances = meta.get("feature_importances",{})
            if importances:
                fig_imp = plot_feature_importance(importances, top_k=6)
                fig_imp.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
                st.plotly_chart(fig_imp, use_container_width=True)
                crit_imp = get_feature_importance_criteria(importances, top_k=6)
                render_criteria_footer(crit_imp, key_suffix="feature_importance", show_details=False)
        except Exception:
            st.caption("Feature importance unavailable.")

st.divider()

# ─────────────────────────────────────────
# WORKLIST & AUDIT (collapsed)
# ─────────────────────────────────────────
render_hospital_worklist(expanded=False)
st.divider()
render_audit_trail(expanded=False)
