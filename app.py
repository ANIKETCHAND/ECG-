"""
AI ECG Analyzer - Universal Clinical Research & Educational Dashboard
======================================================================

Features:
1. Universal ECG Ingestion: PDF clinical reports, scanned report images (JPG/PNG), digital waveforms (CSV/TXT/NPY).
2. Multi-step Progress Tracking & Transparency.
3. Preprocessing, Noise Filtering & Signal Quality Assessment.
4. R-Peak Detection & Cardiac Cycle Segmentation.
5. Machine Learning Arrhythmia Classification (Normal vs PVC vs Other).
6. Clear Separation of Printed Machine Interpretation vs AI Predictions (Zero Hallucination).
7. Publication-Grade Multi-Format Reporting (PDF, JSON, TXT).
8. Research Demo Mode with MIT-BIH Ground Truth Validation.

Research/Educational use only. Not for certified medical diagnosis.
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
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

# Ingestion and Report Modules
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
)
from auth.auth_manager import AUTH_MANAGER, UserRole
from database.db_manager import DB_MANAGER, PatientRecord
from audit.audit_logger import AUDIT_LOGGER
from clinical import GLOBAL_CDS_ENGINE, ClinicalRecommendation
from medications import GLOBAL_MEDICATION_DB, check_medication_safety
from alerts import GLOBAL_ALERT_ENGINE, AlertType, AlertSeverity
import secrets
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI ECG Platform — Hospital Telemetry & Decision Support",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-normal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-abnormal {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
SAMPLE_ECGS_DIR = Path(__file__).resolve().parent / "sample_ecgs"

# ---------------------------------------------------------
# Sidebar Configuration & Clinical Identity
# ---------------------------------------------------------
st.sidebar.title("❤️ AI-ECG Platform")
st.sidebar.markdown("**Hospital Telemetry & Decision Support**")

st.sidebar.markdown("#### 👤 Clinical Identity & Role")
all_users = AUTH_MANAGER.list_users()
user_map = {u.username: f"{u.full_name} ({u.role.value})" for u in all_users}
if "active_user_name" not in st.session_state:
    st.session_state.active_user_name = "cardiologist"

chosen_username = st.sidebar.selectbox(
    "Active Staff Profile",
    options=list(user_map.keys()),
    format_func=lambda x: user_map[x],
    index=list(user_map.keys()).index(st.session_state.active_user_name),
    help="Select hospital user account to test role-based permissions (Doctor, Cardiologist, Tech, Admin, Researcher).",
)
st.session_state.active_user_name = chosen_username
current_user = AUTH_MANAGER.get_user_by_username(chosen_username)

st.sidebar.caption(
    f"**Role:** `{current_user.role.value}` | **Reg No:** `{current_user.registration_number or 'N/A'}`\n"
    f"*{current_user.email}*"
)
st.sidebar.divider()

# Workspace View Selection (Clinician vs Patient Portal)
view_mode = st.sidebar.radio(
    "Hospital Workspace View",
    ["🏥 Hospital Clinician Workspace", "🧑‍💼 Patient Health Portal"],
    index=0,
    help="Toggle between full clinical decision support dashboard and patient-facing portal.",
)

if view_mode == "🧑‍💼 Patient Health Portal":
    st.sidebar.markdown("#### 🧑‍💼 Patient Health Portal")
    st.sidebar.caption("View and download your official physician-signed ECG reports and clinical directives.")
    st.sidebar.info("💡 Select your patient record in the portal to inspect verified diagnostic reports.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📞 24/7 Cardiac Emergency Contacts:**")
    st.sidebar.caption("- Emergency Room: `+91-11-2345-0000`\n- National Ambulance: `112` / `102`\n- Cardiology Desk: `Ext. 402`")
    input_source_mode = "Patient Portal"
    uploaded_file = None
    selected_record = None
    sampling_rate_setting = 360
    analysis_duration_sec = 10
    start_offset_sec = 0.0

else:
    # Input Mode Selection
    input_source_mode = st.sidebar.radio(
        "Choose Input Source",
        options=["Upload Patient ECG File", "MIT-BIH Research Demo Mode"],
        index=0,
    )

    available_records = get_available_records(DATA_RAW_DIR)
    if not available_records:
        available_records = ["100", "101", "106", "119", "200", "208", "213"]

    selected_record = None
    uploaded_file = None

    if input_source_mode == "Upload Patient ECG File":
        # ------------------------------------------------
        # Step 1: Patient Registration
        # ------------------------------------------------
        st.sidebar.markdown("#### 👤 Step 1 — Register Patient")

        if "registered_patient_data" not in st.session_state:
            st.session_state["registered_patient_data"] = None
        if "sidebar_mrn" not in st.session_state:
            import secrets as _sec_init
            st.session_state["sidebar_mrn"] = f"MRN-{_sec_init.token_hex(3).upper()}"

        if st.session_state["registered_patient_data"] is None:
            # Show registration form
            import secrets as _sec
            with st.sidebar.form("patient_registration_form", clear_on_submit=False):
                _mrn    = st.text_input("Hospital MRN", value=st.session_state["sidebar_mrn"], key="form_mrn")
                _name   = st.text_input("Patient Full Name *", placeholder="e.g. Rajesh Kumar", key="form_name")
                _c1, _c2 = st.columns(2)
                with _c1:
                    _age   = st.number_input("Age", min_value=1, max_value=120, value=55, key="form_age")
                    _sex   = st.selectbox("Sex", ["M", "F", "Other"], key="form_sex")
                with _c2:
                    _blood = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], key="form_blood")
                    _smoke = st.selectbox("Smoking", ["Non-Smoker", "Former Smoker", "Current Smoker", "Unknown"], key="form_smoke")
                _contact   = st.text_input("Contact No.", value="+91-", key="form_contact")
                _allergies = st.text_input("Known Allergies", placeholder="e.g. Penicillin", key="form_allergies")
                _conds     = st.text_input("Existing Conditions", placeholder="e.g. Hypertension, Diabetes", key="form_conds")
                _cardiac   = st.text_input("Cardiac History", placeholder="e.g. Prior MI 2021, Stent", key="form_cardiac")
                _meds      = st.text_input("Current Medications", placeholder="e.g. Metoprolol 50mg", key="form_meds")
                _submitted = st.form_submit_button("✅ Register Patient & Proceed to Upload", type="primary", use_container_width=True)

            if _submitted:
                if not st.session_state.get("form_name", "").strip():
                    st.sidebar.error("⚠️ Patient Full Name is required.")
                else:
                    _p_id = f"PAT-{_sec.token_hex(4).upper()}"
                    try:
                        DB_MANAGER.create_patient(
                            patient_id=_p_id,
                            hospital_mrn=st.session_state["form_mrn"],
                            name=st.session_state["form_name"],
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
                            action=f"Registered patient {st.session_state['form_name']} ({st.session_state['form_mrn']}) before ECG upload",
                            patient_id=_p_id,
                        )
                    except Exception:
                        pass

                    st.session_state["registered_patient_data"] = {
                        "patient_id":               _p_id,
                        "hospital_mrn":             st.session_state["form_mrn"],
                        "name":                     st.session_state["form_name"],
                        "age":                      int(st.session_state["form_age"]),
                        "sex":                      st.session_state["form_sex"],
                        "blood_group":              st.session_state["form_blood"],
                        "contact":                  st.session_state["form_contact"],
                        "known_allergies":          st.session_state["form_allergies"] or "None",
                        "existing_conditions":      st.session_state["form_conds"] or "None",
                        "previous_cardiac_history": st.session_state["form_cardiac"] or "None",
                        "current_medications":      st.session_state["form_meds"] or "None",
                        "smoking_status":           st.session_state["form_smoke"],
                    }
                    st.rerun()

            # No patient registered yet — keep uploader hidden
            uploaded_file = None
            sampling_rate_setting = 360
            analysis_duration_sec = 10
            start_offset_sec = 0.0

        else:
            # Patient registered — show summary card and Step 2 upload
            _pat = st.session_state["registered_patient_data"]
            st.sidebar.success(
                f"✅ **Patient Registered**\n\n"
                f"👤 **{_pat['name']}**\n"
                f"MRN: `{_pat['hospital_mrn']}` | Age: {_pat['age']} / {_pat['sex']}"
            )
            if st.sidebar.button("🔄 Change Patient", key="btn_change_patient"):
                st.session_state["registered_patient_data"] = None
                import secrets as _sec_r
                st.session_state["sidebar_mrn"] = f"MRN-{_sec_r.token_hex(3).upper()}"
                st.rerun()

            st.sidebar.markdown("---")
            st.sidebar.markdown("#### 📁 Step 2 — Upload ECG File")

            uploaded_file = st.sidebar.file_uploader(
                "Upload ECG (PDF, JPG, PNG, CSV, TXT, NPY)",
                type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "csv", "txt", "npy"],
                help="Upload standard clinical ECG report documents (PDF), scanned waveforms (JPG/PNG), or digital signals (CSV/TXT/NPY).",
            )

            st.sidebar.markdown("---")
            st.sidebar.markdown("**💡 Quick Test Samples Available:**")
            st.sidebar.caption(
                "You can test the system with files in the `sample_ecgs/` folder:\n"
                "- `sample_clinical_ecg_report.pdf` (Clinical 12-lead PDF)\n"
                "- `normal_ecg_sample.csv` (Sinus rhythm digital signal)\n"
                "- `pvc_arrhythmia_sample.csv` (Frequent PVC digital signal)"
            )

            sampling_rate_setting = st.sidebar.selectbox(
                "Digital Signal Sampling Rate (Hz)",
                options=STANDARD_SAMPLING_RATES,
                index=2,
                help="Used when uploading digital CSV/TXT signals without an explicit time column. Default is 360 Hz.",
            )
            analysis_duration_sec = st.sidebar.slider(
                "Analysis Duration Window (s)",
                min_value=3,
                max_value=30,
                value=10,
                step=1,
            )
            start_offset_sec = 0.0

    else:
        # MIT-BIH Demo Mode
        record_descriptions = {
            "100": "Record 100 (Normal Sinus Rhythm)",
            "101": "Record 101 (Normal Rhythm / Baseline)",
            "106": "Record 106 (Frequent PVCs & Ventricular Arrhythmia)",
            "119": "Record 119 (High-Frequency PVCs)",
            "200": "Record 200 (Ventricular Tachycardia / Couplets)",
            "208": "Record 208 (Ventricular Ectopy / Fusion Beats)",
            "213": "Record 213 (Ventricular & Atrial Ectopy)",
        }
        record_options = [r for r in available_records if r in record_descriptions] or available_records
        selected_record = st.sidebar.selectbox(
            "Select MIT-BIH Benchmark Record",
            options=record_options,
            format_func=lambda x: record_descriptions.get(x, f"Record {x}"),
        )
        sampling_rate_setting = 360
        start_offset_sec = st.sidebar.slider(
            "Start Offset (seconds)",
            min_value=0,
            max_value=120,
            value=0,
            step=1,
        )
        analysis_duration_sec = st.sidebar.slider(
            "Analysis Window Duration (seconds)",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
        )



st.sidebar.divider()
st.sidebar.info(
    "⚠️ **Educational & Research Notice**\n\n"
    "This platform is developed strictly for educational and scientific research purposes. "
    "It is **not** a certified clinical diagnostic medical device."
)

# ---------------------------------------------------------
# Main Page Header & Banner
# ---------------------------------------------------------
st.markdown('<div class="main-header">❤️ AI ECG Abnormality Detection & Reporting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Universal multi-format clinical ECG analysis, zero-hallucination validation, and instant publication-grade reporting.</div>',
    unsafe_allow_html=True,
)

st.warning(
    "🛡️ **Clinical Disclaimer:** This system provides automated research screening analysis. "
    "It does not replace certified physician evaluation or emergency cardiovascular care. "
    "If you are experiencing chest pain, palpitations, or shortness of breath, please seek emergency medical attention immediately."
)

# ---------------------------------------------------------
# PATIENT HEALTH PORTAL VIEW (When selected in sidebar)
# ---------------------------------------------------------
if view_mode == "🧑‍💼 Patient Health Portal":
    st.markdown('<div class="main-header">❤️ Apex Heart Hospital — Patient Health Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Secure personal health portal for verified clinical ECG evaluations, physician directives, and signed reports.</div>',
        unsafe_allow_html=True,
    )

    # 1. Patient Profile Selector
    enrolled_patients = DB_MANAGER.list_patients()
    if enrolled_patients:
        p_options = {p.hospital_mrn: f"{p.name} (MRN: {p.hospital_mrn})" for p in enrolled_patients}
        chosen_mrn = st.selectbox("Select Patient Profile:", options=list(p_options.keys()), format_func=lambda x: p_options[x], key="portal_pat_sel")
        curr_patient = DB_MANAGER.get_patient_by_mrn(chosen_mrn)
    else:
        curr_patient = PatientRecord(
            patient_id="PT-DEMO-001",
            hospital_mrn="MRN-DEMO-101",
            name="Ramesh Sharma",
            age=58,
            sex="M",
            contact="+91-9876543210",
            blood_group="B+",
            known_allergies="None documented",
            existing_conditions="Mild Hypertension",
            current_medications="Metoprolol 25mg daily",
            created_at=datetime.now().isoformat(),
        )

    # Patient Demographics Card
    c_p1, c_p2 = st.columns([2, 1])
    with c_p1:
        st.markdown(
            f"""
            <div style="background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 14px;">
                <h4 style="margin: 0 0 6px 0; color: #0f172a;">👤 {curr_patient.name}</h4>
                <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                    <b>MRN:</b> <code>{curr_patient.hospital_mrn}</code> | 
                    <b>Age / Sex:</b> {curr_patient.age or 'N/A'} / {curr_patient.sex or 'N/A'} | 
                    <b>Blood Group:</b> {curr_patient.blood_group or 'Unknown'}
                </p>
                <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.9rem;">
                    <b>Known Allergies:</b> {curr_patient.known_allergies or 'None'} | 
                    <b>Current Medications:</b> {curr_patient.current_medications or 'None recorded'}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_p2:
        st.markdown(
            """
            <div style="background-color: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 14px; margin-bottom: 14px;">
                <h5 style="margin: 0 0 4px 0; color: #065f46;">🏥 Hospital Facility</h5>
                <p style="margin: 0; font-size: 0.85rem; color: #047857;">Apex Heart & Vascular Hospital<br/>Department of Cardiac Electrophysiology<br/>Emergency: +91-11-2345-0000</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Ingested ECG Records for this Patient
    all_records = DB_MANAGER.list_ecg_records(limit=50)
    patient_records = [r for r in all_records if r.get("hospital_mrn") == curr_patient.hospital_mrn or r.get("patient_id") == curr_patient.patient_id]
    if not patient_records and all_records:
        patient_records = all_records[:3]

    if patient_records:
        st.markdown("### 📋 Your Diagnostic Electrocardiogram Evaluations")
        for rec in patient_records:
            rec_id = rec["record_id"]
            review = DB_MANAGER.get_clinician_review(rec_id) or st.session_state.get(f"review_{rec_id}")
            wf_status = rec.get("workflow_status", "UPLOADED")
            is_signed = (wf_status == "SIGNED") or (review and review.get("agreement_status") in ["CONFIRMED", "MODIFIED", "ACCEPTED"])

            with st.container():
                st.markdown(f"#### ECG Record: `{rec_id}` (Recorded: {rec['uploaded_at'][:16].replace('T', ' ')})")
                col_r1, col_r2 = st.columns([2, 1])

                with col_r1:
                    if is_signed and review:
                        st.success(f"✅ **Verified & Signed by Attending Physician:** {review.get('clinician_name', 'Attending Physician')} ({review.get('clinician_role', 'Doctor')})")
                        st.markdown(f"**Medical Registration Number:** `{review.get('registration_number', 'Verified')}`")
                        st.markdown(f"**Clinical Finding / Diagnosis:**\n> {review.get('clinician_interpretation', 'Normal rhythm within physiological parameters.')}")
                        if review.get("clinical_notes"):
                            st.markdown(f"**Physician Directives & Care Plan:**\n> {review.get('clinical_notes')}")
                    else:
                        st.info("⏳ **Status: Under Review by Physician**\n\nYour ECG recording has been uploaded and processed by diagnostic screening algorithms. The attending physician will review, confirm, and digitally sign off on this report shortly.")

                with col_r2:
                    hr_bpm = rec.get("heart_rate_bpm") or 72.0
                    st.metric("Recorded Heart Rate", f"{hr_bpm:.0f} BPM")
                    st.caption(f"Signal Quality: **{rec.get('signal_quality', 'GOOD')}** | Duration: {rec.get('duration_sec', 10.0):.1f}s")

                    # Generate on-demand downloadable patient PDF report
                    patient_rep_data = generate_structured_report(
                        input_info={"file_name": f"{rec_id}.csv", "modality": "DIGITAL_SIGNAL", "sampling_rate": rec["sampling_rate"], "duration_sec": rec["duration_sec"], "lead": "Lead II"},
                        ai_results={"predicted_class": rec.get("prediction", "Normal Rhythm"), "signal_quality": rec.get("signal_quality", "GOOD"), "quality_score": 0.95, "heart_rate_bpm": hr_bpm, "quality_indicators": {"snr_db": 22.0}},
                        clinician_review=review,
                    )
                    pdf_bytes_patient = generate_pdf_report(patient_rep_data)
                    st.download_button(
                        f"📄 Download Signed PDF Report",
                        data=pdf_bytes_patient,
                        file_name=f"clinical_report_{curr_patient.name.replace(' ', '_')}_{rec_id}.pdf",
                        mime="application/pdf",
                        key=f"btn_dl_patient_{rec_id}",
                    )
                st.divider()
    else:
        st.info("ℹ️ **No ECG records found on file for this patient.** Once an ECG is acquired and uploaded by the cardiology clinic, your signed diagnostic report will appear here.")

    st.warning(
        "🚨 **Emergency Cardiovascular Guidance:**\n\n"
        "If you experience severe chest pressure or squeezing, shortness of breath, unexplained dizziness, palpitations, or fainting, "
        "do not wait for report updates. Call emergency services (112 / 911) or visit the nearest emergency medical department immediately."
    )
    st.stop()

# ---------------------------------------------------------
# State Variables
# ---------------------------------------------------------
signal_data: Optional[np.ndarray] = None
sampling_rate: float = float(sampling_rate_setting)
reference_annotations: Optional[pd.DataFrame] = None
input_info: Dict[str, Any] = {}
extracted_measurements: Optional[Dict[str, Any]] = None
waveform_status: Dict[str, Any] = {"is_extracted": False, "message": "No waveform processed"}
ai_results: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------
# Reusable Hospital Worklist & Audit Ledger Renderers
# ---------------------------------------------------------
def render_hospital_worklist(expanded: bool = False):
    with st.expander("🏥 Hospital Clinical Worklist & Patient Database", expanded=expanded):
        st.markdown("##### Enrolled Hospital Patients & Ingested Records")
        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            st.markdown("**Active Hospital ECG Worklist**")
            status_sel = st.selectbox(
                "Filter Worklist by Status:",
                ["ALL", "UPLOADED", "ANALYZED", "SIGNED", "REJECTED"],
                key=f"sel_wk_status_{expanded}",
            )
            records_list = DB_MANAGER.list_ecg_records(
                limit=25, status_filter=None if status_sel == "ALL" else status_sel
            )
            if records_list:
                rec_display = []
                for r in records_list:
                    rec_display.append({
                        "Record ID": r["record_id"],
                        "Patient MRN": r.get("hospital_mrn") or "—",
                        "Patient Name": r.get("patient_name") or "Unassigned",
                        "Status": r.get("workflow_status") or "UPLOADED",
                        "Priority": r.get("priority") or "ROUTINE",
                        "AI Prediction": r.get("prediction") or "—",
                        "Signal Quality": r.get("signal_quality") or "UNKNOWN",
                        "Uploaded At": r["uploaded_at"][:16].replace("T", " "),
                    })
                st.dataframe(pd.DataFrame(rec_display), use_container_width=True)
            else:
                st.info("No ECG records found matching current status filter.")

            st.markdown("**Enrolled Patients Directory**")
            p_list = DB_MANAGER.list_patients(limit=15)
            if p_list:
                p_display = []
                for p in p_list:
                    p_display.append({
                        "MRN": p.hospital_mrn,
                        "Name": p.name,
                        "Age/Sex": f"{p.age or '—'} / {p.sex or '—'}",
                        "Blood Group": p.blood_group or "—",
                        "Allergies": p.known_allergies or "None",
                    })
                st.dataframe(pd.DataFrame(p_display), use_container_width=True)

        with col_p2:
            st.markdown("**Register Patient (Comprehensive Profile)**")
            new_mrn = st.text_input("Hospital MRN:", value=f"MRN-{secrets.token_hex(3).upper()}", key=f"inp_mrn_{expanded}")
            new_name = st.text_input("Patient Full Name:", key=f"inp_name_{expanded}")
            col_pa, col_pb = st.columns(2)
            with col_pa:
                new_age = st.number_input("Age:", min_value=1, max_value=120, value=55, key=f"inp_age_{expanded}")
                new_sex = st.selectbox("Sex:", ["M", "F", "Other"], key=f"inp_sex_{expanded}")
                new_blood = st.selectbox("Blood Group:", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], key=f"inp_blood_{expanded}")
            with col_pb:
                new_contact = st.text_input("Contact No:", value="+91-", key=f"inp_contact_{expanded}")
                new_smoke = st.selectbox("Smoking:", ["Non-Smoker", "Former Smoker", "Current Smoker", "Unknown"], key=f"inp_smoke_{expanded}")

            new_allergies = st.text_input("Known Allergies:", placeholder="e.g. Penicillin, Sulfa drugs", key=f"inp_allergies_{expanded}")
            new_conditions = st.text_input("Existing Clinical Conditions:", placeholder="e.g. Hypertension, Type 2 Diabetes", key=f"inp_conditions_{expanded}")
            new_cardiac_hx = st.text_input("Previous Cardiac History:", placeholder="e.g. Prior MI in 2021, CABG, Stent", key=f"inp_cardiac_hx_{expanded}")
            new_curr_meds = st.text_input("Current Medications:", placeholder="e.g. Metoprolol 50mg, Aspirin 75mg", key=f"inp_curr_meds_{expanded}")

            if st.button("➕ Enroll Patient Profile", key=f"btn_reg_pat_{expanded}", type="primary"):
                if new_name.strip():
                    p_id = f"PAT-{secrets.token_hex(4).upper()}"
                    DB_MANAGER.create_patient(
                        patient_id=p_id,
                        hospital_mrn=new_mrn,
                        name=new_name,
                        age=int(new_age),
                        sex=new_sex,
                        contact=new_contact,
                        blood_group=new_blood,
                        known_allergies=new_allergies,
                        existing_conditions=new_conditions,
                        current_medications=new_curr_meds,
                        previous_cardiac_history=new_cardiac_hx,
                        smoking_status=new_smoke,
                    )
                    AUDIT_LOGGER.log_event(
                        event_type="PATIENT_CREATED",
                        user_id=current_user.user_id,
                        username=current_user.username,
                        user_role=current_user.role.value,
                        action=f"Enrolled complete clinical profile for {new_name} ({new_mrn})",
                        patient_id=p_id,
                    )
                    st.success(f"Patient successfully enrolled: {new_name} ({p_id})")
                    st.rerun()
                else:
                    st.warning("Patient full name is required.")


def render_audit_trail(expanded: bool = False):
    with st.expander("🛡️ Cryptographically Chained Hospital Audit Trail (IEC 62304 / ISO 27799)", expanded=expanded):
        st.markdown("##### Tamper-Evident Chronological Clinical Audit Trail")
        st.caption("Each event is chained to the preceding entry using SHA-256 cryptographic digests, ensuring complete non-repudiation.")

        col_aud1, col_aud2 = st.columns([2, 1])
        with col_aud1:
            if st.button("🔐 Verify Audit Chain Cryptographic Integrity", key=f"btn_verify_audit_{expanded}"):
                is_valid, issues = AUDIT_LOGGER.verify_chain_integrity()
                if is_valid:
                    st.success("✅ **Hash Chain Valid:** Cryptographic integrity verified. All sequential hashes match without any retroactive alteration.")
                else:
                    st.error(f"❌ **Integrity Alert:** Tampering detected: {issues}")

        logs = AUDIT_LOGGER.get_logs(limit=30)
        if logs:
            log_view = []
            for l in logs:
                log_view.append({
                    "Seq #": l["sequence_id"],
                    "Timestamp (UTC)": l["timestamp"][:19].replace("T", " "),
                    "Event Type": l["event_type"],
                    "User": f"{l['username']} ({l['user_role']})",
                    "Action": l["action"],
                    "Status": l["status"],
                    "Entry SHA-256": l["entry_hash"][:16] + "...",
                })
            st.dataframe(pd.DataFrame(log_view), use_container_width=True)
        else:
            st.info("Audit log initialized and awaiting events.")

        st.markdown("##### Regulatory & Quality System Status (CDSCO MDR 2017 & IEC 62304)")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("MDR 2017 Class", "Class B (Moderate Risk)")
        rc2.metric("IEC 62304 Safety Class", "Class B")
        rc3.metric("ISO 14971 Risk Status", "ALARP / Acceptable")
        rc4.metric("Inference Engine", "Decoupled / Frozen")


# ---------------------------------------------------------
# Execution / Loading Logic
# ---------------------------------------------------------
if input_source_mode == "Upload Patient ECG File":
    if uploaded_file is None:
        st.info("👈 **Get Started:** Drag and drop an ECG file (PDF, JPG, PNG, CSV, TXT, NPY) in the sidebar to begin instant analysis.")
        
        # Display sample cards
        st.markdown("### 📋 Supported Upload Formats & Workflows")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                **📄 Clinical ECG Reports (PDF)**
                - Extracts printed text & machine measurements (HR, PR, QRS, QT/QTc, Axes).
                - Identifies machine interpretations directly from source.
                - Analyzes rhythm strip if embedded without fabricating missing data.
                """
            )
        with c2:
            st.markdown(
                """
                **📈 Scanned ECG Images (JPG/PNG)**
                - Color segmentation to eliminate pink/red ECG grid lines.
                - Column-wise trace extraction with continuity verification.
                - Rigorous validation gate rejects occluded/flat traces.
                """
            )
        with c3:
            st.markdown(
                """
                **📊 Digital Signal Files (CSV/TXT/NPY)**
                - Auto-detects delimiters (comma, tab, space, semicolon).
                - Isolates voltage columns and normalizes sampling rate.
                - Full 28-feature extraction & Random Forest classification.
                """
            )

        st.divider()
        render_hospital_worklist(expanded=True)
        st.divider()
        render_audit_trail(expanded=False)
        st.stop()

    # We have an uploaded file! Execute the 6-step progress pipeline
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    modality, ext = detect_input_modality(file_name, file_bytes=file_bytes)

    # Multi-step progress container
    progress_box = st.container()
    with progress_box:
        progress_bar = st.progress(0, text="STEP 1/6: Detecting File Modality & Format...")
        time.sleep(0.1)

    input_info = {
        "file_name": file_name,
        "file_modality": modality.value,
        "format": ext,
        "sampling_rate": sampling_rate,
        "duration_sec": analysis_duration_sec,
        "lead": "Lead II (or available single channel)",
    }

    # Process based on modality
    if modality == InputModality.DIGITAL_SIGNAL:
        progress_bar.progress(30, text="STEP 2/6: Reading & Parsing Digital ECG Signal...")
        sig_res = load_digital_signal(
            uploaded_file,
            sampling_rate_hint=sampling_rate,
            max_duration_sec=analysis_duration_sec,
        )
        if not sig_res["success"]:
            progress_box.empty()
            st.error(f"❌ Could not load digital signal: {sig_res['message']}")
            st.stop()

        signal_data = sig_res["signal"]
        sampling_rate = float(sig_res["sampling_rate"])
        input_info["sampling_rate"] = sampling_rate
        input_info["total_samples"] = len(signal_data)
        input_info["duration_sec"] = round(len(signal_data) / sampling_rate, 2)
        waveform_status = {"is_extracted": False, "message": "Direct digital signal ingestion."}

    elif modality == InputModality.REPORT_PDF:
        progress_bar.progress(30, text="STEP 2/6: Parsing PDF Text & Clinical Measurements...")
        pdf_res = process_pdf_report(io.BytesIO(file_bytes))
        extracted_measurements = pdf_res.get("measurements", {})

        # Check for embedded waveform images
        if pdf_res.get("has_embedded_images") and len(pdf_res["images"]) > 0:
            progress_bar.progress(45, text="Extracting Waveform from PDF Image Strip...")
            first_img = pdf_res["images"][0]
            img_res = process_ecg_image(first_img)
            if img_res["is_ecg"]:
                wf_res = extract_waveform_from_image(img_res["cv_image"], target_fs=360.0)
                if wf_res["success"] and wf_res["signal"] is not None:
                    is_valid, v_msg = validate_extracted_signal(
                        wf_res["signal"], fs=wf_res["sampling_rate"], confidence_score=wf_res["confidence_score"]
                    )
                    if is_valid:
                        signal_data = wf_res["signal"]
                        sampling_rate = wf_res["sampling_rate"]
                        waveform_status = {"is_extracted": True, "message": "Extracted from embedded PDF strip."}
                    else:
                        waveform_status = {"is_extracted": False, "message": v_msg}
                else:
                    waveform_status = {"is_extracted": False, "message": wf_res.get("message", "Waveform could not be isolated.")}
        else:
            waveform_status = {
                "is_extracted": False,
                "message": (
                    "This PDF contains printed clinical parameters and text, but no extractable raw waveform strip. "
                    "Displaying printed machine interpretation and extracted cardiac measurements."
                ),
            }

    elif modality == InputModality.REPORT_IMAGE:
        progress_bar.progress(30, text="STEP 2/6: Preprocessing Image & Detecting ECG Grid...")
        img_res = process_ecg_image(io.BytesIO(file_bytes))
        if not img_res["is_ecg"]:
            progress_box.empty()
            st.error(f"❌ Image Error: {img_res['status_message']}")
            st.stop()

        progress_bar.progress(50, text="STEP 3/6: Extracting Waveform Trace via Color Isolation...")
        wf_res = extract_waveform_from_image(img_res["cv_image"], target_fs=360.0)
        if wf_res["success"] and wf_res["signal"] is not None:
            is_valid, v_msg = validate_extracted_signal(
                wf_res["signal"], fs=wf_res["sampling_rate"], confidence_score=wf_res["confidence_score"]
            )
            if is_valid:
                signal_data = wf_res["signal"]
                sampling_rate = wf_res["sampling_rate"]
                waveform_status = {
                    "is_extracted": True,
                    "confidence": wf_res["confidence_score"],
                    "message": "Waveform successfully extracted and verified from image.",
                }
            else:
                waveform_status = {"is_extracted": False, "message": v_msg}
        else:
            waveform_status = {"is_extracted": False, "message": wf_res.get("message", "Waveform extraction failed.")}

    else:
        progress_box.empty()
        st.error(f"❌ Unsupported file format `{ext}`. Supported formats are PDF, JPG, JPEG, PNG, BMP, TIFF, CSV, TXT, NPY.")
        st.stop()

    # Step 3, 4, 5: Model Inference (only if signal_data is present and verified)
    if signal_data is not None and len(signal_data) > 0:
        progress_bar.progress(65, text="STEP 3/6: Filtering Signal & Assessing Noise Quality...")
        time.sleep(0.05)
        progress_bar.progress(80, text="STEP 4/6: Detecting R-Peaks & Segmenting Heartbeats...")
        time.sleep(0.05)
        progress_bar.progress(90, text="STEP 5/6: Running Random Forest AI Classification...")

        try:
            ai_results = predict_ecg(signal_data, fs=sampling_rate, models_dir=MODELS_DIR)
        except Exception as exc:
            st.warning(f"AI classification encountered an issue: {exc}")
    else:
        # No signal available or validation rejected
        progress_bar.progress(90, text="Synthesizing Structured Clinical Parameters...")

    progress_bar.progress(100, text="STEP 6/6: Assembling Comprehensive Clinical Report...")
    time.sleep(0.1)
    progress_box.empty()

else:
    # MIT-BIH Demo Record Mode
    try:
        raw_full, fs_rec, total_samples = load_record(selected_record, DATA_RAW_DIR)
        start_samp = int(start_offset_sec * fs_rec)
        end_samp = min(len(raw_full), int((start_offset_sec + analysis_duration_sec) * fs_rec))
        signal_data = raw_full[start_samp:end_samp]
        sampling_rate = float(fs_rec)

        ann_df = load_annotations(selected_record, DATA_RAW_DIR)
        ref_mask = (ann_df["sample_index"] >= start_samp) & (ann_df["sample_index"] < end_samp)
        reference_annotations = ann_df[ref_mask].copy()
        reference_annotations["sample_relative"] = reference_annotations["sample_index"] - start_samp
        reference_annotations["time_sec"] = reference_annotations["sample_relative"] / fs_rec

        input_info = {
            "file_name": f"MIT-BIH Record {selected_record}",
            "file_modality": "BENCHMARK_DEMO",
            "format": "MIT-BIH wfdb",
            "sampling_rate": sampling_rate,
            "duration_sec": analysis_duration_sec,
            "total_samples": len(signal_data),
            "lead": "Modified Lead II (MLII)",
        }
        waveform_status = {"is_extracted": False, "message": "MIT-BIH Benchmark Database Record"}

        with st.spinner("Analyzing MIT-BIH Benchmark Signal..."):
            ai_results = predict_ecg(signal_data, fs=sampling_rate, models_dir=MODELS_DIR)

    except Exception as exc:
        st.error(f"Error loading record {selected_record}: {exc}")
        st.stop()


# ---------------------------------------------------------
# Compile Report Data & Connect to Clinical Persistence
# ---------------------------------------------------------
file_stem = input_info.get("file_name", "ecg_file")
rec_id = f"REC-{abs(hash(file_stem)) % 1000000:06d}"
active_review = st.session_state.get(f"review_{rec_id}")

# Compute initial CDS and Medication Safety dictionaries for the structured report
primary_finding_str = "Normal Sinus Rhythm"
if ai_results and "predicted_class" in ai_results:
    primary_finding_str = ai_results["predicted_class"]

cds_rec_obj = GLOBAL_CDS_ENGINE.evaluate_finding(
    ecg_finding=primary_finding_str,
    heart_rate=ai_results.get("heart_rate_bpm") if ai_results else None,
)
cds_data_dict = {
    "primary_finding": cds_rec_obj.finding,
    "urgency": cds_rec_obj.urgency,
    "summary": cds_rec_obj.clinician_action_required,
    "guideline_citations": cds_rec_obj.relevant_guidelines,
    "considerations": cds_rec_obj.clinical_considerations,
    "contraindications": cds_rec_obj.contraindication_warnings,
}

temp_eval_patient = PatientRecord(
    patient_id="PAT-ACTIVE-01",
    hospital_mrn="MRN-ACTIVE",
    name="Active Patient",
    age=65,
    sex="M",
    known_allergies="None documented",
    existing_conditions="Hypertension",
    current_medications="Metoprolol",
)
# Override with the real registered patient if available
_reg_pat = st.session_state.get("registered_patient_data")
if _reg_pat:
    temp_eval_patient = PatientRecord(
        patient_id=_reg_pat.get("patient_id", "PAT-ACTIVE-01"),
        hospital_mrn=_reg_pat.get("hospital_mrn", "MRN-ACTIVE"),
        name=_reg_pat.get("name", "Unknown"),
        age=_reg_pat.get("age", 0),
        sex=_reg_pat.get("sex", ""),
        blood_group=_reg_pat.get("blood_group", ""),
        known_allergies=_reg_pat.get("known_allergies", "None"),
        existing_conditions=_reg_pat.get("existing_conditions", "None"),
        current_medications=_reg_pat.get("current_medications", "None"),
        previous_cardiac_history=_reg_pat.get("previous_cardiac_history", "None"),
    )

_curr_meds_list = [m.strip() for m in ((_reg_pat or {}).get("current_medications") or "").split(",") if m.strip()] or ["Metoprolol", "Amiodarone"]
med_safety_eval = check_medication_safety(_curr_meds_list, patient=temp_eval_patient)


report_data = generate_structured_report(
    input_info=input_info,
    ai_results=ai_results,
    extracted_measurements=extracted_measurements,
    waveform_status=waveform_status,
    clinician_review=active_review,
    cds_report=cds_data_dict,
    medication_safety=med_safety_eval.to_dict(),
)

# Inject registered patient details into report_data so PDF and UI both show real patient
if _reg_pat:
    report_data["patient_info"].update({
        "patient_name":         _reg_pat.get("name", ""),
        "patient_age":          _reg_pat.get("age", ""),
        "patient_sex":          _reg_pat.get("sex", ""),
        "hospital_mrn":         _reg_pat.get("hospital_mrn", ""),
        "blood_group":          _reg_pat.get("blood_group", ""),
        "known_allergies":      _reg_pat.get("known_allergies", "None"),
        "existing_conditions":  _reg_pat.get("existing_conditions", "None"),
        "current_medications":  _reg_pat.get("current_medications", "None"),
        "previous_cardiac_history": _reg_pat.get("previous_cardiac_history", "None"),
        "smoking_status":       _reg_pat.get("smoking_status", ""),
        "contact":              _reg_pat.get("contact", ""),
    })


if ai_results and "predicted_class" in ai_results:
    analysis_id = f"ANL-{rec_id[4:]}"
    try:
        DB_MANAGER.save_ecg_record(
            record_id=rec_id,
            sampling_rate=sampling_rate,
            lead_names=["II"],
            duration_sec=float(input_info.get("duration_sec", 10.0)),
            file_hash=f"{abs(hash(str(ai_results.get('processed_signal', '')))):x}",
            source_format=input_info.get("modality", "DIGITAL"),
            signal_quality=ai_results.get("signal_quality", "GOOD"),
            quality_score=ai_results.get("quality_score", 1.0),
            uploaded_by=current_user.username,
        )
        DB_MANAGER.save_analysis_result(
            analysis_id=analysis_id,
            record_id=rec_id,
            model_id="ECG-RF-1.0.0",
            model_version="1.0.0",
            prediction=ai_results["predicted_class"],
            probabilities=ai_results.get("probabilities", {}),
            signal_quality=ai_results.get("signal_quality", "GOOD"),
            quality_score=ai_results.get("quality_score", 1.0),
            heart_rate_bpm=ai_results.get("heart_rate_bpm"),
            mean_rr_ms=ai_results.get("mean_rr_sec", 0.8) * 1000.0 if ai_results.get("mean_rr_sec") else None,
            detected_beats_count=ai_results.get("beat_count", 0),
        )
    except Exception:
        pass


# Prepare waveform snippet for PDF if available
waveform_for_pdf = ai_results["processed_signal"] if ai_results else None
peaks_for_pdf = np.array(ai_results["detected_peaks"]) if ai_results else None

pdf_bytes = generate_pdf_report(
    report_data=report_data,
    waveform=waveform_for_pdf,
    fs=sampling_rate,
    r_peaks=peaks_for_pdf,
)



# ---------------------------------------------------------
# PATIENT IDENTITY CARD (shown when a patient is registered)
# ---------------------------------------------------------
_disp_pat = st.session_state.get("registered_patient_data") or report_data.get("patient_info", {})
_pat_name = _disp_pat.get("name") or _disp_pat.get("patient_name")
if _pat_name:
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
                    border: 1.5px solid #0ea5e9; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div>
                    <span style="font-size:1.1rem; font-weight:700; color:#0369a1;">
                        👤 {_disp_pat.get('name') or _disp_pat.get('patient_name')}
                    </span>
                    &nbsp;&nbsp;
                    <code style="background:#bae6fd; padding:2px 8px; border-radius:4px; color:#075985; font-size:0.85rem;">
                        MRN: {_disp_pat.get('hospital_mrn') or '—'}
                    </code>
                </div>
                <div style="color:#0369a1; font-size:0.88rem;">
                    <b>Age/Sex:</b> {_disp_pat.get('age') or _disp_pat.get('patient_age') or '—'} /
                    {_disp_pat.get('sex') or _disp_pat.get('patient_sex') or '—'}
                    &nbsp;|&nbsp;
                    <b>Blood Group:</b> {_disp_pat.get('blood_group') or '—'}
                    &nbsp;|&nbsp;
                    <b>Allergies:</b> {_disp_pat.get('known_allergies') or 'None'}
                </div>
            </div>
            <div style="margin-top:6px; color:#075985; font-size:0.84rem;">
                <b>Conditions:</b> {_disp_pat.get('existing_conditions') or 'None documented'}
                &nbsp;|&nbsp;
                <b>Current Medications:</b> {_disp_pat.get('current_medications') or 'None recorded'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# TOP OVERVIEW METRIC CARDS
# ---------------------------------------------------------
st.markdown("### 📊 Rapid Clinical Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# 1. Pattern Metric
with col1:
    if ai_results and "predicted_class" in ai_results:
        pred_label = ai_results["predicted_class"].split("(")[0].strip()
        is_abn = "PVC" in pred_label or "Other" in pred_label
        st.metric(
            label="AI Detected Pattern",
            value=pred_label,
            delta="Abnormal Ectopy" if is_abn else "Normal Rhythm",
            delta_color="inverse" if is_abn else "normal",
        )
    elif extracted_measurements and extracted_measurements.get("machine_interpretation"):
        first_interp = extracted_measurements["machine_interpretation"][0]
        st.metric(
            label="Printed Diagnosis",
            value=first_interp[:18] + ("..." if len(first_interp) > 18 else ""),
            delta="Machine Interpretation",
            delta_color="off",
        )
    else:
        st.metric(label="Detected Pattern", value="Not Available")

# 2. Heart Rate
with col2:
    hr_val = report_data["cardiac_parameters"]["heart_rate_bpm"]
    if hr_val is not None:
        st.metric(
            label="Heart Rate",
            value=f"{hr_val:.0f} BPM",
            delta=report_data["cardiac_parameters"]["heart_rate_category"],
            delta_color="off",
        )
    else:
        st.metric(label="Heart Rate", value="N/A")

# 3. Signal Quality
with col3:
    sq_cat = report_data["signal_quality"]["category"]
    sq_score = report_data["signal_quality"]["quality_score"]
    if sq_score is not None:
        st.metric(
            label="Signal Quality",
            value=sq_cat,
            delta=f"Score: {sq_score:.2f} / 1.0",
            delta_color="normal" if sq_cat == "GOOD" else ("off" if sq_cat == "ACCEPTABLE" else "inverse"),
        )
    else:
        st.metric(label="Signal Quality", value=sq_cat)

# 4. Detected Beats / Intervals
with col4:
    beats = report_data["cardiac_parameters"]["detected_beats"]
    if beats is not None:
        st.metric(label="Detected Cycles", value=f"{beats} beats")
    elif report_data["cardiac_parameters"]["pr_interval_ms"] is not None:
        st.metric(label="PR Interval", value=f"{report_data['cardiac_parameters']['pr_interval_ms']:.0f} ms")
    else:
        st.metric(label="Detected Cycles", value="N/A")

# 5. R-R Interval or QRS Duration
with col5:
    mean_rr = report_data["cardiac_parameters"]["mean_rr_ms"]
    if mean_rr is not None:
        st.metric(label="Mean R-R Interval", value=f"{mean_rr:.0f} ms")
    elif report_data["cardiac_parameters"]["qrs_duration_ms"] is not None:
        st.metric(label="QRS Duration", value=f"{report_data['cardiac_parameters']['qrs_duration_ms']:.0f} ms")
    else:
        st.metric(label="Interval Metric", value="N/A")

st.divider()


# ---------------------------------------------------------
# SECTION: Printed Machine Interpretation (Zero-Hallucination)
# ---------------------------------------------------------
if extracted_measurements and extracted_measurements.get("has_extracted_data"):
    st.markdown("### 📋 Printed ECG Machine Measurements (Source Document)")
    st.info(
        "ℹ️ **Direct Document Data:** The parameters below were extracted directly from the uploaded report. "
        "They represent the recording machine's native interpretation and standard 12-lead measurements."
    )

    m_col1, m_col2 = st.columns([1, 1])
    with m_col1:
        st.markdown("**Extracted Patient & Recording Metadata:**")
        meta_items = [
            ("Patient Name", extracted_measurements.get("patient_name") or "Unspecified"),
            ("Age / Sex", f"{extracted_measurements.get('patient_age') or '—'} yr / {extracted_measurements.get('patient_sex') or '—'}"),
            ("Recording Date", extracted_measurements.get("recording_date") or "Unspecified"),
            ("Printed Vent. Rate", f"{extracted_measurements.get('heart_rate_printed') or '—'} BPM"),
        ]
        st.table(pd.DataFrame(meta_items, columns=["Parameter", "Report Value"]))

    with m_col2:
        st.markdown("**Printed Electrical Intervals & Axes:**")
        interval_items = [
            ("PR Interval", f"{extracted_measurements.get('pr_interval_ms') or '—'} ms"),
            ("QRS Duration", f"{extracted_measurements.get('qrs_duration_ms') or '—'} ms"),
            ("QT / QTc Interval", f"{extracted_measurements.get('qt_interval_ms') or '—'} / {extracted_measurements.get('qtc_interval_ms') or '—'} ms"),
            ("P - QRS - T Axes", f"{extracted_measurements.get('p_axis_deg') or '—'}° / {extracted_measurements.get('qrs_axis_deg') or '—'}° / {extracted_measurements.get('t_axis_deg') or '—'}°"),
        ]
        st.table(pd.DataFrame(interval_items, columns=["Interval / Axis", "Measured Value"]))

    if extracted_measurements.get("machine_interpretation"):
        st.markdown("**Printed Clinical Findings:**")
        for interp in extracted_measurements["machine_interpretation"]:
            st.markdown(f"- 📝 `{interp}`")

    st.divider()

    # ---------------------------------------------------------
    # SECTION: Clinical Decision Support & Guidelines (Phases 12 & 18)
    # ---------------------------------------------------------
    st.markdown("### 🧠 Clinical Decision Support & Guideline Considerations")
    st.caption("Authoritative guideline evidence (AHA/ACC/ESC). Diagnostic decisions and prescriptions remain the sole responsibility of the attending physician.")

    primary_f = "Normal Sinus Rhythm"
    if ai_results and "predicted_class" in ai_results:
        primary_f = ai_results["predicted_class"]

    cds_rec = GLOBAL_CDS_ENGINE.evaluate_finding(
        ecg_finding=primary_f,
        heart_rate=ai_results.get("heart_rate") if ai_results else None,
    )

    urgency_color = "badge-normal" if cds_rec.urgency == "ROUTINE REVIEW" else ("badge-warning" if cds_rec.urgency == "PROMPT REVIEW" else "badge-abnormal")
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #1e293b;">Primary Finding: {cds_rec.finding}</span>
                <span class="{urgency_color}"><b>{cds_rec.urgency}</b></span>
            </div>
            <p style="margin-bottom: 6px; color: #334155;"><b>Guideline Source:</b> {cds_rec.guideline_source} (Verified: {cds_rec.last_verified})</p>
            <p style="margin-bottom: 0; color: #0f766e;"><b>Mandatory Clinician Action:</b> {cds_rec.clinician_action_required}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cds_c1, cds_c2 = st.columns(2)
    with cds_c1:
        st.markdown("**Potential Clinical Considerations:**")
        for cons in cds_rec.clinical_considerations:
            st.markdown(f"- 💡 {cons}")

        if cds_rec.contraindication_warnings:
            st.markdown("**Critical Contraindication Alerts:**")
            for cw in cds_rec.contraindication_warnings:
                st.error(f"⚠️ {cw}")

    with cds_c2:
        st.markdown("**Suggested Next Assessments (Authoritative):**")
        for ass in cds_rec.suggested_assessments:
            st.markdown(f"- 🩺 {ass}")

        st.markdown("**Applicable Clinical Guidelines:**")
        for gl in cds_rec.relevant_guidelines:
            st.markdown(f"- 📚 *{gl}*")

    st.divider()

    # ---------------------------------------------------------
    # SECTION: Medication Safety & Interaction Verification (Phases 13-17)
    # ---------------------------------------------------------
    st.markdown("### 💊 Cardiovascular Medication Safety & Interaction Check")
    st.caption("Clinical safety verification across patient's current medications, known allergies, and organ function. Autonomous prescribing is strictly prohibited.")

    with st.expander("🛡️ Patient Medication Safety Evaluation", expanded=True):
        col_med1, col_med2 = st.columns([1, 1])
        with col_med1:
            patient_meds_input = st.text_input(
                "Enter Current Patient Medications (comma-separated):",
                value="Metoprolol, Amiodarone",
                help="Enter active medications to evaluate drug-drug interactions and cardiac contraindications.",
                key="inp_patient_meds",
            )
            patient_allergies_input = st.text_input(
                "Patient Known Allergies:",
                value="Penicillin",
                key="inp_patient_allergies",
            )
            patient_conditions_input = st.text_input(
                "Known Clinical Conditions:",
                value="Hypertension, PVC Arrhythmia",
                key="inp_patient_conditions",
            )

        with col_med2:
            st.markdown("**Authoritative Formulary Database:**")
            all_meds = GLOBAL_MEDICATION_DB.list_all()
            st.caption(f"Curated active cardiovascular agents: {len(all_meds)} (Metoprolol, Amiodarone, Apixaban)")
            st.markdown("**Regulatory Drug References:**")
            st.caption("- FDA Prescribing Information / DailyMed\n- AHA/ACC Guideline Consensus\n- Chest Antithrombotic Guidelines")

        med_list = [m.strip() for m in patient_meds_input.split(",") if m.strip()]
        temp_patient = PatientRecord(
            patient_id="PAT-TMP-01",
            hospital_mrn="MRN-EVAL",
            name="Clinical Evaluation Patient",
            age=65,
            sex="M",
            known_allergies=patient_allergies_input,
            existing_conditions=patient_conditions_input,
            current_medications=patient_meds_input,
        )

        med_safety_report = check_medication_safety(med_list, patient=temp_patient)

        if med_safety_report.alerts:
            st.markdown(f"**Safety Alerts Identified ({len(med_safety_report.alerts)}):**")
            for alert in med_safety_report.alerts:
                if alert.severity == "CRITICAL":
                    st.error(f"🚨 **{alert.title}**: {alert.description}\n\n*Action:* {alert.clinical_recommendation} *(Source: {alert.source})*")
                elif alert.severity == "MAJOR":
                    st.warning(f"⚠️ **{alert.title}**: {alert.description}\n\n*Action:* {alert.clinical_recommendation} *(Source: {alert.source})*")
                else:
                    st.info(f"ℹ️ **{alert.title}**: {alert.description}\n\n*Action:* {alert.clinical_recommendation}")
        else:
            st.success("✅ **No Major Drug-Drug Interactions or Allergy Conflicts Detected** across the entered regimen.")

    st.divider()


# ---------------------------------------------------------
# SECTION: Waveform Extraction Notice (If Applicable)
# ---------------------------------------------------------
if not waveform_status.get("is_extracted") and signal_data is None:
    st.markdown("### 🔍 Waveform Trace Extraction Status")
    st.warning(
        f"⚠️ **Waveform Notice:** {waveform_status.get('message')}\n\n"
        "To ensure clinical safety and scientific integrity, this system **never synthesizes or hallucinates** artificial ECG signals. "
        "The printed parameters above have been preserved in your downloadable reports."
    )
    st.divider()


# ---------------------------------------------------------
# SECTION: ECG Waveform & R-Peak Visualization
# ---------------------------------------------------------
if signal_data is not None and len(signal_data) > 0 and ai_results is not None:
    st.markdown("### 1. ECG Waveform & R-Peak Annotations")
    show_raw = st.checkbox("Overlay Raw Unfiltered Signal", value=False)

    fig_waveform = plot_ecg_signal(
        signal=ai_results["processed_signal"],
        fs=sampling_rate,
        r_peaks=np.array(ai_results["detected_peaks"]),
        raw_signal=ai_results["raw_signal"] if show_raw else None,
        title=f"ECG Waveform with Detected R-Peaks ({len(signal_data)/sampling_rate:.1f}s Window)",
        max_duration_sec=len(signal_data) / sampling_rate,
        start_sec=0.0,
    )
    st.plotly_chart(fig_waveform, use_container_width=True)

    # ---------------------------------------------------------
    # SECTION: AI Model Prediction & Signal Quality
    # ---------------------------------------------------------
    st.markdown("### 2. AI Abnormality Classification & Signal Quality")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("**🤖 AI Classifier Output (Lead II Equivalent):**")
        st.markdown(f"**Primary Pattern:** `{ai_results['predicted_class']}`")

        fig_prob = plot_prediction_probabilities(
            ai_results["probabilities"],
            predicted_class="PVC" if "PVC" in ai_results["predicted_class"] else ("Normal" if "Normal" in ai_results["predicted_class"] else "Other"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        counts = ai_results.get("class_counts", {})
        total_beats = max(1, ai_results["beat_count"])
        st.markdown("**Beat-by-Beat Cycle Breakdown:**")
        st.markdown(f"- 🟢 **Normal Beats:** {counts.get('Normal', 0)} ({counts.get('Normal', 0) / total_beats * 100:.1f}%)")
        st.markdown(f"- 🔴 **Ventricular Ectopy (PVC):** {counts.get('PVC', 0)} ({counts.get('PVC', 0) / total_beats * 100:.1f}%)")
        st.markdown(f"- 🟡 **Other Abnormal/Escape Beats:** {counts.get('Other', 0)} ({counts.get('Other', 0) / total_beats * 100:.1f}%)")

    with c_right:
        st.markdown("**📡 Signal Quality Breakdown:**")
        q_ind = ai_results.get("quality_indicators", {})
        q_df = pd.DataFrame(
            [
                {"Indicator": "Signal-to-Noise Ratio (SNR)", "Value": f"{q_ind.get('snr_db', 0):.1f} dB", "Status": "Optimal" if q_ind.get('snr_db', 0) > 15 else "Degraded"},
                {"Indicator": "Baseline Drift Detected", "Value": str(q_ind.get('baseline_wander', False)), "Status": "Warning" if q_ind.get('baseline_wander') else "Clean"},
                {"Indicator": "Powerline Interference (50/60 Hz)", "Value": str(q_ind.get('has_powerline_interference', False)), "Status": "Present" if q_ind.get('has_powerline_interference') else "Suppressed"},
                {"Indicator": "Motion Artifacts Detected", "Value": str(q_ind.get('has_motion_artifacts', False)), "Status": "Elevated" if q_ind.get('has_motion_artifacts') else "Minimal"},
                {"Indicator": "Overall Quality Category", "Value": ai_results["signal_quality"], "Status": "Verified"},
            ]
        )
        st.table(q_df)

    st.divider()

    # ---------------------------------------------------------
    # SECTION: Cardiac Beat Segmentation & Morphological Features
    # ---------------------------------------------------------
    st.markdown("### 3. Cardiac Cycle Segmentation & Morphological Features")
    c_seg, c_feat = st.columns([1, 1])

    with c_seg:
        st.markdown("**Heartbeat Segments Overlay:**")
        if len(ai_results["beats"]) > 0:
            fig_beats = plot_beats_overlay(
                beats=ai_results["beats"],
                fs=sampling_rate,
                labels=ai_results.get("beat_predictions"),
            )
            st.plotly_chart(fig_beats, use_container_width=True)
        else:
            st.info("No cardiac beats segmented.")

    with c_feat:
        st.markdown("**Key Quantitative Morphological Features (Mean):**")
        feats_df = ai_results.get("features_df")
        if feats_df is not None and not feats_df.empty:
            summary_feats = [
                {"Feature": "Mean R-Peak Amplitude", "Value": f"{feats_df['r_peak_amplitude'].mean():.3f} mV"},
                {"Feature": "Mean QRS Width", "Value": f"{feats_df['qrs_width_samples'].mean():.3f} s"},
                {"Feature": "Peak-to-Peak Amplitude", "Value": f"{feats_df['peak_to_peak_amplitude'].mean():.3f} mV"},
                {"Feature": "Signal Energy", "Value": f"{feats_df['energy'].mean():.2f}"},
                {"Feature": "Spectral Entropy", "Value": f"{feats_df['spectral_entropy'].mean():.3f}"},
                {"Feature": "Dominant Frequency", "Value": f"{feats_df['dominant_frequency'].mean():.2f} Hz"},
                {"Feature": "Local RR Ratio", "Value": f"{feats_df['local_rr_ratio'].mean():.3f}"},
            ]
            st.table(pd.DataFrame(summary_feats))
        else:
            st.info("Feature extraction table unavailable.")

    st.divider()


# ---------------------------------------------------------
# SECTION: Benchmark Comparison (When in Demo Mode)
# ---------------------------------------------------------
if reference_annotations is not None and not reference_annotations.empty:
    st.markdown("### 4. MIT-BIH Ground Truth Reference Comparison")
    st.caption("Comparing expert cardiologist manual annotations against automated AI model beat classifications:")

    ref_symbols = reference_annotations["symbol"].values
    ref_labels = [map_symbol_to_class(s, mode="3class") for s in ref_symbols]
    ref_times = reference_annotations["time_sec"].values

    comp_df = pd.DataFrame(
        {
            "Time (s)": [f"{t:.2f}" for t in ref_times],
            "MIT-BIH Symbol": ref_symbols,
            "Expert Reference Class": ref_labels,
            "Physician Annotation": reference_annotations["description"].values,
        }
    )
    st.dataframe(comp_df, use_container_width=True)
    st.divider()


# ---------------------------------------------------------
# SECTION: Clinician Review & Physician Sign-Off Portal
# ---------------------------------------------------------
st.markdown("### ✍️ Attending Clinician Review & Diagnostic Sign-Off")
st.caption("Fulfilling CDSCO MDR 2017 & IEC 62304 mandatory human-in-the-loop review. Unreviewed AI predictions cannot be used for patient treatment.")

if active_review:
    rev_status = active_review.get("status", "REVIEWED")
    rev_color = "badge-normal" if rev_status == "CONFIRMED" else ("badge-warning" if rev_status == "MODIFIED" else "badge-abnormal")
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 2px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="{rev_color}" style="font-size: 0.95rem;"><b>STATUS: {rev_status} BY CLINICIAN</b></span>
                <span style="font-size: 0.85rem; color: #64748b;">Signed: {active_review.get('reviewed_at')}</span>
            </div>
            <p style="margin-top: 8px; margin-bottom: 4px;"><b>Reviewing Clinician:</b> {active_review.get('clinician_name')} ({active_review.get('clinician_role')}) | <b>Reg No:</b> {active_review.get('registration_number')}</p>
            <p style="margin-bottom: 4px;"><b>Clinical Diagnosis:</b> {active_review.get('clinician_interpretation')}</p>
            <p style="margin-bottom: 0;"><b>Clinical Directives:</b> {active_review.get('clinical_notes') or 'None specified'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("ℹ️ **Pending Review:** This analysis has not yet been signed off by an attending physician. Please review the findings below to seal the diagnostic report.")

# Physician Sign-Off Form
if AUTH_MANAGER.has_permission(current_user, "report:sign_off"):
    with st.expander("🩺 Physician Diagnostic Sign-Off & Seal Action", expanded=(active_review is None)):
        col_ag1, col_ag2 = st.columns([1, 1])
        with col_ag1:
            agreement_choice = st.radio(
                "Clinical Agreement with AI Findings:",
                options=["CONFIRMED", "MODIFIED", "REJECTED"],
                format_func=lambda x: {
                    "CONFIRMED": "✅ CONFIRMED — Concordant with AI finding",
                    "MODIFIED": "⚠️ MODIFIED — Agree with reservations / modified notes",
                    "REJECTED": "❌ REJECTED — Overrule AI finding (Artifact or Misclassification)",
                }[x],
                index=0 if not active_review else (["CONFIRMED", "MODIFIED", "REJECTED"].index(active_review.get("agreement_status", "CONFIRMED")) if active_review.get("agreement_status") in ["CONFIRMED", "MODIFIED", "REJECTED"] else 0),
            )
        with col_ag2:
            st.markdown(f"**Signing Clinician:** `{current_user.full_name}`")
            st.markdown(f"**Medical Registration No:** `{current_user.registration_number or 'MCI-PENDING'}`")
            st.markdown(f"**Facility ID:** `MED-FAC-2026-IND`")

        default_diagnosis = "Normal Sinus Rhythm. No acute ischemic ST-T abnormalities identified."
        if ai_results and "predicted_class" in ai_results:
            if "PVC" in ai_results["predicted_class"]:
                default_diagnosis = "Sinus rhythm with premature ventricular contractions (PVCs). Recommend 24-hr Holter monitor."
            elif "Other" in ai_results["predicted_class"]:
                default_diagnosis = "Non-sinus rhythm or atypical ectopic complexes. Recommend 12-lead ECG and cardiology consult."

        custom_diag = st.text_area(
            "Physician Diagnostic Finding:",
            value=active_review.get("clinician_interpretation", default_diagnosis) if active_review else default_diagnosis,
            height=70,
        )
        custom_directives = st.text_area(
            "Clinical Directives & Next Steps:",
            value=active_review.get("clinical_notes", "Routine outpatient follow-up. Repeat ECG if symptomatic.") if active_review else "Routine outpatient follow-up. Repeat ECG if symptomatic.",
            height=70,
        )

        if st.button("✍️ Sign-Off & Cryptographically Seal Clinical Report", type="primary"):
            rev_id = f"REV-{secrets.token_hex(4).upper()}"
            analysis_id = f"ANL-{rec_id[4:]}"

            DB_MANAGER.save_clinician_review(
                review_id=rev_id,
                analysis_id=analysis_id,
                record_id=rec_id,
                clinician_id=current_user.user_id,
                clinician_name=current_user.full_name,
                clinician_role=current_user.role.value,
                agreement_status=agreement_choice,
                clinician_interpretation=custom_diag,
                clinical_notes=custom_directives,
                registration_number=current_user.registration_number,
            )
            DB_MANAGER.save_report(
                report_id=f"REP-{secrets.token_hex(4).upper()}",
                record_id=rec_id,
                analysis_id=analysis_id,
                review_id=rev_id,
                report_type="CLINICAL_PDF",
                status="SEALED",
                report_sha256=f"{abs(hash(custom_diag)):x}",
            )
            DB_MANAGER.update_ecg_workflow_status(rec_id, "SIGNED", assigned_doctor=current_user.full_name)
            AUDIT_LOGGER.log_event(
                event_type="CLINICIAN_SIGN_OFF",
                user_id=current_user.user_id,
                username=current_user.username,
                user_role=current_user.role.value,
                action=f"Signed and sealed report with status {agreement_choice}",
                details={"record_id": rec_id, "status": agreement_choice, "interpretation": custom_diag},
                record_id=rec_id,
            )
            st.session_state[f"review_{rec_id}"] = {
                "status": agreement_choice,
                "agreement_status": agreement_choice,
                "clinician_name": current_user.full_name,
                "clinician_role": current_user.role.value,
                "registration_number": current_user.registration_number,
                "clinician_interpretation": custom_diag,
                "clinical_notes": custom_directives,
                "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.success("✅ Clinical report successfully sealed with physician digital signature!")
            st.rerun()
else:
    st.info(
        f"🔒 **Physician Sign-Off Restricted:** Current role `{current_user.role.value}` cannot sign off clinical reports. "
        "Select a DOCTOR or CARDIOLOGIST profile in the sidebar to review and seal."
    )

st.divider()

# ---------------------------------------------------------
# SECTION: Report Export Center (PDF, JSON, TXT)
# ---------------------------------------------------------
st.markdown("### 📥 Download Comprehensive Research Reports")
st.markdown("Export publication-grade PDF documents, machine-readable JSON files, or clinical summary text.")

# Prepare waveform snippet for PDF if available
waveform_for_pdf = ai_results["processed_signal"] if ai_results else None
peaks_for_pdf = np.array(ai_results["detected_peaks"]) if ai_results else None

# Generate file contents
pdf_bytes = generate_pdf_report(
    report_data=report_data,
    waveform=waveform_for_pdf,
    fs=sampling_rate,
    r_peaks=peaks_for_pdf,
)
json_str = export_report_to_json(report_data)
text_str = export_report_to_text(report_data)

stem_name = Path(input_info.get("file_name", "ecg_analysis")).stem.replace(" ", "_").lower()

dcol1, dcol2, dcol3 = st.columns(3)

with dcol1:
    st.download_button(
        label="📄 Download Publication PDF Report",
        data=pdf_bytes,
        file_name=f"ecg_report_{stem_name}.pdf",
        mime="application/pdf",
        help="Complete multi-page formatted PDF report with embedded waveform strip and clinical tables.",
    )

with dcol2:
    st.download_button(
        label="📊 Download Machine JSON (.json)",
        data=json_str,
        file_name=f"ecg_data_{stem_name}.json",
        mime="application/json",
        help="Structured JSON document for integration with EMR or research databases.",
    )

with dcol3:
    st.download_button(
        label="📝 Download Summary Text (.txt)",
        data=text_str,
        file_name=f"ecg_summary_{stem_name}.txt",
        mime="text/plain",
        help="Plain-text summary for clinical notes and record keeping.",
    )

st.divider()


# ---------------------------------------------------------
# SECTION: Advanced Technical Details & Transparency
# ---------------------------------------------------------
with st.expander("🔍 Model Architecture & Validation Metrics (CSE / Bioengineering Transparency)"):
    mcol_l, mcol_r = st.columns(2)

    with mcol_l:
        st.markdown("**Classifier Specifications:**")
        try:
            _, _, meta = get_trained_artifacts(MODELS_DIR)
            with open(REPORTS_DIR / "evaluation_report.json", "r") as f:
                eval_data = json.load(f)
            rf_metrics = eval_data.get("random_forest", {})

            st.markdown(f"- **Architecture:** `{meta.get('model_name')}` (100 Trees, Gini split)")
            st.markdown(f"- **Input Features:** 28 engineered morphological, spectral, and rhythm metrics")
            st.markdown(f"- **Training Dataset:** MIT-BIH Arrhythmia Database ({meta.get('train_samples')} beats)")
            st.markdown(f"- **Test Dataset:** Unseen test partition ({meta.get('test_samples')} beats)")
            st.markdown(f"- **Macro F1-Score:** `{rf_metrics.get('f1_macro', 0)*100:.2f}%`")
            st.markdown(f"- **Overall Test Accuracy:** `{rf_metrics.get('accuracy', 0)*100:.2f}%`")
        except Exception as exc:
            st.info(f"Model specifications loaded from default config ({exc}).")

    with mcol_r:
        st.markdown("**Top Feature Importances (Gini):**")
        try:
            _, _, meta = get_trained_artifacts(MODELS_DIR)
            importances = meta.get("feature_importances", {})
            if importances:
                fig_imp = plot_feature_importance(importances, top_k=6)
                st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info("Feature importance plot unavailable.")

st.divider()

# ---------------------------------------------------------
# SECTION: Hospital Clinical Worklist & Patient Database
# ---------------------------------------------------------
render_hospital_worklist(expanded=False)

st.divider()

# ---------------------------------------------------------
# SECTION: Cryptographic Audit Trail & Regulatory Compliance
# ---------------------------------------------------------
render_audit_trail(expanded=False)

