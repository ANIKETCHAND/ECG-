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

# Page configuration
st.set_page_config(
    page_title="AI ECG Analyzer — Universal Clinical & Research System",
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
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.title("❤️ AI ECG Analyzer")
st.sidebar.markdown("**Universal Multi-Format Ingestion System**")
st.sidebar.divider()

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
        index=2,  # 360 Hz
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
# Compile Report Data
# ---------------------------------------------------------
report_data = generate_structured_report(
    input_info=input_info,
    ai_results=ai_results,
    extracted_measurements=extracted_measurements,
    waveform_status=waveform_status,
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
