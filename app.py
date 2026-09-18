"""
AI ECG Analyzer - Interactive Streamlit Dashboard
==================================================

Phase 16: Clinical Research & Educational Dashboard for:
1. ECG Preprocessing & Signal Quality Assessment
2. R-Peak Detection & Heartbeat Segmentation
3. Machine Learning Abnormality Classification (Normal vs PVC vs Other)
4. Feature Importance & Model Transparency
5. Research Demo Mode with Expert MIT-BIH Ground Truth Annotations

Research/Educational use only. Not for medical diagnosis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup system path
SRC_DIR = Path(__file__).parent / "src"
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

# Page configuration
st.set_page_config(
    page_title="AI ECG Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_RAW_DIR = Path(__file__).parent / "data" / "raw"
MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("❤️ AI ECG Analyzer")
st.sidebar.markdown("**Bioengineering + CSE Educational Prototype**")
st.sidebar.divider()

# Mode selection
input_source = st.sidebar.radio(
    "Select ECG Input Source",
    options=["Demo Record (MIT-BIH)", "Upload Custom ECG Signal (.csv / .npy)"],
    index=0,
)

available_records = get_available_records(DATA_RAW_DIR)
if not available_records:
    available_records = ["100", "101", "106", "119", "200", "208", "213"]

selected_record = None
uploaded_file = None

if input_source == "Demo Record (MIT-BIH)":
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
        "Select MIT-BIH ECG Record",
        options=record_options,
        format_func=lambda x: record_descriptions.get(x, f"Record {x}"),
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload 1D ECG Signal (.csv or .npy)",
        type=["csv", "npy", "txt"],
    )

st.sidebar.divider()
st.sidebar.subheader("⚙️ Signal & Analysis Settings")

sampling_rate = st.sidebar.number_input(
    "Sampling Rate (Hz)",
    min_value=50,
    max_value=1000,
    value=360,
    step=10,
    help="MIT-BIH recordings are sampled at 360 Hz.",
)

start_offset_sec = st.sidebar.slider(
    "Signal Start Offset (seconds)",
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
    "⚠️ **Educational & Research Notice**\n"
    "This software is developed strictly for educational and scientific research purposes. "
    "It is NOT a certified medical diagnostic device."
)

# ---------------------------------------------------------
# Load Signal
# ---------------------------------------------------------
signal_data = None
reference_annotations = None
record_meta_info = {}

if input_source == "Demo Record (MIT-BIH)" and selected_record:
    try:
        raw_full, fs_rec, total_samples = load_record(selected_record, DATA_RAW_DIR)
        start_samp = int(start_offset_sec * fs_rec)
        end_samp = min(len(raw_full), int((start_offset_sec + analysis_duration_sec) * fs_rec))
        signal_data = raw_full[start_samp:end_samp]
        sampling_rate = fs_rec

        # Load expert annotations for demo comparison
        ann_df = load_annotations(selected_record, DATA_RAW_DIR)
        ref_mask = (ann_df["sample_index"] >= start_samp) & (ann_df["sample_index"] < end_samp)
        reference_annotations = ann_df[ref_mask].copy()
        reference_annotations["sample_relative"] = reference_annotations["sample_index"] - start_samp
        reference_annotations["time_sec"] = reference_annotations["sample_relative"] / fs_rec

        record_meta_info = {
            "Record ID": selected_record,
            "Total Record Duration": f"{total_samples / fs_rec:.1f} s",
            "Analyzed Window": f"{start_offset_sec:.1f}s – {start_offset_sec + analysis_duration_sec:.1f}s",
            "Sampling Rate": f"{fs_rec} Hz",
        }
    except Exception as exc:
        st.error(f"Error loading record {selected_record}: {exc}")

elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".npy"):
            signal_data = np.load(uploaded_file).flatten()
        elif uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
            df_up = pd.read_csv(uploaded_file, header=None)
            # Find first numeric column
            signal_data = df_up.iloc[:, 0].dropna().values.astype(float)

        max_samples = int(analysis_duration_sec * sampling_rate)
        signal_data = signal_data[:max_samples]
        record_meta_info = {
            "File Name": uploaded_file.name,
            "Analyzed Samples": len(signal_data),
            "Sampling Rate": f"{sampling_rate} Hz",
        }
    except Exception as exc:
        st.error(f"Error parsing uploaded file: {exc}")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("❤️ AI ECG Analyzer")
st.markdown(
    "#### Real-Time ECG Signal Quality Assessment & Cardiac Abnormality Detection"
)

st.warning(
    "**Disclaimer:** This application is intended for **educational and research purposes only**. "
    "It is not a certified medical diagnostic device and must not be used to make medical decisions or clinical diagnoses."
)

if signal_data is None or len(signal_data) == 0:
    st.info("👈 Please select a demo record or upload an ECG recording from the sidebar to begin analysis.")
    st.stop()

# Run Prediction Pipeline
with st.spinner("Analyzing ECG Signal (Preprocessing, Peak Detection, Feature Extraction, Model Inference)..."):
    try:
        results = predict_ecg(signal_data, fs=sampling_rate, models_dir=MODELS_DIR)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# ---------------------------------------------------------
# TOP METRIC BAR
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

pred_label = results["predicted_class"]
is_abnormal = "PVC" in pred_label or "Other" in pred_label

with col1:
    st.metric(
        label="Predicted Pattern",
        value=results["predicted_class"].split("(")[0].strip(),
        delta="Abnormal Pattern" if is_abnormal else "Normal Rhythm",
        delta_color="inverse" if is_abnormal else "normal",
    )

with col2:
    q_label = results["signal_quality"]
    st.metric(
        label="Signal Quality",
        value=q_label,
        delta=f"Score: {results['quality_score']:.2f}",
        delta_color="normal" if q_label == "GOOD" else ("off" if q_label == "ACCEPTABLE" else "inverse"),
    )

with col3:
    st.metric(
        label="Estimated Heart Rate",
        value=f"{results['heart_rate_bpm']:.1f} BPM",
        help="Calculated from detected R-R intervals.",
    )

with col4:
    st.metric(
        label="Mean R-R Interval",
        value=f"{results['mean_rr_sec']:.3f} s",
        help="Average duration between consecutive heartbeats.",
    )

with col5:
    st.metric(
        label="Detected Beats",
        value=f"{results['beat_count']} beats",
        help="Number of cardiac cycles detected in current window.",
    )

st.divider()

# ---------------------------------------------------------
# SECTION 1: ECG SIGNAL VISUALIZATION
# ---------------------------------------------------------
st.subheader("1. ECG Waveform & R-Peak Detection")

show_raw = st.checkbox("Overlay Raw Unfiltered Signal", value=False)
fig_waveform = plot_ecg_signal(
    signal=results["processed_signal"],
    fs=sampling_rate,
    r_peaks=np.array(results["detected_peaks"]),
    raw_signal=results["raw_signal"] if show_raw else None,
    title=f"Preprocessed ECG Waveform with Detected R-Peaks ({analysis_duration_sec}s Window)",
    max_duration_sec=analysis_duration_sec,
    start_sec=0.0,
)
st.plotly_chart(fig_waveform, use_container_width=True)

# ---------------------------------------------------------
# SECTION 2 & 3: SIGNAL QUALITY & AI PREDICTION
# ---------------------------------------------------------
c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("2. Signal Quality Assessment")
    q_indicators = results.get("quality_indicators", {})

    st.write(f"**Overall Quality Category:** `{results['signal_quality']}` (Confidence Score: `{results['quality_score']:.2f}` / 1.00)")

    q_df = pd.DataFrame(
        [
            {"Metric": "Signal-to-Noise Ratio (SNR)", "Measured Value": f"{q_indicators.get('snr_db', 0):.2f} dB", "Status": "Optimal" if q_indicators.get('snr_db', 0) > 15 else "Degraded"},
            {"Metric": "Baseline Drift Detected", "Measured Value": str(q_indicators.get('baseline_wander', False)), "Status": "Warning" if q_indicators.get('baseline_wander') else "Normal"},
            {"Metric": "Low-Freq Power (<2 Hz)", "Measured Value": f"{q_indicators.get('lf_power', 0):.4f}", "Status": "Acceptable"},
            {"Metric": "Powerline Interference (50/60 Hz)", "Measured Value": str(q_indicators.get('has_powerline_interference', False)), "Status": "Present" if q_indicators.get('has_powerline_interference') else "Clean"},
            {"Metric": "Motion Artifacts Ratio", "Measured Value": f"{q_indicators.get('artifact_outlier_ratio', 0)*100:.1f}%", "Status": "Elevated" if q_indicators.get('has_motion_artifacts') else "Minimal"},
        ]
    )
    st.table(q_df)

with c_right:
    st.subheader("3. AI Abnormality Prediction")
    st.write(f"**Primary Classification:** `{results['predicted_class']}`")

    fig_prob = plot_prediction_probabilities(
        results["probabilities"],
        predicted_class="PVC" if "PVC" in results["predicted_class"] else ("Normal" if "Normal" in results["predicted_class"] else "Other"),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    counts = results.get("class_counts", {})
    st.write("**Beat-by-Beat Class Breakdown:**")
    st.write(f"- 🟢 **Normal Beats:** {counts.get('Normal', 0)} ({counts.get('Normal', 0) / max(1, results['beat_count']) * 100:.1f}%)")
    st.write(f"- 🔴 **Ventricular Ectopy (PVC):** {counts.get('PVC', 0)} ({counts.get('PVC', 0) / max(1, results['beat_count']) * 100:.1f}%)")
    st.write(f"- 🟡 **Other Abnormal/Escape Beats:** {counts.get('Other', 0)} ({counts.get('Other', 0) / max(1, results['beat_count']) * 100:.1f}%)")

st.divider()

# ---------------------------------------------------------
# SECTION 4 & 5: BEAT SEGMENTATION & EXTRACTED FEATURES
# ---------------------------------------------------------
c_seg, c_feat = st.columns([1, 1])

with c_seg:
    st.subheader("4. Heartbeat Segments Overlay")
    st.write("Extracted individual cardiac cycles aligned at the R-peak (±0.2s pre, +0.4s post):")
    if len(results["beats"]) > 0:
        fig_beats = plot_beats_overlay(
            beats=results["beats"],
            fs=sampling_rate,
            labels=results.get("beat_predictions"),
        )
        st.plotly_chart(fig_beats, use_container_width=True)
    else:
        st.info("No beat segments available.")

with c_feat:
    st.subheader("5. Extracted Morphological Features")
    st.write("Sample of quantitative ECG features computed for classification:")
    feats_df = results["features_df"]
    if not feats_df.empty:
        summary_feats = {
            "Mean R-Peak Amplitude": f"{feats_df['r_peak_amplitude'].mean():.3f}",
            "Mean QRS Width": f"{feats_df['qrs_width_samples'].mean():.3f} s",
            "Mean Peak-to-Peak": f"{feats_df['peak_to_peak_amplitude'].mean():.3f}",
            "Mean Energy": f"{feats_df['energy'].mean():.2f}",
            "Spectral Entropy": f"{feats_df['spectral_entropy'].mean():.3f}",
            "Dominant Frequency": f"{feats_df['dominant_frequency'].mean():.2f} Hz",
            "Mean Local RR Ratio": f"{feats_df['local_rr_ratio'].mean():.3f}",
        }
        st.table(pd.DataFrame(list(summary_feats.items()), columns=["Feature Name", "Average Value"]))
    else:
        st.info("No feature table available.")

st.divider()

# ---------------------------------------------------------
# SECTION 6 & 7: MODEL INFO & FEATURE IMPORTANCES
# ---------------------------------------------------------
col_mod_info, col_feat_imp = st.columns([1, 1])

with col_mod_info:
    st.subheader("6. Model Architecture & Metrics")
    try:
        _, _, meta = get_trained_artifacts(MODELS_DIR)
        with open(REPORTS_DIR / "evaluation_report.json", "r") as f:
            eval_data = json.load(f)

        rf_metrics = eval_data.get("random_forest", {})

        st.markdown(f"**Trained Classifier:** `{meta.get('model_name')}` (100 Trees, Gini split)")
        st.markdown(f"**Training Dataset:** `{meta.get('dataset')}`")
        st.markdown(f"**Training Records:** `{', '.join(meta.get('train_records', []))}` ({meta.get('train_samples')} beats)")
        st.markdown(f"**Unseen Test Records:** `{', '.join(meta.get('test_records', []))}` ({meta.get('test_samples')} beats)")

        metrics_table = [
            {"Metric": "Test Accuracy", "Score": f"{rf_metrics.get('accuracy', 0)*100:.2f}%"},
            {"Metric": "Macro Precision", "Score": f"{rf_metrics.get('precision_macro', 0)*100:.2f}%"},
            {"Metric": "Macro Recall", "Score": f"{rf_metrics.get('recall_macro', 0)*100:.2f}%"},
            {"Metric": "Macro F1-Score", "Score": f"{rf_metrics.get('f1_macro', 0)*100:.2f}%"},
            {"Metric": "Weighted F1-Score", "Score": f"{rf_metrics.get('f1_weighted', 0)*100:.2f}%"},
        ]
        st.table(pd.DataFrame(metrics_table))
    except Exception as exc:
        st.info(f"Model metadata available once training evaluation completes ({exc}).")

with col_feat_imp:
    st.subheader("7. Top Feature Importances")
    st.write("Ranking of morphological and rhythm features driving classifier decisions:")
    try:
        _, _, meta = get_trained_artifacts(MODELS_DIR)
        importances = meta.get("feature_importances", {})
        if importances:
            fig_imp = plot_feature_importance(importances, top_k=8)
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Feature importance plot unavailable.")

st.divider()

# ---------------------------------------------------------
# SECTION 8: RESEARCH DEMO MODE (Reference Annotation vs AI)
# ---------------------------------------------------------
if reference_annotations is not None and not reference_annotations.empty:
    st.subheader("8. Research Demo Mode: MIT-BIH Reference Annotation Comparison")
    st.write("Comparing expert physician annotations against the automated AI classifier predictions:")

    # Map annotations to 3class labels
    ref_symbols = reference_annotations["symbol"].values
    ref_labels = [map_symbol_to_class(s, mode="3class") for s in ref_symbols]
    ref_times = reference_annotations["time_sec"].values

    demo_df = pd.DataFrame(
        {
            "Time (s)": [f"{t:.2f}" for t in ref_times],
            "Original Symbol": ref_symbols,
            "Reference Class": ref_labels,
            "Physician Annotation": reference_annotations["description"].values,
        }
    )
    st.dataframe(demo_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# SECTION 9: REPORT EXPORT
# ---------------------------------------------------------
st.subheader("9. Export Comprehensive Research Report")

report_text = f"""# AI ECG Analyzer — Research & Educational Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Recording Details
- Source: {input_source} {f'({selected_record})' if selected_record else ''}
- Sampling Rate: {sampling_rate} Hz
- Window Analyzed: {analysis_duration_sec} seconds
- Total Beats Detected: {results['beat_count']}

## Signal Quality Assessment
- Quality Classification: {results['signal_quality']} (Score: {results['quality_score']:.2f}/1.00)
- Signal-to-Noise Ratio: {results.get('quality_indicators', {}).get('snr_db', 0):.2f} dB
- Baseline Wander: {results.get('quality_indicators', {}).get('baseline_wander', False)}
- Motion Artifacts: {results.get('quality_indicators', {}).get('has_motion_artifacts', False)}

## AI Abnormality Prediction
- Overall Result: {results['predicted_class']}
- Model Probabilities:
  * Normal: {results['probabilities'].get('Normal', 0)*100:.2f}%
  * PVC: {results['probabilities'].get('PVC', 0)*100:.2f}%
  * Other: {results['probabilities'].get('Other', 0)*100:.2f}%

## Beat Statistics
- Estimated Heart Rate: {results['heart_rate_bpm']:.1f} BPM
- Average R-R Interval: {results['mean_rr_sec']:.3f} s
- Beat Counts: {json.dumps(results.get('class_counts', {}))}

## Medical Disclaimer
This report was generated for educational and scientific research purposes only.
It is NOT a medical diagnosis and should never be used as a substitute for certified medical evaluation.
"""

st.download_button(
    label="📄 Download Analysis Summary (.txt)",
    data=report_text,
    file_name=f"ecg_analysis_report_{selected_record or 'custom'}.txt",
    mime="text/plain",
)
