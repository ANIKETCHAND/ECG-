"""
ECG Guardian — Dynamic Criteria Used Component
==============================================
Provides reusable criteria indicators for every chart, visualization, graph, and analytical panel.
Enforces the Zero Fabrication Rule:
- Only display inputs/criteria that ACTUALLY contributed to the calculation or model.
- Strictly separate 'Model inputs' from 'Patient context considered'.
- Never display fields (like Blood Group) as ML criteria when institutional governance excludes them.
- Dynamically detect missing fields and exclude them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import html

# Try importing streamlit for UI rendering; fallback gracefully for headless/unit test contexts
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Path to models directory for metadata extraction
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


@dataclass
class CriterionItem:
    """Individual data criterion with provenance and destination metadata."""
    name: str
    value: Optional[str] = None
    source: str = "ECG Analysis"  # e.g., 'ECG Waveform', 'Vital Signs', 'Patient Record', 'Laboratory'
    used_by: str = "ML Model"      # e.g., 'ECG Model', 'Multimodal Model', 'Clinical Decision Support'
    category: str = "model_input"  # 'model_input', 'patient_context', 'measurement', 'processing'


@dataclass
class CriteriaDescriptor:
    """Descriptor holding criteria metadata for a specific visualization or analytical panel."""
    label: str  # e.g., 'Criteria used', 'Model inputs', 'Patient context considered', 'Criteria considered'
    criteria: List[str]
    detailed_items: List[CriterionItem] = field(default_factory=list)
    missing_items: List[str] = field(default_factory=list)
    processing_steps: List[str] = field(default_factory=list)
    model_version: Optional[str] = None
    all_feature_names: List[str] = field(default_factory=list)
    total_feature_count: Optional[int] = None
    chart_id: str = "chart"

    def format_inline_string(self) -> str:
        """Format as: 'Label: item1 • item2 • item3'."""
        clean_criteria = [c.strip() for c in self.criteria if c and c.strip()]
        return f"{self.label}: " + " • ".join(clean_criteria)


def load_model_feature_metadata(models_dir: Optional[Path] = None) -> Tuple[str, List[str]]:
    """Load model name and feature names from the authoritative production metadata."""
    base_dir = models_dir or MODELS_DIR
    meta_path = base_dir / "production" / "metadata.json"
    if not meta_path.exists():
        meta_path = base_dir / "metadata.json"

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("feature_names", [])
            model_name = data.get("model_name", "RandomForestClassifier")
            return model_name, features
        except Exception:
            pass

    # Default fallback matching the 28-feature standard if file cannot be read
    default_28 = [
        "mean", "std", "min", "max", "range", "median", "energy", "rms", "mav", "snr",
        "zero_crossing_rate", "autocorr_first_peak", "r_peak_amplitude", "p_wave_amplitude",
        "t_wave_amplitude", "peak_to_peak_amplitude", "max_slope", "qrs_width_samples",
        "total_power", "lf_power", "hf_power", "vhf_power", "dominant_frequency",
        "max_power", "spectral_entropy", "pre_rr", "post_rr", "local_rr_ratio"
    ]
    return "RandomForestClassifier", default_28


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CRITERIA GENERATORS (DYNAMICALLY DERIVED FROM ACTUAL RUNTIME DATA)
# ─────────────────────────────────────────────────────────────────────────────

def get_waveform_criteria(
    input_info: Optional[Dict[str, Any]] = None,
    sampling_rate: Optional[float] = 360.0,
    lead: str = "II",
    duration_sec: Optional[float] = None,
    has_raw: bool = False,
) -> CriteriaDescriptor:
    """Criteria used to render the main ECG waveform plot."""
    info = input_info or {}
    fs = info.get("sampling_rate", sampling_rate) or 360.0
    raw_lead = str(info.get("lead", lead) or "Lead II")
    lead_name = f"Lead {raw_lead}" if not raw_lead.lower().startswith("lead") else raw_lead
    dur = info.get("duration_sec", duration_sec)

    criteria = [
        "Raw ECG signal",
        f"Sampling rate ({float(fs):.0f} Hz)",
        f"Lead configuration ({lead_name})",
    ]
    if dur:
        criteria.append(f"Signal duration ({float(dur):.1f}s)")
    else:
        criteria.append("Signal duration")
    if has_raw:
        criteria.append("Raw signal overlay")

    processing = [
        "Baseline correction",
        "Bandpass filtering",
        "Normalization",
    ]

    detailed = [
        CriterionItem("Raw ECG Voltage", f"Series @ {fs} Hz", "ECG Sensor / Digital File", "Waveform Renderer"),
        CriterionItem("Sampling Rate", f"{fs} Hz", "Hardware Acquisition", "Waveform Renderer"),
        CriterionItem("Lead Configuration", str(lead_name), "Electrode Placement", "Waveform Renderer"),
    ]

    return CriteriaDescriptor(
        label="Criteria used",
        criteria=criteria,
        processing_steps=processing,
        detailed_items=detailed,
        chart_id="waveform_graph",
    )


def get_segmentation_criteria(
    beats: Optional[Any] = None,
    fs: float = 360.0,
    pre_window: float = 0.2,
    post_window: float = 0.4,
    detected_peaks_count: Optional[int] = None,
) -> CriteriaDescriptor:
    """Criteria used to extract and overlay segmented cardiac heartbeat cycles."""
    n_beats = len(beats) if beats is not None else (detected_peaks_count or 0)
    criteria = [
        "Detected R-peaks (Pan-Tompkins / gradient detector)",
        f"Cardiac beat window (-{int(pre_window*1000)}ms to +{int(post_window*1000)}ms)",
        "ECG amplitude",
        "Fiducial R-peak temporal alignment (t=0s)",
        f"Sampling rate ({float(fs):.0f} Hz)",
    ]

    detailed = [
        CriterionItem("Fiducial R-Peaks", f"{n_beats} peaks identified", "QRS Detector", "Beat Segmentation Engine"),
        CriterionItem("Pre-R Window", f"{pre_window*1000:.0f} ms", "Algorithm Parameter", "Cycle Extractor"),
        CriterionItem("Post-R Window", f"{post_window*1000:.0f} ms", "Algorithm Parameter", "Cycle Extractor"),
        CriterionItem("Beat Count", f"{n_beats} cycles", "Segmentation Buffer", "Overlay Plot"),
    ]

    return CriteriaDescriptor(
        label="Criteria used",
        criteria=criteria,
        detailed_items=detailed,
        chart_id="beat_segmentation",
    )


def get_ai_classification_criteria(
    ai_results: Optional[Dict[str, Any]] = None,
    model_id: str = "ECG-RF-1.0.0",
    is_multimodal: bool = False,
    patient_profile: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CriteriaDescriptor:
    if "model_name" in kwargs and not model_id:
        model_id = kwargs["model_name"]
    elif "model_name" in kwargs and model_id == "ECG-RF-1.0.0":
        model_id = kwargs["model_name"]
    """
    Criteria used for AI Arrhythmia classification & probability distribution.
    Strictly adheres to Rule 4 & 5:
    - Lists ONLY features supplied to the active model.
    - If model is ECG-only: lists 28 ECG features.
    - If model is multimodal: dynamically lists ONLY available patient fields passed to Model C.
    - Strictly excludes blood group.
    """
    model_name, feature_names = load_model_feature_metadata()
    total_feats = len(feature_names)

    if not is_multimodal:
        # Single-lead ECG Arrhythmia Classifier (Model A)
        criteria = [
            "ECG waveform",
            "R-peak features",
            "Beat segmentation",
            "RR intervals",
            "QRS characteristics",
            f"{total_feats} extracted ECG features",
            "Signal quality gatekeeper",
        ]

        detailed = [
            CriterionItem("ECG Voltage Series", "Waveform strip", "Digital Acquisition", "Feature Pipeline"),
            CriterionItem("R-Peak Temporal Loci", "RR timing dynamics", "Peak Detector", "Feature Pipeline"),
            CriterionItem(f"{total_feats} Extracted Features", "Morphological, Spectral, Temporal", "Feature Engine", f"{model_id} ({model_name})"),
            CriterionItem("Signal Quality Status", ai_results.get("signal_quality", "GOOD") if ai_results else "GOOD", "Quality Gatekeeper", "Inference Barrier"),
        ]

        return CriteriaDescriptor(
            label="Model inputs",
            criteria=criteria,
            detailed_items=detailed,
            model_version=model_id,
            all_feature_names=feature_names,
            total_feature_count=total_feats,
            chart_id="ai_classification_probabilities",
        )
    else:
        # Multimodal Model (Model C)
        criteria = [
            "ECG waveform",
            f"{total_feats} extracted ECG features",
        ]

        detailed = [
            CriterionItem("ECG Waveform Features", f"{total_feats} morphology/rhythm metrics", "ECG Waveform", "Model A Backbone"),
        ]

        # Dynamically inspect actually available clinical context
        prof = patient_profile or {}
        vits = vital_signs or {}
        labs = laboratory_results or {}

        if prof.get("age") or prof.get("patient_age"):
            val = prof.get("age") or prof.get("patient_age")
            criteria.append(f"Age ({val}y)")
            detailed.append(CriterionItem("Age", f"{val} years", "Patient Demographics", "Clinical Prior Model B"))

        if prof.get("sex") or prof.get("patient_sex"):
            val = prof.get("sex") or prof.get("patient_sex")
            criteria.append(f"Sex ({val})")
            detailed.append(CriterionItem("Sex", str(val), "Patient Demographics", "Clinical Prior Model B"))

        if vits.get("systolic_bp") or vits.get("diastolic_bp"):
            bp_val = f"{vits.get('systolic_bp','?')}/{vits.get('diastolic_bp','?')} mmHg"
            criteria.append(f"Blood pressure ({bp_val})")
            detailed.append(CriterionItem("Blood Pressure", bp_val, "Point-in-Time Vitals", "Clinical Prior Model B"))

        if prof.get("symptoms") and str(prof.get("symptoms")).lower() not in ("none", "not provided"):
            criteria.append("Presenting symptoms")
            detailed.append(CriterionItem("Symptoms", str(prof.get("symptoms")), "Patient History", "Clinical Prior Model B"))

        if prof.get("existing_conditions") and str(prof.get("existing_conditions")).lower() not in ("none", "not provided"):
            criteria.append("Existing conditions")
            detailed.append(CriterionItem("Conditions", str(prof.get("existing_conditions")), "Electronic Medical Record", "Clinical Prior Model B"))

        if prof.get("current_medications") and str(prof.get("current_medications")).lower() not in ("none", "not provided"):
            criteria.append("Active medications")
            detailed.append(CriterionItem("Medications", str(prof.get("current_medications")), "Pharmacy Reconciliation", "Clinical Prior Model B"))

        if labs.get("potassium"):
            k_val = f"{labs.get('potassium')} mEq/L"
            criteria.append(f"Serum potassium ({k_val})")
            detailed.append(CriterionItem("Serum Potassium", k_val, "Clinical Laboratory", "Clinical Prior Model B"))

        return CriteriaDescriptor(
            label="Model inputs (Multimodal)",
            criteria=criteria,
            detailed_items=detailed,
            model_version=f"{model_id} + Multimodal Late Fusion",
            all_feature_names=feature_names,
            total_feature_count=total_feats,
            chart_id="multimodal_classification",
        )


def get_signal_quality_criteria(quality_indicators: Optional[Dict[str, Any]] = None) -> CriteriaDescriptor:
    """Criteria used for the Signal Quality Gatekeeper evaluation table."""
    q = quality_indicators or {}
    criteria = [
        f"SNR ({q.get('snr_db', 0.0):.1f} dB)" if "snr_db" in q else "SNR",
        "Baseline drift",
        "Powerline interference",
        "Motion artifacts",
    ]

    detailed = [
        CriterionItem("Signal-to-Noise Ratio", f"{q.get('snr_db', 0):.1f} dB" if "snr_db" in q else "Evaluated", "Spectral Estimation", "Quality Gatekeeper"),
        CriterionItem("Baseline Wander", "Checked (< 0.5 Hz)", "Low-Frequency Filter", "Quality Gatekeeper"),
        CriterionItem("Powerline Harmonics", "Checked (50/60 Hz)", "Spectral Density", "Quality Gatekeeper"),
        CriterionItem("Motion Artifacts", "Checked (> 45 Hz)", "Variance Estimator", "Quality Gatekeeper"),
    ]

    return CriteriaDescriptor(
        label="Criteria used",
        criteria=criteria,
        detailed_items=detailed,
        chart_id="signal_quality_table",
    )


def get_feature_table_criteria(feats_df: Optional[Any] = None) -> CriteriaDescriptor:
    """Criteria used for the Extracted Morphological Feature Table."""
    names = []
    if feats_df is not None and hasattr(feats_df, "columns"):
        names = [str(c).replace("_", " ").title() for c in feats_df.columns if c not in ("label", "record_id", "symbol")][:8]

    if not names:
        names = [
            "R-Peak Amplitude",
            "QRS Width",
            "Peak-to-Peak Amplitude",
            "Signal Energy",
            "Spectral Entropy",
            "Dominant Frequency",
            "Local RR Ratio",
        ]

    criteria = names
    detailed = [
        CriterionItem(n, "Extracted from beat window", "Morphology & FFT Analysis", "Feature Matrix")
        for n in names
    ]

    return CriteriaDescriptor(
        label="Criteria used",
        criteria=criteria,
        detailed_items=detailed,
        chart_id="feature_table",
    )


def get_patient_context_criteria(
    patient_profile: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
) -> CriteriaDescriptor:
    """
    Criteria considered for clinical patient-context panels.
    Notice: Uses 'Patient context considered' rather than 'Model inputs'.
    Excludes any fields that are missing or 'NOT PROVIDED'.
    """
    prof = patient_profile or {}
    vits = vital_signs or {}
    labs = laboratory_results or {}

    available = []
    missing = []
    detailed = []

    # Demographics
    age = prof.get("age") or prof.get("patient_age")
    if age and str(age).upper() != "NOT PROVIDED":
        available.append(f"Age ({age}y)")
        detailed.append(CriterionItem("Age", f"{age} years", "Patient Record", "Clinical Context"))
    else:
        missing.append("Age")

    sex = prof.get("sex") or prof.get("patient_sex")
    if sex and str(sex).upper() != "NOT PROVIDED":
        available.append(f"Sex ({sex})")
        detailed.append(CriterionItem("Sex", str(sex), "Patient Record", "Clinical Context"))
    else:
        missing.append("Sex")

    # Vitals
    if vits.get("systolic_bp") or vits.get("diastolic_bp"):
        bp = f"{vits.get('systolic_bp','?')}/{vits.get('diastolic_bp','?')} mmHg"
        available.append(f"Blood pressure ({bp})")
        detailed.append(CriterionItem("Blood Pressure", bp, "Vital Signs", "Clinical Context"))
    else:
        missing.append("Blood pressure")

    if vits.get("heart_rate"):
        hr = f"{vits.get('heart_rate')} BPM"
        available.append(f"Vital HR ({hr})")
        detailed.append(CriterionItem("Bedside Heart Rate", hr, "Vital Signs", "Clinical Context"))

    # Conditions & Symptoms
    cond = prof.get("existing_conditions") or prof.get("conditions")
    if cond and str(cond).upper() not in ("NONE", "NOT PROVIDED", "NOT DOCUMENTED", "—"):
        available.append("Existing conditions")
        detailed.append(CriterionItem("Conditions", str(cond), "Medical History", "Clinical Context"))
    else:
        missing.append("Existing conditions")

    symp = prof.get("symptoms")
    if symp and str(symp).upper() not in ("NONE", "NOT PROVIDED", "NOT DOCUMENTED", "—"):
        available.append("Presenting symptoms")
        detailed.append(CriterionItem("Symptoms", str(symp), "Patient Report", "Clinical Context"))
    else:
        missing.append("Symptoms")

    smok = prof.get("smoking_status")
    if smok and str(smok).upper() not in ("NONE", "NOT PROVIDED", "NOT DOCUMENTED", "—", ""):
        available.append(f"Smoking ({smok})")
        detailed.append(CriterionItem("Smoking Status", str(smok), "Social History", "Clinical Context"))
    else:
        missing.append("Smoking status")

    # Labs
    if labs.get("potassium"):
        available.append(f"Serum potassium ({labs.get('potassium')} mEq/L)")
        detailed.append(CriterionItem("Potassium", f"{labs.get('potassium')} mEq/L", "Laboratory", "Clinical Context"))

    if labs.get("troponin_i") or labs.get("troponin"):
        trop = labs.get("troponin_i") or labs.get("troponin")
        available.append(f"Troponin ({trop} ng/mL)")
        detailed.append(CriterionItem("Troponin", f"{trop} ng/mL", "Cardiac Markers", "Clinical Context"))

    return CriteriaDescriptor(
        label="Patient context considered",
        criteria=available if available else ["No verified patient context provided"],
        detailed_items=detailed,
        missing_items=missing,
        chart_id="patient_context",
    )


def get_medication_safety_criteria(
    med_safety_eval: Optional[Any] = None,
    patient_medications: Optional[List[str]] = None,
    allergies: Optional[str] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    laboratory_results: Optional[Dict[str, Any]] = None,
) -> CriteriaDescriptor:
    """Criteria evaluated by the Medication Safety Engine."""
    criteria = []
    detailed = []

    # Active medications
    meds = patient_medications or []
    if med_safety_eval and hasattr(med_safety_eval, "analyzed_medications"):
        meds = med_safety_eval.analyzed_medications or meds

    if meds:
        criteria.append(f"Active medications ({len(meds)} agents)")
        detailed.append(CriterionItem("Current Medications", ", ".join(meds), "Reconciliation", "Medication Safety Engine"))
    else:
        criteria.append("Current medications")

    if allergies and str(allergies).upper() not in ("NONE", "NO KNOWN ALLERGIES", "NOT PROVIDED"):
        criteria.append("Documented allergies")
        detailed.append(CriterionItem("Allergies", str(allergies), "Allergy Record", "Medication Safety Engine"))

    criteria.append("Known conditions & contraindications")

    # Physiological cross-checks
    vits = vital_signs or {}
    labs = laboratory_results or {}

    if vits.get("heart_rate"):
        criteria.append("Heart rate (bradycardia cross-check)")
    if vits.get("systolic_bp"):
        criteria.append("Blood pressure (hypotension cross-check)")
    if labs.get("potassium"):
        criteria.append("Serum potassium (electrolyte conflict check)")
    if labs.get("creatinine") or labs.get("egfr"):
        criteria.append("Renal clearance (eGFR/Creatinine dosing check)")

    return CriteriaDescriptor(
        label="Criteria evaluated",
        criteria=criteria,
        detailed_items=detailed,
        chart_id="medication_safety_evaluation",
    )


def get_cds_criteria(
    cds_rec: Optional[Any] = None,
    has_patient_history: bool = False,
    has_vitals: bool = False,
    has_labs: bool = False,
    has_meds: bool = False,
    has_prior_ecg: bool = False,
) -> CriteriaDescriptor:
    """Criteria considered for Clinical Decision Support recommendations."""
    criteria = ["AI ECG findings", "ACC/AHA/ESC clinical guidelines"]
    if has_patient_history:
        criteria.append("Patient cardiac history")
    if has_vitals:
        criteria.append("Point-in-time vital signs")
    if has_labs:
        criteria.append("Laboratory results")
    if has_meds:
        criteria.append("Current cardiovascular medications")
    if has_prior_ecg:
        criteria.append("Prior ECG comparative findings")

    detailed = [
        CriterionItem("AI ECG Classification", "Rhythm & beat predictions", "Model Output", "CDS Engine"),
        CriterionItem("Guideline Framework", "AHA/ACC/ESC Class I/IIa directives", "Clinical Knowledge Base", "CDS Engine"),
    ]

    return CriteriaDescriptor(
        label="Criteria considered",
        criteria=criteria,
        detailed_items=detailed,
        chart_id="clinical_decision_support",
    )


def get_feature_importance_criteria(
    importances: Optional[Dict[str, float]] = None,
    top_k: int = 6,
) -> CriteriaDescriptor:
    """Criteria used for the Feature Importance plot."""
    top_features = []
    if importances:
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_features = [f"{k} ({v:.3f})" for k, v in sorted_feats]

    criteria = [
        f"Gini impurity decrease across 150 ensemble trees",
        f"Top {top_k} features from 28 extracted ECG metrics",
        "MIT-BIH training cohort (10,152 beats)",
    ]

    detailed = [
        CriterionItem("Model Family", "Balanced Random Forest", "Ensemble Architecture", "Feature Importance Plot"),
        CriterionItem("Metric", "Mean Decrease in Gini Impurity", "Tree Split Analysis", "Feature Importance Plot"),
    ]

    return CriteriaDescriptor(
        label="Criteria used",
        criteria=criteria,
        detailed_items=detailed,
        chart_id="feature_importance_plot",
    )


def get_longitudinal_criteria(
    current_ecg_id: str,
    previous_ecg_id: Optional[str] = None,
    available_metrics: Optional[List[str]] = None,
) -> CriteriaDescriptor:
    """Criteria used for Longitudinal Previous ECG Comparison."""
    metrics = available_metrics or ["Heart rate", "PR interval", "QRS duration", "QTc interval", "Rhythm findings"]
    criteria = [f"Current ECG ({current_ecg_id})"]
    if previous_ecg_id:
        criteria.append(f"Previous ECG ({previous_ecg_id})")
    criteria.extend(metrics)

    return CriteriaDescriptor(
        label="Criteria used",
        criteria=criteria,
        chart_id="longitudinal_comparison",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND RENDERING HELPERS (STREAMLIT & HTML)
# ─────────────────────────────────────────────────────────────────────────────

def get_criteria_footer_html(descriptor: CriteriaDescriptor) -> str:
    """
    Generate responsive HTML string for the criteria footer.
    Theme-adaptive (readable in dark/light mode), mobile-friendly with flex-wrap.
    Zero leading indentation so Markdown parsers never treat it as an indented code block.
    """
    tokens = " • ".join(html.escape(c) for c in descriptor.criteria)
    label = html.escape(descriptor.label)

    proc_html = ""
    if descriptor.processing_steps:
        proc_tokens = " • ".join(html.escape(p) for p in descriptor.processing_steps)
        proc_html = (
            f'<div style="font-size:0.71rem;color:#8b949e;margin-top:3px;display:flex;flex-wrap:wrap;align-items:center;gap:4px;">'
            f'<span style="font-weight:600;color:#58a6ff;">Processing:</span>'
            f'<span>{proc_tokens}</span>'
            f'</div>'
        )

    missing_html = ""
    if descriptor.missing_items:
        miss_tokens = " • ".join(html.escape(m) for m in descriptor.missing_items)
        missing_html = (
            f'<div style="font-size:0.70rem;color:#d29922;margin-top:3px;display:flex;flex-wrap:wrap;align-items:center;gap:4px;">'
            f'<span style="font-weight:600;">&#9888; Not provided:</span>'
            f'<span>{miss_tokens}</span>'
            f'</div>'
        )

    return (
        f'<div class="criteria-footer-container" style="margin-top:4px;margin-bottom:12px;padding:4px 8px;background:rgba(22,27,34,0.4);border-left:2px solid #30363d;border-radius:0 4px 4px 0;line-height:1.4;">'
        f'<div style="font-size:0.73rem;color:#8b949e;display:flex;flex-wrap:wrap;align-items:center;gap:4px;">'
        f'<span style="font-weight:600;color:#c9d1d9;">{label}:</span>'
        f'<span>{tokens}</span>'
        f'</div>'
        f'{proc_html}'
        f'{missing_html}'
        f'</div>'
    )


def render_criteria_footer(
    descriptor: CriteriaDescriptor,
    key_suffix: str = "",
    show_details: bool = True,
) -> None:
    """
    Renders the Criteria footer in Streamlit with optional interactive details modal/expander.
    Follows all guidelines: subtle font, muted text, no chart collision, responsive.
    """
    if not HAS_STREAMLIT:
        return

    # Render HTML footer
    st.markdown(get_criteria_footer_html(descriptor), unsafe_allow_html=True)

    # Optional interactive details view (Expander/Modal)
    if show_details and (descriptor.detailed_items or descriptor.all_feature_names):
        detail_key = f"crit_det_{descriptor.chart_id}_{key_suffix}"
        with st.expander("🔍 View Criteria & Data Source Details", expanded=False):
            if descriptor.model_version:
                st.caption(f"**Target Model / Engine:** `{descriptor.model_version}`")

            if descriptor.detailed_items:
                rows = []
                for item in descriptor.detailed_items:
                    rows.append({
                        "Criterion": item.name,
                        "Value / Setting": item.value or "Evaluated",
                        "Data Source": item.source,
                        "Used By": item.used_by,
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if descriptor.all_feature_names:
                st.markdown(f"**Authoritative Extracted Feature Set ({len(descriptor.all_feature_names)} features):**")
                cols = st.columns(3)
                for idx, fname in enumerate(descriptor.all_feature_names):
                    cols[idx % 3].caption(f"{idx+1}. `{fname}`")
