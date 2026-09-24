"""
Unit Tests for Reusable CriteriaUsed Component
=============================================
Tests dynamic criteria generation, zero fabrication rule enforcement,
model inputs vs patient context separation, and HTML/text formatting.
"""

import pytest
from src.components.analysis.criteria_used import (
    CriteriaDescriptor,
    get_waveform_criteria,
    get_segmentation_criteria,
    get_ai_classification_criteria,
    get_signal_quality_criteria,
    get_feature_table_criteria,
    get_patient_context_criteria,
    get_medication_safety_criteria,
    get_cds_criteria,
    get_feature_importance_criteria,
    get_criteria_footer_html,
    load_model_feature_metadata,
)


def test_waveform_criteria_generation():
    desc = get_waveform_criteria(input_info={"sampling_rate": 360, "lead": "II", "duration_sec": 10.0}, has_raw=True)
    assert desc.label == "Criteria used"
    s = desc.format_inline_string()
    assert "Raw ECG signal" in s
    assert "360 Hz" in s
    assert "Lead II" in s
    assert "Raw signal overlay" in s
    assert len(desc.processing_steps) >= 3


def test_ai_classification_criteria_single_lead_zero_fabrication():
    # Model A: Must strictly list 28 ECG features and NO patient context fields (Age, BP, etc.)
    desc = get_ai_classification_criteria(model_id="ECG-RF-1.0.0", is_multimodal=False)
    assert desc.label == "Model inputs"
    s = desc.format_inline_string()
    assert "ECG waveform" in s
    assert "28 extracted ECG features" in s
    assert "R-peak features" in s
    assert "RR intervals" in s
    assert "QRS characteristics" in s
    assert "Age" not in s
    assert "Blood pressure" not in s
    assert "Blood group" not in s
    assert len(desc.all_feature_names) == 28


def test_ai_classification_criteria_multimodal_dynamic_inputs():
    # Model C: Should dynamically reflect ONLY available patient fields and strictly exclude Blood Group
    prof = {
        "age": 67,
        "sex": "Male",
        "blood_group": "B+ (Positive)",  # Stored in profile but MUST NOT be listed in model inputs
        "symptoms": "Chest pain",
        "existing_conditions": "Hypertension",
        "current_medications": "Lisinopril",
    }
    vits = {"systolic_bp": 138, "diastolic_bp": 84}
    labs = {"potassium": 4.2}

    desc = get_ai_classification_criteria(
        model_id="ECG-RF-1.0.0",
        is_multimodal=True,
        patient_profile=prof,
        vital_signs=vits,
        laboratory_results=labs,
    )
    assert "Model inputs" in desc.label
    s = desc.format_inline_string()
    assert "Age (67y)" in s
    assert "Sex (Male)" in s
    assert "Blood pressure (138/84 mmHg)" in s
    assert "Presenting symptoms" in s
    assert "Active medications" in s
    assert "Serum potassium (4.2 mEq/L)" in s
    # CRITICAL: Blood group must NOT be in model criteria!
    assert "Blood group" not in s
    assert "B+" not in s


def test_patient_context_criteria_missing_fields_exclusion():
    # If Blood pressure, Smoking status, and Labs were NOT entered, DO NOT display them!
    prof = {"age": 55, "sex": "Female", "existing_conditions": "None"}
    desc = get_patient_context_criteria(patient_profile=prof, vital_signs=None, laboratory_results=None)
    assert desc.label == "Patient context considered"
    s = desc.format_inline_string()
    assert "Age (55y)" in s
    assert "Sex (Female)" in s
    assert "Blood pressure" not in s
    assert "Smoking" not in s
    assert "Potassium" not in s
    assert "Blood pressure" in desc.missing_items


def test_signal_quality_criteria():
    desc = get_signal_quality_criteria({"snr_db": 18.5, "baseline_wander": False})
    s = desc.format_inline_string()
    assert "SNR (18.5 dB)" in s
    assert "Baseline drift" in s
    assert "Powerline interference" in s
    assert "Motion artifacts" in s


def test_medication_safety_criteria():
    desc = get_medication_safety_criteria(
        patient_medications=["Metoprolol", "Lisinopril"],
        allergies="Penicillin",
        vital_signs={"heart_rate": 55, "systolic_bp": 105},
        laboratory_results={"potassium": 5.4, "creatinine": 1.4},
    )
    s = desc.format_inline_string()
    assert "Active medications (2 agents)" in s
    assert "Documented allergies" in s
    assert "bradycardia cross-check" in s
    assert "hypotension cross-check" in s
    assert "electrolyte conflict" in s
    assert "Renal clearance" in s


def test_cds_criteria_uses_considered_not_model():
    desc = get_cds_criteria(has_patient_history=True, has_vitals=True, has_labs=True)
    assert "considered" in desc.label.lower()
    assert "model" not in desc.label.lower()
    s = desc.format_inline_string()
    assert "AI ECG findings" in s
    assert "ACC/AHA/ESC clinical guidelines" in s
    assert "Point-in-time vital signs" in s


def test_criteria_footer_html_rendering():
    desc = CriteriaDescriptor(
        label="Criteria used",
        criteria=["ECG waveform", "R-peak features", "RR intervals"],
        processing_steps=["Bandpass filtering", "Baseline correction"],
        missing_items=["Blood pressure"],
    )
    html_out = get_criteria_footer_html(desc)
    assert "Criteria used:" in html_out
    assert "ECG waveform • R-peak features • RR intervals" in html_out
    assert "Processing:" in html_out
    assert "Bandpass filtering • Baseline correction" in html_out
    assert "Not provided:" in html_out
    assert "Blood pressure" in html_out
