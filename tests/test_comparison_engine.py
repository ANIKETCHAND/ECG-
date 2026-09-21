"""
Unit Tests for Phase 7: ECG Machine vs. AI Verification & Disagreement Detection
================================================================================
Validates:
1. Machine statement parser (rhythm categories, measurements, unconfirmed status).
2. Concordance detection (AGREE on normal, AGREE on ectopy).
3. Discrepancy detection (SIGNIFICANT_DISAGREEMENT on PVC vs Normal).
4. Safety alerting on conditions outside AI scope (e.g. Infarct/STEMI, AFib).
5. Fail-safe behavior on missing/empty machine statements (UNABLE_TO_COMPARE).
"""

import pytest

from src.comparison.ai_machine_comparison import compare_ai_and_machine
from src.comparison.disagreement_detector import ComparisonStatus
from src.comparison.machine_interpretation import parse_machine_interpretation


def test_parse_machine_interpretation_normal():
    raw = "UNCONFIRMED REPORT: Normal Sinus Rhythm, HR: 72 bpm, PR: 160, QRS: 88, QTc: 412"
    res = parse_machine_interpretation(raw)
    assert res.normalized_category == "NORMAL_SINUS"
    assert res.reported_heart_rate == 72.0
    assert res.reported_pr_interval_ms == 160.0
    assert res.reported_qrs_duration_ms == 88.0
    assert res.reported_qtc_ms == 412.0
    assert res.is_unconfirmed is True


def test_parse_machine_interpretation_pvc():
    raw = "Sinus rhythm with occasional premature ventricular contractions (PVCs)"
    res = parse_machine_interpretation(raw)
    assert res.normalized_category == "VENTRICULAR_ECTOPY"


def test_parse_machine_interpretation_infarct():
    raw = "Acute Anteroseptal Myocardial Infarction - ST Elevation"
    res = parse_machine_interpretation(raw)
    assert res.normalized_category == "ISCHEMIA_INFARCT"


def test_comparison_agree_normal():
    raw = "Normal Sinus Rhythm, HR=75"
    interp = parse_machine_interpretation(raw)
    ai_res = {
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 74.0,
    }
    cmp = compare_ai_and_machine(ai_res, interp)
    assert cmp.status == ComparisonStatus.AGREE
    assert "agree: Normal" in cmp.summary


def test_comparison_agree_pvc():
    raw = "Ventricular Premature Beats detected"
    interp = parse_machine_interpretation(raw)
    ai_res = {
        "prediction": "Premature Ventricular Contraction",
        "heart_rate": 80.0,
    }
    cmp = compare_ai_and_machine(ai_res, interp)
    assert cmp.status == ComparisonStatus.AGREE
    assert "agree: Ventricular Ectopy" in cmp.summary


def test_comparison_disagreement_ai_pvc_machine_normal():
    raw = "Normal Sinus Rhythm"
    interp = parse_machine_interpretation(raw)
    ai_res = {
        "prediction": "Premature Ventricular Contraction",
        "heart_rate": 78.0,
    }
    cmp = compare_ai_and_machine(ai_res, interp)
    assert cmp.status == ComparisonStatus.SIGNIFICANT_DISAGREEMENT
    assert "AI detected Ventricular Ectopy" in cmp.summary
    assert len(cmp.discrepancy_details) > 0


def test_comparison_machine_infarct_scope_warning():
    raw = "Acute ST-Elevation Infarct in V1-V4"
    interp = parse_machine_interpretation(raw)
    ai_res = {
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 70.0,
    }
    cmp = compare_ai_and_machine(ai_res, interp)
    assert cmp.status == ComparisonStatus.SIGNIFICANT_DISAGREEMENT
    assert "CRITICAL DISCREPANCY" in cmp.summary
    assert cmp.coverage_limitation_warning is not None
    assert "STEMI" in cmp.coverage_limitation_warning


def test_comparison_missing_machine_statement():
    interp = parse_machine_interpretation(None)
    ai_res = {
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 70.0,
    }
    cmp = compare_ai_and_machine(ai_res, interp)
    assert cmp.status == ComparisonStatus.UNABLE_TO_COMPARE
    assert "missing" in cmp.summary.lower()


def test_comparison_minor_difference_heart_rate():
    raw = "Normal Sinus Rhythm, HR=60"
    interp = parse_machine_interpretation(raw)
    ai_res = {
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 85.0,  # 25 bpm difference
    }
    cmp = compare_ai_and_machine(ai_res, interp)
    assert cmp.status == ComparisonStatus.MINOR_DIFFERENCE
    assert cmp.rate_difference_bpm == 25.0
