"""
Unit Tests for Phase 8: Longitudinal ECG & Patient History
==========================================================
Validates:
1. Serial comparison deltas (HR, RR, QRS, QTc, rhythm shifts).
2. Detection of SIGNIFICANT_CHANGE vs. STABLE vs. INCOMPATIBLE_COMPARISON.
3. Multi-recording trend analysis and slope calculation.
4. PatientECGHistoryStore history sequence management.
"""

import pytest

from src.longitudinal.ecg_comparison import (
    LongitudinalChangeStatus,
    compare_serial_ecgs,
)
from src.longitudinal.ecg_history import PatientECGHistoryStore
from src.longitudinal.trend_analysis import calculate_serial_trends


def test_serial_comparison_stable():
    prior = {
        "analysis_id": "ANL-001",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 72.0,
        "rr_intervals": [830.0, 835.0, 832.0],
        "qrs_duration_ms": 85.0,
        "qtc_ms": 410.0,
    }
    current = {
        "analysis_id": "ANL-002",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 74.0,  # +2 bpm
        "rr_intervals": [810.0, 812.0, 815.0],
        "qrs_duration_ms": 86.0,  # +1 ms
        "qtc_ms": 412.0,  # +2 ms
    }

    res = compare_serial_ecgs(current, prior, patient_id="PT-1001")
    assert res.status == LongitudinalChangeStatus.STABLE
    assert res.delta_heart_rate_bpm == 2.0
    assert res.rhythm_transition_detected is False
    assert "STABLE" in res.objective_change_summary


def test_serial_comparison_rhythm_shift():
    prior = {
        "analysis_id": "ANL-001",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 70.0,
    }
    current = {
        "analysis_id": "ANL-002",
        "lead_analyzed": "II",
        "prediction": "Premature Ventricular Contraction",
        "heart_rate": 75.0,
    }

    res = compare_serial_ecgs(current, prior, patient_id="PT-1001")
    assert res.status == LongitudinalChangeStatus.SIGNIFICANT_CHANGE
    assert res.rhythm_transition_detected is True
    assert "Rhythm shift" in res.notable_deltas[0]


def test_serial_comparison_heart_rate_surge():
    prior = {
        "analysis_id": "ANL-001",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 65.0,
    }
    current = {
        "analysis_id": "ANL-002",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 95.0,  # +30 bpm
    }

    res = compare_serial_ecgs(current, prior, patient_id="PT-1001")
    assert res.status == LongitudinalChangeStatus.SIGNIFICANT_CHANGE
    assert res.delta_heart_rate_bpm == 30.0


def test_serial_comparison_qrs_widening():
    prior = {
        "analysis_id": "ANL-001",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "qrs_duration_ms": 80.0,
    }
    current = {
        "analysis_id": "ANL-002",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "qrs_duration_ms": 110.0,  # +30 ms
    }

    res = compare_serial_ecgs(current, prior, patient_id="PT-1001")
    assert res.status == LongitudinalChangeStatus.SIGNIFICANT_CHANGE
    assert res.delta_qrs_duration_ms == 30.0


def test_serial_comparison_incompatible_leads():
    prior = {
        "analysis_id": "ANL-001",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
    }
    current = {
        "analysis_id": "ANL-002",
        "lead_analyzed": "V1",  # Incompatible!
        "prediction": "Normal Sinus Rhythm",
    }

    res = compare_serial_ecgs(current, prior, patient_id="PT-1001")
    assert res.status == LongitudinalChangeStatus.INCOMPATIBLE_COMPARISON
    assert "Incompatible leads" in res.objective_change_summary


def test_trend_analysis_multiple_records():
    history = [
        {"timestamp": "2026-09-01", "heart_rate": 60.0, "prediction": "Normal Sinus Rhythm"},
        {"timestamp": "2026-09-05", "heart_rate": 72.0, "prediction": "Normal Sinus Rhythm"},
        {"timestamp": "2026-09-10", "heart_rate": 84.0, "prediction": "Premature Ventricular Contraction"},
    ]
    report = calculate_serial_trends(history, patient_id="PT-999")
    assert report.total_recordings == 3
    assert report.hr_trend_slope is not None
    assert report.hr_trend_slope > 0
    assert report.overall_stability in ["PROGRESSIVE_TACHYCARDIA", "EVOLVING_RHYTHM"]


def test_patient_history_store():
    store = PatientECGHistoryStore()
    store.add_analysis("PT-100", {
        "analysis_id": "ANL-1",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 70.0,
    })
    store.add_analysis("PT-100", {
        "analysis_id": "ANL-2",
        "lead_analyzed": "II",
        "prediction": "Normal Sinus Rhythm",
        "heart_rate": 72.0,
    })

    assert len(store.get_history("PT-100")) == 2
    cmp = store.get_latest_comparison("PT-100")
    assert cmp is not None
    assert cmp.status == LongitudinalChangeStatus.STABLE
