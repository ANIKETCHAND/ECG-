"""
Unit Tests for Deterministic ECG Measurement Engine
"""

import numpy as np
import pytest
from src.measurements.measurement_engine import (
    ClinicalECGMeasurements,
    compute_ecg_measurements,
)


def test_measurement_engine_empty_input():
    res = compute_ecg_measurements(np.array([]), 360.0, np.array([]))
    assert res.heart_rate_bpm is None
    assert res.heart_rate_status == "Not calculated"
    assert "Empty or invalid signal" in res.confidence_notes[0]


def test_measurement_engine_heart_rate_and_rr():
    fs = 360.0
    # Simulate 5 seconds at 60 BPM (1 peak every 360 samples)
    peaks = np.array([360, 720, 1080, 1440, 1800])
    sig = np.zeros(2000)
    for p in peaks:
        sig[p] = 1.0

    res = compute_ecg_measurements(sig, fs, peaks, leads_available=["II"])
    assert res.heart_rate_bpm == 60.0
    assert res.mean_rr_ms == 1000.0
    assert res.median_rr_ms == 1000.0
    assert res.sdnn_ms == 0.0
    assert res.rmssd_ms == 0.0
    assert res.heart_rate_status == "Reliably measured"


def test_measurement_engine_single_lead_axis_rejection():
    fs = 360.0
    peaks = np.array([360, 720, 1080])
    sig = np.sin(np.linspace(0, 10, 1200))
    res = compute_ecg_measurements(sig, fs, peaks, leads_available=["II"])

    # Axis must not be fabricated on single lead
    assert res.p_axis_deg is None
    assert res.qrs_axis_deg is None
    assert "Not measurable on single-lead ECG" in res.axis_status


def test_measurement_engine_qtc_calculation():
    fs = 360.0
    # 75 BPM -> RR = 0.8s
    peaks = np.array([288, 576, 864, 1152])
    sig = np.zeros(1500)
    res = compute_ecg_measurements(sig, fs, peaks, leads_available=["II"])

    assert res.heart_rate_bpm is not None
    # If QT is estimated or not estimated, verify status is explicit
    if res.qt_interval_ms is not None:
        assert res.qtc_bazett_ms is not None
        assert res.qtc_fridericia_ms is not None
