"""
Unit Tests for Unified ECG Core Domain Entities
"""

import numpy as np
import pytest
from src.ecg_core.models import ECGRecording, ECGAnalysisResult, ClinicianReview


def test_ecg_recording_creation_and_hash():
    # Create synthetic 1-lead signal
    fs = 360.0
    sig = np.sin(2 * np.pi * 1.0 * np.arange(fs * 5) / fs)
    rec = ECGRecording(
        record_id="REC-001",
        sampling_rate=fs,
        duration=5.0,
        lead_names=["II"],
        number_of_leads=1,
        signals=sig,
    )
    assert rec.record_id == "REC-001"
    assert rec.number_of_leads == 1
    assert rec.duration == 5.0
    assert len(rec.data_hash) == 64  # SHA-256 hex digest
    valid, errors = rec.validate()
    assert valid
    assert len(errors) == 0


def test_ecg_recording_multilead_and_aliases():
    fs = 500.0
    sig_12 = np.zeros((12, 5000))
    rec = ECGRecording(
        record_id="REC-12L",
        sampling_rate=fs,
        duration=10.0,
        lead_names=[],
        number_of_leads=12,
        signals=sig_12,
    )
    assert rec.number_of_leads == 12
    assert len(rec.lead_names) == 12
    assert rec.has_lead("II")
    assert rec.has_lead("Lead II")
    assert rec.has_lead("MLII")
    assert rec.get_lead("II") is not None
    assert rec.get_lead("V1") is not None
    assert rec.get_lead("NONEXISTENT") is None


def test_ecg_recording_validation_failures():
    # Empty signal
    rec_empty = ECGRecording(
        record_id="REC-EMPTY",
        sampling_rate=360.0,
        duration=0.0,
        lead_names=["II"],
        number_of_leads=1,
        signals=np.array([]),
    )
    valid, errors = rec_empty.validate()
    assert not valid
    assert any("empty" in e.lower() for e in errors)

    # Invalid sampling rate
    rec_bad_fs = ECGRecording(
        record_id="REC-BAD-FS",
        sampling_rate=0.0,
        duration=1.0,
        lead_names=["II"],
        number_of_leads=1,
        signals=np.ones(100),
    )
    valid, errors = rec_bad_fs.validate()
    assert not valid


def test_ecg_analysis_result_and_clinician_review():
    analysis = ECGAnalysisResult(
        analysis_id="ANA-101",
        record_id="REC-001",
        model_id="ECG-RF",
        model_version="1.0.0",
        preprocessing_version="1.0.0",
        lead_analyzed="II",
        signal_quality="GOOD",
        quality_score=0.92,
        quality_indicators={"snr_db": 18.5},
        prediction="Normal",
        model_probabilities={"Normal": 0.95, "PVC": 0.05, "Other": 0.0},
        heart_rate_bpm=72.0,
        mean_rr_ms=833.0,
        detected_beats_count=10,
        detected_r_peaks=[0, 360, 720],
    )
    d = analysis.to_dict()
    assert d["analysis_id"] == "ANA-101"
    assert d["prediction"] == "Normal"

    review = ClinicianReview(
        review_id="REV-201",
        analysis_id="ANA-101",
        record_id="REC-001",
        clinician_id="USR-DOC",
        clinician_name="Dr. Ananya Roy",
        clinician_role="CARDIOLOGIST",
        agreement_status="CONFIRMED",
        clinician_interpretation="Normal Sinus Rhythm. Verified.",
        registration_number="MCI-CARD-4421",
    )
    rd = review.to_dict()
    assert rd["agreement_status"] == "CONFIRMED"
    assert rd["registration_number"] == "MCI-CARD-4421"
