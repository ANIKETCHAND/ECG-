"""
Tests for dataset-grounded feature builders (training/real_features.py).

These are pure functions over measured inputs, so they can be verified with
synthetic-but-known inputs even when the upstream PhysioNet datasets are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.real_features import (
    AF_FEATURE_NAMES,
    PTBXL_FEATURE_NAMES,
    PTBXL_SUPERCLASSES,
    QUALITY_FEATURE_NAMES,
    ST_FEATURE_NAMES,
    af_label_from_rhythm_aux,
    af_rhythm_features,
    ptbxl_labels_from_scp_codes,
    ptbxl_lead_features,
    quality_features,
    quality_label_from_features,
    st_features,
    st_label_from_features,
    template_for,
)


# --------------------------------------------------------------------- AF labels
def test_af_label_mapping_covers_documented_rhythms():
    assert af_label_from_rhythm_aux("AFIB") == "AFib"
    assert af_label_from_rhythm_aux("AFL") == "AFib"
    assert af_label_from_rhythm_aux("N") == "Non-AFib"
    assert af_label_from_rhythm_aux("(N") == "Non-AFib" or af_label_from_rhythm_aux("(N") is None


def test_af_label_returns_none_for_unmodelled_rhythms():
    assert af_label_from_rhythm_aux(None) is None
    assert af_label_from_rhythm_aux("") is None
    assert af_label_from_rhythm_aux("VFIB") is None


def test_af_label_mapping_tolerates_whitespace_and_case():
    # WFDB aux notes are commonly prefixed with a delimiter character.
    assert af_label_from_rhythm_aux("  afib  ") == "AFib"
    assert af_label_from_rhythm_aux("N ") == "Non-AFib"


# ---------------------------------------------------------------------- AF features
def test_af_features_require_enough_intervals():
    assert af_rhythm_features([0.8, 0.8]) is None
    assert af_rhythm_features([]) is None


def test_af_features_regular_rhythm_has_low_variability():
    regular = af_rhythm_features([0.8] * 30)
    assert regular is not None
    assert regular["rr_cv"] == pytest.approx(0.0, abs=1e-9)
    assert regular["rr_rmssd"] == pytest.approx(0.0, abs=1e-9)
    assert regular["pnn50"] == pytest.approx(0.0, abs=1e-9)


def test_af_features_irregular_rhythm_has_higher_variability():
    rng = np.random.default_rng(0)
    irregular = af_rhythm_features(list(0.8 + rng.normal(0, 0.25, 60)))
    regular = af_rhythm_features([0.8] * 60)
    assert irregular["rr_cv"] > regular["rr_cv"]
    assert irregular["rr_rmssd"] > regular["rr_rmssd"]
    assert irregular["pnn50"] >= regular["pnn50"]


def test_af_feature_names_match_returned_keys():
    features = af_rhythm_features([0.8] * 20)
    assert set(features.keys()) == set(AF_FEATURE_NAMES)


# ---------------------------------------------------------------------- ST features
def _synthetic_beat(fs: float = 360.0, st_elevation_mv: float = 0.0) -> tuple[np.ndarray, int]:
    """Build a beat with a flat baseline, a QRS spike and a controllable ST level."""
    n = int(1.0 * fs)
    signal = np.zeros(n, dtype=float)
    r_index = int(0.4 * fs)
    signal[r_index - 3 : r_index + 4] = 1.0  # QRS
    # ST segment region (J point onwards) shifted by the requested amount
    st_start = r_index + int(0.08 * fs)
    signal[st_start : st_start + int(0.2 * fs)] = st_elevation_mv
    return signal, r_index


def test_st_features_measure_j_point_and_st60():
    signal, r_index = _synthetic_beat(st_elevation_mv=0.30)
    features = st_features(signal, 360.0, r_index)
    assert features is not None
    assert features["st60_mv"] > 0.20
    assert set(features.keys()) == set(ST_FEATURE_NAMES)


def test_st_features_none_when_beat_truncated_at_boundary():
    signal = np.zeros(200)
    assert st_features(signal, 360.0, 0) is None


def test_st_labels_threshold_both_directions():
    assert st_label_from_features({"st60_mv": 0.25}) == "ST_Elevation"
    assert st_label_from_features({"st60_mv": -0.20}) == "ST_Depression"
    assert st_label_from_features({"st60_mv": 0.01}) == "Normal"


# ------------------------------------------------------------------ PTB-XL features
def test_ptbxl_feature_names_cover_twelve_leads():
    assert len(PTBXL_FEATURE_NAMES) == 24
    assert "V6_st_dev_mv" in PTBXL_FEATURE_NAMES


def test_ptbxl_lead_features_reads_named_leads():
    lead_names = ["I", "II", "V1", "V2"]
    matrix = np.zeros((4, 1000))
    matrix[0, 500] = 1.5  # lead I amplitude
    matrix[1, 500] = -2.0  # lead II amplitude
    features = ptbxl_lead_features(matrix, 360.0, lead_names)
    assert features is not None
    assert features["I_qrs_amp_mv"] == pytest.approx(1.5)
    assert features["II_qrs_amp_mv"] == pytest.approx(2.0)
    # Leads not supplied are absent, not zero-filled.
    assert "V6_qrs_amp_mv" not in features


def test_ptbxl_lead_features_normalises_lead_name_variants():
    features = ptbxl_lead_features(np.ones((2, 500)), 360.0, [" i ", "avr"])
    assert features is not None
    assert any(name.startswith("I_") for name in features)


def test_ptbxl_lead_features_none_when_no_standard_lead_matched():
    assert ptbxl_lead_features(np.ones((1, 500)), 360.0, ["MLII"]) is None


def test_ptbxl_scp_code_aggregation_to_superclasses():
    labels = ptbxl_labels_from_scp_codes({"IMI": 100.0, "LAFB": 50.0, "NDT": 25.0})
    assert "MI" in labels
    assert "CD" in labels
    assert "STTC" in labels
    assert set(labels).issubset(set(PTBXL_SUPERCLASSES))


def test_ptbxl_scp_codes_respects_score_threshold():
    # A zero-confidence statement must not create a label.
    assert ptbxl_labels_from_scp_codes({"IMI": 0.0}) == []


def test_ptbxl_unknown_codes_are_dropped_not_guessed():
    assert ptbxl_labels_from_scp_codes({"NOT_A_REAL_CODE": 100.0}) == []


# ------------------------------------------------------------------ quality features
def _clean_ecg(fs: float = 360.0, seconds: float = 10.0) -> np.ndarray:
    n = int(fs * seconds)
    t = np.arange(n) / fs
    signal = 0.02 * np.sin(2 * np.pi * 1.2 * t)
    for centre in np.arange(0.8, seconds, 0.8):
        signal += 1.0 * np.exp(-0.5 * ((t - centre) / 0.012) ** 2)
    return signal


def test_quality_features_are_measured_not_invented():
    features = quality_features(_clean_ecg(), 360.0)
    assert features is not None
    assert set(features.keys()) == set(QUALITY_FEATURE_NAMES)
    assert features["clipping_ratio"] >= 0.0


def test_quality_features_reject_short_or_flat_signals():
    assert quality_features(np.zeros(10), 360.0) is None
    assert quality_features(np.zeros(4000), 360.0) is None  # zero variance


def test_quality_label_degrades_with_noise():
    rng = np.random.default_rng(1)
    clean = quality_features(_clean_ecg(), 360.0)
    noisy_signal = _clean_ecg() * 0.05 + rng.normal(0, 1.0, 3600)
    noisy = quality_features(noisy_signal, 360.0)

    assert clean is not None and noisy is not None
    # Band-SNR is a relative measure, so assert the ordering it is meant to capture
    # rather than claiming a specific absolute category for a synthetic waveform.
    assert clean["snr_db"] > noisy["snr_db"]
    assert quality_label_from_features(noisy) in {"POOR", "UNUSABLE"}
    assert clean["clipping_ratio"] < 5.0


def test_clipping_detector_does_not_flag_a_clean_ecg():
    # Regression: an earlier implementation measured distance from the signal's
    # own minimum, which flagged a clean ECG's isoelectric baseline as 42% clipped.
    features = quality_features(_clean_ecg(), 360.0)
    assert features is not None
    assert features["clipping_ratio"] < 1.0


def test_clipping_detector_flags_saturation():
    signal = _clean_ecg()
    saturated = np.clip(signal, -0.25, 0.25)
    features = quality_features(saturated, 360.0)
    clean_features = quality_features(signal, 360.0)
    assert features is not None and clean_features is not None
    assert features["clipping_ratio"] > clean_features["clipping_ratio"]
    assert features["clipping_ratio"] > 2.0
    assert quality_label_from_features(features) in {"POOR", "UNUSABLE"}


def test_quality_label_is_monotonic_in_snr():
    base = {"clipping_ratio": 0.0, "baseline_wander_ratio": 0.01, "powerline_ratio": 0.01, "motion_spike_ratio": 0.0}
    assert quality_label_from_features({"snr_db": 30.0, **base}) == "GOOD"
    assert quality_label_from_features({"snr_db": 14.0, **base}) == "ACCEPTABLE"
    assert quality_label_from_features({"snr_db": 6.0, **base}) == "POOR"
    assert quality_label_from_features({"snr_db": 1.0, **base}) == "UNUSABLE"


# -------------------------------------------------------------------------- schema
def test_template_for_returns_ordered_vector():
    names, values = template_for(["b", "a"], {"a": 1.0, "b": 2.0})
    assert names == ["b", "a"]
    assert values == [2.0, 1.0]


def test_template_for_raises_rather_than_defaulting_to_zero():
    # A silently zero-filled feature vector is indistinguishable from a measured
    # one downstream, so this must be an error.
    with pytest.raises(KeyError, match="incomplete"):
        template_for(["a", "b"], {"a": 1.0})
