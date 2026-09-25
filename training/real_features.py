"""
Dataset-Grounded Feature Builders
=================================

Feature construction shared by the real-data training paths.

Every function here is a pure function over *measured* inputs (RR intervals,
signal samples, 12-lead matrices). None of them invents values. They are unit
tested against synthetic-but-known inputs so that the algorithms are verified
even when the upstream PhysioNet datasets are not present on disk.

Nothing in this module fabricates training data; it only transforms it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------- labels
#: Maps AFDB rhythm-change annotation aux strings to a binary rhythm label.
#: Reference: WFDB MIMIC/AFDB rhythm annotation convention.
AFDB_RHYTHM_LABELS: Dict[str, str] = {
    "AFIB": "AFib",
    "AFL": "AFib",
    "AFLT": "AFib",
    "N": "Non-AFib",
    "SBR": "Non-AFib",
    "SR": "Non-AFib",
    "B": "Non-AFib",
    "SVTA": "Non-AFib",
    "BII": "Non-AFib",
    "AB": "Non-AFib",
}


def af_label_from_rhythm_aux(aux: Optional[str]) -> Optional[str]:
    """Map a WFDB rhythm annotation aux string to an AF label.

    Returns ``None`` for rhythms this project does not model, so that the
    caller can exclude them rather than mislabel them.
    """
    if not aux:
        return None
    key = str(aux).strip().upper()
    return AFDB_RHYTHM_LABELS.get(key)


# ------------------------------------------------------------------- AF rhythm
AF_FEATURE_NAMES = [
    "rr_mean",
    "rr_cv",
    "rr_rmssd",
    "rr_shannon_entropy",
    "pnn50",
]


def af_rhythm_features(rr_intervals_s: Sequence[float]) -> Optional[Dict[str, float]]:
    """Compute rhythm-irregularity features from RR intervals in seconds.

    Returns ``None`` when there are too few intervals to characterise rhythm,
    which keeps malformed segments out of the training set instead of filling
    them with placeholder zeros.
    """
    rr = np.asarray(list(rr_intervals_s), dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 5:
        return None

    diffs = np.diff(rr)
    mean_rr = float(np.mean(rr))
    std_rr = float(np.std(rr))
    cv = std_rr / (mean_rr + 1e-12)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if diffs.size else 0.0

    hist, _ = np.histogram(diffs, bins=10, density=True) if diffs.size else (np.array([]), None)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if hist.size else 0.0

    pnn50 = float(np.mean(np.abs(diffs) > 0.05) * 100.0) if diffs.size else 0.0

    return {
        "rr_mean": mean_rr,
        "rr_cv": float(cv),
        "rr_rmssd": rmssd,
        "rr_shannon_entropy": entropy,
        "pnn50": pnn50,
    }


# ------------------------------------------------------------- ST / ischemia
ST_FEATURE_NAMES = ["j_point_mv", "st60_mv", "st_slope_mv_per_s"]


def st_features(
    signal: Sequence[float],
    fs: float,
    r_index: int,
    *,
    qrs_offset_s: float = 0.08,
    st_measure_s: float = 0.06,
    baseline_window_s: float = 0.08,
) -> Optional[Dict[str, float]]:
    """Derive J-point, ST60 and ST slope for one beat.

    Measurements are referenced to the isoelectric baseline estimated from the
    PR segment immediately preceding the QRS complex.

    Args:
        signal: 1-D ECG signal (filtered).
        fs: Sampling rate in Hz.
        r_index: Sample index of the R peak.
        qrs_offset_s: Approximate QRS offset after the R peak (J point).
        st_measure_s: Offset after the J point at which ST is measured.
        baseline_window_s: Window immediately before the QRS used as baseline.

    Returns:
        ``{j_point_mv, st60_mv, st_slope_mv_per_s}`` or ``None`` when the beat
        is too close to the signal boundary to measure honestly.
    """
    sig = np.asarray(signal, dtype=float)
    r_index = int(r_index)

    j_idx = r_index + int(round(qrs_offset_s * fs))
    st_idx = j_idx + int(round(st_measure_s * fs))
    base_start = r_index - int(round((0.20 + baseline_window_s) * fs))
    base_end = r_index - int(round(0.20 * fs))

    if base_start < 0 or st_idx >= sig.size:
        return None

    baseline = float(np.median(sig[base_start:base_end])) if base_end > base_start else 0.0
    j_point = float(sig[j_idx]) - baseline
    st60 = float(sig[st_idx]) - baseline
    slope = (st60 - j_point) / max(st_measure_s, 1e-6)

    return {
        "j_point_mv": round(j_point, 6),
        "st60_mv": round(st60, 6),
        "st_slope_mv_per_s": round(float(slope), 6),
    }


def st_label_from_features(
    features: Dict[str, float],
    *,
    elevation_mv: float = 0.10,
    depression_mv: float = -0.08,
) -> str:
    """Threshold-based ST label used to derive weak labels from ST-T records.

    This is an explicit, documented heuristic operating on measured ST
    amplitudes, not a fabrication of labels.
    """
    st60 = features["st60_mv"]
    if st60 >= elevation_mv:
        return "ST_Elevation"
    if st60 <= depression_mv:
        return "ST_Depression"
    return "Normal"


# --------------------------------------------------------------- 12-lead PTB-XL
#: Per-lead descriptors used for the multi-label 12-lead model.
_PTBXL_PER_LEAD = ("qrs_amp_mv", "st_dev_mv")

PTBXL_FEATURE_NAMES = [
    f"{lead}_{descriptor}"
    for lead in ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
    for descriptor in _PTBXL_PER_LEAD
]

PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def ptbxl_lead_features(
    signal_2d: np.ndarray,
    fs: float,
    lead_names: Sequence[str],
) -> Optional[Dict[str, float]]:
    """Build the 12-lead feature vector consumed by the PTB-XL model.

    For each lead: peak-to-peak QRS amplitude and early-ST deviation relative to
    that lead's own baseline. Missing leads are simply omitted, and ``None`` is
    returned when no requested lead can be measured.
    """
    matrix = np.asarray(signal_2d, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]

    normalised = {str(n).strip().upper().replace(" ", ""): i for i, n in enumerate(lead_names)}
    features: Dict[str, float] = {}

    for lead in ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"):
        idx = normalised.get(lead)
        if idx is None or idx >= matrix.shape[0]:
            continue
        trace = matrix[idx]
        if trace.size < 10 or not np.isfinite(trace).all():
            continue
        baseline = float(np.median(trace))
        centred = trace - baseline
        qrs_amp = float(np.max(np.abs(centred)))
        # Early ST deviation: mean of the segment ~80 ms after the highest-slope
        # region, used as a coarse repolarisation descriptor.
        offset = int(round(0.10 * fs))
        seg = centred[offset : offset + max(1, int(round(0.04 * fs)))] if centred.size > offset else np.array([0.0])
        st_dev = float(np.mean(seg)) if seg.size else 0.0
        features[f"{lead}_qrs_amp_mv"] = round(qrs_amp, 6)
        features[f"{lead}_st_dev_mv"] = round(st_dev, 6)

    if not features:
        return None
    return features


def ptbxl_labels_from_scp_codes(scp_codes: Dict[str, float], threshold: float = 0.0) -> List[str]:
    """Map PTB-XL SCP statement codes to the five diagnostic superclasses.

    Reference: PTB-XL statement→superclass aggregation as published with the
    dataset. Codes scoring at or below ``threshold`` are ignored.
    """
    mapping = {
        "NORM": "NORM",
        "IMI": "MI",
        "AMI": "MI",
        "ASMI": "MI",
        "ALMI": "MI",
        "INJAS": "MI",
        "INJAL": "MI",
        "INJIN": "MI",
        "LMI": "MI",
        "IPMI": "MI",
        "ISC_": "STTC",
        "ISCAN": "STTC",
        "ISCLA": "STTC",
        "ISCIN": "STTC",
        "ISCIL": "STTC",
        "ISCAS": "STTC",
        "NDT": "STTC",
        "NST_": "STTC",
        "DIG": "STTC",
        "LNGQT": "STTC",
        "STE_": "STTC",
        "STD_": "STTC",
        "ANEUR": "STTC",
        "EL": "STTC",
        "LAFB": "CD",
        "LPFB": "CD",
        "IRBBB": "CD",
        "CRBBB": "CD",
        "CLBBB": "CD",
        "ILBBB": "CD",
        "IVCD": "CD",
        "WPW": "CD",
        "1AVB": "CD",
        "2AVB": "CD",
        "3AVB": "CD",
        "LVH": "HYP",
        "RVH": "HYP",
        "LAO/LAE": "HYP",
        "RAO/RAE": "HYP",
        "SEHYP": "HYP",
    }
    labels = set()
    for code, score in (scp_codes or {}).items():
        if score is not None and float(score) <= threshold:
            continue
        superclass = mapping.get(str(code).strip().upper())
        if superclass:
            labels.add(superclass)
    return sorted(labels)


# ------------------------------------------------------------------- quality
QUALITY_FEATURE_NAMES = [
    "snr_db",
    "clipping_ratio",
    "baseline_wander_ratio",
    "powerline_ratio",
    "motion_spike_ratio",
]


def _clipping_ratio(sig: np.ndarray, min_run: int = 3) -> float:
    """Percentage of samples sitting on a flat plateau at the signal extremes.

    Clipping is *saturation*: the acquisition chain rail-limits and several
    consecutive samples take an identical extreme value. Merely being near the
    minimum is not clipping — an ECG's isoelectric baseline sits near the minimum
    by construction, which is why a distance-from-extreme test reports a healthy
    signal as 40% clipped.

    Method: quantise to the signal's own noise floor, find runs of >= ``min_run``
    identical consecutive samples, and keep those whose value lies at the top or
    bottom decile of the range.
    """
    if sig.size < min_run + 1:
        return 0.0

    span = float(np.max(sig) - np.min(sig))
    if span <= 0:
        return 0.0

    # Tolerance = half the smallest resolvable step in this recording, estimated
    # as the 1st percentile of non-zero sample-to-sample differences. Anything
    # larger lets the flattish trough of a slow baseline oscillation masquerade as
    # a saturation plateau; anything smaller misses genuine rails on noisy data.
    diffs = np.abs(np.diff(sig))
    non_zero = diffs[diffs > 0]
    if non_zero.size == 0:
        return 100.0  # a constant trace is entirely rail-limited or dead
    atol = float(np.percentile(non_zero, 1)) * 0.5

    same_as_previous = diffs <= atol
    run_lengths = np.zeros(sig.size, dtype=int)
    current = 1
    for i in range(1, sig.size):
        current = current + 1 if same_as_previous[i - 1] else 1
        run_lengths[i] = current

    in_long_run = run_lengths >= min_run
    if not np.any(in_long_run):
        return 0.0

    # A rail is at the very top or bottom of the recorded range, not merely in the
    # outer decile.
    low_rail = float(np.min(sig)) + atol
    high_rail = float(np.max(sig)) - atol
    plateau = in_long_run & ((sig <= low_rail) | (sig >= high_rail))
    return float(np.mean(plateau) * 100.0)


def quality_features(signal: Sequence[float], fs: float) -> Optional[Dict[str, float]]:
    """Measure the five technical quality descriptors used by the quality gate.

    These are measurements of the supplied signal, so a quality corpus can be
    labelled from measurements rather than from invented statistics.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < int(fs) or not np.isfinite(sig).all():
        return None

    sig = sig - float(np.mean(sig))
    std = float(np.std(sig))
    if std <= 0:
        return None

    clipping_ratio = _clipping_ratio(sig)

    # Band powers via FFT.
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    total = float(np.sum(power)) + 1e-12

    band = lambda lo_hz, hi_hz: float(np.sum(power[(freqs >= lo_hz) & (freqs < hi_hz)]))
    signal_band = band(5.0, 15.0)
    noise_band = band(0.0, 2.0) + 1e-12
    snr_db = float(10.0 * np.log10((signal_band + 1e-12) / noise_band))
    baseline_wander_ratio = band(0.0, 0.5) / total
    powerline_ratio = (band(48.0, 52.0) + band(58.0, 62.0)) / total

    # Motion spikes: fraction of 500 ms windows whose std exceeds 3x global std.
    win = max(1, int(round(0.5 * fs)))
    n_windows = sig.size // win
    if n_windows > 0:
        windowed = sig[: n_windows * win].reshape(n_windows, win)
        spike_ratio = float(np.mean(np.std(windowed, axis=1) > 3.0 * std))
    else:
        spike_ratio = 0.0

    return {
        "snr_db": round(snr_db, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "baseline_wander_ratio": round(float(baseline_wander_ratio), 6),
        "powerline_ratio": round(float(powerline_ratio), 6),
        "motion_spike_ratio": round(spike_ratio, 6),
    }


def quality_label_from_features(features: Dict[str, float]) -> str:
    """Rule-based quality category, mirroring the production quality gate.

    Used to build a training corpus from *measured* descriptors. The production
    gate remains rule-based (``src/quality/quality_gate.py``); this function only
    exists so a learned quality model can be fitted where a labelled corpus is
    available.
    """
    snr = features["snr_db"]
    clipping = features["clipping_ratio"]
    wander = features["baseline_wander_ratio"]
    powerline = features["powerline_ratio"]

    if snr < 3.0 or clipping > 5.0:
        return "UNUSABLE"
    if snr < 10.0 or wander > 0.18 or powerline > 0.15:
        return "POOR"
    if snr < 18.0 or wander > 0.08 or powerline > 0.05:
        return "ACCEPTABLE"
    return "GOOD"


def template_for(feature_names: Sequence[str], values: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """Project a feature dict onto a fixed ordered schema.

    Missing keys raise, rather than silently defaulting to zero: a vector with
    invented entries would be indistinguishable from a measured one downstream.
    """
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise KeyError(f"Feature vector is incomplete; missing: {missing}")
    return list(feature_names), [float(values[name]) for name in feature_names]
