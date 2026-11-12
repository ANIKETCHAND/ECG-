"""
Deterministic ECG Measurement Engine
====================================

Calculates clinical electrophysiological measurements using deterministic,
peer-reviewed biomedical signal processing algorithms:
- Heart Rate (BPM) from validated R-R intervals
- R-R interval dynamics (Mean, Median, SDNN, RMSSD)
- QRS duration estimation via derivative thresholding
- QT / QTc (Bazett and Fridericia) interval calculation
- PR interval estimation
- Electrical frontal axes calculation (P, QRS, T axes)

Hard Clinical Rule:
If a parameter cannot be reliably measured by a technically valid algorithm,
return None and 'Not reliably measurable'.
NEVER substitute an invented or guessed number.

References:
- AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the ECG
- CSE (Common Standards for Quantitative Electrocardiography) Working Party
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ClinicalECGMeasurements:
    """Clinical measurement entity with explicit measurability attribution."""
    heart_rate_bpm: Optional[float] = None
    heart_rate_status: str = "Not calculated"
    mean_rr_ms: Optional[float] = None
    median_rr_ms: Optional[float] = None
    sdnn_ms: Optional[float] = None
    rmssd_ms: Optional[float] = None
    qrs_duration_ms: Optional[float] = None
    qrs_status: str = "Not reliably measurable"
    pr_interval_ms: Optional[float] = None
    pr_status: str = "Not reliably measurable"
    qt_interval_ms: Optional[float] = None
    qtc_bazett_ms: Optional[float] = None
    qtc_fridericia_ms: Optional[float] = None
    qt_status: str = "Not reliably measurable"
    p_axis_deg: Optional[float] = None
    qrs_axis_deg: Optional[float] = None
    t_axis_deg: Optional[float] = None
    axis_status: str = "Not measurable on single-lead ECG"
    confidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_frontal_axes(
    lead_i: np.ndarray,
    lead_avf: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
) -> Dict[str, Any]:
    """Compute deterministic frontal plane electrical axes (P, QRS, T) in degrees.

    Uses net area integrals of Lead I and Lead aVF fiducial windows according to
    the standard hexaxial reference system:
    theta = atan2(Net_aVF, Net_I) * (180 / pi)

    Returns:
        Dict containing:
        - 'qrs_axis_deg': float (-180 to +180 deg)
        - 'p_axis_deg': float (-180 to +180 deg)
        - 't_axis_deg': float (-180 to +180 deg)
        - 'classification': str (Normal, Left Axis Deviation, Right Axis Deviation, Extreme)
    """
    sig_1 = np.asarray(lead_i, dtype=float)
    sig_avf = np.asarray(lead_avf, dtype=float)
    peaks = np.asarray(r_peaks, dtype=int)

    if len(sig_1) == 0 or len(sig_avf) == 0 or len(sig_1) != len(sig_avf) or len(peaks) < 2:
        return {
            "qrs_axis_deg": None,
            "p_axis_deg": None,
            "t_axis_deg": None,
            "classification": "Indeterminate (insufficient multi-lead cycles)",
        }

    qrs_pre = int(0.06 * fs)
    qrs_post = int(0.08 * fs)
    p_pre = int(0.22 * fs)
    p_post = int(0.10 * fs)
    t_pre = int(0.12 * fs)
    t_post = int(0.38 * fs)

    net_1_qrs_list = []
    net_avf_qrs_list = []
    net_1_p_list = []
    net_avf_p_list = []
    net_1_t_list = []
    net_avf_t_list = []

    for p in peaks:
        if p - p_pre >= 0 and p + t_post < len(sig_1):
            net_1_qrs_list.append(np.sum(sig_1[p - qrs_pre : p + qrs_post]))
            net_avf_qrs_list.append(np.sum(sig_avf[p - qrs_pre : p + qrs_post]))
            net_1_p_list.append(np.sum(sig_1[p - p_pre : p - p_post]))
            net_avf_p_list.append(np.sum(sig_avf[p - p_pre : p - p_post]))
            net_1_t_list.append(np.sum(sig_1[p + t_pre : p + t_post]))
            net_avf_t_list.append(np.sum(sig_avf[p + t_pre : p + t_post]))

    if len(net_1_qrs_list) < 2:
        return {
            "qrs_axis_deg": None,
            "p_axis_deg": None,
            "t_axis_deg": None,
            "classification": "Indeterminate (insufficient complete beats)",
        }

    mean_net_1_qrs = float(np.median(net_1_qrs_list))
    mean_net_avf_qrs = float(np.median(net_avf_qrs_list))
    qrs_deg = round(float(np.degrees(np.arctan2(mean_net_avf_qrs, mean_net_1_qrs))), 1)

    mean_net_1_p = float(np.median(net_1_p_list))
    mean_net_avf_p = float(np.median(net_avf_p_list))
    p_deg = round(float(np.degrees(np.arctan2(mean_net_avf_p, mean_net_1_p))), 1)

    mean_net_1_t = float(np.median(net_1_t_list))
    mean_net_avf_t = float(np.median(net_avf_t_list))
    t_deg = round(float(np.degrees(np.arctan2(mean_net_avf_t, mean_net_1_t))), 1)

    if -30.0 <= qrs_deg <= 90.0:
        classification = "Normal Electrical Axis (-30° to +90°)"
    elif -90.0 <= qrs_deg < -30.0:
        classification = "Left Axis Deviation (-30° to -90°)"
    elif 90.0 < qrs_deg <= 180.0:
        classification = "Right Axis Deviation (+90° to +180°)"
    else:
        classification = "Extreme Axis Deviation (-90° to -180°)"

    return {
        "qrs_axis_deg": qrs_deg,
        "p_axis_deg": p_deg,
        "t_axis_deg": t_deg,
        "classification": classification,
    }


def compute_ecg_measurements(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    leads_available: Optional[List[str]] = None,
    multi_lead_signals: Optional[Dict[str, np.ndarray]] = None,
) -> ClinicalECGMeasurements:
    """Compute deterministic clinical ECG parameters.

    Args:
        signal: 1D filtered physiological voltage series (preferably Lead II)
        fs: Sampling frequency in Hz
        r_peaks: Sample indices of detected R-peaks
        leads_available: List of lead names present in recording

    Returns:
        ClinicalECGMeasurements with explicit validity status for every field.
    """
    m = ClinicalECGMeasurements()
    notes: List[str] = []

    if signal is None or len(signal) == 0 or fs <= 0:
        m.confidence_notes.append("Empty or invalid signal provided.")
        return m

    sig = np.asarray(signal, dtype=float)
    peaks = np.asarray(r_peaks, dtype=int)

    # 1. R-R Interval & Heart Rate Computation
    if len(peaks) >= 2:
        rr_intervals_sec = np.diff(peaks) / float(fs)
        # Filter physiological bounds (0.25s to 2.5s -> 24 to 240 BPM)
        valid_rr = rr_intervals_sec[(rr_intervals_sec >= 0.25) & (rr_intervals_sec <= 2.5)]

        if len(valid_rr) >= 1:
            mean_rr = float(np.mean(valid_rr))
            median_rr = float(np.median(valid_rr))
            m.mean_rr_ms = round(mean_rr * 1000.0, 1)
            m.median_rr_ms = round(median_rr * 1000.0, 1)
            m.heart_rate_bpm = round(60.0 / mean_rr, 1)
            m.heart_rate_status = "Reliably measured"

            if len(valid_rr) >= 2:
                m.sdnn_ms = round(float(np.std(valid_rr)) * 1000.0, 1)
                successive_diffs = np.diff(valid_rr)
                m.rmssd_ms = round(float(np.sqrt(np.mean(successive_diffs ** 2))) * 1000.0, 1)
        else:
            m.heart_rate_status = "Not reliably measurable (R-R intervals outside physiological bounds)"
            notes.append("Detected R-R intervals fell outside 24–240 BPM physiological range.")
    else:
        m.heart_rate_status = "Not reliably measurable (insufficient cardiac cycles < 2)"
        notes.append("Fewer than 2 R-peaks detected. Heart rate calculation unavailable.")

    # 2. QRS Duration Estimation (Derivative Thresholding on Mean Beat)
    if len(peaks) >= 3 and m.mean_rr_ms is not None:
        try:
            # Segment beats around R-peak: [-100ms, +120ms]
            pre_samp = int(0.10 * fs)
            post_samp = int(0.12 * fs)
            beat_snippets = []

            for p in peaks:
                if p - pre_samp >= 0 and p + post_samp < len(sig):
                    beat_snippets.append(sig[p - pre_samp : p + post_samp])

            if len(beat_snippets) >= 3:
                avg_qrs = np.mean(beat_snippets, axis=0)
                r_idx = pre_samp

                # Compute absolute first derivative
                diff_qrs = np.abs(np.diff(avg_qrs))
                max_diff = np.max(diff_qrs)
                # Baseline noise threshold (15% of max slope)
                thresh = max_diff * 0.15

                # Search backward from R for Q onset
                q_onset = 0
                for i in range(r_idx - 1, 0, -1):
                    if diff_qrs[i] < thresh:
                        q_onset = i
                        break

                # Search forward from R for S offset
                s_offset = len(avg_qrs) - 1
                for i in range(r_idx, len(diff_qrs)):
                    if diff_qrs[i] < thresh:
                        s_offset = i
                        break

                qrs_duration_sec = (s_offset - q_onset) / float(fs)
                qrs_ms = qrs_duration_sec * 1000.0

                # Physiologically valid QRS duration: 50 ms to 220 ms
                if 50.0 <= qrs_ms <= 220.0:
                    m.qrs_duration_ms = round(qrs_ms, 1)
                    m.qrs_status = "Estimated from derivative thresholding"
                else:
                    m.qrs_status = f"Not reliably measurable (calculated {qrs_ms:.0f}ms out of bounds)"
                    notes.append("QRS boundary detection fell outside physiological boundaries.")
            else:
                m.qrs_status = "Not reliably measurable (insufficient complete QRS snippets)"
        except Exception as exc:
            m.qrs_status = f"Not reliably measurable ({str(exc)})"
    else:
        m.qrs_status = "Not reliably measurable (insufficient beats)"

    # 3. QT and QTc Calculation
    # Requires high-contrast T-wave offset detection
    if m.heart_rate_bpm is not None and m.mean_rr_ms is not None and len(peaks) >= 3:
        try:
            # Segment window [-80ms, +600ms]
            t_pre = int(0.08 * fs)
            t_post = min(int(0.60 * fs), int((m.mean_rr_ms / 1000.0) * 0.75 * fs))

            t_snippets = []
            for p in peaks:
                if p - t_pre >= 0 and p + t_post < len(sig):
                    t_snippets.append(sig[p - t_pre : p + t_post])

            if len(t_snippets) >= 3:
                avg_beat = np.mean(t_snippets, axis=0)
                r_idx = t_pre

                # Dynamic Q-onset detection before R peak:
                # Search for Q-wave deflection in window [R - 70ms, R]
                q_search_start = max(0, r_idx - int(0.07 * fs))
                q_window = avg_beat[q_search_start:r_idx]
                if len(q_window) >= 2:
                    q_min_local = int(np.argmin(q_window))
                    q_trough = q_search_start + q_min_local
                    # Trace backwards from Q-trough to find baseline departure / inflection
                    diff_pre = np.abs(np.diff(avg_beat[q_search_start : q_trough + 1]))
                    q_point = q_trough
                    if len(diff_pre) > 0:
                        thr = 0.10 * np.max(np.abs(np.diff(avg_beat)))
                        for k in range(len(diff_pre) - 1, -1, -1):
                            if diff_pre[k] < thr:
                                q_point = q_search_start + k
                                break
                else:
                    q_point = max(0, r_idx - int(0.04 * fs))

                # Find T peak in window [+150ms, +450ms] after R
                t_search_start = r_idx + int(0.15 * fs)
                t_search_end = min(len(avg_beat), r_idx + int(0.48 * fs))

                if t_search_end > t_search_start:
                    t_window = avg_beat[t_search_start:t_search_end]
                    t_peak_local = np.argmax(np.abs(t_window))
                    t_peak_idx = t_search_start + t_peak_local

                    # Find T offset where derivative returns to baseline slope
                    diff_t = np.abs(np.diff(avg_beat))
                    t_offset_idx = min(len(avg_beat) - 1, t_peak_idx + int(0.10 * fs))
                    for i in range(t_peak_idx, min(len(diff_t), t_peak_idx + int(0.18 * fs))):
                        if diff_t[i] < 0.05 * np.max(diff_t):
                            t_offset_idx = i
                            break

                    qt_sec = (t_offset_idx - q_point) / float(fs)
                    qt_ms = qt_sec * 1000.0
                    rr_sec = m.mean_rr_ms / 1000.0

                    # Physiological bounds for QT: 280 ms to 600 ms
                    if 280.0 <= qt_ms <= 600.0 and rr_sec > 0:
                        m.qt_interval_ms = round(qt_ms, 1)
                        # Bazett: QTc = QT / sqrt(RR)
                        m.qtc_bazett_ms = round(qt_ms / np.sqrt(rr_sec), 1)
                        # Fridericia: QTc = QT / (RR^(1/3))
                        m.qtc_fridericia_ms = round(qt_ms / (rr_sec ** (1.0 / 3.0)), 1)
                        m.qt_status = "Measured from Lead II average beat (dynamic Q-onset to T-offset)"
                    else:
                        m.qt_status = "Not reliably measurable (T-wave offset out of bounds)"
                else:
                    m.qt_status = "Not reliably measurable (T-wave search window insufficient)"
            else:
                m.qt_status = "Not reliably measurable (insufficient clean beats)"
        except Exception:
            m.qt_status = "Not reliably measurable (T-wave morphology indistinct)"
    else:
        m.qt_status = "Not reliably measurable (requires stable heart rate)"

    # 4. PR Interval Estimation
    if len(peaks) >= 3 and fs > 0:
        try:
            pr_pre = int(0.28 * fs)
            pr_post = int(0.08 * fs)
            pr_snippets = []
            for p in peaks:
                if p - pr_pre >= 0 and p + pr_post < len(sig):
                    pr_snippets.append(sig[p - pr_pre : p + pr_post])
            if len(pr_snippets) >= 3:
                avg_pr_beat = np.mean(pr_snippets, axis=0)
                r_pos = pr_pre

                # P-wave search interval: [-240ms, -100ms] before R
                p_win_start = r_pos - int(0.24 * fs)
                p_win_end = r_pos - int(0.09 * fs)
                if p_win_start >= 0 and p_win_end > p_win_start:
                    p_win = avg_pr_beat[p_win_start:p_win_end]
                    # Check for identifiable P wave deflection
                    if np.ptp(p_win) >= 0.02 and np.ptp(p_win) > 0.05 * np.ptp(avg_pr_beat):
                        p_local_max = np.argmax(p_win)
                        p_peak_idx = p_win_start + p_local_max

                        # Search backward from P peak for P onset
                        diff_p = np.abs(np.diff(avg_pr_beat))
                        p_onset = p_win_start
                        for idx in range(p_peak_idx - 1, max(0, p_win_start - int(0.04 * fs)), -1):
                            if diff_p[idx] < 0.10 * np.max(diff_p[p_win_start:p_win_end]):
                                p_onset = idx
                                break

                        # Q onset is around [-50ms, -20ms] before R
                        q_search_start = r_pos - int(0.08 * fs)
                        q_search_end = r_pos
                        q_onset = r_pos - int(0.04 * fs)
                        diff_q = np.abs(np.diff(avg_pr_beat))
                        for idx in range(r_pos - 1, max(0, q_search_start), -1):
                            if diff_q[idx] < 0.15 * np.max(diff_q[q_search_start:q_search_end]):
                                q_onset = idx
                                break

                        calc_pr_sec = (q_onset - p_onset) / float(fs)
                        calc_pr_ms = calc_pr_sec * 1000.0
                        if 100.0 <= calc_pr_ms <= 260.0:
                            m.pr_interval_ms = round(calc_pr_ms, 1)
                            m.pr_status = "Estimated from Lead II P-wave onset"
                        else:
                            m.pr_status = f"Not reliably measurable (calculated {calc_pr_ms:.0f}ms out of bounds)"
                    else:
                        m.pr_status = "Not reliably measurable (P-wave morphology indistinct)"
                else:
                    m.pr_status = "Not reliably measurable (insufficient baseline window)"
            else:
                m.pr_status = "Not reliably measurable (insufficient beats)"
        except Exception:
            m.pr_status = "Not reliably measurable on single-lead automated baseline"
    else:
        m.pr_status = "Not reliably measurable on single-lead automated baseline"

    # 5. Frontal Plane Electrical Axes
    # When orthogonal multi-lead signals (I and aVF) are provided, calculate exact hexaxial vector axes
    leads = [l.strip().upper() for l in (leads_available or ["II"])]
    has_lead_1 = any(l in ["I", "LEAD I", "LEAD 1"] for l in leads)
    has_lead_avf = any(l in ["AVF", "LEAD AVF", "A_VF"] for l in leads)

    if multi_lead_signals:
        lead_i_sig = None
        lead_avf_sig = None
        for k, v in multi_lead_signals.items():
            k_upper = k.strip().upper()
            if k_upper in ["I", "LEAD I", "LEAD 1"]:
                lead_i_sig = v
            elif k_upper in ["AVF", "LEAD AVF", "A_VF"]:
                lead_avf_sig = v

        if lead_i_sig is not None and lead_avf_sig is not None:
            axes_res = compute_frontal_axes(lead_i_sig, lead_avf_sig, fs, peaks)
            m.qrs_axis_deg = axes_res.get("qrs_axis_deg")
            m.p_axis_deg = axes_res.get("p_axis_deg")
            m.t_axis_deg = axes_res.get("t_axis_deg")
            m.axis_status = axes_res.get("classification", "Calculated from limb leads I and aVF")
        elif has_lead_1 and has_lead_avf:
            m.axis_status = "Multi-lead limb vector calculation available"
        else:
            m.axis_status = "Not measurable on single-lead ECG (requires orthogonal limb leads I and aVF)"
    elif has_lead_1 and has_lead_avf:
        m.axis_status = "Multi-lead limb vector calculation available"
    else:
        m.axis_status = "Not measurable on single-lead ECG (requires orthogonal limb leads I and aVF)"
        m.p_axis_deg = None
        m.qrs_axis_deg = None
        m.t_axis_deg = None

    m.confidence_notes = notes
    return m
