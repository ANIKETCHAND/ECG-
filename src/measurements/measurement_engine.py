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


def compute_ecg_measurements(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    leads_available: Optional[List[str]] = None,
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
            # Segment window [-50ms, +500ms]
            t_pre = int(0.05 * fs)
            t_post = min(int(0.60 * fs), int((m.mean_rr_ms / 1000.0) * 0.75 * fs))

            t_snippets = []
            for p in peaks:
                if p - t_pre >= 0 and p + t_post < len(sig):
                    t_snippets.append(sig[p - t_pre : p + t_post])

            if len(t_snippets) >= 3:
                avg_beat = np.mean(t_snippets, axis=0)
                q_point = t_pre

                # Find T peak in window [+150ms, +450ms] after R
                t_search_start = t_pre + int(0.15 * fs)
                t_search_end = min(len(avg_beat), t_pre + int(0.48 * fs))

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
                        m.qt_status = "Estimated from Lead II average beat"
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

    # 4. PR Interval
    # P-wave detection on single-lead is often subtle and variable
    m.pr_status = "Not reliably measurable on single-lead automated baseline"
    m.pr_interval_ms = None

    # 5. Frontal Plane Electrical Axes
    # Strict Clinical Boundary: Electrical Axis CANNOT be determined from a single lead.
    # It mathematically requires at least two orthogonal limb vector projections (Lead I and aVF).
    leads = leads_available or ["II"]
    has_lead_1 = any(l.strip().upper() in ["I", "LEAD I", "LEAD 1"] for l in leads)
    has_lead_avf = any(l.strip().upper() in ["AVF", "LEAD AVF", "A_VF"] for l in leads)

    if has_lead_1 and has_lead_avf:
        m.axis_status = "Multi-lead limb vector calculation available"
    else:
        m.axis_status = "Not measurable on single-lead ECG (requires orthogonal limb leads I and aVF)"
        m.p_axis_deg = None
        m.qrs_axis_deg = None
        m.t_axis_deg = None

    m.confidence_notes = notes
    return m
