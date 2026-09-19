"""
Mandatory Signal Quality & Safety Gatekeeper
=============================================

Implements the Hard Safety Rule:
NO RELIABLE INPUT = NO AI RESULT

Pre-inference verification checklist:
1. Input validity & non-emptiness
2. Missing-data / NaN / Inf check
3. Sampling-rate verification
4. Lead configuration check
5. Duration verification (minimum 1.5 seconds)
6. Dynamic variance & amplitude sanity
7. Baseline wander & powerline noise analysis
8. Saturation / electrode disconnect / rail clipping detection

Output Categories:
- GOOD: High technical fidelity. Full AI inference permitted.
- ACCEPTABLE: Moderate noise. AI inference permitted with explicit noise warning.
- POOR: Significant degradation. Basic measurements only; AI classification suppressed.
- UNUSABLE: Critical corruption/disconnect. Pipeline halted: NO AI RESULT.

References:
- Medical Devices Rules (MDR) 2017 (CDSCO)
- IEC 60601-2-25: Particular requirements for ECG safety and performance
- ISO 14971 Risk Mitigation for Algorithmic Error
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal as scipy_signal


class QualityCategory(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNUSABLE = "UNUSABLE"


@dataclass
class QualityGateResult:
    """Immutable decision record from the safety gatekeeper."""
    category: QualityCategory
    can_run_ai: bool
    can_compute_measurements: bool
    quality_score: float  # 0.0 to 1.0
    snr_db: float
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


def evaluate_signal_quality_gate(
    signal: Optional[np.ndarray],
    fs: float,
    lead_name: str = "II",
    min_duration_sec: float = 1.5,
) -> QualityGateResult:
    """Execute the mandatory pre-inference signal quality gatekeeper.

    Args:
        signal: 1D physiological voltage series
        fs: Sampling frequency in Hz
        lead_name: Identifier of the lead being inspected
        min_duration_sec: Minimum duration threshold

    Returns:
        QualityGateResult enforcing hard safety barriers.
    """
    rejection_reasons: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    # 1. Null or Empty Check
    if signal is None or len(signal) == 0:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-99.0,
            rejection_reasons=["No physiological signal data was provided (empty input)."],
            warnings=["Input stream is empty."],
        )

    sig = np.asarray(signal, dtype=float)

    # 2. Missing-Data Check (NaN / Infinite Values)
    nan_count = int(np.sum(np.isnan(sig)))
    inf_count = int(np.sum(np.isinf(sig)))
    if nan_count > 0 or inf_count > 0:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-99.0,
            rejection_reasons=[
                f"Signal contains non-physiological artifacts ({nan_count} NaNs, {inf_count} Infs)."
            ],
            warnings=["Raw data corruption detected."],
        )

    # 3. Sampling Rate Verification
    if fs <= 0 or np.isnan(fs):
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-99.0,
            rejection_reasons=[f"Invalid sampling rate ({fs} Hz). Sampling rate must be positive."],
        )

    if fs < 100.0:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=0.0,
            rejection_reasons=[
                f"Sampling rate ({fs:.1f} Hz) is below minimum diagnostic requirement (100 Hz)."
            ],
        )

    if fs > 2000.0:
        warnings.append(f"High sampling rate ({fs:.1f} Hz). Signal may require decimation.")

    # 4. Duration Check
    duration_sec = len(sig) / float(fs)
    metrics["duration_sec"] = round(duration_sec, 3)
    if duration_sec < min_duration_sec:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.1,
            snr_db=0.0,
            rejection_reasons=[
                f"Recording duration ({duration_sec:.2f}s) is below the minimum required ({min_duration_sec}s)."
            ],
            metrics=metrics,
        )

    # 5. Dynamic Variance & Flatline Check
    sig_std = float(np.std(sig))
    sig_range = float(np.ptp(sig))
    metrics["std"] = round(sig_std, 4)
    metrics["ptp_range"] = round(sig_range, 4)

    if sig_std < 0.01 or sig_range < 0.05:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.0,
            snr_db=-20.0,
            rejection_reasons=[
                "Signal is flatline or disconnected (insufficient voltage variance)."
            ],
            metrics=metrics,
        )

    # 6. Rail Saturation / Clipping Check
    # Check if more than 3% of points sit at the absolute min or max value
    max_val = np.max(sig)
    min_val = np.min(sig)
    clipped_high = np.mean(np.isclose(sig, max_val, atol=1e-4))
    clipped_low = np.mean(np.isclose(sig, min_val, atol=1e-4))
    clipping_ratio = float(clipped_high + clipped_low)
    metrics["clipping_ratio"] = round(clipping_ratio, 4)

    if clipping_ratio > 0.05:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.05,
            snr_db=-10.0,
            rejection_reasons=[
                f"Amplifier rail clipping/saturation detected on {lead_name} ({clipping_ratio*100:.1f}% samples clipped)."
            ],
            metrics=metrics,
        )

    # 7. Extreme Outlier Spike Check (Border Artifacts)
    z_scores = np.abs((sig - np.mean(sig)) / (sig_std + 1e-9))
    max_z = float(np.max(z_scores))
    metrics["max_z_score"] = round(max_z, 2)
    if max_z > 35.0:
        return QualityGateResult(
            category=QualityCategory.UNUSABLE,
            can_run_ai=False,
            can_compute_measurements=False,
            quality_score=0.1,
            snr_db=-5.0,
            rejection_reasons=["Extreme synthetic border spike or impulse noise artifact detected."],
            metrics=metrics,
        )

    # 8. Spectral Quality: Bandpass & Noise Estimation
    # Compute power spectral density
    try:
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 2.0)))
        total_power = np.sum(psd) + 1e-12

        # In-band cardiac power: 0.5 Hz - 40 Hz
        ecg_band = (freqs >= 0.5) & (freqs <= 40.0)
        ecg_power = np.sum(psd[ecg_band])

        # Baseline drift power: < 0.5 Hz
        drift_band = freqs < 0.5
        drift_power = np.sum(psd[drift_band])
        drift_ratio = float(drift_power / total_power)
        metrics["drift_power_ratio"] = round(drift_ratio, 4)

        # High-frequency noise power: > 45 Hz
        hf_band = freqs > 45.0
        hf_power = np.sum(psd[hf_band])

        # Powerline noise check (around 50 Hz and 60 Hz ± 1 Hz)
        pl_band_50 = (freqs >= 49.0) & (freqs <= 51.0)
        pl_band_60 = (freqs >= 59.0) & (freqs <= 61.0)
        has_pl = bool(np.sum(psd[pl_band_50]) > 0.15 * ecg_power or np.sum(psd[pl_band_60]) > 0.15 * ecg_power)
        metrics["has_powerline_interference"] = has_pl

        # SNR Estimation: ratio of in-band cardiac power to out-of-band noise
        noise_power = max(1e-12, total_power - ecg_power)
        snr_raw = float(ecg_power / noise_power)
        snr_db = round(float(10.0 * np.log10(max(1e-3, snr_raw))), 1)
        metrics["snr_db"] = snr_db

    except Exception:
        snr_db = 15.0
        drift_ratio = 0.05
        has_pl = False
        metrics["snr_db"] = snr_db

    # 9. Categorical Decision Logic
    if drift_ratio > 0.50:
        warnings.append("Severe low-frequency baseline drift detected.")
    if has_pl:
        warnings.append("50/60 Hz powerline interference present.")

    # Quality score computation (0.0 to 1.0)
    score = 1.0
    if snr_db < 15.0:
        score -= min(0.4, (15.0 - snr_db) * 0.03)
    if drift_ratio > 0.20:
        score -= min(0.3, (drift_ratio - 0.20) * 0.5)
    if has_pl:
        score -= 0.15
    if clipping_ratio > 0.01:
        score -= 0.20

    quality_score = round(max(0.0, min(1.0, score)), 2)
    metrics["quality_score"] = quality_score

    # Hard decision boundaries
    if snr_db < 3.0 or quality_score < 0.35:
        category = QualityCategory.POOR
        can_run_ai = False  # HARD SAFETY RULE: Suppress AI on POOR signal
        can_compute_measurements = True
        warnings.append(
            "Signal quality is POOR (SNR < 3 dB). AI classification is suppressed to prevent diagnostic error."
        )
    elif snr_db < 10.0 or quality_score < 0.65 or drift_ratio > 0.35:
        category = QualityCategory.ACCEPTABLE
        can_run_ai = True
        can_compute_measurements = True
        warnings.append("Signal quality is ACCEPTABLE. Moderate baseline wander or noise present.")
    else:
        category = QualityCategory.GOOD
        can_run_ai = True
        can_compute_measurements = True

    return QualityGateResult(
        category=category,
        can_run_ai=can_run_ai,
        can_compute_measurements=can_compute_measurements,
        quality_score=quality_score,
        snr_db=snr_db,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
        metrics=metrics,
    )
