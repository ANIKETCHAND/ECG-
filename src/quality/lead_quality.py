"""
Lead Quality Assessment Module
==============================
Evaluates an individual ECG lead across electrical, physiological, and technical dimensions.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

try:
    from src.quality.artifact_detection import detect_motion_and_muscle_artifacts
    from src.quality.baseline_wander import analyze_baseline_wander
    from src.quality.clipping_detection import detect_clipping_and_flatline
    from src.quality.noise_detection import analyze_noise_and_powerline
except ImportError:
    from quality.artifact_detection import detect_motion_and_muscle_artifacts
    from quality.baseline_wander import analyze_baseline_wander
    from quality.clipping_detection import detect_clipping_and_flatline
    from quality.noise_detection import analyze_noise_and_powerline


def evaluate_single_lead_quality(
    signal: np.ndarray,
    fs: float,
    lead_name: str = "II",
) -> Dict[str, Any]:
    """Inspect one ECG lead and return categorical grade and technical metrics."""
    if signal is None or len(signal) == 0:
        return {
            "lead": lead_name,
            "status": "UNUSABLE",
            "quality_score": 0.0,
            "snr_db": -99.0,
            "issues": ["Empty or null signal"],
            "recommendation": f"Check electrode connection on {lead_name}.",
            "metrics": {},
        }

    sig = np.asarray(signal, dtype=float)
    issues: List[str] = []
    penalties = 0.0

    # 1. Check NaNs / Infs
    if np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
        return {
            "lead": lead_name,
            "status": "UNUSABLE",
            "quality_score": 0.0,
            "snr_db": -99.0,
            "issues": ["Signal contains invalid NaN or Infinite values"],
            "recommendation": f"Re-acquire digital signal on {lead_name}.",
            "metrics": {},
        }

    # 2. Clipping and Flatline
    is_clipped, is_flatline, clip_metrics = detect_clipping_and_flatline(sig)
    if is_flatline:
        return {
            "lead": lead_name,
            "status": "UNUSABLE",
            "quality_score": 0.0,
            "snr_db": -99.0,
            "issues": ["Electrode disconnected / continuous flatline"],
            "recommendation": f"Inspect electrode contact and lead wire for {lead_name}.",
            "metrics": clip_metrics,
        }


    if is_clipped:
        issues.append("Amplifier clipping / rail saturation detected")
        penalties += 0.35

    # 3. Dynamic range check
    sig_std = float(np.std(sig))
    sig_range = float(np.ptp(sig))
    if sig_std < 0.03 or sig_range < 0.15:
        issues.append("Low amplitude / attenuated voltage trace")
        penalties += 0.30
    elif sig_range > 15.0:
        issues.append("Excessive voltage amplitude / artifact (>15 mV)")
        penalties += 0.40

    # 4. Noise and Powerline
    snr_db, has_powerline, noise_metrics = analyze_noise_and_powerline(sig, fs)
    if snr_db < 0.0:
        issues.append(f"Severe noise degradation (SNR: {snr_db:.1f} dB)")
        penalties += 0.40
    elif snr_db < 8.0:
        issues.append(f"Moderate noise (SNR: {snr_db:.1f} dB)")
        penalties += 0.20

    if has_powerline:
        issues.append("50/60 Hz powerline interference detected")
        penalties += 0.15

    # 5. Baseline Wander
    has_drift, drift_ratio, drift_metrics = analyze_baseline_wander(sig, fs)
    if has_drift:
        issues.append(f"Excessive baseline respiration drift (drift ratio: {drift_ratio:.2f})")
        penalties += 0.25

    # 6. Muscle and Motion Artifacts
    has_muscle, has_spikes, art_metrics = detect_motion_and_muscle_artifacts(sig, fs)
    if has_spikes:
        issues.append("Electrode motion artifact spikes detected")
        penalties += 0.25
    if has_muscle:
        issues.append("Somatic tremor / muscle tremor artifact detected")
        penalties += 0.20

    # Compute overall quality score (0.0 to 1.0)
    score = max(0.0, min(1.0, 1.0 - penalties))

    if score >= 0.80 and snr_db >= 10.0:
        status = "GOOD"
        rec = "Optimal signal quality. Proceed with clinical interpretation."
    elif score >= 0.50 and snr_db >= 4.0:
        status = "ACCEPTABLE"
        rec = "Acceptable trace with minor noise. Review baseline before sign-off."
    elif score >= 0.20:
        status = "POOR"
        rec = f"Significant artifact on {lead_name}. Check patient movement and lead cables."
    else:
        status = "UNUSABLE"
        rec = f"Unusable signal on {lead_name}. Repeat ECG acquisition."

    return {
        "lead": lead_name,
        "status": status,
        "quality_score": round(score, 2),
        "snr_db": snr_db,
        "issues": issues,
        "recommendation": rec,
        "metrics": {**clip_metrics, **noise_metrics, **drift_metrics, **art_metrics},
    }
