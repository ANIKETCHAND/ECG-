"""
Feature Attribution & Evidence Subsystem
========================================
Quantifies specific feature deviations that drove the model's abnormal classification.
Enables clinician explainability: explains WHY the AI flagged ectopy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np


def compute_feature_attribution_for_beat(
    beat_features: np.ndarray,
    baseline_features: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Compute feature deviations for an aberrant beat relative to normal beats.

    Args:
        beat_features: 1D array of feature values for the candidate beat.
        baseline_features: 2D array of feature values for normal/baseline beats.
        feature_names: List of feature names.
        top_k: Number of highest-deviation features to return.

    Returns:
        List of dictionaries detailing top contributing features, deviations, and clinical interpretations.
    """
    if len(baseline_features) == 0:
        return []

    mean_baseline = np.mean(baseline_features, axis=0)
    std_baseline = np.std(baseline_features, axis=0)
    std_baseline[std_baseline == 0] = 1e-6

    # Z-scores relative to baseline
    z_scores = (beat_features - mean_baseline) / std_baseline
    abs_z = np.abs(z_scores)

    # Sort indices by absolute z-score
    top_indices = np.argsort(abs_z)[::-1][:top_k]

    attributions = []
    clinical_explanations = {
        "local_rr_ratio": "Prematurity index relative to surrounding sinus rhythm",
        "pre_rr": "Coupling interval duration preceding the ventricular depolarization",
        "post_rr": "Post-extrasystolic pause duration",
        "qrs_width_samples": "Depolarization duration (broad QRS indicates ventricular origin)",
        "autocorr_first_peak": "Rhythm periodicity disruption",
        "spectral_entropy": "Complexity and disorganization in spectral frequency content",
        "max_power": "Peak power concentration in frequency spectrum",
        "r_peak_amplitude": "Voltage magnitude of initial R wave deflection",
        "peak_to_peak_amplitude": "Total voltage swing across QRS complex",
    }

    for idx in top_indices:
        feat_name = feature_names[idx]
        val = float(beat_features[idx])
        base_val = float(mean_baseline[idx])
        z = float(z_scores[idx])

        attributions.append({
            "feature_name": feat_name,
            "beat_value": round(val, 4),
            "normal_baseline_mean": round(base_val, 4),
            "z_score_deviation": round(z, 2),
            "clinical_significance": clinical_explanations.get(
                feat_name, "Quantitative morphological or spectral metric deviation"
            ),
        })

    return attributions
