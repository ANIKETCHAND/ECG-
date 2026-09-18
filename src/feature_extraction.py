"""
Feature Extraction Module
===========================

Extracts features from heartbeat segments:
- Time-domain features (mean, std, RMS, energy, etc.)
- ECG-specific features (amplitude, width, etc.)
- Frequency-domain features (FFT-based)

Research/educational use only.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from typing import Dict, Optional


def extract_time_domain_features(beat: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract time-domain features from a heartbeat segment.

    Args:
        beat: 1D array of beat samples
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary of time-domain features
    """
    beat = beat.flatten()
    n = len(beat)

    features = {}

    # Basic statistics
    features["mean"] = np.mean(beat)
    features["std"] = np.std(beat)
    features["min"] = np.min(beat)
    features["max"] = np.max(beat)
    features["range"] = features["max"] - features["min"]
    features["median"] = np.median(beat)

    # Signal energy
    features["energy"] = np.sum(beat ** 2)

    # RMS (Root Mean Square)
    features["rms"] = np.sqrt(np.mean(beat ** 2))

    # Mean absolute value
    features["mav"] = np.mean(np.abs(beat))

    # Signal-to-noise ratio (approximate)
    if features["std"] > 0:
        features["snr"] = features["mean"] / features["std"]
    else:
        features["snr"] = 0.0

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(beat)) != 0)
    features["zero_crossing_rate"] = zero_crossings / n if n > 0 else 0.0

    # Autocorrelation (for periodicity)
    if n > 1:
        autocorr = np.correlate(beat - np.mean(beat), beat - np.mean(beat), mode="full")
        autocorr = autocorr[n - 1:]  # Take positive lags
        # Find first peak after zero lag
        peaks, _ = find_peaks(autocorr[1:])
        if len(peaks) > 0:
            features["autocorr_first_peak"] = autocorr[peaks[0] + 1] / autocorr[1] if autocorr[1] != 0 else 0.0
        else:
            features["autocorr_first_peak"] = 0.0
    else:
        features["autocorr_first_peak"] = 0.0

    return features


def extract_ecg_specific_features(beat: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract ECG-specific features from a heartbeat segment.

    Args:
        beat: 1D array of beat samples
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary of ECG-specific features
    """
    beat = beat.flatten()
    features = {}

    # R-peak amplitude (max value)
    r_peak_idx = np.argmax(beat)
    features["r_peak_amplitude"] = beat[r_peak_idx]

    # P-wave amplitude (negative of min, for approximation)
    # This is a simplified approximation
    features["p_wave_amplitude"] = abs(np.min(beat))

    # T-wave amplitude
    # Search in second half for T-wave
    second_half = beat[r_peak_idx + 1:]
    if len(second_half) > 0:
        features["t_wave_amplitude"] = np.max(second_half)
    else:
        features["t_wave_amplitude"] = 0.0

    # Peak-to-peak amplitude (max - min)
    features["peak_to_peak_amplitude"] = np.max(beat) - np.min(beat)

    # Slope features (approximate QRS complex width)
    # Find rising and falling slopes
    slopes = np.diff(beat)
    max_slope = np.max(np.abs(slopes))
    features["max_slope"] = max_slope

    # Approximate QRS width (where slope is high)
    high_slope = np.abs(slopes) > 0.1 * max_slope
    if np.sum(high_slope) > 0:
        features["qrs_width_samples"] = np.sum(high_slope) / fs
    else:
        features["qrs_width_samples"] = 0.0

    return features


def extract_frequency_domain_features(beat: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract frequency-domain features from a heartbeat segment.

    Args:
        beat: 1D array of beat samples
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary of frequency-domain features
    """
    beat = beat.flatten()
    n = len(beat)

    features = {}

    # FFT
    fft_values = fft(beat)
    freqs = fftfreq(n, d=1/fs)

    # Power spectral density
    power_spectrum = np.abs(fft_values) ** 2

    # Total power
    features["total_power"] = np.sum(power_spectrum)

    # Power in specific bands
    # Low frequency (0.5-1 Hz)
    lf_band = np.where((np.abs(freqs) >= 0.5) & (np.abs(freqs) < 1.0))[0]
    if len(lf_band) > 0:
        features["lf_power"] = np.sum(power_spectrum[lf_band])
    else:
        features["lf_power"] = 0.0

    # High frequency (1-10 Hz)
    hf_band = np.where((np.abs(freqs) >= 1.0) & (np.abs(freqs) < 10.0))[0]
    if len(hf_band) > 0:
        features["hf_power"] = np.sum(power_spectrum[hf_band])
    else:
        features["hf_power"] = 0.0

    # Very high frequency (>10 Hz)
    vhf_band = np.where(np.abs(freqs) >= 10.0)[0]
    if len(vhf_band) > 0:
        features["vhf_power"] = np.sum(power_spectrum[vhf_band])
    else:
        features["vhf_power"] = 0.0

    # Dominant frequency
    if len(freqs) > 0:
        features["dominant_frequency"] = freqs[np.argmax(power_spectrum)]
        features["max_power"] = np.max(power_spectrum)
    else:
        features["dominant_frequency"] = 0.0
        features["max_power"] = 0.0

    # Spectral entropy (approximate)
    if features["total_power"] > 0:
        psd_normalized = power_spectrum / features["total_power"]
        # Remove zeros for entropy calculation
        psd_nonzero = psd_normalized[psd_normalized > 0]
        if len(psd_nonzero) > 0:
            features["spectral_entropy"] = -np.sum(psd_nonzero * np.log2(psd_nonzero))
        else:
            features["spectral_entropy"] = 0.0
    else:
        features["spectral_entropy"] = 0.0

    return features


def extract_all_features(
    beat: np.ndarray,
    fs: float,
    pre_rr: Optional[float] = None,
    post_rr: Optional[float] = None,
    local_rr_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """Extract all features from a heartbeat segment.

    Combines time-domain, ECG-specific, frequency-domain, and rhythm features.

    Args:
        beat: 1D array of beat samples
        fs: Sampling frequency (Hz)
        pre_rr: Time in seconds from previous R-peak (optional)
        post_rr: Time in seconds to next R-peak (optional)
        local_rr_ratio: Ratio of pre_rr to local mean RR (optional)

    Returns:
        Dictionary of all extracted features
    """
    time_features = extract_time_domain_features(beat, fs)
    ecg_features = extract_ecg_specific_features(beat, fs)
    freq_features = extract_frequency_domain_features(beat, fs)

    features = {**time_features, **ecg_features, **freq_features}

    # RR interval features
    features["pre_rr"] = float(pre_rr) if pre_rr is not None and np.isfinite(pre_rr) else 0.8
    features["post_rr"] = float(post_rr) if post_rr is not None and np.isfinite(post_rr) else 0.8
    features["local_rr_ratio"] = float(local_rr_ratio) if local_rr_ratio is not None and np.isfinite(local_rr_ratio) else 1.0

    # Sanitize all values: replace NaNs and Infs
    for k, v in features.items():
        if np.isnan(v):
            features[k] = 0.0
        elif np.isinf(v):
            features[k] = 1e5 if v > 0 else -1e5
        else:
            features[k] = float(v)

    return features


def extract_features_batch(
    beats: np.ndarray,
    fs: float,
    r_peaks: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Extract features from multiple heartbeat segments.

    Args:
        beats: Array of shape (n_beats, window_size)
        fs: Sampling frequency (Hz)
        r_peaks: Array of R-peak sample indices (optional, for RR metrics)

    Returns:
        Dictionary mapping feature names to numpy arrays of values
    """
    all_features: Dict[str, list] = {}
    n_beats = len(beats)

    if r_peaks is not None and len(r_peaks) == n_beats and n_beats > 1:
        r_diffs = np.diff(r_peaks) / fs
        avg_rr = float(np.mean(r_diffs)) if len(r_diffs) > 0 else 0.8
        for i in range(n_beats):
            pre_rr = float(r_diffs[i - 1]) if i > 0 else avg_rr
            post_rr = float(r_diffs[i]) if i < len(r_diffs) else avg_rr
            ratio = pre_rr / avg_rr if avg_rr > 0 else 1.0
            feat = extract_all_features(beats[i], fs, pre_rr=pre_rr, post_rr=post_rr, local_rr_ratio=ratio)
            for k, v in feat.items():
                if k not in all_features:
                    all_features[k] = []
                all_features[k].append(v)
    else:
        for i in range(n_beats):
            feat = extract_all_features(beats[i], fs)
            for k, v in feat.items():
                if k not in all_features:
                    all_features[k] = []
                all_features[k].append(v)

    return {k: np.array(v) for k, v in all_features.items()}