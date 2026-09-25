"""
Waveform Digitisation Round-Trip Validation
===========================================

Answers a question the rest of the pipeline assumes away: *how accurate is the
image-to-waveform extraction, actually?*

Method
------
1. Synthesise an ECG with known ground truth (known heart rate, known beat
   times, known morphology).
2. Render it as a calibrated ECG strip image — pink 1 mm / 5 mm grid at the
   clinical standard 25 mm/s and 10 mm/mV, dark trace.
3. Run the production extractor (:func:`extract_waveform_from_image`) on that
   image.
4. Compare recovered beats against ground truth.

What is measured
----------------
* **Beat timing** — R-peak count and position error in milliseconds. This is the
  clinically meaningful quantity and is independent of amplitude scaling.
* **Morphology** — Pearson correlation between the recovered and true waveform
  after amplitude normalisation.

What is deliberately *not* claimed
----------------------------------
Absolute voltage recovery. The extractor rescales the trace so its peak equals
1.5 mV, so the amplitude axis is not calibrated to the printed signal. The report
records this as ``amplitude_recovery: NOT_CALIBRATED`` with the reason, rather
than printing a plausible-looking voltage gain.

Usage::

    python training/validate_digitization.py --duration 10 --heart-rate 75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from src.ecg_input.waveform_extractor import extract_waveform_from_image
from src.peak_detection import detect_r_peaks

DEFAULT_FS = 360.0

#: Clinical ECG paper standard.
MM_PER_SECOND = 25.0
MM_PER_MV = 10.0


def synthesise_ecg(
    duration_s: float = 10.0,
    heart_rate_bpm: float = 75.0,
    fs: float = DEFAULT_FS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic ECG with known-beat ground truth.

    Returns:
        ``(signal_mv, time_s, true_r_peak_samples)``. The R peak is placed at the
        centre of each beat window.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * fs))
    time = np.arange(n_samples) / fs
    signal = np.zeros(n_samples, dtype=float)

    rr_s = 60.0 / heart_rate_bpm
    beat_centres = np.arange(rr_s, duration_s, rr_s)

    def gaussian(centre: float, amplitude: float, width: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((time - centre) / width) ** 2)

    for centre in beat_centres:
        # P wave, QRS complex, T wave — a coarse but recognisable morphology.
        signal += gaussian(centre - 0.16, 0.12, 0.025)
        signal += gaussian(centre - 0.02, -0.15, 0.008)
        signal += gaussian(centre, 1.10, 0.010)
        signal += gaussian(centre + 0.02, -0.20, 0.010)
        signal += gaussian(centre + 0.24, 0.28, 0.045)

    # Small physiological baseline wander plus a little noise, so the extractor is
    # not validated on an unrealistically pristine signal.
    signal += 0.05 * np.sin(2 * np.pi * 0.25 * time)
    signal += rng.normal(0.0, 0.004, n_samples)

    true_peaks = np.array([int(round(c * fs)) for c in beat_centres], dtype=int)
    return signal, time, true_peaks


def render_strip_image(
    signal_mv: np.ndarray,
    fs: float,
    *,
    px_per_sec: float = 250.0,
    px_per_mv: float = 100.0,
    background: int = 255,
) -> np.ndarray:
    """Render a signal as a BGR ECG strip image with a standard red grid."""
    n_samples = len(signal_mv)
    duration_s = n_samples / fs
    width = int(round(duration_s * px_per_sec))
    height = int(round(3.0 * px_per_mv))  # roughly a 3 mV tall plotting area

    # White background, BGR
    image = np.full((height, width, 3), background, dtype=np.uint8)

    baseline_y = height // 2
    px_per_mm_x = px_per_sec / MM_PER_SECOND
    px_per_mm_y = px_per_mv / MM_PER_MV

    grid_minor = np.array([200, 200, 255], dtype=np.uint8)  # light pink/red in BGR
    grid_major = np.array([160, 170, 240], dtype=np.uint8)

    # Vertical grid lines: minor every 1 mm, major every 5 mm.
    step_minor_x = max(1, int(round(px_per_mm_x)))
    for x in range(0, width, step_minor_x):
        colour = grid_major if (x // step_minor_x) % 5 == 0 else grid_minor
        image[:, x] = colour

    step_minor_y = max(1, int(round(px_per_mm_y)))
    for y in range(0, height, step_minor_y):
        colour = grid_major if ((baseline_y - y) // step_minor_y) % 5 == 0 else grid_minor
        image[y, :] = colour

    # Trace: dark pixels, 2 px thick for continuity across the columns.
    resampled = np.interp(
        np.linspace(0, n_samples - 1, width),
        np.arange(n_samples),
        signal_mv,
    )
    trace_y = baseline_y - (resampled * px_per_mv)
    trace_y = np.clip(trace_y, 1, height - 2).astype(int)

    for x in range(width):
        y = trace_y[x]
        image[max(0, y - 1) : min(height, y + 2), x] = np.array([0, 0, 0], dtype=np.uint8)

    return image


def _peak_timing_error(true_peaks: np.ndarray, recovered_peaks: np.ndarray, fs: float) -> Dict[str, Any]:
    """Match recovered peaks to true peaks and report timing error in ms."""
    if len(true_peaks) == 0 or len(recovered_peaks) == 0:
        return {
            "matched_beats": 0,
            "mean_absolute_error_ms": None,
            "max_absolute_error_ms": None,
        }

    errors_ms: List[float] = []
    for true_peak in true_peaks:
        nearest = recovered_peaks[np.argmin(np.abs(recovered_peaks - true_peak))]
        errors_ms.append(abs(float(nearest - true_peak)) / fs * 1000.0)

    return {
        "matched_beats": int(len(errors_ms)),
        "mean_absolute_error_ms": round(float(np.mean(errors_ms)), 2),
        "max_absolute_error_ms": round(float(np.max(errors_ms)), 2),
    }


def _morphology_correlation(reference: np.ndarray, recovered: np.ndarray) -> Optional[float]:
    """Scale-invariant waveform similarity."""
    if reference.size == 0 or recovered.size == 0:
        return None
    resampled = np.interp(
        np.linspace(0, reference.size - 1, recovered.size),
        np.arange(reference.size),
        reference,
    )
    resampled = resampled - np.mean(resampled)
    recovered = recovered - np.mean(recovered)
    denominator = float(np.std(resampled) * np.std(recovered))
    if denominator <= 0:
        return None
    return round(float(np.mean(resampled * recovered) / denominator), 4)


def validate_round_trip(
    duration_s: float = 10.0,
    heart_rate_bpm: float = 75.0,
    fs: float = DEFAULT_FS,
) -> Dict[str, Any]:
    """Run the full render -> extract -> compare loop."""
    signal, _time, true_peaks = synthesise_ecg(duration_s, heart_rate_bpm, fs)
    image = render_strip_image(signal, fs)

    extraction = extract_waveform_from_image(image, target_fs=fs, assumed_duration_sec=duration_s)

    report: Dict[str, Any] = {
        "task": "waveform_digitisation_round_trip",
        "ground_truth": {
            "duration_s": duration_s,
            "heart_rate_bpm": heart_rate_bpm,
            "sampling_rate_hz": fs,
            "true_beats": int(len(true_peaks)),
        },
        "extraction": {
            "success": bool(extraction.get("success")),
            "confidence_score": extraction.get("confidence_score"),
            "message": extraction.get("message"),
        },
        "amplitude_recovery": {
            "status": "NOT_CALIBRATED",
            "reason": (
                "The extractor rescales the trace so that its peak equals 1.5 mV, so absolute voltage is "
                "not recoverable from the image. Only beat timing and morphology are validated."
            ),
        },
    }

    if not extraction.get("success") or extraction.get("signal") is None:
        report["timing"] = {"status": "NOT_MEASURED", "reason": "Extraction did not produce a signal."}
        report["morphology_correlation"] = None
        return report

    recovered = np.asarray(extraction["signal"], dtype=float)
    recovered_fs = float(extraction.get("sampling_rate", fs))

    recovered_peaks, _ = detect_r_peaks(recovered, fs=recovered_fs)
    recovered_peaks = np.asarray(recovered_peaks, dtype=int)

    timing = _peak_timing_error(true_peaks, recovered_peaks, recovered_fs)
    timing["true_beat_count"] = int(len(true_peaks))
    timing["recovered_beat_count"] = int(len(recovered_peaks))
    timing["beat_count_absolute_error"] = int(abs(len(recovered_peaks) - len(true_peaks)))
    timing["status"] = "MEASURED"

    if len(recovered_peaks) >= 2:
        recovered_hr = 60.0 * recovered_fs / float(np.mean(np.diff(recovered_peaks)))
        timing["recovered_heart_rate_bpm"] = round(recovered_hr, 2)
        timing["heart_rate_absolute_error_bpm"] = round(abs(recovered_hr - heart_rate_bpm), 2)

    report["timing"] = timing
    report["morphology_correlation"] = _morphology_correlation(signal, recovered)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ECG image digitisation round-trip accuracy")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--heart-rate", type=float, default=75.0)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    args = parser.parse_args()

    result = validate_round_trip(duration_s=args.duration, heart_rate_bpm=args.heart_rate, fs=args.fs)
    print(json.dumps(result, indent=2))
