"""
Simulated Real-Time ECG Streaming
=================================

Demonstrates the online path: instead of analysing a finished recording in one
shot, signal arrives in windows and beats are classified as they are detected.

This is a simulation, not a device driver — it replays a finite buffer as if it
were arriving live. The purpose is to show the pipeline holds up under streaming
semantics, which differ from batch analysis in two ways that matter:

* **State.** Beats already reported must not be reported twice when the next
  window overlaps the previous one. The streamer tracks the last emitted R-peak
  sample and only emits genuinely new beats.
* **Latency.** Each window is processed with only the samples available so far, so
  the per-window quality gate and peak detector see a short context, exactly as a
  live monitor would.

Usage::

    from src.streaming.simulated_stream import run_streaming_demo
    result = run_streaming_demo(duration_s=30, heart_rate_bpm=78)
    print(result["summary"])

Attaching this to a Bluetooth or serial source is then a matter of replacing the
window generator, not the analysis path.
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from feature_extraction import extract_all_features
from peak_detection import detect_r_peaks
from preprocessing import preprocess_pipeline
from segmentation import extract_beats

MODELS_DIR = _SRC_DIR.parent / "models"


@dataclass
class BeatEvent:
    beat_index: int
    sample_index: int
    timestamp_s: float
    prediction: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamWindowResult:
    window_index: int
    window_start_s: float
    window_end_s: float
    quality_category: str
    beats_detected: int
    new_beats: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stream_window_indices(
    n_samples: int,
    fs: float,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, end)`` sample bounds for successive overlapping windows."""
    window = max(1, int(round(window_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    start = 0
    while start < n_samples:
        end = min(n_samples, start + window)
        yield start, end
        if end >= n_samples:
            break
        start += step


def _load_production_model():
    """Load the production classifier and scaler, or ``(None, None, {})``."""
    from inference.inference_engine import get_model_artifacts

    try:
        return get_model_artifacts()
    except Exception:
        return None, None, {}


def stream_beats(
    signal: np.ndarray,
    fs: float,
    *,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    lead_name: str = "II",
) -> Iterator[Dict[str, Any]]:
    """Classify beats incrementally as windows arrive.

    Yields:
        Dicts describing either a window result (``event: "window"``) or a newly
        detected beat (``event: "beat"``). Windows whose signal fails the
        pre-inference quality gate are reported as skipped and emit no beats —
        the same safety rule the batch path applies.
    """
    signal = np.asarray(signal, dtype=float)
    classifier, scaler, metadata = _load_production_model()
    feature_names = metadata.get("feature_names", [])
    classes = list(getattr(classifier, "classes_", [])) if classifier is not None else []

    try:
        from safety.signal_quality_gate import evaluate_signal_quality_gate
        from safety.signal_quality_gate import QualityCategory
    except ImportError:  # pragma: no cover - path variants
        evaluate_signal_quality_gate = None
        QualityCategory = None

    last_emitted_sample = -1
    beat_index = 0

    # Overlapping windows re-detect the same beat at slightly different sample
    # positions (filter transients shift the peak by a few samples), so a strict
    # "newer than the last emitted beat" test still admits near-duplicates. A
    # refractory gap is what the batch detector already uses physiologically; here
    # it also removes cross-window duplicates. 250 ms is safe to ~240 bpm.
    refractory_samples = max(1, int(round(0.25 * fs)))

    for window_index, (start, end) in enumerate(stream_window_indices(len(signal), fs, window_sec, step_sec)):
        window = signal[start:end]

        quality_category = "NOT_ASSESSED"
        can_run_ai = True
        if evaluate_signal_quality_gate is not None and len(window) > 0:
            gate = evaluate_signal_quality_gate(window, fs, lead_name=lead_name)
            quality_category = gate.category.value
            can_run_ai = bool(gate.can_run_ai) and gate.category != QualityCategory.UNUSABLE

        new_beats = 0
        events: List[BeatEvent] = []

        if can_run_ai and classifier is not None and window.size > 0:
            processed = preprocess_pipeline(window, fs=fs)
            peaks, _ = detect_r_peaks(processed, fs=fs)
            beats, valid_beat_indices = extract_beats(processed, peaks, fs=fs, pre_window=0.2, post_window=0.4)

            # ``extract_beats`` returns positions within the supplied peak array,
            # not sample indices, so map back through ``peaks`` before converting
            # to an absolute position in the overall stream.
            for beat, peak_position in zip(beats, np.asarray(valid_beat_indices, dtype=int)):
                if peak_position < 0 or peak_position >= len(peaks):
                    continue
                absolute_sample = start + int(peaks[peak_position])
                if absolute_sample <= last_emitted_sample + refractory_samples:
                    continue  # already reported by an earlier overlapping window

                features = extract_all_features(beat, fs=fs)
                row = [float(features.get(name, 0.0)) for name in feature_names] if feature_names else None

                prediction = "UNCLASSIFIED"
                confidence = 0.0
                probabilities: Dict[str, float] = {}
                if row is not None and scaler is not None:
                    try:
                        matrix = scaler.transform(np.asarray([row], dtype=float))
                        prediction = str(classifier.predict(matrix)[0])
                        probs = classifier.predict_proba(matrix)[0]
                        confidence = float(np.max(probs))
                        probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
                    except Exception:
                        prediction = "CLASSIFICATION_FAILED"

                beat_index += 1
                new_beats += 1
                last_emitted_sample = absolute_sample
                events.append(
                    BeatEvent(
                        beat_index=beat_index,
                        sample_index=absolute_sample,
                        timestamp_s=round(absolute_sample / fs, 3),
                        prediction=prediction,
                        confidence=round(confidence, 4),
                        probabilities=probabilities,
                    )
                )

        yield {
            "event": "window",
            **StreamWindowResult(
                window_index=window_index,
                window_start_s=round(start / fs, 3),
                window_end_s=round(end / fs, 3),
                quality_category=quality_category,
                beats_detected=len(events),
                new_beats=new_beats,
            ).to_dict(),
            "ai_allowed": can_run_ai,
        }

        for event in events:
            yield {"event": "beat", **event.to_dict()}


def summarise_stream(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stream events into a session summary."""
    windows = [e for e in events if e.get("event") == "window"]
    beats = [e for e in events if e.get("event") == "beat"]

    counts: Dict[str, int] = {}
    for beat in beats:
        counts[beat["prediction"]] = counts.get(beat["prediction"], 0) + 1

    confidences = [beat["confidence"] for beat in beats if beat.get("confidence")]
    sorted_beats = sorted(beats, key=lambda b: b["timestamp_s"])
    intervals = np.diff([b["timestamp_s"] for b in sorted_beats]) if len(sorted_beats) > 1 else np.array([])

    return {
        "windows_processed": len(windows),
        "windows_skipped_by_quality_gate": sum(1 for w in windows if not w.get("ai_allowed", True)),
        "beats_classified": len(beats),
        "prediction_counts": counts,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "estimated_heart_rate_bpm": round(float(60.0 / np.mean(intervals)), 2) if intervals.size else None,
        "duration_covered_s": round(sorted_beats[-1]["timestamp_s"], 2) if sorted_beats else 0.0,
        "note": (
            "Simulated stream. Beat timing derives from detected R-peaks within each window; no gating "
            "or anti-aliasing is applied beyond the standard preprocessing chain."
        ),
    }


def run_streaming_demo(
    signal: Optional[np.ndarray] = None,
    fs: float = 360.0,
    duration_s: float = 30.0,
    heart_rate_bpm: float = 78.0,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> Dict[str, Any]:
    """Run the streaming path over a signal, synthesising one when none is given.

    The synthesised signal is a demonstration waveform, clearly labelled as such
    in the returned metadata. It is never presented as patient data.
    """
    synthetic = signal is None
    if synthetic:
        n_samples = int(round(duration_s * fs))
        time = np.arange(n_samples) / fs
        signal = np.zeros(n_samples, dtype=float)
        rr = 60.0 / heart_rate_bpm
        for centre in np.arange(rr, duration_s, rr):
            signal += 1.10 * np.exp(-0.5 * ((time - centre) / 0.010) ** 2)
            signal += 0.28 * np.exp(-0.5 * ((time - (centre + 0.24)) / 0.045) ** 2)
            signal += 0.12 * np.exp(-0.5 * ((time - (centre - 0.16)) / 0.025) ** 2)
        signal += 0.05 * np.sin(2 * np.pi * 0.3 * time)
        signal += np.random.default_rng(42).normal(0.0, 0.004, n_samples)

    events = list(stream_beats(np.asarray(signal, dtype=float), fs, window_sec=window_sec, step_sec=step_sec))
    summary = summarise_stream(events)

    return {
        "status": "COMPLETED",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "signal_provenance": "synthesised demonstration waveform" if synthetic else "caller-supplied signal",
        "sampling_rate_hz": fs,
        "window_sec": window_sec,
        "step_sec": step_sec,
        "summary": summary,
        "events": events,
    }
