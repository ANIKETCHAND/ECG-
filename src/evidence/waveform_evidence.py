"""
Waveform Evidence & Visual Snippet Subsystem
============================================
Extracts cropped waveform snippets, bounding timestamps, and visualization coordinates
for aberrant beats to facilitate rapid visual inspection by clinicians.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class WaveformSnippet:
    beat_number: int
    r_peak_sample: int
    start_sample: int
    end_sample: int
    start_time_seconds: float
    end_time_seconds: float
    signal_slice: List[float]
    sampling_rate: float
    lead_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_waveform_snippet(
    signal: np.ndarray,
    peak_sample: int,
    fs: float,
    beat_number: int,
    lead_name: str = "II",
    window_before_sec: float = 0.3,
    window_after_sec: float = 0.5,
) -> WaveformSnippet:
    """Extract a cropped ECG waveform snippet around a specific cardiac beat."""
    sig = np.asarray(signal, dtype=float)
    pre_samples = int(window_before_sec * fs)
    post_samples = int(window_after_sec * fs)

    start = max(0, peak_sample - pre_samples)
    end = min(len(sig), peak_sample + post_samples)

    snippet_signal = sig[start:end].tolist()

    return WaveformSnippet(
        beat_number=beat_number,
        r_peak_sample=peak_sample,
        start_sample=start,
        end_sample=end,
        start_time_seconds=round(start / fs, 3),
        end_time_seconds=round(end / fs, 3),
        signal_slice=snippet_signal,
        sampling_rate=fs,
        lead_name=lead_name,
    )
