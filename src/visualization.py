"""
Visualization Module
====================

Provides interactive Plotly charts and visual components for the Streamlit dashboard:
- Raw vs Filtered ECG signal
- R-peak annotations on ECG waveform
- Extracted heartbeat segments overlay
- Prediction probability bar chart
- Signal quality gauge/metric cards
- Feature importance visualization

Research/educational use only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_ecg_signal(
    signal: np.ndarray,
    fs: float,
    r_peaks: Optional[np.ndarray] = None,
    raw_signal: Optional[np.ndarray] = None,
    title: str = "ECG Signal",
    max_duration_sec: float = 10.0,
    start_sec: float = 0.0,
) -> go.Figure:
    """Create interactive Plotly waveform plot for ECG with optional R-peaks.

    Args:
        signal: Preprocessed ECG signal array
        fs: Sampling rate (Hz)
        r_peaks: Indices of detected R-peaks
        raw_signal: Optional raw signal for comparison
        title: Plot title
        max_duration_sec: Duration in seconds to display
        start_sec: Start time offset in seconds

    Returns:
        Plotly Figure
    """
    start_sample = int(start_sec * fs)
    end_sample = min(len(signal), int((start_sec + max_duration_sec) * fs))

    t = np.arange(start_sample, end_sample) / fs
    sig_slice = signal[start_sample:end_sample]

    fig = go.Figure()

    if raw_signal is not None and len(raw_signal) >= end_sample:
        raw_slice = raw_signal[start_sample:end_sample]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=raw_slice,
                name="Raw Signal",
                mode="lines",
                line=dict(color="#b0bec5", width=1),
                opacity=0.7,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=sig_slice,
            name="Processed ECG",
            mode="lines",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    if r_peaks is not None and len(r_peaks) > 0:
        mask = (r_peaks >= start_sample) & (r_peaks < end_sample)
        visible_peaks = r_peaks[mask]
        if len(visible_peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=visible_peaks / fs,
                    y=signal[visible_peaks],
                    mode="markers",
                    name="Detected R-Peaks",
                    marker=dict(color="#d62728", size=9, symbol="triangle-up"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV / norm)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_beats_overlay(
    beats: np.ndarray,
    fs: float,
    pre_window: float = 0.2,
    post_window: float = 0.4,
    labels: Optional[List[str]] = None,
    max_beats: int = 40,
) -> go.Figure:
    """Plot overlaid individual heartbeat segments with average waveform."""
    if len(beats) == 0:
        fig = go.Figure()
        fig.update_layout(title="No beats available to display")
        return fig

    n_samples = beats.shape[1]
    time_axis = (np.arange(n_samples) - int(pre_window * fs)) / fs

    fig = go.Figure()

    display_count = min(len(beats), max_beats)
    colors = {
        "Normal": "rgba(44, 160, 44, 0.25)",
        "PVC": "rgba(214, 39, 40, 0.35)",
        "Other": "rgba(255, 127, 14, 0.35)",
    }

    for idx in range(display_count):
        beat_lbl = labels[idx] if labels and idx < len(labels) else "Normal"
        color = colors.get(beat_lbl, "rgba(31, 119, 180, 0.2)")
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=beats[idx],
                mode="lines",
                showlegend=False,
                line=dict(color=color, width=1),
                hoverinfo="skip",
            )
        )

    # Average waveform
    mean_beat = np.mean(beats, axis=0)
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=mean_beat,
            mode="lines",
            name="Mean Beat Profile",
            line=dict(color="#000000", width=2.5),
        )
    )

    fig.add_vline(x=0.0, line_width=1, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Heartbeat Segments Overlay ({display_count} beats)",
        xaxis_title="Time relative to R-peak (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def plot_prediction_probabilities(
    probabilities: Dict[str, float],
    predicted_class: str,
) -> go.Figure:
    """Horizontal bar chart of prediction probabilities."""
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]

    palette = []
    for c in classes:
        if c == predicted_class:
            palette.append("#e63946" if c == "PVC" else ("#2a9d8f" if c == "Normal" else "#f4a261"))
        else:
            palette.append("#adb5bd")

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker=dict(color=palette),
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Model Class Probabilities",
        xaxis=dict(title="Probability (%)", range=[0, 115]),
        yaxis=dict(title="", autorange="reversed"),
        template="plotly_white",
        height=240,
        margin=dict(l=60, r=40, t=40, b=40),
    )

    return fig


def plot_feature_importance(
    importances: Dict[str, float],
    top_k: int = 10,
) -> go.Figure:
    """Plot top-k feature importances."""
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
    sorted_feats = sorted_feats[::-1]  # reverse for bottom-to-top bar plot

    names = [k for k, _ in sorted_feats]
    values = [v for _, v in sorted_feats]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color="#457b9d"),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Top {top_k} Feature Importances",
        xaxis=dict(title="Importance Weight"),
        yaxis=dict(title=""),
        template="plotly_white",
        height=320,
        margin=dict(l=140, r=40, t=40, b=40),
    )

    return fig
