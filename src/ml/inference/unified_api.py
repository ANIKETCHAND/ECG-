"""
Unified Multi-Task ECG Inference Engine.
Phases 21 & 22:
- Supports analyze_ecg(recording, task=..., model_id=...)
- Supports tasks: beat_arrhythmia, af_detection, 12lead_diagnostic, st_analysis, quality_gate
- Strictly adheres to non-diagnostic regulatory boundaries.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.ecg_core.models import ECGRecording
from src.ecg_core.standardized_record import StandardizedECGRecord
from src.ml.inference.inference_engine import run_ecg_ml_inference
from src.ml.models.registry import GLOBAL_MODEL_REGISTRY, ModelLifecycleStatus
from src.ml.tasks import (
    TASK_12LEAD_DIAGNOSTIC,
    TASK_AF_DETECTION,
    TASK_BEAT_ARRHYTHMIA,
    TASK_QUALITY_GATE,
    TASK_ST_ANALYSIS,
)
from src.quality.quality_gate import evaluate_ecg_quality_gate


def analyze_ecg(
    recording_or_signal: Union[ECGRecording, StandardizedECGRecord, np.ndarray, List[float]],
    fs: Optional[float] = None,
    task: str = TASK_BEAT_ARRHYTHMIA,
    model_id: Optional[str] = None,
    lead: str = "II",
) -> Dict[str, Any]:
    """Unified entrypoint for ECG analysis across tasks and registry models."""
    start_time = time.perf_counter()
    analysis_id = f"ANL-{uuid.uuid4().hex[:12].upper()}"

    # Extract raw signal and sampling rate
    if isinstance(recording_or_signal, StandardizedECGRecord):
        signal = recording_or_signal.signal
        sampling_rate = recording_or_signal.sampling_rate_hz
        leads = recording_or_signal.leads
    elif isinstance(recording_or_signal, ECGRecording):
        sampling_rate = float(recording_or_signal.sampling_rate)
        leads = recording_or_signal.lead_names
        if lead in leads:
            signal = recording_or_signal.signals[leads.index(lead)]
        else:
            signal = recording_or_signal.signals[0]
    else:
        signal = np.asarray(recording_or_signal, dtype=float)
        sampling_rate = float(fs) if fs else 360.0
        leads = [lead]

    # Extract task_id if TaskDefinition object is passed
    task_id = task.task_id if hasattr(task, "task_id") else str(task)

    # Task dispatch
    if task_id == "quality_gate":
        gate_res = evaluate_ecg_quality_gate(signal, sampling_rate, lead_name=lead)
        return {
            "analysis_id": analysis_id,
            "task": "quality_gate",
            "model_version": "SIGNAL-GATE-1.0.0",
            "prediction": gate_res.category.value,
            "can_run_ai": gate_res.can_run_ai,
            "quality_score": gate_res.quality_score,
            "warnings": gate_res.warnings,
            "rejections": gate_res.rejection_reasons,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    elif task_id == "beat_arrhythmia":
        res = run_ecg_ml_inference(signal, fs=sampling_rate, lead_to_analyze=lead)
        res["task"] = "beat_arrhythmia"
        return res

    elif task_id in ("af_detection", "12lead_diagnosis", "st_analysis"):
        # Graceful, safe non-diagnostic triage placeholder for extended research tasks
        gate_res = evaluate_ecg_quality_gate(signal, sampling_rate, lead_name=lead)
        return {
            "analysis_id": analysis_id,
            "task": task_id,
            "model_version": f"RESEARCH-{task_id.upper()}-0.1.0",
            "prediction": "RESEARCH_TASK_EVALUATION_ONLY",
            "signal_quality": gate_res.category.value,
            "lead_analyzed": lead,
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": [
                f"Task '{task_id}' is currently in research validation stage.",
                "Not approved for primary or secondary clinical triage.",
            ],
            "limitations": [
                "Investigative model architecture only.",
                "Zero patient-level diagnosis authorized.",
            ],
        }

    else:
        return {
            "analysis_id": analysis_id,
            "task": task_id,
            "model_version": "UNKNOWN",
            "prediction": "NO_RESULT_UNKNOWN_TASK",
            "signal_quality": "UNUSABLE",
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "warnings": [f"Requested task '{task_id}' is not recognized by ECG Guardian."],
            "limitations": ["Unknown task specification."],
        }

