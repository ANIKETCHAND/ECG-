"""
ECG Machine Learning Tasks Registry
===================================
Explicit task configurations mapping datasets to clinical diagnostic objectives.
"""

from src.ml.tasks.base_task import TaskDefinition

TASK_BEAT_ARRHYTHMIA = TaskDefinition(
    task_id="beat_arrhythmia",
    name="Beat-Level Arrhythmia Screening",
    description="Classifies individual cardiac cycles into Normal, PVC, SVEB, or Other.",
    target_classes=["Normal", "PVC", "SVEB", "Other"],
    is_multilabel=False,
    required_leads=["II", "MLII"],
    recommended_sampling_rate=360.0,
    compatible_datasets=["mit_bih_arrhythmia", "mit_bih_svdb", "mit_bih_nsrdb"],
)

TASK_AF_DETECTION = TaskDefinition(
    task_id="af_detection",
    name="Atrial Fibrillation Rhythm Screening",
    description="Detects episodes of Atrial Fibrillation / Flutter in rhythm strips.",
    target_classes=["AFib", "Non-AFib"],
    is_multilabel=False,
    required_leads=["II", "ECG1"],
    recommended_sampling_rate=250.0,
    compatible_datasets=["mit_bih_afdb", "mit_bih_arrhythmia"],
)

TASK_12LEAD_DIAGNOSTIC = TaskDefinition(
    task_id="12lead_diagnosis",
    name="12-Lead Multi-Label Clinical Diagnosis",
    description="Multi-label classification of standard 12-lead ECG into 5 clinical superclasses (NORM, MI, STTC, CD, HYP).",
    target_classes=["NORM", "MI", "STTC", "CD", "HYP"],
    is_multilabel=True,
    required_leads=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    recommended_sampling_rate=500.0,
    compatible_datasets=["ptb_xl", "ptbdb", "ptb_xl_plus"],
)

TASK_ST_ANALYSIS = TaskDefinition(
    task_id="st_analysis",
    name="ST-Segment & Ischemia Analysis",
    description="Identifies transient and acute ST-elevation / depression events.",
    target_classes=["Normal", "ST_Elevation", "ST_Depression"],
    is_multilabel=False,
    required_leads=["II", "V5"],
    recommended_sampling_rate=360.0,
    compatible_datasets=["mit_bih_stdb", "european_st_t"],
)

TASK_QUALITY_GATE = TaskDefinition(
    task_id="quality_gate",
    name="Pre-Inference Signal Trust Gatekeeper",
    description="Evaluates technical diagnostic quality and halts processing on corrupted signals.",
    target_classes=["GOOD", "ACCEPTABLE", "POOR", "UNUSABLE"],
    is_multilabel=False,
    required_leads=[],
    recommended_sampling_rate=0.0,  # Any
    compatible_datasets=["mit_bih_arrhythmia", "ptb_xl"],
)

ALL_TASKS = {
    "beat_arrhythmia": TASK_BEAT_ARRHYTHMIA,
    "af_detection": TASK_AF_DETECTION,
    "12lead_diagnosis": TASK_12LEAD_DIAGNOSTIC,
    "st_analysis": TASK_ST_ANALYSIS,
    "quality_gate": TASK_QUALITY_GATE,
}

__all__ = [
    "TaskDefinition",
    "TASK_BEAT_ARRHYTHMIA",
    "TASK_AF_DETECTION",
    "TASK_12LEAD_DIAGNOSTIC",
    "TASK_ST_ANALYSIS",
    "TASK_QUALITY_GATE",
    "ALL_TASKS",
]
