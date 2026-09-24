"""
Dataset Registry Subsystem
==========================
Comprehensive catalog of approved ECG databases for training, tuning, and clinical validation.
Maintains licenses, official sources, sampling rates, lead configurations, and task suitability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetEntry:
    dataset_id: str
    name: str
    description: str
    source_url: str
    physionet_slug: Optional[str]
    version: str
    license_type: str
    lead_count: int
    lead_names: List[str]
    sampling_rates: List[float]
    total_records: int
    task_suitability: List[str]  # beat_arrhythmia, af_detection, 12lead_diagnosis, st_analysis, quality_gate
    status: str = "REGISTERED"  # REGISTERED, AVAILABLE, PARTIAL, MANUAL_DOWNLOAD_REQUIRED
    download_size_mb: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


APPROVED_DATASETS: Dict[str, DatasetEntry] = {
    # Dataset Group A — Beat / Rhythm
    "mit_bih_arrhythmia": DatasetEntry(
        dataset_id="mit_bih_arrhythmia",
        name="MIT-BIH Arrhythmia Database",
        description="Gold-standard ambulatory 2-channel ECG recordings with beat-by-beat expert annotations.",
        source_url="https://physionet.org/content/mitdb/1.0.0/",
        physionet_slug="mitdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["MLII", "V1"],
        sampling_rates=[360.0],
        total_records=48,
        task_suitability=["beat_arrhythmia", "quality_gate"],
        download_size_mb=100.0,
        notes="Validated for ventricular and supraventricular ectopy screening.",
    ),
    "mit_bih_svdb": DatasetEntry(
        dataset_id="mit_bih_svdb",
        name="MIT-BIH Supraventricular Arrhythmia Database",
        description="78 2-channel ECG recordings designed specifically to benchmark supraventricular ectopy (SVEB).",
        source_url="https://physionet.org/content/svdb/1.0.0/",
        physionet_slug="svdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["MLII", "V1"],
        sampling_rates=[128.0],
        total_records=78,
        task_suitability=["beat_arrhythmia"],
        download_size_mb=90.0,
        notes="Essential for mitigating Class 'Other' false negatives and detecting atrial premature beats.",
    ),
    "mit_bih_afdb": DatasetEntry(
        dataset_id="mit_bih_afdb",
        name="MIT-BIH Atrial Fibrillation Database",
        description="25 10-hour ambulatory ECG recordings with rhythm-level AFib/AFL annotations.",
        source_url="https://physionet.org/content/afdb/1.0.0/",
        physionet_slug="afdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["ECG1", "ECG2"],
        sampling_rates=[250.0],
        total_records=25,
        task_suitability=["af_detection"],
        download_size_mb=350.0,
        notes="Provides ground-truth episode boundaries for Atrial Fibrillation.",
    ),
    "mit_bih_nsrdb": DatasetEntry(
        dataset_id="mit_bih_nsrdb",
        name="MIT-BIH Normal Sinus Rhythm Database",
        description="18 long-term 2-channel recordings from healthy subjects with no significant cardiac arrhythmias.",
        source_url="https://physionet.org/content/nsrdb/1.0.0/",
        physionet_slug="nsrdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["ECG1", "ECG2"],
        sampling_rates=[128.0],
        total_records=18,
        task_suitability=["beat_arrhythmia", "quality_gate"],
        download_size_mb=120.0,
        notes="Provides verified normal sinus rhythm control population.",
    ),
    "mit_bih_ltdb": DatasetEntry(
        dataset_id="mit_bih_ltdb",
        name="MIT-BIH Long-Term ECG Database",
        description="7 24-hour recordings with annotated beat types to evaluate long-term rhythm stability.",
        source_url="https://physionet.org/content/ltdb/1.0.0/",
        physionet_slug="ltdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["ECG1", "ECG2"],
        sampling_rates=[128.0],
        total_records=7,
        task_suitability=["beat_arrhythmia"],
        download_size_mb=300.0,
        notes="Used for multi-hour stability and diurnal heart rate variation testing.",
    ),

    # Dataset Group B — Diagnostic 12-Lead ECG
    "ptb_xl": DatasetEntry(
        dataset_id="ptb_xl",
        name="PTB-XL: A Large Publicly Available Electrocardiography Dataset",
        description="21,837 clinical 12-lead 10-second ECG recordings from 18,885 patients with comprehensive SCP-ECG statements.",
        source_url="https://physionet.org/content/ptb-xl/1.0.3/",
        physionet_slug="ptb-xl",
        version="1.0.3",
        license_type="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        lead_count=12,
        lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        sampling_rates=[100.0, 500.0],
        total_records=21837,
        task_suitability=["12lead_diagnosis", "quality_gate"],
        download_size_mb=3100.0,
        notes="Primary benchmark for hospital 12-lead multi-label diagnosis (MI, STTC, CD, HYP, NORM).",
    ),
    "ptbdb": DatasetEntry(
        dataset_id="ptbdb",
        name="PTB Diagnostic ECG Database",
        description="549 high-resolution 15-channel recordings from 290 subjects (healthy and MI patients).",
        source_url="https://physionet.org/content/ptbdb/1.0.0/",
        physionet_slug="ptbdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=15,
        lead_names=["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6", "vx", "vy", "vz"],
        sampling_rates=[1000.0],
        total_records=549,
        task_suitability=["12lead_diagnosis", "st_analysis"],
        download_size_mb=1200.0,
        notes="High-fidelity 1000 Hz diagnostic data for acute ischemia and infarction validation.",
    ),
    "ptb_xl_plus": DatasetEntry(
        dataset_id="ptb_xl_plus",
        name="PTB-XL+: A Comprehensive Companion Dataset for PTB-XL",
        description="Extended annotations for PTB-XL including discrete fiducial points, wave onsets/offsets, and segment measurements.",
        source_url="https://physionet.org/content/ptb-xl-plus/1.0.1/",
        physionet_slug="ptb-xl-plus",
        version="1.0.1",
        license_type="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        lead_count=12,
        lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        sampling_rates=[500.0],
        total_records=21837,
        task_suitability=["12lead_diagnosis", "quality_gate"],
        download_size_mb=450.0,
        notes="Supplements PTB-XL with median beat waveforms and manual interval ground truth.",
    ),

    # Dataset Group C — ST/T Analysis
    "mit_bih_stdb": DatasetEntry(
        dataset_id="mit_bih_stdb",
        name="MIT-BIH ST Change Database",
        description="28 2-channel recordings with expert annotations of transient ischemic ST-segment shifts.",
        source_url="https://physionet.org/content/stdb/1.0.0/",
        physionet_slug="stdb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["ECG1", "ECG2"],
        sampling_rates=[360.0],
        total_records=28,
        task_suitability=["st_analysis"],
        download_size_mb=80.0,
        notes="Evaluates algorithmic sensitivity to ischemic ST elevation and depression.",
    ),
    "european_st_t": DatasetEntry(
        dataset_id="european_st_t",
        name="European ST-T Database",
        description="90 2-hour 2-channel recordings with annotated ST and T-wave changes from myocardial ischemia.",
        source_url="https://physionet.org/content/edb/1.0.0/",
        physionet_slug="edb",
        version="1.0.0",
        license_type="Open PhysioNet (ODC-BY 1.0)",
        lead_count=2,
        lead_names=["ECG1", "ECG2"],
        sampling_rates=[250.0],
        total_records=90,
        task_suitability=["st_analysis"],
        download_size_mb=250.0,
        notes="Standard ESC benchmark for transient myocardial ischemia monitoring.",
    ),

    # Dataset Group D — Hospital-Level & Credentialed Benchmarks
    "mimic_iv_ecg": DatasetEntry(
        dataset_id="mimic_iv_ecg",
        name="MIMIC-IV-ECG: Diagnostic Electrocardiogram Database",
        description="Large-scale hospital database of ~800,000 diagnostic 12-lead ECGs matched to clinical records.",
        source_url="https://physionet.org/content/mimic-iv-ecg/1.0/",
        physionet_slug="mimic-iv-ecg",
        version="1.0",
        license_type="PhysioNet Credentialed Health Data License 1.5.0",
        lead_count=12,
        lead_names=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        sampling_rates=[500.0],
        total_records=800000,
        task_suitability=["12lead_diagnosis", "quality_gate"],
        status="MANUAL_ACCESS_REQUIRED",
        download_size_mb=45000.0,
        notes="Requires CITI 'Data or Specimens Only Research' certification and formal PhysioNet DUA approval.",
    ),
    "mimic_iv_clinical": DatasetEntry(
        dataset_id="mimic_iv_clinical",
        name="MIMIC-IV Clinical Database",
        description="Comprehensive de-identified ICU and emergency clinical database (labs, medications, conditions).",
        source_url="https://physionet.org/content/mimiciv/2.2/",
        physionet_slug="mimiciv",
        version="2.2",
        license_type="PhysioNet Credentialed Health Data License 1.5.0",
        lead_count=1,
        lead_names=["CLINICAL_RECORD"],
        sampling_rates=[0.0],
        total_records=299712,
        task_suitability=["clinical_decision_support", "medication_safety"],
        status="MANUAL_ACCESS_REQUIRED",
        download_size_mb=35000.0,
        notes="Credentialed clinical database for correlating ECG electrophysiology with ICU outcomes and medications.",
    ),
}


class DatasetRegistry:
    """Registry manager for dataset discovery, query, and status tracking."""

    def __init__(self, base_data_dir: Optional[Path | str] = None) -> None:
        self.base_dir = (
            Path(base_data_dir)
            if base_data_dir
            else Path(__file__).resolve().parent.parent.parent / "data" / "datasets"
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._entries = dict(APPROVED_DATASETS)

    def list_datasets(self) -> List[DatasetEntry]:
        """Return list of all registered dataset entries."""
        return list(self._entries.values())

    def get_dataset(self, dataset_id: str) -> Optional[DatasetEntry]:
        """Fetch metadata for a dataset by ID."""
        return self._entries.get(dataset_id.lower().strip())

    def get_datasets_for_task(self, task: str) -> List[DatasetEntry]:
        """Return all datasets suitable for a specific ML task."""
        t_clean = task.lower().strip()
        return [e for e in self._entries.values() if t_clean in e.task_suitability]

    def get_local_path(self, dataset_id: str) -> Path:
        """Get the designated root directory for a dataset."""
        p = self.base_dir / dataset_id.lower().strip()
        p.mkdir(parents=True, exist_ok=True)
        (p / "raw").mkdir(parents=True, exist_ok=True)
        (p / "processed").mkdir(parents=True, exist_ok=True)
        (p / "metadata").mkdir(parents=True, exist_ok=True)
        (p / "annotations").mkdir(parents=True, exist_ok=True)
        return p


DATASET_REGISTRY = DatasetRegistry()
