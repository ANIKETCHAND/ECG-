"""
Dataset Management Subsystem
============================
Multi-dataset discovery, download, inspection, validation, and standard conversion.
"""

from src.datasets.downloader import GLOBAL_DATASET_DOWNLOADER as DATASET_DOWNLOADER, DatasetDownloader
from src.datasets.inspector import GLOBAL_DATASET_INSPECTOR as DATASET_INSPECTOR, DatasetInspector, InspectionReport
from src.datasets.label_mapper import GLOBAL_LABEL_MAPPER, LabelMapper
from src.datasets.patient_index import GLOBAL_PATIENT_INDEX, PatientIndex, resolve_patient_id
from src.datasets.registry import APPROVED_DATASETS, DATASET_REGISTRY, DatasetEntry, DatasetRegistry

__all__ = [
    "APPROVED_DATASETS",
    "DATASET_REGISTRY",
    "DatasetEntry",
    "DatasetRegistry",
    "GLOBAL_PATIENT_INDEX",
    "PatientIndex",
    "resolve_patient_id",
    "GLOBAL_LABEL_MAPPER",
    "LabelMapper",
    "DATASET_INSPECTOR",
    "DatasetInspector",
    "InspectionReport",
    "DatasetDownloader",
]
