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
from src.datasets.multimodal_record import MultimodalECGRecord
from src.datasets.patient_linker import GLOBAL_PATIENT_LINKER, PatientLinker
from src.datasets.timestamp_mapper import GLOBAL_TIMESTAMP_MAPPER, TimestampMapper
from src.datasets.ecg_feature_builder import GLOBAL_ECG_FEATURE_BUILDER, ECGFeatureBuilder
from src.datasets.clinical_feature_builder import (
    CLINICAL_FEATURE_NAMES,
    BLOOD_GROUP_FEATURE_NAMES,
    GLOBAL_CLINICAL_FEATURE_BUILDER,
    ClinicalFeatureBuilder,
)
from src.datasets.loader import GLOBAL_MULTIMODAL_LOADER, MultimodalDatasetLoader

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
    "MultimodalECGRecord",
    "GLOBAL_PATIENT_LINKER",
    "PatientLinker",
    "GLOBAL_TIMESTAMP_MAPPER",
    "TimestampMapper",
    "GLOBAL_ECG_FEATURE_BUILDER",
    "ECGFeatureBuilder",
    "CLINICAL_FEATURE_NAMES",
    "BLOOD_GROUP_FEATURE_NAMES",
    "GLOBAL_CLINICAL_FEATURE_BUILDER",
    "ClinicalFeatureBuilder",
    "GLOBAL_MULTIMODAL_LOADER",
    "MultimodalDatasetLoader",
]

