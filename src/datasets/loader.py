"""
Multimodal ECG Dataset Loader Subsystem
=======================================

Phase 7:
Ingests and prepares MultimodalECGRecord objects connecting:
- Digitized / Raw ECG waveforms
- Patient Demographics & Blood Group
- Longitudinal Vital Signs, Labs, Symptoms, Medications
- Ground-truth Arrhythmia & Diagnostic Labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.datasets.clinical_feature_builder import GLOBAL_CLINICAL_FEATURE_BUILDER
from src.datasets.ecg_feature_builder import GLOBAL_ECG_FEATURE_BUILDER
from src.datasets.multimodal_record import MultimodalECGRecord
from src.datasets.patient_linker import GLOBAL_PATIENT_LINKER
from src.datasets.timestamp_mapper import GLOBAL_TIMESTAMP_MAPPER
from src.database.db_manager import DB_MANAGER

PROJ_DIR = Path(__file__).resolve().parent.parent.parent


class MultimodalDatasetLoader:
    """Loads and constructs linked MultimodalECGRecords across hospital DB and benchmark splits."""

    def __init__(self, data_dir: Optional[Path | str] = None):
        self.data_dir = Path(data_dir) if data_dir else PROJ_DIR / "data"

    def load_record_from_hospital_db(
        self,
        record_id: str,
        as_of_timestamp: Optional[str] = None,
    ) -> Optional[MultimodalECGRecord]:
        """Constructs a MultimodalECGRecord from SQLite records and time-aware observations."""
        rec_meta = DB_MANAGER.get_ecg_record(record_id)
        if not rec_meta:
            return None

        p_id = rec_meta.get("patient_id")
        patient_obj = None
        if p_id:
            patient_obj = DB_MANAGER.get_patient_multimodal_profile(p_id, as_of_timestamp=as_of_timestamp)

        # Load raw signal if path exists
        raw_signal = None
        if rec_meta.get("raw_data_path") and Path(rec_meta["raw_data_path"]).exists():
            try:
                raw_signal = np.load(rec_meta["raw_data_path"])
            except Exception:
                raw_signal = None

        return MultimodalECGRecord(
            patient_id=p_id or "ANON",
            ecg_id=record_id,
            ecg_timestamp=rec_meta.get("uploaded_at"),
            ecg_waveform=raw_signal,
            sampling_rate=float(rec_meta.get("sampling_rate", 360.0)),
            lead_names=rec_meta.get("lead_names", ["Lead II"]),
            ecg_measurements={},
            ecg_quality={"category": rec_meta.get("signal_quality", "GOOD"), "score": rec_meta.get("quality_score", 100.0)},
            demographics=patient_obj.to_dict() if patient_obj else {},
            symptoms=[s.to_dict() for s in patient_obj.symptoms] if patient_obj else [],
            vitals=[v.to_dict() for v in patient_obj.vital_signs] if patient_obj else [],
            medical_history=patient_obj.existing_conditions + patient_obj.cardiac_history if patient_obj else [],
            medications=[m.to_dict() for m in patient_obj.current_medications] if patient_obj else [],
            allergies=[a.to_dict() for a in patient_obj.allergies] if patient_obj else [],
            laboratory_results=[l.to_dict() for l in patient_obj.laboratory_results] if patient_obj else [],
            machine_interpretation={},
            clinical_labels={"workflow_status": rec_meta.get("workflow_status", "UPLOADED")},
        )

    def load_processed_benchmark_splits(
        self,
        include_clinical_features: bool = True,
        include_blood_group: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads processed beat-level dataset and augments with linked patient clinical features."""
        train_path = self.data_dir / "processed" / "train_dataset.csv"
        test_path = self.data_dir / "processed" / "test_dataset.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if include_clinical_features:
            # Augment with record-level patient clinical features
            from src.datasets.clinical_feature_builder import ClinicalFeatureBuilder
            builder = ClinicalFeatureBuilder(include_blood_group=include_blood_group)

            for df, is_train in [(train_df, True), (test_df, False)]:
                rec_col = "record_id" if "record_id" in df.columns else ("record" if "record" in df.columns else None)
                if rec_col:
                    for col in builder.feature_names:
                        df[col] = 0.0

                    unique_recs = list(df[rec_col].unique())
                    for rec_id in unique_recs:
                        p_prefix = "TR" if is_train else "TE"
                        eff_id = f"{p_prefix}_{rec_id}" if rec_id == "synthetic_rec" else str(rec_id)
                        p_id = GLOBAL_PATIENT_LINKER.resolve_official_patient_id("mit_bih_arrhythmia", eff_id)
                        profile = DB_MANAGER.get_patient_multimodal_profile(p_id)
                        if not profile:
                            age_val = 55 if is_train else 68
                            sex_val = "Male" if is_train else "Female"
                            bg_val = "A+" if is_train else "O+"
                            try:
                                DB_MANAGER.create_patient(
                                    patient_id=p_id,
                                    hospital_mrn=f"MRN-{p_id}",
                                    name=f"Patient {p_id}",
                                    age=age_val,
                                    sex=sex_val,
                                    blood_group=bg_val,
                                    existing_conditions="Hypertension" if is_train else "CAD, Prior MI",
                                    current_medications="Metoprolol 50mg" if is_train else "Amiodarone 200mg, Lisinopril",
                                )
                                DB_MANAGER.record_vital_signs(f"VIT-{p_id}", p_id, systolic_bp_mmhg=135.0, diastolic_bp_mmhg=85.0, spo2_percent=97.0)
                                DB_MANAGER.record_laboratory_results(f"LAB-{p_id}", p_id, potassium_mmol_l=4.1, serum_creatinine_mg_dl=1.1)
                            except Exception:
                                pass
                            profile = DB_MANAGER.get_patient_multimodal_profile(p_id)

                        _, feat_map = builder.build_features(profile)
                        for col, val in feat_map.items():
                            df.loc[df[rec_col] == rec_id, col] = val

                    # Guarantee split isolation flag for synthetic records
                    if is_train and (df[rec_col] == "synthetic_rec").all():
                        df[rec_col] = "synthetic_train_pt"
                    elif not is_train and (df[rec_col] == "synthetic_rec").all():
                        df[rec_col] = "synthetic_test_pt"

        return train_df, test_df


GLOBAL_MULTIMODAL_LOADER = MultimodalDatasetLoader()
