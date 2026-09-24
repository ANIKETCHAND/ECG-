"""
Clinical Context Model Subsystem
================================

Phase 11:
Independent clinical context classifier trained strictly on patient demographics,
vital signs, laboratory biomarkers, symptoms, and medical history.

Key Architectural Principles (Rule 2 & Phase 11):
- Operates independently from the ECG waveform model.
- Evaluates clinical pre-test risk without seeing raw waveform morphology.
- Uses explicit missingness indicators so unrecorded values are never assumed normal.
- Never computes a generic uncalibrated "health score".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.datasets.clinical_feature_builder import (
    CLINICAL_FEATURE_NAMES,
    ClinicalFeatureBuilder,
)


class ClinicalContextModel:
    """Clinical tabular context classifier evaluating pre-test arrhythmia / abnormality probability."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        classes: Optional[List[str]] = None,
        include_blood_group: bool = False,
    ):
        self.model_type = model_type
        self.classes = classes or ["Normal", "Other", "PVC"]
        self.include_blood_group = include_blood_group
        self.feature_builder = ClinicalFeatureBuilder(include_blood_group=include_blood_group)
        self.feature_names = self.feature_builder.feature_names
        
        self.scaler = StandardScaler()
        if model_type == "logistic":
            self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        elif model_type == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        else:
            self.classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
        self.is_fitted = False

    def fit(self, X_clinical: np.ndarray, y: np.ndarray) -> ClinicalContextModel:
        """Fit scaler and classifier strictly on training clinical feature matrix."""
        X_scaled = self.scaler.fit_transform(X_clinical)
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X_clinical: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ClinicalContextModel must be fitted before predict()")
        X_scaled = self.scaler.transform(X_clinical)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X_clinical: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ClinicalContextModel must be fitted before predict_proba()")
        X_scaled = self.scaler.transform(X_clinical)
        return self.classifier.predict_proba(X_scaled)

    def evaluate_patient_proba(
        self,
        patient: Any,
        as_of_timestamp: Optional[str] = None,
    ) -> Dict[str, float]:
        """Inference for a single patient record."""
        feat_vec, _ = self.feature_builder.build_features(patient, as_of_timestamp=as_of_timestamp)
        feat_matrix = feat_vec.reshape(1, -1)
        probs = self.predict_proba(feat_matrix)[0]
        return {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, p / "clinical_classifier.pkl")
        joblib.dump(self.scaler, p / "clinical_scaler.pkl")
        meta = {
            "model_type": self.model_type,
            "classes": self.classes,
            "feature_names": self.feature_names,
            "include_blood_group": self.include_blood_group,
        }
        with open(p / "clinical_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str) -> ClinicalContextModel:
        p = Path(model_dir)
        with open(p / "clinical_metadata.json", "r") as f:
            meta = json.load(f)
        inst = cls(
            model_type=meta["model_type"],
            classes=meta["classes"],
            include_blood_group=meta.get("include_blood_group", False),
        )
        inst.classifier = joblib.load(p / "clinical_classifier.pkl")
        inst.scaler = joblib.load(p / "clinical_scaler.pkl")
        inst.is_fitted = True
        return inst
