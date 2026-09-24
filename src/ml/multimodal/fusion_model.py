"""
Multimodal ECG + Clinical Context Fusion Model
==============================================

Phase 12:
Implements the decoupled Multimodal Architecture:
- Sub-network A: ECG Waveform Model (28 features -> ECG Class Probabilities)
- Sub-network B: Clinical Context Model (29 tabular features -> Clinical Pre-test Probabilities)
- Fusion Layer C: Calibrated Multimodal Fusion Network combining both representations

Adheres to:
- Rule 2: NEVER train one giant black-box model. Sub-networks are independently testable.
- Phase 13: Directly facilitates 3-Model Comparison (ECG-only vs Clinical-only vs Multimodal).
- Phase 14: Directly supports Feature Ablation testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ml.clinical_context.model import ClinicalContextModel


class MultimodalECGFusionModel:
    """Multimodal Fusion Classifier combining ECG representations with clinical context."""

    def __init__(
        self,
        ecg_model: Any,
        ecg_scaler: Any,
        clinical_model: Optional[ClinicalContextModel] = None,
        fusion_type: str = "calibrated_logistic",
        classes: Optional[List[str]] = None,
    ):
        self.ecg_model = ecg_model
        self.ecg_scaler = ecg_scaler
        self.classes = classes or ["Normal", "Other", "PVC"]
        self.clinical_model = clinical_model or ClinicalContextModel(classes=self.classes)
        self.fusion_type = fusion_type
        
        self.fusion_scaler = StandardScaler()
        if fusion_type == "gradient_boosting":
            self.fusion_classifier = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
            )
        else:
            self.fusion_classifier = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        self.is_fitted = False

    def build_fusion_features(
        self,
        X_ecg: np.ndarray,
        X_clinical: np.ndarray,
    ) -> np.ndarray:
        """Constructs multimodal fusion representation:
        [ECG_Probabilities (3), Clinical_Probabilities (3), Key_ECG_Features (Top 5), Key_Clinical_Features (Top 5)]
        """
        # 1. ECG predictions/probabilities
        X_ecg_scaled = self.ecg_scaler.transform(X_ecg)
        p_ecg = self.ecg_model.predict_proba(X_ecg_scaled)  # (N, 3)

        # 2. Clinical context probabilities
        p_clinical = self.clinical_model.predict_proba(X_clinical)  # (N, 3)

        # 3. Selected salient features for direct multimodal residual connection
        # ECG: local_rr_ratio (idx 27), pre_rr (idx 25), autocorr_first_peak (idx 11)
        salient_ecg = X_ecg_scaled[:, [11, 25, 27]] if X_ecg_scaled.shape[1] > 27 else X_ecg_scaled[:, :3]

        # Clinical: age (idx 0), systolic_bp (idx 4), potassium (idx 10)
        salient_clin = X_clinical[:, [0, 4, 10]] if X_clinical.shape[1] > 10 else X_clinical[:, :3]

        # Concatenate into fusion feature vector
        return np.hstack([p_ecg, p_clinical, salient_ecg, salient_clin])

    def fit(
        self,
        X_ecg_train: np.ndarray,
        X_clinical_train: np.ndarray,
        y_train: np.ndarray,
    ) -> MultimodalECGFusionModel:
        """Fits clinical model and fusion layer strictly on training folds."""
        # 1. Ensure clinical context model is fitted
        if not self.clinical_model.is_fitted:
            self.clinical_model.fit(X_clinical_train, y_train)

        # 2. Generate fusion representation
        X_fusion = self.build_fusion_features(X_ecg_train, X_clinical_train)

        # 3. Fit fusion scaler and classifier
        X_fusion_scaled = self.fusion_scaler.fit_transform(X_fusion)
        self.fusion_classifier.fit(X_fusion_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(
        self,
        X_ecg: np.ndarray,
        X_clinical: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MultimodalECGFusionModel must be fitted before predict()")
        X_fusion = self.build_fusion_features(X_ecg, X_clinical)
        X_fusion_scaled = self.fusion_scaler.transform(X_fusion)
        return self.fusion_classifier.predict(X_fusion_scaled)

    def predict_proba(
        self,
        X_ecg: np.ndarray,
        X_clinical: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MultimodalECGFusionModel must be fitted before predict_proba()")
        X_fusion = self.build_fusion_features(X_ecg, X_clinical)
        X_fusion_scaled = self.fusion_scaler.transform(X_fusion)
        return self.fusion_classifier.predict_proba(X_fusion_scaled)

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.fusion_classifier, p / "fusion_classifier.pkl")
        joblib.dump(self.fusion_scaler, p / "fusion_scaler.pkl")
        self.clinical_model.save(p / "clinical_submodel")
        meta = {
            "fusion_type": self.fusion_type,
            "classes": self.classes,
            "architecture": "Decoupled ECG Model + Clinical Context Model + Calibrated Fusion Layer",
            "regulatory_note": "Rule 2 Compliant — Sub-models are independently evaluated.",
        }
        with open(p / "fusion_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls,
        model_dir: Path | str,
        ecg_model: Any,
        ecg_scaler: Any,
    ) -> MultimodalECGFusionModel:
        p = Path(model_dir)
        with open(p / "fusion_metadata.json", "r") as f:
            meta = json.load(f)
        clinical_sub = ClinicalContextModel.load(p / "clinical_submodel")
        inst = cls(
            ecg_model=ecg_model,
            ecg_scaler=ecg_scaler,
            clinical_model=clinical_sub,
            fusion_type=meta["fusion_type"],
            classes=meta["classes"],
        )
        inst.fusion_classifier = joblib.load(p / "fusion_classifier.pkl")
        inst.fusion_scaler = joblib.load(p / "fusion_scaler.pkl")
        inst.is_fitted = True
        return inst
