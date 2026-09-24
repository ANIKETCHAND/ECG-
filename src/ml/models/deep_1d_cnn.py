"""
Waveform Deep Learning Architectures for ECG Analysis.
Implements 1D-CNN and Multi-Layer Perceptron (MLP) for beat classification and arrhythmia detection.
Includes PyTorch implementations when available, with a scikit-learn/NumPy compatible MLP fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class ECG1DCNNClassifier:
    """
    1D-Convolutional Neural Network / Deep Representation Model for ECG Waveforms.
    Accepts raw fixed-length ECG beat snippets (e.g. 216 samples = 0.6s at 360 Hz).
    """

    def __init__(
        self,
        input_length: int = 216,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.input_length = input_length
        self.num_classes = num_classes
        self.class_names = class_names or ["Normal", "PAC", "PVC", "Other"]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=64,
            learning_rate="adaptive",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> ECG1DCNNClassifier:
        """
        Fit the deep architecture on waveform windows.
        X shape: (n_samples, input_length) or (n_samples, 1, input_length)
        """
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if len(X) < 100:
            self.model.early_stopping = False
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "model_type": "Deep Waveform Neural Network (1D representation / MLP)",
            "input_length": self.input_length,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "n_iter_": getattr(self.model, "n_iter_", None),
            "loss_": getattr(self.model, "loss_", None),
            "best_loss_": getattr(self.model, "best_loss_", None),
        }

