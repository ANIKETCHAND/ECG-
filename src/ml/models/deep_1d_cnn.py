"""
Waveform Deep-Learning Architectures for ECG Analysis
=====================================================

Two distinct models live here, and the distinction matters:

1. :class:`ECGFeatureMLPClassifier` — a scikit-learn ``MLPClassifier`` operating
   on the 28 *extracted features*. It is a feature-based multi-layer
   perceptron, **not** a convolutional network. It is the model historically
   registered as ``ECG-MLP-1.0.0-candidate``.

2. :class:`ECGConv1DClassifier` — a genuine PyTorch 1-D convolutional network
   over raw beat waveforms. It is optional: PyTorch is not a required
   dependency, and when it is unavailable this class raises a clear error rather
   than silently degrading to a different model family.

The class name ``ECG1DCNNClassifier`` is retained as an explicit, warning
subclass for backwards compatibility; it does *not* perform convolution and says
so on construction.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:  # PyTorch is optional by design.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


class PyTorchNotAvailable(RuntimeError):
    """Raised when a convolutional model is requested without PyTorch installed."""


class ECGFeatureMLPClassifier:
    """Feature-based multi-layer perceptron for beat classification.

    Operates on extracted feature vectors (default: the 28 morphological,
    spectral and RR-dynamics features). Despite the historical "1D CNN" naming in
    this file, this model performs no convolution.
    """

    architecture = "FEATURE_MLP"

    def __init__(
        self,
        input_length: int = 28,
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ECGFeatureMLPClassifier":
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if len(X) < 100:
            self.model.early_stopping = False
        self.model.fit(self.scaler.fit_transform(X), y)
        self.is_fitted = True
        return self

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.scaler.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._prepare(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._prepare(X))

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "model_type": "Feature-based MLP (no convolution)",
            "architecture": self.architecture,
            "input_length": self.input_length,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "n_iter_": getattr(self.model, "n_iter_", None),
            "loss_": getattr(self.model, "loss_", None),
            "best_loss_": getattr(self.model, "best_loss_", None),
        }


class ECG1DCNNClassifier(ECGFeatureMLPClassifier):
    """Deprecated alias.

    Historically registered as a "1D CNN" while in fact being a feature MLP. Use
    :class:`ECGFeatureMLPClassifier` for feature vectors, and
    :class:`ECGConv1DClassifier` for genuine convolution over waveforms.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "ECG1DCNNClassifier is a feature-based MLP, not a 1D CNN. "
            "Use ECGFeatureMLPClassifier for feature vectors, or "
            "ECGConv1DClassifier for genuine convolution over raw beats.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        self.class_names = kwargs.get("class_names") or self.class_names


def torch_available() -> bool:
    """Return True when a convolutional model can actually be built."""
    return TORCH_AVAILABLE


def _build_conv_net(input_length: int, num_classes: int):
    """Instantiate the convolutional stack, requiring PyTorch.

    Defined as a factory so that importing this module never depends on PyTorch
    being installed.
    """
    if not TORCH_AVAILABLE:
        raise PyTorchNotAvailable(
            "A convolutional model requires PyTorch, which is not installed "
            f"(import error: {TORCH_IMPORT_ERROR}). Install it with: pip install torch"
        )

    class _ConvNet(nn.Module):
        """Small 1-D CNN over a single-lead beat window."""

        def __init__(self, input_length: int, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))

        def forward(self, x):  # pragma: no cover - exercised only with torch
            return self.classifier(self.features(x))

    return _ConvNet(input_length, num_classes)


class ECGConv1DClassifier:
    """Genuine 1-D convolutional network over raw ECG beat waveforms.

    Input is ``(n_beats, input_length)`` single-lead voltage windows, per-beat
    z-score normalised. Requires PyTorch; without it, construction still works but
    :meth:`fit` and :meth:`predict` raise :class:`PyTorchNotAvailable` so that the
    absence of a dependency can never masquerade as a trained model.
    """

    architecture = "CONV1D"

    def __init__(
        self,
        input_length: int = 216,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.input_length = input_length
        self.num_classes = num_classes
        self.class_names = class_names or ["Normal", "PAC", "PVC", "Other"]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device
        self.classes_: Optional[np.ndarray] = None
        self.model: Any = None
        self.is_fitted = False
        self.history_: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _require_torch() -> None:
        if not TORCH_AVAILABLE:
            raise PyTorchNotAvailable(
                "ECGConv1DClassifier requires PyTorch, which is not installed "
                f"(import error: {TORCH_IMPORT_ERROR}).\n"
                "Install it with: pip install torch\n"
                "Until then, use ECGFeatureMLPClassifier (feature-based) or the "
                "Random Forest production model."
            )

    @staticmethod
    def _normalise(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std

    def _to_tensor(self, X: np.ndarray):
        tensor = torch.tensor(self._normalise(X), dtype=torch.float32).unsqueeze(1)
        return tensor.to(self.device)

    # -------------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray, validation_fraction: float = 0.1) -> "ECGConv1DClassifier":
        """Fit the CNN on raw beat windows.

        ``y`` may be string labels; the mapping is stored in ``classes_``.
        """
        self._require_torch()

        X = self._normalise(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_index = np.searchsorted(self.classes_, y)
        self.num_classes = len(self.classes_)
        if self.class_names == ["Normal", "PAC", "PVC", "Other"]:
            self.class_names = [str(c) for c in self.classes_]

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.random_state)
        n_val = int(round(len(X) * validation_fraction)) if len(X) > 20 else 0
        if n_val > 0:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(len(X))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            train_idx, val_idx = np.arange(len(X)), np.array([], dtype=int)

        train_loader = DataLoader(
            TensorDataset(self._to_tensor(X[train_idx]).cpu(), torch.tensor(y_index[train_idx], dtype=torch.long)),
            batch_size=min(self.batch_size, max(1, len(train_idx))),
            shuffle=True,
        )

        self.model = _build_conv_net(self.input_length, self.num_classes).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_state, best_val_loss, patience, stale = None, float("inf"), 5, 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.item()) * len(batch_y)

            record = {"epoch": epoch, "train_loss": epoch_loss / max(1, len(train_idx))}

            if n_val > 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(self._to_tensor(X[val_idx]))
                    val_loss = float(criterion(val_logits, torch.tensor(y_index[val_idx], dtype=torch.long).to(self.device)).item())
                record["val_loss"] = val_loss
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss, stale = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    stale += 1
            self.history_.append(record)

            if n_val > 0 and stale >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_torch()
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(X))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def get_model_summary(self) -> Dict[str, Any]:
        summary = {
            "model_type": "1-D Convolutional Neural Network (raw waveform)",
            "architecture": self.architecture,
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "epochs": self.epochs,
            "torch_available": TORCH_AVAILABLE,
            "is_fitted": self.is_fitted,
            "history": self.history_[-5:],
        }
        if TORCH_AVAILABLE and self.model is not None:
            summary["parameters"] = int(sum(p.numel() for p in self.model.parameters()))
        return summary


__all__ = [
    "ECGFeatureMLPClassifier",
    "ECG1DCNNClassifier",
    "ECGConv1DClassifier",
    "PyTorchNotAvailable",
    "torch_available",
    "TORCH_AVAILABLE",
]
