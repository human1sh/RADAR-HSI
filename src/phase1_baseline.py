"""
src/phase1_baseline.py - Physics-based, Classical Machine Learning, and Deep Learning
                         Baseline Models for Hyperspectral Classification (Phase 1).
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


# Explicitly declare all 9 ground-truth target classes for Pavia University (1..9)
PAVIA_LABELS = list(range(1, 10))


class SpectralAngleMapper:
    """Spectral Angle Mapper (SAM) physics-based vector classifier."""

    def __init__(self):
        self.class_means = {}
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calculates normalized mean reference spectrum per land-cover class."""
        self.classes = np.unique(y)
        for c in self.classes:
            class_pixels = X[y == c]
            mean_vector = np.mean(class_pixels, axis=0)
            norm = np.linalg.norm(mean_vector)
            self.class_means[c] = mean_vector / (norm + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns class with minimum spectral angle theta."""
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        angles = []
        for c in self.classes:
            ref = self.class_means[c]
            cos_theta = np.clip(np.dot(X_norm, ref), -1.0, 1.0)
            angles.append(np.arccos(cos_theta))

        angles = np.column_stack(angles)
        pred_indices = np.argmin(angles, axis=1)
        return self.classes[pred_indices]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "kappa": float(cohen_kappa_score(y_test, preds, labels=PAVIA_LABELS)),
            "f1_macro": float(f1_score(
                y_test,
                preds,
                labels=PAVIA_LABELS,
                average="macro",
                zero_division=0
            ))
        }


class PCASVMClassifier:
    """Principal Component Analysis (PCA) + RBF Support Vector Machine Classifier."""

    def __init__(self, n_components: int = 15, C: float = 10.0, seed: int = 42):
        self.n_components = n_components
        self.C = C
        self.seed = seed
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.clf = SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_pca = self.pca.fit_transform(X)
        self.clf.fit(X_pca, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X)
        return self.clf.predict(X_pca)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "kappa": float(cohen_kappa_score(y_test, preds, labels=PAVIA_LABELS)),
            "f1_macro": float(f1_score(
                y_test,
                preds,
                labels=PAVIA_LABELS,
                average="macro",
                zero_division=0
            ))
        }


class Spectral1DCNN(nn.Module):
    """Deep 1D Convolutional Neural Network for per-pixel spectral sequence classification."""

    def __init__(self, in_channels: int = 103, num_classes: int = 9):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.feature_extractor(x).squeeze(-1)
        return self.classifier(feat)


class PyTorchSpectralTrainer:
    """Trainer wrapper providing standardized fit/predict interfaces for PyTorch 1D models."""

    def __init__(self, model: nn.Module, lr: float = 3e-3, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 35, batch_size: int = 256):
        # Maps 1-indexed targets (1..9) -> PyTorch indices (0..8)
        ds_tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train - 1, dtype=torch.int64)
        )
        loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            preds = torch.argmax(self.model(tx), dim=1).cpu().numpy()
        return preds + 1  # Returns 1-indexed predictions (1..9)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "kappa": float(cohen_kappa_score(y_test, preds, labels=PAVIA_LABELS)),
            "f1_macro": float(f1_score(
                y_test,
                preds,
                labels=PAVIA_LABELS,
                average="macro",
                zero_division=0
            ))
        }
