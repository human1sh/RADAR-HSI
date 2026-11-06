"""
src/phase3_spatial_nn.py - 3D Spatial-Spectral CNN Architecture & PyTorch Trainer
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


class SpatialSpectral3DCNN(nn.Module):
    """
    Joint 3D Spatial-Spectral Convolutional Neural Network with dynamic channel calculation.
    """

    def __init__(self, in_channels: int = 103, num_classes: int = 9, patch_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Dynamically scale spectral kernel size if in_channels is small
        k1_spec = min(7, in_channels) if in_channels >= 7 else 3
        k2_spec = min(5, in_channels - (k1_spec - 1)) if in_channels >= 11 else 3

        self.conv3d_layer = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(k1_spec, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=(k2_spec, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.GELU(),
        )

        # Output spectral dimension calculation
        spec_dim = in_channels - (k1_spec - 1) - (k2_spec - 1)
        in_conv2d_channels = 32 * spec_dim

        self.conv2d_layer = nn.Sequential(
            nn.Conv2d(in_conv2d_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (Batch, 1, Channels, Height, Width)

        x = self.conv3d_layer(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)  # Flatten 3D spatial-spectral features to 2D channels

        feat = self.conv2d_layer(x).squeeze(-1).squeeze(-1)
        return self.classifier(feat)


class PyTorchSpatialSpectralTrainer:
    """Trainer wrapper providing standardized fit/predict interfaces & persistence for 3D CNNs."""

    def __init__(self, model: nn.Module, lr: float = 2e-3, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 30, batch_size: int = 128):
        ds_tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64)
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
        preds_list = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                bx = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32).to(self.device)
                preds = torch.argmax(self.model(bx), dim=1).cpu().numpy()
                preds_list.append(preds)
        return np.concatenate(preds_list)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        preds = self.predict(X_test)
        num_classes = getattr(self.model, "num_classes", 9)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "kappa": float(cohen_kappa_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, labels=list(range(num_classes)), average="macro", zero_division=0))
        }

    def save(self, path: Union[str, Path]):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "in_channels": getattr(self.model, "in_channels", 103),
            "num_classes": getattr(self.model, "num_classes", 9),
            "patch_size": getattr(self.model, "patch_size", 3),
        }
        torch.save(checkpoint, save_path)
        print(f"      • Model checkpoint saved successfully to '{save_path.resolve()}'")

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None):
        save_path = Path(path)
        checkpoint = torch.load(save_path, map_location=device or "cpu")

        model = SpatialSpectral3DCNN(
            in_channels=checkpoint.get("in_channels", 103),
            num_classes=checkpoint.get("num_classes", 9),
            patch_size=checkpoint.get("patch_size", 3)
        )
        model.load_state_dict(checkpoint["state_dict"])

        trainer = cls(model=model, device=device)
        print(f"      • Model checkpoint loaded successfully from '{save_path.resolve()}'")
        return trainer


def extract_spatial_patches(
    cube: np.ndarray,
    gt: np.ndarray,
    patch_size: int = 3,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 3D spatial-spectral patches (patch_size x patch_size x B) from a hyperspectral cube.

    Args:
        cube: Hyperspectral image cube of shape (H, W, B).
        gt: Ground truth mask of shape (H, W).
        patch_size: Spatial dimensions of square patch (default: 3).
        mask: Optional boolean mask indicating pixels to extract (defaults to gt > 0).

    Returns:
        Tuple of (patches, labels):
            - patches: Array of shape (N, B, patch_size, patch_size) formatted for 3D CNNs.
            - labels: 1D array of 0-indexed PyTorch target labels (0..num_classes-1).
    """
    H, W, B = cube.shape
    half_p = patch_size // 2

    padded_cube = np.pad(
        cube,
        ((half_p, half_p), (half_p, half_p), (0, 0)),
        mode="reflect"
    )

    if mask is None:
        mask = gt > 0

    valid_coords = np.argwhere(mask)
    patches_list, labels_list = [], []

    for r, c in valid_coords:
        label = int(gt[r, c])
        if label < 1:
            continue

        patch = padded_cube[r : r + patch_size, c : c + patch_size, :]
        patch_tensor = np.transpose(patch, (2, 0, 1))  # (B, H, W)

        patches_list.append(patch_tensor)
        labels_list.append(label - 1)  # Convert GT [1..9] -> PyTorch Target [0..8]

    patches = np.array(patches_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    return patches, labels
