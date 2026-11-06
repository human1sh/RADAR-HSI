"""
src/dataset.py - Pavia University dataset loading, normalization, and leakage-aware splitting.
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

class PaviaDataset:
    """Handles dataset loading, min-max reflectance scaling, and spatial validation splits."""

    CLASS_NAMES = ["Unlabeled", "Asphalt", "Meadows", "Gravel", "Trees",
                    "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.cube, self.gt = self._load_data()
        self.H, self.W, self.B = self.cube.shape
        self.wavelengths = np.linspace(430.0, 860.0, self.B, dtype=np.float32)

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        cube_path = self.data_dir / "paviaU.mat"
        gt_path = self.data_dir / "paviaU_gt.mat"
        if not cube_path.exists() or not gt_path.exists():
            raise FileNotFoundError(f"Dataset files not found in '{self.data_dir.resolve()}'. "
                "Expected 'paviaU.mat' and 'paviaU_gt.mat'.")
        cube = loadmat(str(cube_path))["paviaU"].astype(np.float32)
        gt = loadmat(str(gt_path))["paviaU_gt"].astype(np.int32)
        # Scale raw digital counts to normalized [0.0, 1.0] reflectance range
        c_min = cube.min()
        c_max = cube.max()
        if c_max > c_min:
            cube_norm = (cube - c_min) / (c_max - c_min)
        else:
            cube_norm = cube
        return cube_norm, gt

    def get_random_split(self, test_size: float=0.3, seed: int=42, return_indices: bool=False):
        """
        Stratified random pixel split over annotated regions (gt > 0).
        Optionally returns test pixel indices to preserve precise indexing when evaluating degraded cubes.
        """
        mask = self.gt > 0
        X = self.cube[mask]
        y = self.gt[mask]
        indices = np.arange(len(y))
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X, y, indices, test_size=test_size, random_state=seed, stratify=y)
        if return_indices:
            return X_tr, X_te, y_tr, y_te, idx_tr, idx_te
        return X_tr, X_te, y_tr, y_te

    def get_spatial_block_split(self, split_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Spatially disjoint block split (isolates spatial autocorrelation contamination)."""
        split_row = int(self.H * split_ratio)
        mask = self.gt > 0
        train_mask = mask & (np.arange(self.H)[:, None] < split_row)
        test_mask = mask & (np.arange(self.H)[:, None] >= split_row)
        X_train, y_train = self.cube[train_mask], self.gt[train_mask]
        X_test, y_test = self.cube[test_mask], self.gt[test_mask]
        return X_train, X_test, y_train, y_test
