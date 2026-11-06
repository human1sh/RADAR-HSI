"""
src/phase4a_unconstrained_slic.py - Pure Double-Blind SLIC Engine
"""

import numpy as np
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score


class UnconstrainedSLICEngine:
    def __init__(self, n_segments: int = 350, n_components: int = 3, compactness: float = 10.0):
        self.n_segments = n_segments
        self.n_components = n_components
        self.compactness = compactness

    def fit_predict(self, cube: np.ndarray) -> np.ndarray:
        H, W, B = cube.shape
        flat_cube = cube.reshape(-1, B)

        # 1. Run PCA without StandardScaler to preserve variance weighting
        pca = PCA(n_components=self.n_components, random_state=42)
        cube_pca = pca.fit_transform(flat_cube).reshape(H, W, self.n_components)

        # 2. SLIC Superpixel Segmentation
        cluster_map = slic(
            cube_pca,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=1,
            channel_axis=-1
        )
        return cluster_map

    def evaluate_alignment(self, gt: np.ndarray, cluster_map: np.ndarray) -> float:
        valid_mask = gt > 0
        return float(normalized_mutual_info_score(gt[valid_mask], cluster_map[valid_mask]))