"""
src/phase2_restoration.py - Adaptive Radiometric Restoration Engine.
"""

import numpy as np
from scipy.signal import savgol_filter


class RadiometricRestorer:
    """
    Adaptive restoration engine providing column-wise moment matching destriping
    and spectral smoothing tailored for spaceborne optical payloads.
    """

    def __init__(
        self,
        apply_destripe: bool = True,
        apply_smoothing: bool = True,
        sg_window: int = 5,
        sg_poly: int = 2,
        stripe_threshold: float = 0.02
    ):
        self.apply_destripe = apply_destripe
        self.apply_smoothing = apply_smoothing
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.stripe_threshold = stripe_threshold

    def moment_matching_destripe(self, cube: np.ndarray) -> np.ndarray:
        """Removes pushbroom column striping only when column gain deviation exceeds threshold."""
        H, W, B = cube.shape
        destriped = cube.copy()

        for b in range(B):
            band = destriped[:, :, b]
            global_mean = np.mean(band)
            global_std = np.std(band) + 1e-8

            col_means = np.mean(band, axis=0)
            col_stds = np.std(band, axis=0) + 1e-8

            # Only adjust columns where relative mean deviation indicates true pushbroom striping
            col_dev = np.abs(col_means - global_mean) / (global_mean + 1e-8)
            stripe_mask = col_dev > self.stripe_threshold

            if np.any(stripe_mask):
                band[:, stripe_mask] = (
                    (band[:, stripe_mask] - col_means[stripe_mask]) *
                    (global_std / col_stds[stripe_mask]) + global_mean
                )
            destriped[:, :, b] = band

        return np.clip(destriped, 0.0, 1.0)

    def spectral_smoothing(self, cube: np.ndarray) -> np.ndarray:
        """Applies conservative Savitzky-Golay filtering to preserve fine absorption features."""
        H, W, B = cube.shape
        flat_cube = cube.reshape(-1, B)
        smoothed = savgol_filter(flat_cube, window_length=self.sg_window, polyorder=self.sg_poly, axis=1)
        return np.clip(smoothed.reshape(H, W, B), 0.0, 1.0)

    def process(self, noisy_cube: np.ndarray) -> np.ndarray:
        """Executes adaptive restoration pipeline."""
        restored = noisy_cube.copy()

        if self.apply_destripe:
            restored = self.moment_matching_destripe(restored)

        if self.apply_smoothing:
            restored = self.spectral_smoothing(restored)

        return restored