"""
src/simulator.py - Forward physics engine modeling optical sensor noise and ADC quantization.
"""

import numpy as np


class SensorSimulator:
    """Simulates optoelectronic flight payload noise chain and 12-bit ADC quantization."""

    def __init__(
        self,
        shot_noise: bool = True,
        read_noise_std: float = 0.008,
        stripe_factor: float = 0.015,
        adc_bits: int = 12,
        seed: int = 42
    ):
        self.shot_noise = shot_noise
        self.read_noise_std = read_noise_std
        self.stripe_factor = stripe_factor
        self.adc_bits = adc_bits
        self.seed = seed

    def apply_forward_model(self, clean_cube: np.ndarray) -> np.ndarray:
        """Applies detector striping, Poisson shot noise, readout noise, and ADC quantization."""
        rng = np.random.default_rng(self.seed)
        noisy = clean_cube.copy()
        _, W, _ = noisy.shape

        # 1. Detector Column Non-Uniformity (Pushbroom Striping)
        if self.stripe_factor > 0:
            column_gains = rng.normal(1.0, self.stripe_factor, size=(1, W, 1))
            noisy = noisy * column_gains

        # 2. Photon Shot Noise (Poisson-distributed signal variance)
        if self.shot_noise:
            full_well_capacity = 10000.0
            photons = rng.poisson(np.clip(noisy, 0.0, 1.0) * full_well_capacity)
            noisy = photons / full_well_capacity

        # 3. Additive Electronic Readout Noise
        if self.read_noise_std > 0:
            read_noise = rng.normal(0, self.read_noise_std, size=noisy.shape)
            noisy = noisy + read_noise

        # 4. 12-Bit ADC Quantization & Dynamic Range Clipping
        noisy = np.clip(noisy, 0.0, 1.0)
        max_dn = (2 ** self.adc_bits) - 1
        quantized = np.round(noisy * max_dn) / max_dn

        return quantized
