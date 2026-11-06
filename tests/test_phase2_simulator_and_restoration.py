"""Unit tests for Phase 2 Sensor Simulation & Radiometric Restoration."""

import numpy as np
import pytest
from src.phase2_simulator import SensorSimulator
from src.phase2_restoration import RadiometricRestorer
from tests.test_synthetic import make_fake_pavia


def test_sensor_simulator_noise_chain(tmp_path):
    ds = make_fake_pavia(tmp_path)
    simulator = SensorSimulator(shot_noise=True, read_noise_std=0.01, stripe_factor=0.015, adc_bits=12, seed=42)
    
    noisy_cube = simulator.apply_forward_model(ds.cube)
    
    assert noisy_cube.shape == ds.cube.shape
    assert noisy_cube.min() >= 0.0 and noisy_cube.max() <= 1.0
    
    # Check 12-bit ADC quantization levels
    max_dn = (2 ** 12) - 1
    levels = noisy_cube * max_dn
    assert np.allclose(levels, np.round(levels), atol=1e-3)


def test_radiometric_restoration_pipeline(tmp_path):
    ds = make_fake_pavia(tmp_path)
    simulator = SensorSimulator(shot_noise=True, read_noise_std=0.01, stripe_factor=0.015, seed=42)
    noisy_cube = simulator.apply_forward_model(ds.cube)
    
    restorer = RadiometricRestorer(apply_destripe=True, apply_smoothing=True, sg_window=5, sg_poly=2)
    restored_cube = restorer.process(noisy_cube)
    
    assert restored_cube.shape == ds.cube.shape
    assert restored_cube.min() >= 0.0 and restored_cube.max() <= 1.0