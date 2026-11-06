"""Unit tests for src/dataset.py."""

import numpy as np
import pytest
from src.dataset import PaviaDataset
from tests.test_synthetic import make_fake_pavia


def test_dataset_loading_and_normalization(tmp_path):
    ds = make_fake_pavia(tmp_path)
    assert ds.cube.min() >= 0.0 and ds.cube.max() <= 1.0
    assert ds.cube.dtype == np.float32
    assert ds.H > 0 and ds.W > 0 and ds.B == 30
    assert len(ds.wavelengths) == ds.B
    assert ds.wavelengths[0] == 430.0 and ds.wavelengths[-1] == 860.0


def test_random_split_generation(tmp_path):
    ds = make_fake_pavia(tmp_path)
    X_tr, X_te, y_tr, y_te = ds.get_random_split(test_size=0.3, seed=42)
    
    assert len(X_tr) + len(X_te) == (ds.gt > 0).sum()
    assert X_tr.shape[1] == ds.B
    assert set(np.unique(y_tr)).issubset(set(range(1, 10)))
    assert set(np.unique(y_te)).issubset(set(range(1, 10)))


def test_spatial_block_split(tmp_path):
    ds = make_fake_pavia(tmp_path)
    X_tr, X_te, y_tr, y_te = ds.get_spatial_block_split(split_ratio=0.5)
    
    assert X_tr.shape[1] == ds.B
    assert X_te.shape[1] == ds.B
    assert len(X_tr) > 0 and len(X_te) > 0


def test_missing_files_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        PaviaDataset(tmp_path)