"""Unit tests for Phase 3 3D Spatial-Spectral CNN and Model Persistence."""

import numpy as np
import pytest
from src.phase3_spatial_nn import SpatialSpectral3DCNN, PyTorchSpatialSpectralTrainer
from src.phase4b_constrained_discovery import extract_constrained_subclasses, extract_object_level_split
from tests.test_synthetic import make_fake_pavia


def test_3d_cnn_fit_predict_save_load(tmp_path):
    pytest.importorskip("torch")
    ds = make_fake_pavia(tmp_path)
    
    # Generate constrained sub-clusters and extract object-stratified 3D patches (patch_size=3)
    constrained_map, _, _ = extract_constrained_subclasses(
        cube=ds.cube, gt=ds.gt, n_components=3, min_subsegments=2, max_subsegments=3
    )
    X_tr, y_tr, X_te, y_te = extract_object_level_split(
        ds.cube, ds.gt, constrained_map, test_ratio=0.3, patch_size=3, seed=42
    )
    
    # Instantiate 3D-CNN and trainer
    model = SpatialSpectral3DCNN(in_channels=ds.B, num_classes=9, patch_size=3)
    trainer = PyTorchSpatialSpectralTrainer(model=model, lr=1e-3)
    
    # Fit model on a subset of training patches
    trainer.fit(X_tr[:100], y_tr[:100], epochs=2, batch_size=32)
    metrics = trainer.evaluate(X_te[:50], y_te[:50])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    
    # Test checkpoint save and load persistence
    ckpt_path = tmp_path / "model.pt"
    trainer.save(ckpt_path)
    assert ckpt_path.exists()
    
    loaded_trainer = PyTorchSpatialSpectralTrainer.load(ckpt_path)
    preds_orig = trainer.predict(X_te[:50])
    preds_load = loaded_trainer.predict(X_te[:50])
    
    assert np.array_equal(preds_orig, preds_load)