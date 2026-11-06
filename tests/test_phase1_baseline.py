"""Unit tests for Phase 1 baseline classifiers (src/phase1_baselines.py)."""

import numpy as np
import pytest
from src.phase1_baseline import SpectralAngleMapper, PCASVMClassifier, Spectral1DCNN, PyTorchSpectralTrainer
from tests.test_synthetic import make_fake_pavia


def test_sam_and_svm_baselines(tmp_path):
    ds = make_fake_pavia(tmp_path)
    X_tr, X_te, y_tr, y_te = ds.get_random_split(test_size=0.3, seed=42)
    
    # SAM
    sam = SpectralAngleMapper()
    sam.fit(X_tr, y_tr)
    metrics_sam = sam.evaluate(X_te, y_te)
    assert 0.0 <= metrics_sam["accuracy"] <= 1.0
    
    # PCA + SVM
    svm = PCASVMClassifier(n_components=5, C=1.0, seed=42)
    svm.fit(X_tr, y_tr)
    metrics_svm = svm.evaluate(X_te, y_te)
    assert metrics_svm["accuracy"] > 0.50


def test_1d_spectral_cnn_trainer(tmp_path):
    pytest.importorskip("torch")
    ds = make_fake_pavia(tmp_path)
    X_tr, X_te, y_tr, y_te = ds.get_random_split(test_size=0.3, seed=42)
    
    # Pass 1-indexed targets (1..9) directly; PyTorchSpectralTrainer handles the -1 conversion internally
    trainer = PyTorchSpectralTrainer(Spectral1DCNN(in_channels=ds.B, num_classes=9), lr=1e-3)
    trainer.fit(X_tr, y_tr, epochs=2, batch_size=64)
    
    preds = trainer.predict(X_te)
    metrics = trainer.evaluate(X_te, y_te)
    
    assert len(preds) == len(y_te)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "kappa" in metrics and "f1_macro" in metrics