"""Unit tests for Phase 4 Unsupervised Discovery Pipeline."""

import numpy as np
from src.phase4a_unconstrained_slic import UnconstrainedSLICEngine
from src.phase4b_constrained_discovery import extract_constrained_subclasses, extract_object_level_split
from tests.test_synthetic import make_fake_pavia


def test_unconstrained_slic_extraction(tmp_path):
    ds = make_fake_pavia(tmp_path)
    
    engine = UnconstrainedSLICEngine(n_segments=30, n_components=3, compactness=10.0)
    cluster_map = engine.fit_predict(ds.cube)
    
    assert cluster_map.shape == (ds.H, ds.W)
    assert len(np.unique(cluster_map)) > 1
    
    nmi_score = engine.evaluate_alignment(ds.gt, cluster_map)
    assert 0.0 <= nmi_score <= 1.0


def test_constrained_subclass_discovery_and_split(tmp_path):
    ds = make_fake_pavia(tmp_path)
    
    constrained_map, parent_gt_map, registry = extract_constrained_subclasses(
        cube=ds.cube,
        gt=ds.gt,
        n_components=3,
        min_subsegments=2,
        max_subsegments=3
    )
    
    assert constrained_map.shape == (ds.H, ds.W)
    assert parent_gt_map.shape == (ds.H, ds.W)
    assert len(registry) > 0
    
    X_tr, y_tr, X_te, y_te = extract_object_level_split(
        ds.cube, ds.gt, constrained_map, test_ratio=0.3, patch_size=3, seed=42
    )
    
    assert X_tr.shape[1:] == (ds.B, 3, 3)
    assert X_te.shape[1:] == (ds.B, 3, 3)
    assert len(X_tr) > 0 and len(X_te) > 0
    assert set(np.unique(y_tr)).issubset(set(range(9)))
    assert set(np.unique(y_te)).issubset(set(range(9)))