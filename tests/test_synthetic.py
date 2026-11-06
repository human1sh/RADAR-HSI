"""Small separable fake Pavia-like dataset for unit tests (9 classes laid out in a grid)."""

import numpy as np
from scipy.io import savemat
from src.dataset import PaviaDataset


def make_fake_pavia(tmp_path, cells=6, B=30, seed=0):
    rng = np.random.default_rng(seed)
    cell = 20
    H = W = cells * cell
    
    # Generate distinct class mean spectra across B bands
    means = np.cumsum(rng.standard_normal((9, B)), axis=1) * 300 + 3000
    classes = rng.permutation(np.resize(np.arange(1, 10), cells * cells)).reshape(cells, cells)
    gt = np.kron(classes, np.ones((cell, cell), int)).astype(np.int32)
    
    # Build raw digital counts
    cube = means[gt - 1] + 150 * rng.standard_normal((H, W, B))
    cube = np.clip(cube, 0, 8000).astype(np.float32)
    
    savemat(str(tmp_path / "paviaU.mat"), {"paviaU": cube})
    savemat(str(tmp_path / "paviaU_gt.mat"), {"paviaU_gt": gt})
    
    return PaviaDataset(tmp_path)