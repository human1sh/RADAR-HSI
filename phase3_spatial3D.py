"""
run_spatial_experiment.py - Phase 3 execution driver for 3D Spatial-Spectral CNN benchmarking,
                            spatial block split generalization evaluation, and map rendering.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.dataset import PaviaDataset
from src.phase1_baseline import Spectral1DCNN, PyTorchSpectralTrainer
from src.phase3_spatial_nn import SpatialSpectral3DCNN, PyTorchSpatialSpectralTrainer, extract_spatial_patches
from src.visualize import plot_classification_map

OUTPUT_DIR = Path("./outputs")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print("\n" + "=" * 75)
    print(" PHASE 3: 3D SPATIAL-SPECTRAL CNN BENCHMARK & GENERALIZATION")
    print("=" * 75)

    # 1. Load Data
    ds = PaviaDataset(data_dir="./data")
    print(f"[1/4] Data Loaded: Cube {ds.cube.shape} | Ground Truth {ds.gt.shape}")

    # 2. Extract 3D Spatial-Spectral Patches (3x3 spatial context)
    patch_size = 3
    print(f"\n[2/4] Extracting {patch_size}x{patch_size} 3D Spatial-Spectral Patches...")

    # In phase3_spatial3D.py (lines 34-40):
    split_row = int(ds.H * 0.5)
    half_p = patch_size // 2
    mask = ds.gt > 0

    # Exclude guard band rows [split_row - half_p, split_row + half_p - 1]
    train_mask = mask & (np.arange(ds.H)[:, None] < (split_row - half_p))
    test_mask = mask & (np.arange(ds.H)[:, None] >= (split_row + half_p))

    X_tr_patches, y_tr_spt = extract_spatial_patches(ds.cube, ds.gt, patch_size=patch_size, mask=train_mask)
    X_te_patches, y_te_spt = extract_spatial_patches(ds.cube, ds.gt, patch_size=patch_size, mask=test_mask)

    print(f"      • Spatial Block Train Patches : {X_tr_patches.shape}")
    print(f"      • Spatial Block Test Patches  : {X_te_patches.shape}")

    # 3. Train 1D Spectral CNN Baseline for Comparison
    print("\n[3/4] Benchmarking Models on Spatial Block Split (Top/Bottom Cut)...")
    X_tr_spt, X_te_spt, y_tr_1d, y_te_1d = ds.get_spatial_block_split(split_ratio=0.5)
    trainer_1d = PyTorchSpectralTrainer(Spectral1DCNN(in_channels=ds.B, num_classes=9))
    trainer_1d.fit(X_tr_spt, y_tr_1d, epochs=30, batch_size=256)
    metrics_1d = trainer_1d.evaluate(X_te_spt, y_te_1d)

    # 4. Train 3D Spatial-Spectral CNN
    print("      • Training PyTorch 3D Spatial-Spectral CNN...")
    trainer_3d = PyTorchSpatialSpectralTrainer(
        SpatialSpectral3DCNN(in_channels=ds.B, num_classes=9, patch_size=patch_size)
    )
    trainer_3d.fit(X_tr_patches, y_tr_spt, epochs=30, batch_size=128)
    metrics_3d = trainer_3d.evaluate(X_te_patches, y_te_spt)

    # Summary Report
    print("\n" + "=" * 75)
    print(" PHASE 3 BENCHMARK EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Model Architecture':<25} | {'Split Strategy':<15} | {'Accuracy':<10} | {'Kappa':<7} | {'Macro F1':<8}")
    print("-" * 75)
    print(f"{'PyTorch 1D Spectral CNN':<25} | {'Spatial Block':<15} | {metrics_1d['accuracy']*100:6.2f}%    | {metrics_1d['kappa']:.4f}  | {metrics_1d['f1_macro']:.4f}")
    print(f"{'PyTorch 3D Spatial-Spectral':<25} | {'Spatial Block':<15} | {metrics_3d['accuracy']*100:6.2f}%    | {metrics_3d['kappa']:.4f}  | {metrics_3d['f1_macro']:.4f}")
    print("=" * 75)

    boost = (metrics_3d['accuracy'] - metrics_1d['accuracy']) * 100
    print(f"\n--- Key Finding: Spatial Context Generalization Boost ---")
    print(f"• 3D Spatial-Spectral Accuracy Recovery: +{boost:.2f}% over 1D-CNN")
    print("=" * 75 + "\n")

    # 5. Render 2D Prediction Map
    print("[4/4] Rendering 2D Spatial Classification Map for 3D CNN...")
    all_patches, _ = extract_spatial_patches(ds.cube, ds.gt, patch_size=patch_size, mask=mask)
    all_preds = trainer_3d.predict(all_patches)
    pred_map_3d = np.zeros_like(ds.gt)
    pred_map_3d[mask] = all_preds + 1  # Convert (0..8) back to GT space (1..9)

    save_path = OUTPUT_DIR / "phase3_cnn_3d_spatial_classification_map.png"
    plot_classification_map(
        gt=ds.gt,
        pred_map=pred_map_3d,
        title="PyTorch 3D Spatial-Spectral CNN (Spatial Block Split)",
        save_path=save_path,
        split_row=split_row
    )
    print(f"      • Saved 3D CNN classification map to '{save_path.resolve()}'\n")


if __name__ == "__main__":
    main()
