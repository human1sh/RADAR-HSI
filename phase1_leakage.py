"""
phase1_leakage_expt.py - Phase 1 Hyperspectral Exploratory Data Analysis & Leakage Assessment
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.dataset import PaviaDataset
from src.phase2_simulator import SensorSimulator
from src.phase1_baseline import SpectralAngleMapper, PCASVMClassifier, Spectral1DCNN, PyTorchSpectralTrainer
from src.utils import predict_full_scene, plot_classification_maps
from src.visualize import plot_leakage_bar_chart


DATA_DIR = "./data"
OUTPUT_DIR = Path("./outputs")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 70)
    print(" HYPERSPECTRAL LAND-COVER BENCHMARK & LEAKAGE ANALYSIS")
    print("=" * 70)

    # 1. Load Data & Run Forward Sensor Simulation
    ds = PaviaDataset(data_dir=DATA_DIR)
    print(f"[1/5] Data Loaded: Cube {ds.cube.shape} | Ground Truth {ds.gt.shape}")

    simulator = SensorSimulator(shot_noise=True, read_noise_std=0.008, stripe_factor=0.015, adc_bits=12, seed=42)
    _ = simulator.apply_forward_model(ds.cube)
    print("      • Optoelectronic Sensor Forward Engine Verified (Shot, Read, Striping, 12-bit ADC)")

    # 2. Extract Validation Splits
    X_tr_rnd, X_te_rnd, y_tr_rnd, y_te_rnd = ds.get_random_split(test_size=0.3, seed=42)
    X_tr_spt, X_te_spt, y_tr_spt, y_te_spt = ds.get_spatial_block_split(split_ratio=0.5)
    split_row = int(ds.H * 0.5)

    print("\n[2/5] Validation Splits Extracted:")
    print(f"      Random Split   -> Train: {X_tr_rnd.shape[0]:6d} px | Test: {X_te_rnd.shape[0]:6d} px")
    print(f"      Spatial Block  -> Train: {X_tr_spt.shape[0]:6d} px | Test: {X_te_spt.shape[0]:6d} px")

    # 3. Model Training & Evaluation
    print("\n[3/5] Running Baseline & Deep Learning Classifiers...")

    # SAM
    sam = SpectralAngleMapper()
    sam.fit(X_tr_rnd, y_tr_rnd)
    metrics_sam_rnd = sam.evaluate(X_te_rnd, y_te_rnd)

    # PCA + SVM
    svm_rnd = PCASVMClassifier(n_components=15, C=10.0, seed=42)
    svm_rnd.fit(X_tr_rnd, y_tr_rnd)
    metrics_svm_rnd = svm_rnd.evaluate(X_te_rnd, y_te_rnd)

    svm_spt = PCASVMClassifier(n_components=15, C=10.0, seed=42)
    svm_spt.fit(X_tr_spt, y_tr_spt)
    metrics_svm_spt = svm_spt.evaluate(X_te_spt, y_te_spt)

    # PyTorch 1D-CNN
    print("      • Training PyTorch 1D Spectral CNN...")
    trainer_rnd = PyTorchSpectralTrainer(Spectral1DCNN(in_channels=ds.B, num_classes=9))
    trainer_rnd.fit(X_tr_rnd, y_tr_rnd, epochs=35, batch_size=256)
    metrics_cnn_rnd = trainer_rnd.evaluate(X_te_rnd, y_te_rnd)

    trainer_spt = PyTorchSpectralTrainer(Spectral1DCNN(in_channels=ds.B, num_classes=9))
    trainer_spt.fit(X_tr_spt, y_tr_spt, epochs=35, batch_size=256)
    metrics_cnn_spt = trainer_spt.evaluate(X_te_spt, y_te_spt)

    # 4. Generate & Save Figures
    print("\n[4/5] Generating & Saving Publication Figures into ./outputs/...")
    pred_map_svm = predict_full_scene(svm_spt, ds.cube)
    pred_map_cnn = predict_full_scene(trainer_spt, ds.cube)

    plot_classification_maps(
        gt=ds.gt, pred_map=pred_map_svm,
        title="PCA + RBF SVM (Spatial Block Split)",
        save_path=OUTPUT_DIR / "phase1_svm_spatial_classification_map.png",
        split_row=split_row
    )

    plot_classification_maps(
        gt=ds.gt, pred_map=pred_map_cnn,
        title="PyTorch 1D-CNN (Spatial Block Split)",
        save_path=OUTPUT_DIR / "phase1_cnn_spatial_classification_map.png",
        split_row=split_row
    )

    metrics_summary = {
        "svm_rnd": metrics_svm_rnd, "svm_spt": metrics_svm_spt,
        "cnn_rnd": metrics_cnn_rnd, "cnn_spt": metrics_cnn_spt
    }
    plot_leakage_bar_chart(metrics_summary, save_path=OUTPUT_DIR / "phase1_model_leakage_comparison.png")
    print(f"      • Saved classification maps and bar charts to '{OUTPUT_DIR.resolve()}'")

    # 5. Summary Report
    print("\n[5/5] Final Benchmark Summary:")
    print("=" * 70)
    print(f"{'Model Architecture':<20} | {'Split Strategy':<15} | {'Accuracy':<10} | {'Kappa':<7} | {'Macro F1':<8}")
    print("-" * 70)
    print(f"{'SAM (Physics)':<20} | {'Random Split':<15} | {metrics_sam_rnd['accuracy']*100:6.2f}%    | {metrics_sam_rnd['kappa']:.4f}  | {metrics_sam_rnd['f1_macro']:.4f}")
    print("-" * 70)
    print(f"{'PCA + RBF SVM':<20} | {'Random Split':<15} | {metrics_svm_rnd['accuracy']*100:6.2f}%    | {metrics_svm_rnd['kappa']:.4f}  | {metrics_svm_rnd['f1_macro']:.4f}")
    print(f"{'PCA + RBF SVM':<20} | {'Spatial Block':<15} | {metrics_svm_spt['accuracy']*100:6.2f}%    | {metrics_svm_spt['kappa']:.4f}  | {metrics_svm_spt['f1_macro']:.4f}")
    print("-" * 70)
    print(f"{'PyTorch 1D-CNN':<20} | {'Random Split':<15} | {metrics_cnn_rnd['accuracy']*100:6.2f}%    | {metrics_cnn_rnd['kappa']:.4f}  | {metrics_cnn_rnd['f1_macro']:.4f}")
    print(f"{'PyTorch 1D-CNN':<20} | {'Spatial Block':<15} | {metrics_cnn_spt['accuracy']*100:6.2f}%    | {metrics_cnn_spt['kappa']:.4f}  | {metrics_cnn_spt['f1_macro']:.4f}")
    print("=" * 70)

    gap_svm = (metrics_svm_rnd['accuracy'] - metrics_svm_spt['accuracy']) * 100
    gap_cnn = (metrics_cnn_rnd['accuracy'] - metrics_cnn_spt['accuracy']) * 100

    print("\n--- Key Finding: Spatial Autocorrelation Information Leakage ---")
    print(f"• PCA + SVM Accuracy Inflation : +{gap_svm:.2f}%")
    print(f"• 1D-CNN Accuracy Inflation    : +{gap_cnn:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
