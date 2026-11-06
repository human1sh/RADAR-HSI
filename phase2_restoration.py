"""
run_degradation_experiment.py - Evaluates classifier robustness under varying sensor SNR levels,
                                 quantifies accuracy recovery via radiometric restoration,
                                 and renders 2D spatial prediction comparison maps.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from src.dataset import PaviaDataset
from src.phase2_simulator import SensorSimulator
from src.phase2_restoration import RadiometricRestorer
from src.phase1_baseline import Spectral1DCNN, PyTorchSpectralTrainer
from src.utils import predict_full_scene
from src.visualize import plot_restoration_comparison_map

OUTPUT_DIR = Path("./outputs")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print("\n" + "=" * 75)
    print(" PHASE 2: SENSOR NOISE DEGRADATION & RADIOMETRIC RESTORATION EXPERIMENT")
    print("=" * 75)

    ds = PaviaDataset(data_dir="./data")
    
    # 1. Extract exact test indices alongside train/test pixel arrays
    X_tr_rnd, X_te_rnd, y_tr_rnd, y_te_rnd, _, idx_te = ds.get_random_split(
        test_size=0.3, seed=42, return_indices=True
    )

    # 2. Train baseline PyTorch 1D-CNN on clean reference reflectance spectra
    print("[1/4] Training baseline PyTorch 1D-CNN on clean reference spectra...")
    trainer = PyTorchSpectralTrainer(Spectral1DCNN(in_channels=ds.B, num_classes=9))
    trainer.fit(X_tr_rnd, y_tr_rnd, epochs=35, batch_size=256)

    # 3. Tested Gaussian Readout Noise levels corresponding to degrading SNR
    read_noise_levels = [0.001, 0.01, 0.03, 0.08, 0.15]
    unrestored_accs = []
    restored_accs = []

    restorer = RadiometricRestorer(apply_destripe=True, apply_smoothing=True)

    print("\n[2/4] Evaluating accuracy across degrading sensor noise levels...")
    mask = ds.gt > 0

    for sigma in read_noise_levels:
        # Forward sensor noise simulator
        simulator = SensorSimulator(shot_noise=True, read_noise_std=sigma, stripe_factor=0.015, adc_bits=12, seed=42)
        noisy_cube = simulator.apply_forward_model(ds.cube)

        # Extract exact noisy test spectra via idx_te
        X_noisy_all = noisy_cube[mask]
        X_te_noisy = X_noisy_all[idx_te]
        acc_noisy = trainer.evaluate(X_te_noisy, y_te_rnd)["accuracy"] * 100
        unrestored_accs.append(acc_noisy)

        # Extract exact restored test spectra via idx_te
        restored_cube = restorer.process(noisy_cube)
        X_restored_all = restored_cube[mask]
        X_te_restored = X_restored_all[idx_te]
        acc_restored = trainer.evaluate(X_te_restored, y_te_rnd)["accuracy"] * 100
        restored_accs.append(acc_restored)

        print(f"      • Read Noise sigma = {sigma:.3f} | Unrestored Acc: {acc_noisy:6.2f}% | Restored Acc: {acc_restored:6.2f}% (+{acc_restored - acc_noisy:.2f}%)")

    # 4. Plot SNR Degradation Curve
    print("\n[3/4] Generating SNR Degradation & Restoration Curve...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(read_noise_levels, unrestored_accs, 'o--', color='#c44e52', linewidth=2, label='Degraded Payload (Unrestored)')
    ax.plot(read_noise_levels, restored_accs, 's-', color='#4c72b0', linewidth=2, label='Restored Payload (Destriped + SG Filter)')

    ax.set_xlabel('Readout Noise Std Dev (sigma)', fontsize=12)
    ax.set_ylabel('1D-CNN Accuracy (%)', fontsize=12)
    ax.set_title('Payload Noise Degradation & Radiometric Restoration Recovery', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=11)

    plt.tight_layout()
    curve_save_path = OUTPUT_DIR / "phase2_snr_degradation_restoration_curve.png"
    plt.savefig(curve_save_path, dpi=300)
    plt.close()
    print(f"      • Saved degradation curve to '{curve_save_path.resolve()}'")

    # 5. Render 3-Panel 2D Prediction Maps at Operational Noise Level (sigma = 0.010)
    print("\n[4/4] Generating 2D Prediction Map Comparisons (sigma = 0.010)...")
    target_sigma = 0.010
    sim_target = SensorSimulator(shot_noise=True, read_noise_std=target_sigma, stripe_factor=0.015, adc_bits=12, seed=42)
    noisy_cube_target = sim_target.apply_forward_model(ds.cube)
    restored_cube_target = restorer.process(noisy_cube_target)

    unrestored_full_map = predict_full_scene(trainer, noisy_cube_target)
    restored_full_map = predict_full_scene(trainer, restored_cube_target)

    map_save_path = OUTPUT_DIR / "phase2_restoration_prediction_map.png"
    plot_restoration_comparison_map(
        gt=ds.gt,
        unrestored_map=unrestored_full_map,
        restored_map=restored_full_map,
        sigma=target_sigma,
        save_path=map_save_path
    )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()