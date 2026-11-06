"""
src/visualize.py - Publication-grade visualization utilities with phase-prefixed image saving.
"""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CLASS_COLORS = [
    "#000000",  # 0: Unlabeled (Black)
    "#4682B4",  # 1: Asphalt (Steel Blue)
    "#32CD32",  # 2: Meadows (Lime Green)
    "#D2691E",  # 3: Gravel (Chocolate)
    "#228B22",  # 4: Trees (Forest Green)
    "#FF69B4",  # 5: Painted metal sheets (Hot Pink)
    "#A0522D",  # 6: Bare Soil (Sienna)
    "#708090",  # 7: Bitumen (Slate Gray)
    "#FF4500",  # 8: Self-Blocking Bricks (Orange Red)
    "#FFD700",  # 9: Shadows (Gold)
]

CLASS_NAMES = [
    "Unlabeled",
    "1: Asphalt",
    "2: Meadows",
    "3: Gravel",
    "4: Trees",
    "5: Painted metal sheets",
    "6: Bare Soil",
    "7: Bitumen",
    "8: Self-Blocking Bricks",
    "9: Shadows",
]


def make_rgb_composite(cube: np.ndarray, bands: List[int] = [60, 30, 10]) -> np.ndarray:
    """Generates a contrast-normalized RGB composite image from specified spectral band indices."""
    rgb = cube[:, :, bands].copy()
    for i in range(3):
        channel = rgb[:, :, i]
        p2, p98 = np.percentile(channel, (2, 98))
        rgb[:, :, i] = np.clip((channel - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb


def plot_classification_map(
    gt: np.ndarray,
    pred_map: np.ndarray,
    title: str = "Spatial Classification Map",
    save_path: Optional[Union[str, Path]] = None,
    split_row: Optional[int] = None
):
    """Plots Ground Truth alongside 2D Predicted Classification Map."""
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    # Panel 1: Ground Truth Map
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title("Ground Truth Map (paviaU_gt)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Prediction Map
    pred_masked = np.zeros_like(gt)
    pred_masked[gt > 0] = pred_map[gt > 0]
    axes[1].imshow(pred_masked, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title(title, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    if split_row is not None:
        axes[1].axhline(
            y=split_row, color="red", linestyle="--", linewidth=2,
            label=f"Spatial Split Cut (Row {split_row})"
        )
        axes[1].legend(loc="upper right", framealpha=0.8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i])
        for i in range(1, len(CLASS_NAMES))
    ]
    labels = CLASS_NAMES[1:]
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=10,
        frameon=True, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"      • Saved classification map to '{save_path.resolve()}'")
        plt.close()
    else:
        plt.show()


def plot_leakage_bar_chart(
    metrics_summary: dict[str, dict[str, float]],
    save_path: Optional[Union[str, Path]] = None
):
    """Plots a grouped bar chart comparing Random Split vs. Spatial Block Cut accuracy."""
    models = ["PCA + RBF SVM", "PyTorch 1D-CNN"]
    random_accs = [
        metrics_summary["svm_rnd"]["accuracy"] * 100,
        metrics_summary["cnn_rnd"]["accuracy"] * 100
    ]
    spatial_accs = [
        metrics_summary["svm_spt"]["accuracy"] * 100,
        metrics_summary["cnn_spt"]["accuracy"] * 100
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars1 = ax.bar(x - width / 2, random_accs, width, label="Random Pixel Split (Inflated)", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, spatial_accs, width, label="Spatial Block Cut (Honest)", color="#c44e52")

    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Phase 1: Spatial Autocorrelation Leakage Benchmark", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 120)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"      • Saved leakage bar chart to '{save_path.resolve()}'")
        plt.close()
    else:
        plt.show()


def plot_restoration_comparison_map(
    gt: np.ndarray,
    unrestored_map: np.ndarray,
    restored_map: np.ndarray,
    sigma: float,
    save_path: Optional[Union[str, Path]] = None
):
    """Plots 3-panel comparison: [ Ground Truth | Unrestored Noisy Map | Restored Map ]."""
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=300)

    # Ground Truth
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title("Ground Truth (paviaU_gt)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Unrestored
    unrestored_masked = np.zeros_like(gt)
    unrestored_masked[gt > 0] = unrestored_map[gt > 0]
    axes[1].imshow(unrestored_masked, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title(f"Unrestored Noisy Payload (sigma={sigma})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Restored
    restored_masked = np.zeros_like(gt)
    restored_masked[gt > 0] = restored_map[gt > 0]
    axes[2].imshow(restored_masked, cmap=cmap, vmin=0, vmax=9)
    axes[2].set_title("Restored Payload (Destriped + SG)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i])
        for i in range(1, len(CLASS_NAMES))
    ]
    labels = CLASS_NAMES[1:]
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=10,
        frameon=True, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"      • Saved restoration comparison map to '{save_path.resolve()}'")
        plt.close()
    else:
        plt.show()


def plot_three_panel_map(
    cube: np.ndarray,
    gt: np.ndarray,
    pred_map: np.ndarray,
    title: str = "3-Panel Spatial Classification",
    pred_title: str = "Predicted Map",
    save_path: Optional[Union[str, Path]] = None,
    rgb_bands: List[int] = [60, 30, 10]
):
    """Plots a 3-panel figure: [ RGB Composite | Ground Truth | Model Prediction ]."""
    cmap = ListedColormap(CLASS_COLORS)
    rgb_img = make_rgb_composite(cube, bands=rgb_bands)

    fig, axes = plt.subplots(1, 3, figsize=(21, 9), dpi=300)

    # Panel 1: RGB Image Composite
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"RGB Image Composite (Bands {rgb_bands})", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Panel 2: Ground Truth Map
    axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title("Ground Truth Map (`paviaU_gt)", fontsize=14, fontweight="bold", pad=10)
    axes[1].axis("off")

    # Panel 3: Prediction Map
    axes[2].imshow(pred_map, cmap=cmap, vmin=0, vmax=9)
    axes[2].set_title(pred_title, fontsize=14, fontweight="bold", pad=10)
    axes[2].axis("off")

    # Shared Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i])
        for i in range(1, len(CLASS_NAMES))
    ]
    labels = CLASS_NAMES[1:]
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=11,
        frameon=True, bbox_to_anchor=(0.5, -0.02)
    )

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"      • Saved 3-panel map to '{save_path.resolve()}'")
        plt.close()
