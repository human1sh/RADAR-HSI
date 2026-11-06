"""
src/utils.py - Spatial 2D scene prediction tools and visualization map rendering.
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = [
    "Unlabeled", "Asphalt", "Meadows", "Gravel", "Trees",
    "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"
]

CLASS_COLORS = [
    '#000000', '#4682B4', '#32CD32', '#D2691E', '#228B22',
    '#FF69B4', '#8B4513', '#708090', '#FF4500', '#FFD700'
]


def predict_full_scene(model, cube: np.ndarray) -> np.ndarray:
    """Flattens 3D datacube (H, W, B), runs model inference, and reshapes into 2D map (H, W)."""
    H, W, B = cube.shape
    X_all = cube.reshape(-1, B)
    preds = model.predict(X_all)
    return preds.reshape(H, W)


def plot_classification_maps(
    gt: np.ndarray,
    pred_map: np.ndarray,
    title: str = "Model Classification Map vs Ground Truth",
    save_path: Optional[Union[str, Path]] = None,
    split_row: Optional[int] = None
) -> None:
    """Plots Ground Truth map alongside predicted 2D classification map."""
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    cmap = ListedColormap(CLASS_COLORS)

    pred_masked = np.zeros_like(gt)
    pred_masked[gt > 0] = pred_map[gt > 0]

    # Ground Truth
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title("Ground Truth Map (`paviaU_gt`)", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    # Predictions
    axes[1].imshow(pred_masked, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title(f"Predicted Map: {title}", fontsize=12, fontweight='bold')
    axes[1].axis("off")

    if split_row is not None:
        axes[1].axhline(
            y=split_row, color='red', linestyle='--', linewidth=2,
            label=f"Train/Test Spatial Cut (Row {split_row})"
        )
        axes[1].legend(loc="upper right", framealpha=0.8)

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=f"{i}: {CLASS_NAMES[i]}")
        for i in range(1, 10)
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        fontsize=10
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.close()
