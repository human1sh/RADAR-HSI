"""
phase4_unsupervised_pipeline_expt.py - Master Pipeline with Fixed Panel Geometry, Top-Centered Legend & Figure Titles
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb
from matplotlib.patches import Patch

from src.dataset import PaviaDataset
from src.phase4a_unconstrained_slic import UnconstrainedSLICEngine
from src.phase4b_constrained_discovery import (
    extract_constrained_subclasses,
    extract_object_level_split,
)
from src.phase3_spatial_nn import SpatialSpectral3DCNN, PyTorchSpatialSpectralTrainer

# Directory Setup
DATA_DIR = Path("./data")
FIG_DIR = Path("./outputs")
MODEL_DIR = Path("./models")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Official Pavia University Ground-Truth Class Names & Colors
PAVIA_CLASSES = [
    "Background",
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Painted Metal Sheets",
    "Bare Soil",
    "Bitumen",
    "Self-Blocking Bricks",
    "Shadows"
]

# Distinct palette (Bare Soil: Saddle Brown #8B4513 | Bitumen: Dark Violet #9400D3)
PAVIA_HEX = [
    "#000000", # 0: Background
    "#4682B4", # 1: Asphalt (Steel Blue)
    "#3CB371", # 2: Meadows (Medium Sea Green)
    "#00FFFF", # 3: Gravel (Cyan)
    "#256D1B", # 4: Trees (Forest Green)
    "#FF69B4", # 5: Painted Metal Sheets (Hot Pink)
    "#8B4513", # 6: Bare Soil (Saddle Brown)
    "#9400D3", # 7: Bitumen (Dark Violet)
    "#FF4500", # 8: Self-Blocking Bricks (Orange Red)
    "#FFFF00"  # 9: Shadows (Yellow)
]
pavia_cmap = ListedColormap(PAVIA_HEX)


def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)])


def generate_subcluster_rgb_map(constrained_map: np.ndarray, registry: dict) -> np.ndarray:
    """Generates clean, non-fragmented HSV-shaded variations for each sub-cluster."""
    H, W = constrained_map.shape
    rgb_out = np.zeros((H, W, 3), dtype=np.float32)

    class_groups = {}
    for cid, meta in registry.items():
        pgt = meta["parent_gt_class"]
        class_groups.setdefault(pgt, []).append(cid)

    for pgt, cids in class_groups.items():
        base_rgb = hex_to_rgb(PAVIA_HEX[pgt])
        base_hsv = rgb_to_hsv(base_rgb)
        n_sub = len(cids)

        for idx, cid in enumerate(cids):
            mask = (constrained_map == cid)
            hsv_mod = base_hsv.copy()
            if n_sub > 1:
                val_shift = -0.2 + (idx / (n_sub - 1)) * 0.4
                sat_shift = -0.15 + (idx / (n_sub - 1)) * 0.3
                hsv_mod[2] = np.clip(hsv_mod[2] + val_shift, 0.3, 1.0)
                hsv_mod[1] = np.clip(hsv_mod[1] + sat_shift, 0.2, 1.0)

            rgb_out[mask] = hsv_to_rgb(hsv_mod)

    return rgb_out


def stretch_rgb(rgb_image: np.ndarray) -> np.ndarray:
    stretched = np.zeros_like(rgb_image, dtype=np.float32)
    for b in range(rgb_image.shape[-1]):
        band = rgb_image[:, :, b]
        p_low, p_high = np.percentile(band, (2.0, 98.0))
        if p_high > p_low:
            stretched[:, :, b] = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            stretched[:, :, b] = band
    return stretched


def create_gt_legend_handles():
    return [
        Patch(color=PAVIA_HEX[i], label=f"{i}: {PAVIA_CLASSES[i]}")
        for i in range(1, 10)
    ]


def setup_fixed_3panel_figure(figure_title: str):
    """
    Creates a 3-panel figure with strict canvas geometry and fixed subplots,
    preventing panel dimension shifts regardless of titles or legends.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.0))

    # Assign Figure Main Title at top
    fig.suptitle(figure_title, fontsize=18, fontweight="bold", y=0.95)

    # Lock subplot bounds (left, bottom, right, top)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.74, wspace=0.15)

    return fig, axes


def format_panel(ax, image, title: str, cmap=None, vmin=None, vmax=None):
    if cmap is not None:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    else:
        ax.imshow(image, aspect='equal')

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")


def attach_top_legend(fig):
    fig.legend(
        handles=create_gt_legend_handles(),
        loc="upper center",


        bbox_to_anchor=(0.5, 0.87),
        ncol=9,
        fontsize=11,
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#CCCCCC"
    )


def run_phase4a(ds: PaviaDataset):
    print("\n--------------------------------------------------------------------------")
    print(" [PHASE 4A] RUNNING UNCONSTRAINED DOUBLE-BLIND SLIC DISCOVERY")
    print("--------------------------------------------------------------------------")

    engine_4a = UnconstrainedSLICEngine(n_segments=350, n_components=3, compactness=10.0)
    cluster_map_4a = engine_4a.fit_predict(ds.cube)
    nmi_score = engine_4a.evaluate_alignment(ds.gt, cluster_map_4a)

    print(f" -> Unsupervised NMI Score against Ground Truth: {nmi_score:.4f}")

    rgb_bright = stretch_rgb(ds.cube[:, :, [60, 30, 10]])

    slic_rgb_vis = np.zeros_like(rgb_bright)
    for c_id in np.unique(cluster_map_4a):
        mask = (cluster_map_4a == c_id)
        slic_rgb_vis[mask] = np.mean(rgb_bright[mask], axis=0)

    # Assigned Title: Figure 4A
    fig_title_4a = f"Figure 4A: Pure Unconstrained Double-Blind SLIC Superpixel Segmentation (NMI = {nmi_score:.4f})"
    fig, axes = setup_fixed_3panel_figure(fig_title_4a)

    format_panel(axes[0], rgb_bright, "RGB Image Composite (Bands [60, 30, 10])")
    format_panel(axes[1], ds.gt, "Ground Truth Map (`paviaU_gt`)", cmap=pavia_cmap, vmin=0, vmax=9)
    format_panel(axes[2], slic_rgb_vis, "Per-Superpixel Mean Spectral Reflectance")

    attach_top_legend(fig)

    fig_path = FIG_DIR / "phase4a_unconstrained_discovery_3panel.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f" -> Saved Phase 4A Visual Artifact to: '{fig_path.resolve()}'")


def run_phase4b(ds: PaviaDataset):
    print("\n--------------------------------------------------------------------------")
    print(" [PHASE 4B] RUNNING GROUND-TRUTH CONSTRAINED DISCOVERY & 3D-CNN")
    print("--------------------------------------------------------------------------")

    constrained_map_4b, parent_gt_map, registry = extract_constrained_subclasses(
        cube=ds.cube,
        gt=ds.gt,
        n_components=3,
        min_subsegments=2,
        max_subsegments=3
    )
    print(f" -> Discovered {len(registry)} Intra-Class Sub-Clusters while locking GT perimeters.")

    rgb_bright = stretch_rgb(ds.cube[:, :, [60, 30, 10]])

    # --------------------------------------------------------------------
    # FIGURE 4B-1: Ground Truth Palette Alignment Figure
    # --------------------------------------------------------------------
    fig1_title = "Figure 4B (GT-Aligned): Ground-Truth Perimeter Locking Verification"
    fig1, axes1 = setup_fixed_3panel_figure(fig1_title)

    format_panel(axes1[0], rgb_bright, "RGB Image Composite (Bands [60, 30, 10])")
    format_panel(axes1[1], ds.gt, "Ground Truth Map (`paviaU_gt`)", cmap=pavia_cmap, vmin=0, vmax=9)
    format_panel(axes1[2], parent_gt_map, "Constrained Perimeter Map (GT Palette Aligned)", cmap=pavia_cmap, vmin=0, vmax=9)

    attach_top_legend(fig1)

    fig1_path = FIG_DIR / "phase4b_gt_aligned_3panel.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" -> Saved GT-Aligned Figure to: '{fig1_path.resolve()}'")

    # --------------------------------------------------------------------
    # FIGURE 4B-2: Discovered Intra-Class Sub-Clusters Figure
    # --------------------------------------------------------------------
    subcluster_rgb_vis = generate_subcluster_rgb_map(constrained_map_4b, registry)

    fig2_title = f"Figure 4B (Discovery): GT-Constrained Discovery ({len(registry)} Discovered Intra-Class Sub-Clusters)"
    fig2, axes2 = setup_fixed_3panel_figure(fig2_title)

    format_panel(axes2[0], rgb_bright, "RGB Image Composite (Bands [60, 30, 10])")
    format_panel(axes2[1], ds.gt, "Ground Truth Map (`paviaU_gt`)", cmap=pavia_cmap, vmin=0, vmax=9)
    format_panel(axes2[2], subcluster_rgb_vis, f"Discovered Sub-Cluster Map ({len(registry)} Sub-Classes)")

    attach_top_legend(fig2)

    fig2_path = FIG_DIR / "phase4b_subcluster_discovery_3panel.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" -> Saved Discovered Sub-Clusters Figure to: '{fig2_path.resolve()}'")

    # 3. Object-Stratified Split & Model Training
    print(" -> Building 70/30 object-stratified train/test split (Zero Data Leakage)...")
    X_train, y_train, X_test, y_test = extract_object_level_split(
        ds.cube, ds.gt, constrained_map_4b, test_ratio=0.3, patch_size=3, seed=42
    )

    print(" -> Training 3D Spatial-Spectral CNN model...")
    model = SpatialSpectral3DCNN(in_channels=ds.B, num_classes=9, patch_size=3)
    trainer = PyTorchSpatialSpectralTrainer(model=model, lr=2e-3)
    trainer.fit(X_train, y_train, epochs=30, batch_size=128)

    metrics = trainer.evaluate(X_test, y_test)
    print(f" -> Object-Stratified 3D-CNN Test Accuracy: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['f1_macro']:.4f}")

    model_path = MODEL_DIR / "pavia_3d_spatial_cnn.pt"
    trainer.save(model_path)
    print(f" -> Saved model checkpoint to: '{model_path.resolve()}'")



def main():
    print("==========================================================================")
    print(" PHASE 4 UNIFIED PIPELINE: DISCOVERY, REFINEMENT & OBJECT-LEVEL ML")
    print("==========================================================================")

    ds = PaviaDataset(data_dir=DATA_DIR)
    run_phase4a(ds)
    run_phase4b(ds)

    print("\n==========================================================================")
    print(" PHASE 4 PIPELINE EXECUTION COMPLETE")
    print("==========================================================================")


if __name__ == "__main__":
    main()
