"""
src/phase4b_constrained_discovery.py - Ground-Truth Constrained Discovery Engine & Object Splitter
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def extract_constrained_subclasses(
    cube: np.ndarray,
    gt: np.ndarray,
    n_components: int = 3,
    min_subsegments: int = 2,
    max_subsegments: int = 3,
    compactness: float = 10.0,
    subsegments_per_class: int | None = None,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Sub-clusters ground-truth masks spectrally while smoothing noise to discover
    clean, contiguous physical sub-objects without over-segmentation (max 3 sub-clusters/class).
    """
    H, W, B = cube.shape
    flat_cube = cube.reshape(-1, B)

    # 1. PCA feature extraction
    pca = PCA(n_components=n_components, random_state=seed)
    cube_pca = pca.fit_transform(flat_cube).reshape(H, W, n_components)

    # 2. Compute normalized NDVI feature with dynamic band clamping for small cubes (e.g., unit tests)
    nir_idx = min(68, B - 1)
    red_idx = min(40, max(0, B // 2))

    nir = cube[:, :, nir_idx].astype(np.float32)
    red = cube[:, :, red_idx].astype(np.float32)
    denom = nir + red
    valid_ndvi = denom > 1e-5
    ndvi = np.zeros((H, W), dtype=np.float32)
    ndvi[valid_ndvi] = (nir[valid_ndvi] - red[valid_ndvi]) / denom[valid_ndvi]

    # Combine & apply gentle spatial Gaussian smoothing (sigma=1.5) to prevent over-splitting
    features = np.dstack([cube_pca, ndvi[:, :, None] * 2.0])
    for c in range(features.shape[-1]):
        features[:, :, c] = gaussian_filter(features[:, :, c], sigma=1.5)

    constrained_cluster_map = np.zeros((H, W), dtype=np.int32)
    parent_gt_mapped_map = np.zeros((H, W), dtype=np.int32)
    subclass_registry = {}
    current_cluster_id = 1

    unique_classes = np.unique(gt)
    unique_classes = unique_classes[unique_classes != 0]

    for class_id in unique_classes:
        class_mask = (gt == class_id)
        pixel_count = int(np.sum(class_mask))
        if pixel_count == 0:
            continue

        if subsegments_per_class is not None:
            n_sub = subsegments_per_class
        else:
            if pixel_count > 3000:
                n_sub = max_subsegments
            else:
                n_sub = min_subsegments

        class_features = features[class_mask]

        kmeans = KMeans(n_clusters=n_sub, random_state=seed, n_init=10)
        local_labels = kmeans.fit_predict(class_features) + 1

        coords = np.argwhere(class_mask)
        for idx, (r, c) in enumerate(coords):
            local_id = local_labels[idx]
            global_sub_id = current_cluster_id + (local_id - 1)

            constrained_cluster_map[r, c] = global_sub_id
            parent_gt_mapped_map[r, c] = class_id

        for local_id in range(1, n_sub + 1):
            gid = current_cluster_id + (local_id - 1)
            sub_count = int(np.sum(local_labels == local_id))
            subclass_registry[gid] = {
                "parent_gt_class": int(class_id),
                "subclass_index": int(local_id),
                "pixel_count": sub_count
            }

        current_cluster_id += n_sub

    return constrained_cluster_map, parent_gt_mapped_map, subclass_registry


def extract_object_level_split(
    cube: np.ndarray,
    gt: np.ndarray,
    cluster_map: np.ndarray,
    test_ratio: float = 0.3,
    patch_size: int = 3,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs whole-object stratified train/test split with zero-indexed targets (0..8)."""
    H, W, B = cube.shape
    half_p = patch_size // 2

    padded_cube = np.pad(
        cube,
        ((half_p, half_p), (half_p, half_p), (0, 0)),
        mode="reflect"
    )

    # Class-stratified cluster partitioning
    rng = np.random.default_rng(seed)
    train_objects = set()
    test_objects = set()

    # Stratify sub-clusters per parent ground-truth class
    for class_id in np.unique(gt[gt > 0]):
        class_clusters = np.unique(cluster_map[(gt == class_id) & (cluster_map > 0)])
        if len(class_clusters) == 0:
            continue
        rng.shuffle(class_clusters)

        # Guarantee at least 1 cluster in train and at least 1 in test if >= 2 clusters exist
        if len(class_clusters) == 1:
            train_objects.add(class_clusters[0])
        else:
            n_test = max(1, int(round(len(class_clusters) * test_ratio)))
            test_objects.update(class_clusters[:n_test])
            train_objects.update(class_clusters[n_test:])

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    valid_coords = np.argwhere(gt > 0)

    for r, c in valid_coords:
        label = int(gt[r, c])
        obj_id = int(cluster_map[r, c])

        if label < 1 or label > 9 or obj_id == 0:
            continue

        patch = padded_cube[r : r + patch_size, c : c + patch_size, :]
        patch_tensor = np.transpose(patch, (2, 0, 1))
        target_class = label - 1

        if obj_id in train_objects:
            X_train_list.append(patch_tensor)
            y_train_list.append(target_class)
        elif obj_id in test_objects:
            X_test_list.append(patch_tensor)
            y_test_list.append(target_class)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, y_train, X_test, y_test
