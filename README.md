# RADAR-HSI: Radiometric Analysis, Destriping & Automated Recognition for Hyperspectral Imagery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RADAR-HSI** is a modular scientific machine learning (SciML) pipeline for hyperspectral remote sensing, evaluated on the **Pavia University (`PaviaU`)** airborne ROSIS sensor dataset ($610 \times 340$ pixels across $103$ spectral bands, $9$ ground-truth land-cover classes). 

The repository investigates four foundational challenges in operational hyperspectral telemetry:
1. **Spatial Autocorrelation & Data Leakage:** Quantifying the ~45% accuracy inflation caused by standard random pixel splitting versus strict spatial block cuts.
2. **Optoelectronic Sensor Physics & Radiometric Restoration:** Modeling forward detector degradations (photon shot noise, readout noise, vertical striping, 12-bit ADC quantization) and restoring signal fidelity via spatially adaptive moving-window moment matching.
3. **3D Spatial-Spectral Generalization:** Benchmarking 3D convolutional architectures against 1D spectral models on out-of-distribution spatial regions with strict guard-band isolation.
4. **Object-Based Image Analysis (OBIA) & Sub-Cluster Discovery:** Combining SLIC superpixels with connected-component spatial object stratification to uncover 23 intra-class material sub-clusters without class dropout.

---

## Architecture & Experimental Phases

### Phase 1: Spatial Autocorrelation & Data Leakage Benchmark (`phase1_leakage.py`)
Standard pixel-wise random splits allow neighboring pixels—which share strong spatial autocorrelation—to leak across training and test sets. Phase 1 contrasts stratified random pixel splits against a strict North/South spatial block cut at `row = 305`, evaluating across all 9 target classes (`labels=1..9`, `zero_division=0`) to prevent skewed Macro F1 scores from absent test classes.

| Model Architecture | Split Strategy | Accuracy | Cohen's $\kappa$ | Macro F1 | Accuracy Inflation ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SAM (Physics Baseline)** | Random Split | 56.75% | 0.4740 | 0.6245 | — |
| **PCA + RBF SVM** | Random Split | 93.98% | 0.9198 | 0.9254 | — |
| **PCA + RBF SVM** | Spatial Block Cut | 47.07% | 0.3816 | 0.4123 | **+46.90%** |
| **PyTorch 1D-CNN** | Random Split | 95.31% | 0.9377 | 0.9381 | — |
| **PyTorch 1D-CNN** | Spatial Block Cut | 50.77% | 0.4271 | 0.5138 | **+44.54%** |

---

### Phase 2: Sensor Noise Degradation & Radiometric Restoration (`phase2_restoration.py`)
Simulates realistic airborne/spaceborne payload degradation using isolated pseudo-random generators (`np.random.default_rng`) and evaluates a two-stage radiometric restoration pipeline:
* **Spatially Adaptive Column De-striping:** Uses a localized horizontal moving window ($W = 31$) for column moment matching, eliminating sensor striping while preventing over-smoothing of contiguous natural terrain (e.g., wide meadows).
* **Savitzky-Golay Spectral Smoothing:** Suppresses high-frequency readout noise while preserving narrow mineral and vegetation absorption features.

| Readout Noise ($\sigma$) | Unrestored Accuracy | Restored Accuracy | Net Recovery ($\Delta$) | Regime Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **0.001** | 84.06% | 74.78% | -9.28% | Minor over-smoothing penalty on pristine optical data |
| **0.010** | 47.56% | **51.92%** | **+4.36%** | Significant recovery across urban roofs and roads |
| **0.030** | 13.40% | **17.81%** | **+4.42%** | Suppresses salt-and-pepper spectral collapse |
| **0.080** | 3.97% | **4.85%** | **+0.88%** | Prevents metric inversion under extreme noise |
| **0.150** | 3.31% | 3.28% | -0.03% | Signal saturation near random chance |

---

### Phase 3: 3D Spatial-Spectral CNN Generalization (`phase3_spatial3D.py`)
Extracts $3 \times 3 \times 103$ spatial-spectral cubes around target pixels. To guarantee zero spatial patch leakage across the `row = 305` boundary, a 1-pixel guard band (`guard_row = 305`) purges patch centers touching or adjacent to the split line ($14,608$ train patches / $28,000$ test patches).

| Model Architecture | Split Strategy | Accuracy | Cohen's $\kappa$ | Macro F1 | Generalization Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PyTorch 1D Spectral CNN** | Spatial Block (`guard_row=305`) | 50.15% | 0.4196 | 0.5000 | Baseline |
| **PyTorch 3D Spatial-Spectral CNN** | Spatial Block (`guard_row=305`) | **57.36%** | **0.4997** | **0.6164** | **+7.21% Accuracy** |

---

### Phase 4: Unsupervised & Constrained OBIA Discovery (`phase4_unsupervised_pipeline.py`)
Combines spatial superpixel segmentation with unsupervised intra-class clustering:
* **Phase 4A (Double-Blind SLIC):** Unconstrained SLIC superpixel segmentation ($N_{\text{segments}} = 250$, compactness $m = 10$) locks onto physical object boundaries without ground-truth supervision, achieving $\text{NMI} = 0.4878$.
* **Phase 4B (GT-Constrained Discovery & Object Stratification):** Performs K-Means sub-clustering within ground-truth boundaries to uncover **23 distinct intra-class material sub-clusters** (e.g., separating moisture and canopy density variations within Meadows). Uses `scipy.ndimage.label` connected-component extraction to stratify physical spatial objects per parent class, achieving **100% test class coverage (9/9 classes)** and **53.48% accuracy (0.5482 Macro F1)** on object-level 3D-CNN validation.

---

## Repository Structure

```text
Pavia_Hyperspectral/
├── src/
│   ├── dataset.py                      # PaviaU .mat ingestion, normalization, and spatial splits
│   ├── phase1_baseline.py              # SAM, PCA+SVM, and 1D-CNN leakage benchmarks
│   ├── phase2_simulator.py             # Forward optoelectronic sensor noise engine
│   ├── phase2_restoration.py           # Moving-window column destriping & Savitzky-Golay filter
│   ├── phase3_spatial_nn.py            # 3D Spatial-Spectral CNN & guard-band patch extraction
│   ├── phase4a_unconstrained_slic.py   # Double-blind SLIC superpixel segmentation
│   └── phase4b_constrained_discovery.py # Connected-component stratification & sub-cluster discovery
├── tests/                              # Pytest unit test suite (11 verified tests)
├── outputs/                            # Generated classification maps, SNR curves, and OBIA panels
├── phase1_leakage.py                   # CLI entrypoint for Phase 1
├── phase2_restoration.py               # CLI entrypoint for Phase 2
├── phase3_spatial3D.py                 # CLI entrypoint for Phase 3
├── phase4_unsupervised_pipeline.py     # CLI entrypoint for Phase 4
├── pyproject.toml                      # Project metadata and dependencies
└── README.md
```