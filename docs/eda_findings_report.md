# Exploratory Data Analysis (EDA) Findings Report
**Project Name:** Dog Breeds Classification  
**Project ID:** PRAICP-1007-120  
**Domain:** Deep Learning / Computer Vision / Fine-Grained Visual Categorization  

---

## 1. Executive Summary

This report synthesizes the empirical findings from our exhaustive scan of all **20,580 images** across **120 Stanford dog breeds**, detailing how morphological, resolution, and pixel-intensity distributions directly inform our machine learning preprocessing and modeling pipeline.

---

## 2. Key Empirical Findings

### A. Class Representation & Balance
- **Total Valid Images:** $20,580$
- **Total Breed Classes:** $120$
- **Highest Represented Breed:** Scottish Deerhound / Maltese Dog ($252$ images)
- **Lowest Represented Breed:** Briard / Redbone ($148$ images)
- **Mean Images per Class:** $171.50$
- **Median Images per Class:** $159.50$
- **Imbalance Ratio:** $\frac{252}{148} \approx 1.70$
- **Engineering Verdict:** An imbalance ratio of $1.70$ is considered well-balanced in computer vision. Class weights or synthetic oversampling (e.g., SMOTE) are unnecessary and could introduce unwanted artifacts. Standard stratified splitting guarantees balanced representation in training, validation, and test sets.

### B. Image Dimension & Geometry Profiling
- **Width Range:** $97\text{ px}$ to $3264\text{ px}$ (Mean: $442.53\text{ px}$, Median: $500.0\text{ px}$)
- **Height Range:** $100\text{ px}$ to $2562\text{ px}$ (Mean: $385.86\text{ px}$, Median: $375.0\text{ px}$)
- **Mean Aspect Ratio:** $1.1871$ (slight landscape bias)
- **Extreme Aspect Ratios:** Some portrait banners ($0.33$) and panoramic landscapes ($2.5+$).
- **Engineering Verdict:** Because deep neural networks (CNNs) require uniform matrix dimensions for dense classification layers, images must be standardized to a canonical input resolution of **$224 \times 224$ pixels**. Bilinear interpolation provides the optimal balance between speed and anti-aliasing quality.

### C. Color Space & Channel Integrity
- **Color Channels:** $20,579$ images are 3-channel RGB; exactly $1$ image is 4-channel RGBA with transparency.
- **Corrupted Images:** Exactly $0$ corrupted files detected across all 20,580 scans.
- **Global RGB Intensity Statistics (0–255 scale):**
  - **Red Channel:** Mean = $121.20$, Std = $63.48$
  - **Green Channel:** Mean = $116.04$, Std = $62.25$
  - **Blue Channel:** Mean = $99.95$, Std = $63.15$
- **Engineering Verdict:** 
  1. The blue channel has a lower mean ($99.95$) than red ($121.20$) and green ($116.04$), reflecting warm, outdoor, natural grass, and domestic indoor lighting environments.
  2. All images must be explicitly converted to 3-channel RGB (`img.convert('RGB')`) prior to array conversion to prevent 4-channel RGBA tensor shape crashes.

---

## 3. How EDA Findings Dictate Preprocessing (Milestone 5)

| EDA Finding | Potential Risk | Engineering Preprocessing Decision |
| :--- | :--- | :--- |
| Variable resolutions ($97\text{px}$ to $3264\text{px}$) | Variable tensor dimensions break matrix multiplication | Standardize all inputs to $224 \times 224$ via `tf.image.resize(images, [224, 224])`. |
| 1 image with RGBA channel mode | Tensor dimension mismatch: `(224, 224, 4)` vs expected `(224, 224, 3)` | Enforce `img.convert('RGB')` or slice `tensor[:, :, :3]`. |
| Fine-grained morphological differences between breeds | Overly aggressive augmentation (e.g. heavy shears) distorts canine features | Use realistic, identity-preserving augmentations: random horizontal flip, slight rotation ($\pm 15^\circ$), and modest zoom ($10\%$). Never apply vertical flip. |
| Balanced class distribution ($1.70\times$ ratio) | Imbalanced splits could starve minority classes of validation representation | Apply `StratifiedShuffleSplit` (70% Train, 15% Validation, 15% Test) preserving exact class ratios across all partitions. |
| Model architecture requirements | Incompatible pixel scalings degrade transfer learning performance | Custom CNN uses $[0, 1]$ rescaling (`rescaling(1./255)`), MobileNetV2 uses $[-1, 1]$ scaling (`mobilenet_v2.preprocess_input`). |

---

## 4. Generated Artifacts

1. [outputs/eda/class_distribution.png](../outputs/eda/class_distribution.png) — Overall class frequencies across all 120 breeds.
2. [outputs/eda/image_resolutions.png](../outputs/eda/image_resolutions.png) — Histograms and scatter plots of image dimensions.
3. [outputs/eda/sample_gallery.png](../outputs/eda/sample_gallery.png) — Visual grid of representative dog photographs.
4. [outputs/eda/top_bottom_breeds.png](../outputs/eda/top_bottom_breeds.png) — Top 10 vs Bottom 10 breed representation.
5. [outputs/eda/pixel_intensity_distribution.png](../outputs/eda/pixel_intensity_distribution.png) — Global probability density distributions for R, G, and B channels.
6. [dog-vision.ipynb](../dog-vision.ipynb) — Interactive Jupyter notebook walking through the analysis.
7. [pre_processing.py](../pre_processing.py) — Root executable script orchestrating the pipeline.
