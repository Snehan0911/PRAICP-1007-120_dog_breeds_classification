# Dataset Documentation & Exploration Report
**Project Name:** Dog Breeds Classification  
**Project ID:** PRAICP-1007-120  
**Source Dataset:** Stanford Dogs Dataset / CloudFront Mirror  

---

## 1. Dataset Overview

The dataset consists of **20,580** validated photographs across **120** distinct dog breed categories. Each image is categorized based on biological breed morphology, originating from WordNet taxonomy and ImageNet.

| Metric | Measured Value | Senior Engineer Assessment |
| :--- | :--- | :--- |
| **Total Images** | **20,580** | Large enough for transfer learning; requires CPU-friendly batching |
| **Total Classes** | **120** | Exactly 120 breeds mapped deterministically to indices 0–119 |
| **Corrupted Files** | **0** | Zero corrupt files; 100% byte integrity verified |
| **Min Images / Class** | **148** | Minimum sample count (sufficient support for stratified splits) |
| **Max Images / Class** | **252** | Maximum sample count |
| **Mean Images / Class** | **171.5** | Uniformly distributed around 171.5 images per breed |
| **Imbalance Ratio** | **1.7x** | Low imbalance ratio (148 to 252); weighted cross-entropy not strictly required |

---

## 2. Image Resolution & Dimension Characteristics

Raw images are collected in varied photographic contexts (indoor, outdoor, portrait, action).

- **Width Range:** 97px to 3264px (Mean: 442.53px, Median: 500.0px)
- **Height Range:** 100px to 2562px (Mean: 385.86px, Median: 375.0px)
- **Mean Aspect Ratio:** 1.1871 (Width / Height)
- **Color Channels:** {3: 20579, 4: 1} (Standard 3-channel RGB)

### Engineering Implication for Preprocessing
1. **Resolution Normalization:** Neural network inputs require a fixed spatial tensor (e.g., $224 \times 224 \times 3$). Resizing with bilinear interpolation standardizes arbitrary input scales.
2. **Aspect Ratio Preservation:** To avoid distortion of canine proportions during training, we apply aspect-ratio-preserving smart resizing or standard random crops in augmentation.

---

## 3. Class Imbalance Analysis

With an imbalance ratio of **1.7**, the dataset is remarkably well-balanced for an open-domain vision problem. No single class dominates the distribution (maximum breed holds ~1.2% of the dataset, minimum breed holds ~0.7%).

**Stratified Splitting Strategy:**
We utilize `StratifiedShuffleSplit` (70% train, 15% validation, 15% test) to ensure every single breed is proportionately represented in each data split without leakage.

---

## 4. Generated Artifacts

- **Labels Master Index:** [`labels.csv`](../labels.csv)
- **Class Mapping:** [`model/class_mapping/class_indices.json`](../model/class_mapping/class_indices.json)
- **Class Distribution Chart:** [`outputs/eda/class_distribution.png`](../outputs/eda/class_distribution.png)
- **Resolution Analysis Chart:** [`outputs/eda/image_resolutions.png`](../outputs/eda/image_resolutions.png)
- **Sample Breed Gallery:** [`outputs/eda/sample_gallery.png`](../outputs/eda/sample_gallery.png)
