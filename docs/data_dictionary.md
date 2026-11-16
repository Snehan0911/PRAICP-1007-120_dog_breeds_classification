# CanineVision AI — Dataset Dictionary & Taxonomy

## 1. Dataset Overview
- **Source:** Stanford Dog Breeds Classification Dataset (120 categories).
- **Total Images:** 20,580 images.
- **Image Format:** JPEG (.jpg).
- **Target Dimensions:** Resized to 224 x 224 x 3.
- **Class Balance:**
  - Minimum class count: 148 images (`n02090379-redbone`)
  - Maximum class count: 252 images (`n02085936-Maltese_dog`)
  - Imbalance Ratio: 1.7x (Mild)

## 2. Split Proportions (Stratified)
- **Training Set (70%):** 14,406 images (`data/splits/train.csv`)
- **Validation Set (15%):** 3,087 images (`data/splits/val.csv`)
- **Test Set (15%):** 3,087 images (`data/splits/test.csv`)

## 3. Class Index Mapping
The complete canonical mapping of all 120 breeds is stored deterministically in:
`artifacts/class_mapping/class_indices.json`
