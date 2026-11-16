# CanineVision AI — Experiment Log & MLOps Audit

## Experiment Group: CanineVision-Dog-Breed-Classification
- **Tracking Engine:** MLflow (local store `./mlruns`)
- **Loss Function:** `sparse_categorical_crossentropy`
- **Batch Size:** 32
- **Image Dimensions:** (224, 224, 3)

---

### Experiment 1: Custom CNN Baseline (from Scratch)
- **Architecture:** 4-Block Conv2D (32 -> 64 -> 128 -> 256) + BatchNorm + MaxPool + Dropout + GAP + Dense(512).
- **Parameters:** 1.8M
- **Epochs:** 30
- **Final Val Accuracy:** 42.5%
- **Observations:** Model severely underfits due to lack of visual pretraining on a 120-class dataset with ~170 images/class.

---

### Experiment 2: MobileNetV2 (Transfer Learning)
- **Architecture:** Pretrained MobileNetV2 backbone (frozen in Phase 1, top 30 layers fine-tuned in Phase 2) + GAP + BatchNorm + Dropout(0.3) + Dense(256) + Softmax(120).
- **Parameters:** 3.5M
- **Phase 1 LR:** 1e-3 | **Phase 2 LR:** 1e-4
- **Final Test Accuracy (Top-1):** 84.2% | **Top-3:** 96.1%
- **Latency (CPU):** 48 ms
- **Observations:** Optimal balance of latency, memory, and categorical precision. Chosen for production deployment.

---

### Experiment 3: EfficientNetB0 (Transfer Learning)
- **Architecture:** Pretrained EfficientNetB0 backbone + GAP + BatchNorm + Dropout(0.3) + Dense(256) + Softmax(120).
- **Parameters:** 5.3M
- **Final Test Accuracy (Top-1):** 85.8% | **Top-3:** 97.2%
- **Latency (CPU):** 82 ms
- **Observations:** Marginal accuracy improvement (+1.6%) over MobileNetV2 at the cost of 70% higher CPU inference latency.
