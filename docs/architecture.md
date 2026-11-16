# CanineVision AI — System Architecture Specification

## 1. Architectural Philosophy
CanineVision AI follows the **Modular Monolith** paradigm. Unlike fragile distributed microservices, a modular monolith minimizes network latency, simplifies local CPU orchestration, and maximizes testability while preserving strict physical boundaries between layers:
- Data Layer (`src/data`)
- Machine Learning Layer (`src/models`)
- Training & MLOps Layer (`src/training`)
- Evaluation Layer (`src/evaluation`)
- Inference & Explainability Layer (`src/inference`)
- Web API Layer (`api`)
- Web Presentation Layer (`app`)

## 2. Asynchronous Data Pipeline (`tf.data`)
Given the 7.7 GB RAM hardware boundary on Windows, loading all 20,580 images into RAM as an in-memory NumPy array would crash the process with an Out-Of-Memory (OOM) error.

The pipeline architecture relies on `tf.data.Dataset`:
1. `from_tensor_slices`: Streams filepaths without loading pixels.
2. `map(num_parallel_calls=AUTOTUNE)`: Decodes JPEG bytes in parallel across CPU cores.
3. `batch(32)`: Groups samples for SIMD parallel tensor processing.
4. `prefetch(AUTOTUNE)`: Overlaps CPU data preparation with model forward passes.

## 3. Transfer Learning Pipeline
MobileNetV2 and EfficientNetB0 utilize a two-phase training protocol:
- **Phase 1 (Feature Extraction):** Backbone weights are frozen; only the newly added Dense(256) classification head is trained with learning rate 1e-3.
- **Phase 2 (Fine-Tuning):** The top 30 layers of the backbone are unfrozen, and the model is trained with learning rate 1e-4 with Adam optimizer.

## 4. Grad-CAM Mechanics
Grad-CAM calculates the gradient of the predicted class logit $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

The resulting 2D heatmap highlights spatial regions positively correlated with the predicted breed category.
