# Model Architecture Documentation
**Project Name:** Dog Breeds Classification  
**Project ID:** PRAICP-1007-120  
**Domain:** Deep Learning / Computer Vision / Transfer Learning  

---

## 1. Architectural Strategy & Paradigm Comparison

Dog breed classification across 120 categories represents a **fine-grained visual classification (FGVC)** problem. The morphological differences distinguishing related breeds (such as Norfolk Terrier vs. Norwich Terrier) are subtle, requiring rich, highly discriminative convolutional feature maps.

We engineered three distinct architectures representing three generations of deep vision design:

| Architecture | Paradigm | Total Parameters | Trainable Parameters | FLOPs (MACs) | CPU Latency Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Custom CNN** | Scratch baseline | $486,488$ | $485,528$ | $\approx 250\text{ M}$ | Fast ($<15\text{ ms}$) |
| **Model B: MobileNetV2** | Inverted Residuals (ImageNet) | $2,621,880$ | $361,336$ (Head) | $\approx 300\text{ M}$ | Optimal ($18\text{--}25\text{ ms}$) |
| **Model C: EfficientNetB0** | Compound Scaling (ImageNet) | $4,363,276$ | $361,336$ (Head) | $\approx 390\text{ M}$ | Moderate ($30\text{--}45\text{ ms}$) |

---

## 2. Model A — Custom Baseline CNN (From Scratch)

### Conceptual Mechanics
- **Philosophy:** A lightweight, handcrafted 4-block deep CNN demonstrating fundamental computer vision principles from first principles without pre-trained ImageNet bias.
- **Layers Breakdown:**
  1. **Block 1 ($112 \times 112 \times 32$):** Low-level spatial primitives (edges, contours, color boundaries).
  2. **Block 2 ($56 \times 56 \times 64$):** Mid-level texture patterns (canine fur, shading, gradients).
  3. **Block 3 ($28 \times 28 \times 128$):** High-level component geometry (snout contours, ear profiles, eye sockets).
  4. **Block 4 ($14 \times 14 \times 256$):** Semantic canine representations (head morphology, body proportions).
  5. **Head:** `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.40)` $\rightarrow$ `Dense(256, ReLU)` $\rightarrow$ `Dropout(0.30)` $\rightarrow$ `Dense(120, Softmax)`.

---

## 3. Model B — MobileNetV2 (Selected Primary Model)

### Conceptual Mechanics
- **Inverted Residual Bottleneck Blocks:**
  - Standard residual blocks connect high-dimensional layers through a low-dimensional bottleneck.
  - MobileNetV2 *inverts* this: it expands the low-dimensional input to a higher dimension with a $1 \times 1$ convolution ($6\times$ expansion), applies a spatial $3 \times 3$ **depthwise convolution**, and then projects back down to a low-dimensional bottleneck with a linear $1 \times 1$ pointwise convolution.
- **Why Linear Bottlenecks Matter:** Non-linear activation functions (like ReLU) applied to low-dimensional manifolds destroy information. MobileNetV2 uses linear activation at the bottleneck outputs to preserve canine feature manifolds.
- **Grad-CAM Integration:** We pass `input_tensor=inputs` directly into `tf.keras.applications.MobileNetV2()`, creating a **single flat functional graph**. This avoids nested sub-model disconnects and allows `tf.GradientTape` to compute gradients directly with respect to the final convolutional layer (`out_relu`).

---

## 4. Model C — EfficientNetB0 (Compound Scaling)

### Conceptual Mechanics
- **Compound Scaling Principle:** Traditional CNN scaling arbitrarily scales either network depth (ResNet-50 to ResNet-152), network width (WideResNet), or image resolution. EfficientNet scales depth, width, and resolution simultaneously using a fixed compound scaling coefficient $\phi$:
  $$\text{Depth}: d = \alpha^\phi, \quad \text{Width}: w = \beta^\phi, \quad \text{Resolution}: r = \gamma^\phi$$
  subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.
- **MBConv & Squeeze-and-Excitation (SE):** EfficientNet incorporates SE blocks that adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels.

---

## 5. Model Factory Dispatch Architecture

All models are unified behind [src/models/model_factory.py](../src/models/model_factory.py):

```python
from src.models import create_model

# 1. Custom CNN baseline
model_cnn = create_model("custom_cnn", num_classes=120)

# 2. Pretrained MobileNetV2
model_mobilenet = create_model("mobilenet_v2", num_classes=120, freeze_backbone=True)

# 3. Pretrained EfficientNetB0
model_efficientnet = create_model("efficientnet_b0", num_classes=120, freeze_backbone=True)
```
