"""
Explainable AI (XAI) using Gradient-Weighted Class Activation Mapping (Grad-CAM).
Project: Dog Breeds Classification (PRAICP-1007-120)

Generates localized visual explanations highlighting which image regions (e.g. muzzle,
ears, coat patterns) influenced the convolutional network's breed classification.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.preprocessing import load_and_validate_image, preprocess_mobilenet_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("GradCAM")


class GradCAMExplainer:
    """
    Computes Grad-CAM heatmaps and visual overlay explanations for deep CNN models.
    """

    TARGET_LAYER_CANDIDATES: ClassVar[list[str]] = [
        "out_relu",        # MobileNetV2 final activation
        "top_activation",  # EfficientNetB0 final activation
        "conv4_1",         # Custom CNN final conv layer
        "Conv_1",          # MobileNetV2 alternative final conv
    ]

    def __init__(
        self,
        model: tf.keras.Model | str | Path,
        layer_name: str | None = None,
    ) -> None:
        if isinstance(model, (str, Path)):
            self.model = tf.keras.models.load_model(str(model), compile=False)
        else:
            self.model = model

        self.layer_name = layer_name or self._find_deepest_conv_layer()
        logger.info(f"GradCAMExplainer initialized using target layer: '{self.layer_name}'")

        # Build feature extractor model: maps model input -> [target_layer_output, model_output]
        target_layer = self.model.get_layer(self.layer_name)
        self.gradient_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output],
            name="gradcam_extractor",
        )

    def _find_deepest_conv_layer(self) -> str:
        """Finds the most suitable deep convolutional activation layer in the model."""
        layer_names = [layer.name for layer in self.model.layers]
        for candidate in self.TARGET_LAYER_CANDIDATES:
            if candidate in layer_names:
                return candidate

        # Fallback: find the last Conv2D or ReLU layer
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.ReLU)):
                return layer.name

        raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")

    def compute_heatmap(
        self,
        image_tensor: tf.Tensor | np.ndarray,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Calculates the 2D Grad-CAM heatmap for a preprocessed image tensor.

        Args:
            image_tensor: 4D tensor of shape (1, height, width, 3).
            class_idx: Target category index (0-119). If None, uses top predicted class.

        Returns:
            np.ndarray: 2D normalized heatmap of shape (height, width) with values in [0.0, 1.0].
        """
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.gradient_model(image_tensor)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        # Extract gradients of winning class score w.r.t. convolutional feature maps
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradient computation failed. Check layer connectivity.")

        # Global average pooling on gradients yields importance weights (alphas)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by importance weights
        conv_outputs_val = conv_outputs[0]
        heatmap = tf.reduce_sum(pooled_grads * conv_outputs_val, axis=-1)

        # Apply ReLU to keep only positive contributions towards the target class
        heatmap = tf.nn.relu(heatmap)

        # Normalize heatmap to [0.0, 1.0]
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()

        # Resize heatmap to match spatial dimensions of input image
        target_h, target_w = image_tensor.shape[1], image_tensor.shape[2]
        heatmap_resized = cv2.resize(heatmap_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return np.maximum(heatmap_resized, 0.0)

    @staticmethod
    def overlay_heatmap(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blends a 2D heatmap over the original RGB image.

        Args:
            original_image: RGB image uint8 array of shape (H, W, 3).
            heatmap: 2D float array of shape (H, W) with values in [0.0, 1.0].
            alpha: Transparency factor for heatmap overlay (0.0 to 1.0).
            colormap: OpenCV colormap constant (default cv2.COLORMAP_JET).

        Returns:
            np.ndarray: Blended RGB image uint8 array of shape (H, W, 3).
        """
        # Ensure heatmap matches image spatial shape
        h, w = original_image.shape[:2]
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Scale heatmap to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))

        # Apply colormap (OpenCV outputs BGR)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        # Blend original image and colored heatmap
        blended = np.uint8(alpha * heatmap_colored_rgb + (1.0 - alpha) * original_image)
        return blended

    def explain_image(
        self,
        image_source: str | Path | bytes | io.BytesIO,
        class_idx: int | None = None,
        alpha: float = 0.45,
    ) -> dict[str, Any]:
        """
        Full end-to-end explainability pipeline: loads image, runs inference,
        computes Grad-CAM heatmap, overlays visualization, and converts to Base64 PNG.
        """
        # 1. Load original image as uint8 (224, 224, 3)
        raw_img = load_and_validate_image(image_source, target_size=(224, 224))

        # 2. Preprocess for MobileNetV2 [-1.0, 1.0]
        input_tensor = preprocess_mobilenet_v2(raw_img)
        input_batch = tf.expand_dims(input_tensor, axis=0)

        # 3. Model forward pass
        preds = self.model.predict(input_batch, verbose=0)[0]
        pred_class_id = int(np.argmax(preds)) if class_idx is None else class_idx
        confidence = float(preds[pred_class_id])

        # 4. Compute Heatmap and Overlay
        heatmap = self.compute_heatmap(input_batch, class_idx=pred_class_id)
        overlay = self.overlay_heatmap(raw_img, heatmap, alpha=alpha)

        # 5. Encode Overlay to Base64 PNG for web integration
        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        overlay_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "predicted_class_id": pred_class_id,
            "confidence": round(confidence, 4),
            "heatmap": heatmap,
            "overlay": overlay,
            "overlay_base64": overlay_base64,
        }
