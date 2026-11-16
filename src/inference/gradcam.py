"""
CanineVision AI — Explainable AI via Grad-CAM (Gradient-Weighted Class Activation Mapping).

How Grad-CAM Works:
1. Target Layer Selection:
   Identifies the final convolutional layer in the model (where spatial features have highest
   semantic richness before pooling).

2. Gradient Extraction:
   Using tf.GradientTape, calculates the gradient of the predicted class score with respect
   to the target convolutional layer's activation maps.

3. Channel Importance Pooling:
   Applies global average pooling across spatial dimensions of the gradients to calculate an
   importance weight alpha_k for each feature map channel.

4. Weighted Combination & ReLU:
   Computes a weighted sum of the feature maps, followed by ReLU to capture only features
   that have a positive influence on the target class prediction.

5. Superimposition:
   Resizes the 2D heatmap to match original image dimensions, normalises to [0, 255], applies
   a JET colormap via OpenCV, and blends it onto the original photograph.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAMExplainer:
    """
    Generates explainability heatmaps for image classification decisions.

    Args:
        model: Trained tf.keras.Model.
        target_layer_name: Name of final conv layer. If None, auto-detected.
    """

    def __init__(self, model: tf.keras.Model, target_layer_name: str | None = None) -> None:
        self.model = model
        self.target_layer_name = target_layer_name or self._find_target_conv_layer()
        logger.info("GradCAM initialized with target conv layer: %s", self.target_layer_name)

    def _find_target_conv_layer(self) -> str:
        """Auto-discovers the deepest convolutional layer in the model graph."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return layer.name

        # Fallback names
        for candidate in ["Conv_1", "top_conv", "conv4_1"]:
            try:
                self.model.get_layer(candidate)
                return candidate
            except ValueError:
                continue

        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    def compute_heatmap(
        self,
        img_array: np.ndarray,
        class_index: int | None = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Calculates normalized Grad-CAM heatmap array for a single input image.

        Args:
            img_array: Preprocessed image tensor, shape (1, H, W, C).
            class_index: Target class to explain. If None, uses top predicted class.
            eps: Small epsilon to prevent division by zero during normalization.

        Returns:
            2D numpy array heatmap, values scaled to [0.0, 1.0].
        """
        conv_layer = self.model.get_layer(self.target_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            class_score = predictions[:, class_index]

        # Gradients of target class score w.r.t. conv layer output
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pooling of gradients gives channel importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to retain only positive influences on class score
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    def generate_overlay(
        self,
        original_image_path: str | Path,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays the Grad-CAM heatmap on top of the original RGB image.

        Args:
            original_image_path: Path to original photo.
            heatmap: 2D heatmap array from compute_heatmap().
            alpha: Transparency factor for heatmap overlay (0.0 to 1.0).
            colormap: OpenCV colormap constant.

        Returns:
            RGB numpy array representing superimposed explainability image.
        """
        with Image.open(original_image_path) as img:
            rgb_img = np.array(img.convert("RGB"))

        h, w = rgb_img.shape[:2]

        # Resize heatmap to match original image dimensions
        resized_heatmap = cv2.resize(heatmap, (w, h))
        uint8_heatmap = np.uint8(255 * resized_heatmap)

        # Apply false color heatmap
        colored_heatmap = cv2.applyColorMap(uint8_heatmap, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose
        overlay = np.uint8(alpha * colored_heatmap + (1 - alpha) * rgb_img)
        return overlay
