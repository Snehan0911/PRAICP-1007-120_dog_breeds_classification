"""
Image Preprocessing and Normalization Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Provides robust image loading, RGB enforcement, standardized resizing,
and architecture-specific normalization (Custom CNN, MobileNetV2, EfficientNetB0).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Canonical target resolution for deep CNN models
DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)


def load_and_validate_image(
    image_source: str | Path | bytes | io.BytesIO,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Loads an image from a filepath or byte stream, validates byte integrity,
    converts to 3-channel RGB, resizes to target_size, and returns a uint8 NumPy array.

    Args:
        image_source: Filepath or byte content of the image.
        target_size: Desired (width, height) in pixels.

    Returns:
        np.ndarray: Image array of shape (height, width, 3) and dtype uint8.

    Raises:
        ValueError: If the file is missing, corrupt, or unsupported.
    """
    try:
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
                return np.array(img_resized, dtype=np.uint8)
        else:
            # Handle byte content
            if isinstance(image_source, bytes):
                stream = io.BytesIO(image_source)
            else:
                stream = image_source
            with Image.open(stream) as img:
                img.verify()
            stream.seek(0)
            with Image.open(stream) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
                return np.array(img_resized, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Failed to load or validate image: {exc}") from exc


def preprocess_custom_cnn(image: np.ndarray | tf.Tensor) -> tf.Tensor:
    """
    Normalizes image array to [0.0, 1.0] range for Custom CNN baseline.
    Formula: x / 255.0
    """
    tensor = tf.cast(image, tf.float32)
    return tensor / 255.0


def preprocess_mobilenet_v2(image: np.ndarray | tf.Tensor) -> tf.Tensor:
    """
    Normalizes image array to [-1.0, 1.0] range expected by MobileNetV2.
    Formula: (x / 127.5) - 1.0
    """
    tensor = tf.cast(image, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(tensor)


def preprocess_efficientnet_b0(image: np.ndarray | tf.Tensor) -> tf.Tensor:
    """
    Preprocesses image array for EfficientNetB0.
    EfficientNetB0 architecture includes internal rescaling layers.
    """
    tensor = tf.cast(image, tf.float32)
    return tf.keras.applications.efficientnet.preprocess_input(tensor)


def get_preprocessor(model_name: str) -> Callable[[np.ndarray | tf.Tensor], tf.Tensor]:
    """
    Factory function returning the correct normalization callable for a given model architecture.

    Args:
        model_name: Model identifier ('custom_cnn', 'mobilenet_v2', 'efficientnet_b0').

    Returns:
        Callable normalizing tensors to the model's required numerical range.
    """
    normalized_name = model_name.lower().replace("-", "_").strip()
    if normalized_name in {"custom_cnn", "custom", "baseline"}:
        return preprocess_custom_cnn
    elif normalized_name in {"mobilenet_v2", "mobilenet", "mobilenetv2"}:
        return preprocess_mobilenet_v2
    elif normalized_name in {"efficientnet_b0", "efficientnet", "efficientnetb0"}:
        return preprocess_efficientnet_b0
    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. "
            f"Supported preprocessors: 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0'"
        )
