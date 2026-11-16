"""
CanineVision AI — Image Preprocessing Pipeline.

Responsibility:
    Converts raw image files into normalised numerical arrays
    that neural networks can consume.

    CRITICAL DESIGN DECISION:
    Different neural network architectures require DIFFERENT normalisation:

    - Custom CNN:
        Simple /255 normalisation → values in [0.0, 1.0]
        (Standard for scratch-trained CNNs)

    - MobileNetV2:
        Uses tf.keras.applications.mobilenet_v2.preprocess_input()
        → rescales pixels to [-1.0, 1.0]
        (This is what MobileNetV2 saw during ImageNet training)

    - EfficientNetB0:
        Has built-in rescaling in its first layer, so raw [0, 255]
        pixel values can be fed directly — but using the official
        preprocess_input function is safest.

    If you use the WRONG normalisation for a pretrained model, you feed
    it values it was never trained on. The model's learned features become
    meaningless and accuracy collapses.

    This module contains THREE separate preprocessing functions —
    one per model — to prevent this mistake.

Usage:
    from src.data.preprocessing import get_preprocess_fn
    preprocess_fn = get_preprocess_fn("mobilenet_v2")
    img_array = preprocess_fn(image_path, target_size=(224, 224))
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias
ImageArray = np.ndarray  # Shape: (H, W, 3), dtype: float32


def load_image(
    filepath: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> ImageArray:
    """
    Load an image from disk, convert to RGB, and resize.

    Args:
        filepath: Path to the image file.
        target_size: (width, height) to resize to.

    Returns:
        NumPy array of shape (H, W, 3), dtype uint8, values 0–255.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be opened as an image.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")

    with Image.open(filepath) as img:
        # Convert to RGB — handles grayscale (L), RGBA, and palette (P) modes
        img = img.convert("RGB")
        # LANCZOS is the highest quality downsampling filter
        img = img.resize(target_size, Image.LANCZOS)

    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Preprocessing functions — one per model architecture
# ---------------------------------------------------------------------------

def preprocess_custom_cnn(
    filepath: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> ImageArray:
    """
    Preprocessing for the Custom CNN.

    Normalises pixels from [0, 255] → [0.0, 1.0] by dividing by 255.
    This is the simplest and most common preprocessing for scratch-trained CNNs.

    Args:
        filepath: Path to image file.
        target_size: Resize target (width, height).

    Returns:
        float32 array of shape (H, W, 3), values in [0.0, 1.0].
    """
    img = load_image(filepath, target_size)
    return img.astype(np.float32) / 255.0


def preprocess_mobilenet_v2(
    filepath: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> ImageArray:
    """
    Preprocessing for MobileNetV2.

    Uses the official tf.keras preprocess_input function which rescales
    pixels from [0, 255] → [-1.0, 1.0].
    This matches the exact preprocessing used during MobileNetV2's ImageNet training.

    Args:
        filepath: Path to image file.
        target_size: Resize target (width, height).

    Returns:
        float32 array of shape (H, W, 3), values in [-1.0, 1.0].
    """
    import tensorflow as tf
    img = load_image(filepath, target_size)
    img_float = img.astype(np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_float)


def preprocess_efficientnet(
    filepath: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> ImageArray:
    """
    Preprocessing for EfficientNetB0.

    Uses the official tf.keras preprocess_input function.
    EfficientNet models include internal rescaling layers, but using
    preprocess_input ensures consistent behaviour.

    Args:
        filepath: Path to image file.
        target_size: Resize target (width, height).

    Returns:
        float32 array of shape (H, W, 3).
    """
    import tensorflow as tf
    img = load_image(filepath, target_size)
    img_float = img.astype(np.float32)
    return tf.keras.applications.efficientnet.preprocess_input(img_float)


# ---------------------------------------------------------------------------
# Factory function: get preprocessing by model name
# ---------------------------------------------------------------------------

PREPROCESS_REGISTRY: dict[str, callable] = {
    "normalize_01": preprocess_custom_cnn,
    "custom_cnn": preprocess_custom_cnn,
    "mobilenet_v2": preprocess_mobilenet_v2,
    "efficientnet": preprocess_efficientnet,
    "efficientnet_b0": preprocess_efficientnet,
}


def get_preprocess_fn(model_name: str) -> callable:
    """
    Return the correct preprocessing function for the given model name.

    Args:
        model_name: One of 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0'.

    Returns:
        A callable: preprocess_fn(filepath, target_size) → np.ndarray

    Raises:
        KeyError: If model_name is not recognised.
    """
    key = model_name.lower()
    if key not in PREPROCESS_REGISTRY:
        raise KeyError(
            f"Unknown preprocessing key: '{model_name}'. "
            f"Available: {list(PREPROCESS_REGISTRY.keys())}"
        )
    return PREPROCESS_REGISTRY[key]
