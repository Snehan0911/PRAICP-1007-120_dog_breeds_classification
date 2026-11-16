"""
CanineVision AI — Data Augmentation Pipeline.

Responsibility:
    Builds TensorFlow data augmentation layers that are applied
    ONLY to training images — never to validation or test images.

    WHY AUGMENTATION?
    With ~148–252 images per breed, the model risks memorising the training
    images (overfitting) rather than learning generalisable features.
    Augmentation artificially expands the effective dataset size by creating
    realistic variants of each training image.

    WHY ONLY ON TRAINING DATA?
    Validation and test sets represent real-world data as a user would submit it.
    Augmenting them would change what we're measuring — we'd no longer know
    if good performance is due to the model or due to favourable augmentation.

    WHY THESE SPECIFIC AUGMENTATIONS?
    Chosen to reflect realistic variation in dog photographs:
    - Horizontal flip: Dogs can face left or right
    - Rotation: Camera may be slightly tilted
    - Zoom: Dogs can be closer or further
    - Brightness: Different lighting conditions
    NOT used: vertical flip (dogs are not photographed upside-down),
              extreme colour shifts (would confuse breed-specific colouring).

Usage:
    from src.data.augmentation import build_augmentation_layer
    aug_layer = build_augmentation_layer()
    # Use inside a tf.keras.Sequential model or as a layer in tf.data pipeline
"""

import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_augmentation_layer(
    horizontal_flip: bool = True,
    rotation_factor: float = 0.15,
    zoom_factor: float = 0.15,
    brightness_factor: float = 0.15,
    contrast_factor: float = 0.10,
    translation_factor: float = 0.10,
    seed: int = 42,
) -> tf.keras.Sequential:
    """
    Build a Keras Sequential model containing data augmentation layers.

    These layers are only active during training (training=True).
    During inference (training=False), they pass input through unchanged.
    This means augmentation is built into the model — no separate
    preprocessing pipeline needed for train/val/test distinction.

    Args:
        horizontal_flip:    Whether to randomly flip images horizontally.
        rotation_factor:    Max rotation fraction of 2π (0.15 ≈ ±54°).
                            Clipped to reasonable ±15° for dogs.
        zoom_factor:        Max zoom fraction (0.15 = ±15% zoom in/out).
        brightness_factor:  Max brightness adjustment (0.15 = ±15%).
        contrast_factor:    Max contrast adjustment factor.
        translation_factor: Max horizontal/vertical translation fraction.
        seed:               Random seed for reproducible augmentation.

    Returns:
        A tf.keras.Sequential layer that applies augmentation.
    """
    layers = []

    if horizontal_flip:
        layers.append(
            tf.keras.layers.RandomFlip("horizontal", seed=seed)
        )

    if rotation_factor > 0:
        # Clamp to sensible range for dog photos (±15°)
        actual_factor = min(rotation_factor, 0.15)
        layers.append(
            tf.keras.layers.RandomRotation(factor=actual_factor, seed=seed)
        )

    if zoom_factor > 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_factor, zoom_factor),
                width_factor=(-zoom_factor, zoom_factor),
                seed=seed,
            )
        )

    if translation_factor > 0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=translation_factor,
                width_factor=translation_factor,
                seed=seed,
            )
        )

    if brightness_factor > 0:
        layers.append(
            tf.keras.layers.RandomBrightness(
                factor=brightness_factor, seed=seed
            )
        )

    if contrast_factor > 0:
        layers.append(
            tf.keras.layers.RandomContrast(
                factor=contrast_factor, seed=seed
            )
        )

    logger.info(
        "Built augmentation pipeline with %d layer(s): %s",
        len(layers),
        [type(layer).__name__ for layer in layers],
    )

    return tf.keras.Sequential(layers, name="augmentation")
