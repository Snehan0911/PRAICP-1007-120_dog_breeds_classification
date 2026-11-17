"""
Data Augmentation Pipeline.
Project: Dog Breeds Classification (PRAICP-1007-120)

Builds realistic, morphology-preserving image augmentation layers
applied exclusively during neural network training.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_data_augmentation(
    name: str = "dog_breeds_augmentation",
) -> tf.keras.Sequential:
    """
    Constructs a Keras Sequential model composed of realistic data augmentation layers.

    Transforms applied:
      1. Random horizontal flip (simulates dogs photographed from left vs. right profile).
      2. Subtle rotation (+/- 14.4 degrees) (simulates natural camera tilt).
      3. Modest zoom (+/- 8%) (simulates distance variation between subject and lens).
      4. Subtle translation (+/- 4%) (simulates slight off-center framing).
      5. Subtle contrast adjustment (+/- 8%) (simulates varying daylight / shadows).

    Crucial Engineering Constraints:
      - Never apply vertical flipping (dogs are never naturally inverted).
      - Keep distortion factors subtle to preserve fine-grained breed characteristics
        (e.g., ear erectness, snout length, leg proportions).
      - Augmentation layers automatically deactivate during model.evaluate() and model.predict().
    """
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
            layers.RandomRotation(0.04, fill_mode="reflect", name="aug_rotation"),
            layers.RandomZoom(0.08, fill_mode="reflect", name="aug_zoom"),
            layers.RandomTranslation(
                height_factor=0.04,
                width_factor=0.04,
                fill_mode="reflect",
                name="aug_translation",
            ),
            layers.RandomContrast(0.08, name="aug_contrast"),
        ],
        name=name,
    )
    return augmentation
