"""
CanineVision AI — Model Factory.

Provides a unified interface to instantiate, compile, and configure any
of the three supported architectures by name.
"""

import tensorflow as tf

from src.models.custom_cnn import build_custom_cnn
from src.models.efficientnet_model import build_efficientnet_b0
from src.models.mobilenet_model import build_mobilenet_v2
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 0.001,
    freeze_backbone: bool = True,
    fine_tune_at: int | None = None,
) -> tf.keras.Model:
    """
    Model factory function returning a compiled Keras model.

    Args:
        model_name: One of 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0'.
        input_shape: Image input shape (H, W, C).
        num_classes: Number of output classes (120).
        learning_rate: Initial optimizer learning rate.
        freeze_backbone: Whether to freeze backbone weights (transfer learning).
        fine_tune_at: Layer index from which to unfreeze for fine tuning.

    Returns:
        Compiled tf.keras.Model.

    Raises:
        ValueError: If model_name is not supported.
    """
    key = model_name.lower().replace("-", "_")

    logger.info("Initializing architecture: %s (num_classes=%d)", key, num_classes)

    if key in ("custom_cnn", "cnn"):
        return build_custom_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
        )
    elif key in ("mobilenet_v2", "mobilenet"):
        return build_mobilenet_v2(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone,
            fine_tune_at=fine_tune_at,
        )
    elif key in ("efficientnet_b0", "efficientnet"):
        return build_efficientnet_b0(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone,
            fine_tune_at=fine_tune_at,
        )
    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            "Supported options: 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0'."
        )
