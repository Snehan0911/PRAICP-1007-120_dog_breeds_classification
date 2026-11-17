"""
Model Factory Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Unified factory for instantiating, compiling, and configuring
Custom CNN, MobileNetV2, and EfficientNetB0 neural networks.
"""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf

from .custom_cnn import build_custom_cnn
from .efficientnet_model import build_efficientnet_b0
from .mobilenet_model import build_mobilenet_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ModelFactory")

SUPPORTED_MODELS = ["custom_cnn", "mobilenet_v2", "efficientnet_b0"]


def list_supported_models() -> list[str]:
    """Returns list of all supported architecture identifiers."""
    return list(SUPPORTED_MODELS)


def create_model(
    model_name: str,
    num_classes: int = 120,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    unfreeze_top_n_layers: int = 0,
    **kwargs: Any,
) -> tf.keras.Model:
    """
    Factory function to instantiate and compile any supported neural network model.

    Args:
        model_name: Architecture key ('custom_cnn', 'mobilenet_v2', 'efficientnet_b0').
        num_classes: Number of target categories (default 120).
        input_shape: Spatial tensor dimensions (default 224, 224, 3).
        learning_rate: Initial Adam learning rate.
        dropout_rate: Dropout fraction for regularized head.
        freeze_backbone: Whether to freeze backbone weights (for transfer learning).
        unfreeze_top_n_layers: Number of top layers to unfreeze.
        **kwargs: Extra hyperparameters forwarded to specific builders.

    Returns:
        tf.keras.Model: Compiled functional Keras model ready for training or inference.
    """
    clean_name = model_name.lower().replace("-", "_").strip()

    if clean_name in {"custom_cnn", "custom", "baseline"}:
        return build_custom_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    elif clean_name in {"mobilenet_v2", "mobilenet", "mobilenetv2"}:
        return build_mobilenet_v2(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            unfreeze_top_n_layers=unfreeze_top_n_layers,
            **kwargs,
        )
    elif clean_name in {"efficientnet_b0", "efficientnet", "efficientnetb0"}:
        return build_efficientnet_b0(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            unfreeze_top_n_layers=unfreeze_top_n_layers,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. "
            f"Supported models are: {SUPPORTED_MODELS}"
        )
