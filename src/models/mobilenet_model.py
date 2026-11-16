"""
CanineVision AI — MobileNetV2 Transfer Learning Architecture (Model B).

Architectural Rationale:
    MobileNetV2 uses inverted residual blocks with linear bottlenecks and depthwise
    separable convolutions, drastically reducing FLOPs (300M vs 4B+ in ResNet) and
    parameter footprint (3.4M params).

    Uses a flat functional graph connecting input_tensor directly to base_model,
    allowing full layer access for Grad-CAM explainability and surgical layer-by-layer
    fine-tuning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet_v2(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 0.001,
    freeze_backbone: bool = True,
    fine_tune_at: int | None = None,
) -> tf.keras.Model:
    """
    Builds MobileNetV2 with custom classification head for 120 dog breeds.

    Args:
        input_shape: Dimensions of input image (224, 224, 3).
        num_classes: Number of breed targets (120).
        learning_rate: Initial learning rate for Adam optimizer.
        freeze_backbone: Whether to freeze backbone weights for feature extraction.
        fine_tune_at: Layer index from which to unfreeze backbone for fine-tuning.

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Base pretrained model without top classification layer connected directly to inputs
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet",
    )

    if freeze_backbone:
        base_model.trainable = False
    elif fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # Feature extraction forward pass
    x = base_model.output

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.3, name="head_dropout1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Classifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
