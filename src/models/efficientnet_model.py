"""
CanineVision AI — EfficientNetB0 Transfer Learning Architecture (Model C).

Architectural Rationale:
    EfficientNet uses compound scaling to uniformly scale depth, width, and resolution
    using a fixed compound coefficient. EfficientNetB0 relies on MBConv (Mobile Inverted
    Bottleneck Convolution) blocks enriched with Squeeze-and-Excitation (SE) attention.

    Provides high representational capacity for fine-grained distinction among visually
    similar dog breeds.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_efficientnet_b0(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 0.001,
    freeze_backbone: bool = True,
    fine_tune_at: int | None = None,
) -> tf.keras.Model:
    """
    Builds EfficientNetB0 with custom classification head for 120 dog breeds.

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

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    if freeze_backbone:
        base_model.trainable = False
    elif fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.3, name="head_dropout1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.3, name="head_dropout2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Classifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
