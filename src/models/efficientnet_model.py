"""
EfficientNetB0 Transfer Learning Architecture.
Project: Dog Breeds Classification (PRAICP-1007-120)

Constructs an ImageNet-pretrained EfficientNetB0 backbone with compound scaling
and a regularized classification head for 120 dog breeds.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("EfficientNetB0")


def build_efficientnet_b0(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    unfreeze_top_n_layers: int = 0,
    weights: str = "imagenet",
    label_smoothing: float = 0.05,
    name: str = "efficientnet_b0_dog_breeds",
) -> tf.keras.Model:
    """
    Constructs an ImageNet-pretrained EfficientNetB0 classifier.

    Key Architectural Principles:
      1. Compound Scaling: EfficientNet scales network depth, width, and input resolution
         uniformly using a compound coefficient, yielding high accuracy per FLOP.
      2. MBConv Blocks: Utilizes mobile inverted bottleneck convolutions with Squeeze-and-Excitation
         optimization to capture global feature interdependencies.
      3. Flat Functional Graph: Passing input_tensor=inputs ensures full compatiblity with
         Grad-CAM feature map access ('top_activation').

    Args:
        input_shape: Spatial tensor dimensions (224, 224, 3).
        num_classes: Target categories (120).
        learning_rate: Initial Adam learning rate.
        dropout_rate: Dropout fraction.
        freeze_backbone: Whether to freeze feature extraction layers initially.
        unfreeze_top_n_layers: Number of top layers to unfreeze for fine-tuning.
        weights: Pretrained weights identifier ('imagenet' or None).
        label_smoothing: Label smoothing regularization parameter.
        name: Model identifier string.

    Returns:
        tf.keras.Model: Compiled functional Keras model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Load EfficientNetB0 backbone directly into graph
    base_model = tf.keras.applications.EfficientNetB0(
        input_tensor=inputs,
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    if freeze_backbone:
        base_model.trainable = False
        if unfreeze_top_n_layers > 0:
            for layer in base_model.layers[-unfreeze_top_n_layers:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
            logger.info(
                f"Backbone frozen except top {unfreeze_top_n_layers} non-BatchNorm layers."
            )
        else:
            logger.info(
                "EfficientNetB0 backbone completely frozen for feature extraction."
            )
    else:
        base_model.trainable = True
        logger.info("EfficientNetB0 backbone fully trainable.")

    features = base_model.output

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(rate=dropout_rate, name="head_dropout_1")(x)

    x = layers.Dense(units=256, activation="relu", name="dense_fc1")(x)
    x = layers.Dropout(rate=dropout_rate * 0.75, name="head_dropout_2")(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        name="categorical_crossentropy",
    )
    metrics = [
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info(
        f"Built and compiled {name}: Total Params={model.count_params():,} "
        f"(Trainable: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}, "
        f"Non-Trainable: {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,})"
    )
    return model


if __name__ == "__main__":
    model = build_efficientnet_b0()
    print("\n" + "=" * 65)
    print("EFFICIENTNET_B0 MODEL SUMMARY:")
    print("=" * 65)
    model.summary()
    print("=" * 65)
