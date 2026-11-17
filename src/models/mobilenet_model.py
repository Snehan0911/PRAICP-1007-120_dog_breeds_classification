"""
MobileNetV2 Transfer Learning Architecture.
Project: Dog Breeds Classification (PRAICP-1007-120)

Constructs an ImageNet-pretrained MobileNetV2 backbone with a regularized
custom classification head for 120 dog breeds and fine-tuning support.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MobileNetV2")


def build_mobilenet_v2(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    unfreeze_top_n_layers: int = 0,
    weights: str = "imagenet",
    label_smoothing: float = 0.05,
    name: str = "mobilenet_v2_dog_breeds",
) -> tf.keras.Model:
    """
    Constructs an ImageNet-pretrained MobileNetV2 classifier.

    Key Engineering Architectural Choices:
      1. Single Flat Functional Graph: Passing input_tensor=inputs directly into
         MobileNetV2 connects all inverted residual blocks into one functional graph.
         This eliminates sub-model nesting and allows Grad-CAM to access intermediate
         feature maps ('out_relu') without disconnected graph errors.
      2. Inverted Residual Bottlenecks: Depthwise separable convolutions provide high
         representational power with only ~2.2M base parameters, making it ideal for CPU.
      3. Regularized Head: GlobalAveragePooling2D, Batch Normalization, and Dropout prevent
         overfitting to the 120 fine-grained breed classes.

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

    # Load MobileNetV2 backbone directly into graph
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    # Configure layer trainability
    if freeze_backbone:
        base_model.trainable = False
        if unfreeze_top_n_layers > 0:
            for layer in base_model.layers[-unfreeze_top_n_layers:]:
                # Keep BatchNormalization layers in inference mode to preserve moving stats
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
            logger.info(
                f"Backbone frozen except top {unfreeze_top_n_layers} non-BatchNorm layers."
            )
        else:
            logger.info(
                "MobileNetV2 backbone completely frozen for feature extraction."
            )
    else:
        base_model.trainable = True
        logger.info("MobileNetV2 backbone fully trainable (end-to-end training).")

    # Feature map extraction at deepest convolutional layer
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

    # Compile with Adam, Categorical Crossentropy, Accuracy, and Top-3 Accuracy
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
    model = build_mobilenet_v2()
    print("\n" + "=" * 65)
    print("MOBILENET_V2 TRANSFER LEARNING MODEL SUMMARY:")
    print("=" * 65)
    model.summary()
    print("=" * 65)
