"""
Custom Baseline Convolutional Neural Network (CNN).
Project: Dog Breeds Classification (PRAICP-1007-120)

Implements a 4-block deep CNN from scratch with Batch Normalization,
Global Average Pooling, and Dropout regularization for 120 dog breed classes.
"""

from __future__ import annotations

import io
import logging

import tensorflow as tf
from tensorflow.keras import layers, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("CustomCNN")


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.4,
    label_smoothing: float = 0.05,
    name: str = "custom_cnn_baseline",
) -> tf.keras.Model:
    """
    Constructs and compiles a custom baseline Convolutional Neural Network.

    Architecture Overview:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [112x112x32]
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [56x56x64]
      - Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [28x28x128]
      - Block 4: Conv2D(256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [14x14x256]
      - Classifier: GlobalAveragePooling2D -> Dropout -> Dense(256) -> Dropout -> Dense(120, Softmax)

    Args:
        input_shape: Spatial tensor dimensions (height, width, channels).
        num_classes: Number of target categories (120).
        learning_rate: Initial Adam learning rate.
        dropout_rate: Dropout probability to combat overfitting.
        label_smoothing: Label smoothing regularization parameter.
        name: Model identifier string.

    Returns:
        tf.keras.Model: Compiled functional Keras model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # -------------------------------------------------------------------------
    # Block 1: Low-level features (edges, color boundaries, texture primitives)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv1_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # -------------------------------------------------------------------------
    # Block 2: Mid-level features (patterns, fur textures, simple geometric shapes)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv2_1",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # -------------------------------------------------------------------------
    # Block 3: High-level features (snouts, ear shapes, eye contours, paws)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv3_1",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # -------------------------------------------------------------------------
    # Block 4: Complex semantic representations (head structure, body morphology)
    # -------------------------------------------------------------------------
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="conv4_1",
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # -------------------------------------------------------------------------
    # Classification Head
    # -------------------------------------------------------------------------
    # GlobalAveragePooling2D replaces Flatten, drastically reducing parameters
    # from 14*14*256=50,176 down to 256, eliminating spatial overfitting.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
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

    # Compile model with standard multi-class classification losses and metrics
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
        f"Built and compiled {name}: Parameters={model.count_params():,} "
        f"(Trainable: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,})"
    )
    return model


def get_model_summary_string(model: tf.keras.Model) -> str:
    """Returns the string representation of model.summary()."""
    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    return stream.getvalue()


if __name__ == "__main__":
    model = build_custom_cnn()
    print("\n" + "=" * 65)
    print("CUSTOM CNN BASELINE MODEL SUMMARY:")
    print("=" * 65)
    model.summary()
    print("=" * 65)
