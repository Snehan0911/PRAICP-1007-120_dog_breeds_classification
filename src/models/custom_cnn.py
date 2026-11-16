"""
CanineVision AI — Custom CNN Architecture from Scratch (Model A).

Architectural Rationale:
    A custom convolutional neural network designed specifically for 120-class
    dog breed classification without using pre-trained weights.

Key Architectural Components:
    1. Convolutional Blocks:
       Four sequential conv blocks (32 → 64 → 128 → 256 filters) with 3x3 kernels.
       Early layers learn low-level primitives (edges, textures, fur patterns).
       Deeper layers capture composite semantics (ear shape, snout length, muzzle structure).

    2. Batch Normalization:
       Inserted immediately after convolutions to standardize intermediate activations,
       mitigate internal covariate shift, and smooth the loss landscape.

    3. Spatial Downsampling:
       MaxPooling2D with 2x2 window halves spatial dimensions while preserving dominant activations.

    4. Regularization:
       Spatial dropout and dense dropout (0.25 to 0.5) prevent co-adaptation of neurons.

    5. Classification Head:
       GlobalAveragePooling2D reduces the final feature maps to a compact 1D vector (avoids parameter
       explosion associated with Flatten), followed by Dense(512), BatchNorm, Dropout(0.5), and
       Dense(120, activation='softmax').
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 120,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs and compiles the Custom CNN architecture.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Number of target categories (120).
        learning_rate: Optimizer initial learning rate.

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="act1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="act1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="act2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="act2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="act3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="act3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="act4_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(0.25, name="drop4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="act_dense")(x)
    x = layers.Dropout(0.5, name="drop_dense")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CustomCNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
