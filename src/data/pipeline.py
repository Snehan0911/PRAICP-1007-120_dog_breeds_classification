"""
High-Performance TensorFlow tf.data Pipeline Module.
Project: Dog Breeds Classification (PRAICP-1007-120)

Builds memory-efficient, asynchronously prefetched tf.data.Dataset pipelines
with on-the-fly image decoding, resizing, model-specific normalization,
and training augmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .augmentation import build_data_augmentation
from .preprocessing import DEFAULT_IMAGE_SIZE, get_preprocessor

logger = logging.getLogger("DataPipeline")


def load_and_preprocess_image_tf(
    filepath: tf.Tensor,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    model_name: str = "mobilenet_v2",
) -> tf.Tensor:
    """
    TensorFlow graph-native image loader.
    Reads JPEG bytes, decodes to 3-channel RGB, resizes to target_size,
    and applies model-specific normalization.
    """
    img_bytes = tf.io.read_file(filepath)
    # decode_jpeg with channels=3 ensures 1-channel or 4-channel images convert to 3-channel RGB
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)

    preprocessor = get_preprocessor(model_name)
    img_normalized = preprocessor(img)
    return img_normalized


def build_tf_dataset(
    df_or_path: pd.DataFrame | str | Path,
    num_classes: int = 120,
    model_name: str = "mobilenet_v2",
    batch_size: int = 32,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    is_training: bool = False,
    shuffle_buffer: int = 1024,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset pipeline.

    Args:
        df_or_path: DataFrame containing 'filepath' and 'class_id' columns, or path to split CSV.
        num_classes: Total number of classes for one-hot encoding (default 120).
        model_name: Model architecture for normalization dispatch ('mobilenet_v2', 'custom_cnn').
        batch_size: Number of samples per mini-batch.
        target_size: (width, height) resolution.
        is_training: Whether this dataset is for training (enables shuffling).
        shuffle_buffer: Size of the shuffle buffer.
        augment: Whether to apply data augmentation (only if is_training=True).

    Returns:
        tf.data.Dataset yielding batches of (images, one_hot_labels).
    """
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    filepaths = df["filepath"].tolist()
    class_ids = df["class_id"].tolist()

    # Convert to TensorFlow constant tensors
    ds_files = tf.data.Dataset.from_tensor_slices(filepaths)
    ds_labels = tf.data.Dataset.from_tensor_slices(
        tf.one_hot(class_ids, depth=num_classes, dtype=tf.float32)
    )

    ds = tf.data.Dataset.zip((ds_files, ds_labels))

    if is_training:
        ds = ds.shuffle(
            buffer_size=min(len(df), shuffle_buffer), reshuffle_each_iteration=True
        )

    def _process_sample(
        path: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img = load_and_preprocess_image_tf(
            path, target_size=target_size, model_name=model_name
        )
        return img, label

    # Map with parallel workers
    ds = ds.map(_process_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data augmentation if requested (training only)
    if is_training and augment:
        aug_model = build_data_augmentation()

        def _augment_sample(
            img: tf.Tensor, label: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            return aug_model(img, training=True), label

        ds = ds.map(_augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
