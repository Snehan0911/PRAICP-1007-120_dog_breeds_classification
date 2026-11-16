"""
CanineVision AI — High-Performance tf.data Input Pipeline.

Why this matters:
    Loading 20,580 images into RAM as a raw numpy array requires over 12 GB RAM,
    which would crash immediately on this 7.7 GB RAM system.

    The tf.data API provides an asynchronous streaming pipeline:
    1. Reads image file paths from CSV/DataFrame on disk.
    2. Decodes JPEG bytes in parallel across CPU threads.
    3. Resizes to (224, 224).
    4. Applies architecture-specific normalisation:
       - 'mobilenet_v2': [-1.0, 1.0] scaling
       - 'efficientnet_b0': passthrough / internal scaling
       - 'custom_cnn': [0.0, 1.0] scaling
    5. Applies data augmentation ONLY when training=True.
    6. Batches samples (e.g., batch_size=32).
    7. Prefetches next batch in the background while the model is training.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.data.augmentation import build_augmentation_layer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipelineBuilder:
    """
    Constructs optimized tf.data.Dataset pipelines for training, validation, and testing.

    Args:
        model_name: Name of the architecture ('mobilenet_v2', 'custom_cnn', 'efficientnet_b0').
        image_size: Target (height, width) tuple. Default: (224, 224).
        batch_size: Number of images per batch. Default: 32.
        augment: Whether to apply data augmentation for training sets.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name.lower()
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.seed = seed
        self.augmentation_layer = build_augmentation_layer(seed=seed) if augment else None

    def _load_and_preprocess(self, filepath: tf.Tensor, label: tf.Tensor):
        """
        Reads, decodes, resizes, and normalises a single image using pure TensorFlow ops.
        """
        # Read file from filesystem
        img_bytes = tf.io.read_file(filepath)
        # Decode JPEG to RGB uint8 tensor
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        # Resize to model input shape
        img = tf.image.resize(img, self.image_size, method="bilinear")

        # Apply model-specific normalisation
        if "mobilenet" in self.model_name:
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        elif "efficientnet" in self.model_name:
            img = tf.keras.applications.efficientnet.preprocess_input(img)
        else:
            # Custom CNN: scale to [0, 1]
            img = tf.cast(img, tf.float32) / 255.0

        return img, label

    def create_dataset(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        shuffle_buffer: int = 1024,
    ) -> tf.data.Dataset:
        """
        Build a tf.data.Dataset from a DataFrame containing 'filepath' and 'class_index'.

        Args:
            df: DataFrame containing at least 'filepath' and 'class_index' columns.
            is_training: If True, shuffles and applies augmentation.
            shuffle_buffer: Number of samples to buffer for shuffling.

        Returns:
            Prefetched tf.data.Dataset yielding (batch_images, batch_labels).
        """
        filepaths = df["filepath"].values
        labels = df["class_index"].values

        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

        if is_training:
            dataset = dataset.shuffle(
                buffer_size=min(len(df), shuffle_buffer),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        # Parallel mapping across CPU cores
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(self.batch_size)

        # Apply augmentation on batches if training
        if is_training and self.augmentation_layer is not None:
            dataset = dataset.map(
                lambda x, y: (self.augmentation_layer(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Prefetch next batch asynchronously to avoid CPU bottleneck
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info(
            "Built %s dataset: %d samples, %d batches (batch_size=%d, model=%s)",
            "training" if is_training else "eval",
            len(df),
            len(dataset),
            self.batch_size,
            self.model_name,
        )
        return dataset

    def load_from_splits(
        self, splits_dir: str | Path
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Convenience method to load train, val, and test datasets from saved CSV splits.
        """
        splits_dir = Path(splits_dir)
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        train_ds = self.create_dataset(train_df, is_training=True)
        val_ds = self.create_dataset(val_df, is_training=False)
        test_ds = self.create_dataset(test_df, is_training=False)

        return train_ds, val_ds, test_ds
