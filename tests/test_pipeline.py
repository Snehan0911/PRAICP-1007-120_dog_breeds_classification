"""
Unit tests for tf.data dataset pipeline batching and tensor shapes.
"""

import tensorflow as tf

from src.data.dataset_loader import DatasetLoader


def test_dataset_loader_initialization():
    loader = DatasetLoader()
    assert loader.num_classes == 120
    assert len(loader.breed_names) == 120


def test_tf_data_pipeline_batch_shapes():
    loader = DatasetLoader()
    # Pull a tiny batch of 4 from validation set
    val_ds = loader.get_dataset(
        split="val", model_name="mobilenet_v2", batch_size=4, augment=False
    )

    for images, labels in val_ds.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert labels.shape == (4, 120)
        assert images.dtype == tf.float32
        assert labels.dtype == tf.float32
        # Check normalization bounds for MobileNetV2 [-1.0, 1.0]
        assert float(tf.reduce_min(images)) >= -1.05
        assert float(tf.reduce_max(images)) <= 1.05
        break
