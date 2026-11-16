"""
Tests — tf.data streaming input pipeline.
"""

import tensorflow as tf


def test_pipeline_batch_output_shapes(small_dataset_df):
    """Verify that the tf.data pipeline produces correct batch shapes."""
    from src.data.pipeline import DataPipelineBuilder

    builder = DataPipelineBuilder(
        model_name="mobilenet_v2",
        image_size=(224, 224),
        batch_size=4,
        augment=False,
    )
    dataset = builder.create_dataset(small_dataset_df, is_training=False)

    for images, labels in dataset.take(1):
        assert images.shape == (4, 224, 224, 3)
        assert labels.shape == (4,)
        assert images.dtype == tf.float32
