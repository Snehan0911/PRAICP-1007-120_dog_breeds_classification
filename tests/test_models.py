"""
Unit tests for neural network architectures (Custom CNN).
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.custom_cnn import build_custom_cnn


def test_custom_cnn_output_shape():
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 120)


def test_custom_cnn_forward_pass():
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    # Create dummy batch of 2 synthetic images
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)

    assert preds.shape == (2, 120)
    # Check that softmax probabilities sum to 1.0 for each sample
    for row in preds:
        assert np.sum(row) == pytest.approx(1.0, abs=1e-4)
        assert np.all(row >= 0.0)
        assert np.all(row <= 1.0)


def test_custom_cnn_compiled_loss_and_optimizer():
    model = build_custom_cnn(learning_rate=0.001)
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert isinstance(model.loss, tf.keras.losses.CategoricalCrossentropy)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.001, rel=1e-5)
