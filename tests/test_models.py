"""
Unit tests for neural network architectures (Custom CNN, MobileNetV2, EfficientNetB0, Factory).
Project: Dog Breeds Classification (PRAICP-1007-120)
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.custom_cnn import build_custom_cnn
from src.models.efficientnet_model import build_efficientnet_b0
from src.models.mobilenet_model import build_mobilenet_v2
from src.models.model_factory import create_model, list_supported_models


def test_custom_cnn_output_shape():
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 120)


def test_custom_cnn_forward_pass():
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)

    assert preds.shape == (2, 120)
    for row in preds:
        assert np.sum(row) == pytest.approx(1.0, abs=1e-4)
        assert np.all(row >= 0.0)
        assert np.all(row <= 1.0)


def test_custom_cnn_compiled_loss_and_optimizer():
    model = build_custom_cnn(learning_rate=0.001)
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert isinstance(model.loss, tf.keras.losses.CategoricalCrossentropy)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.001, rel=1e-5)


def test_mobilenet_v2_output_shape():
    model = build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=120, weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 120)
    assert len(model.trainable_weights) > 0


def test_efficientnet_b0_output_shape():
    model = build_efficientnet_b0(input_shape=(224, 224, 3), num_classes=120, weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 120)


def test_model_factory_dispatch():
    supported = list_supported_models()
    assert "custom_cnn" in supported
    assert "mobilenet_v2" in supported
    assert "efficientnet_b0" in supported

    m1 = create_model("custom_cnn", num_classes=120)
    assert m1.output_shape == (None, 120)

    m2 = create_model("mobilenet_v2", num_classes=120, weights=None)
    assert m2.output_shape == (None, 120)

    with pytest.raises(ValueError, match="Unsupported model_name"):
        create_model("non_existent_architecture")
