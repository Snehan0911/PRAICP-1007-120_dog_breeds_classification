"""
Tests — Model Architecture and Model Factory.
"""

import pytest


def test_custom_cnn_output_shape():
    """Verify Custom CNN outputs correct shape (None, 120)."""
    from src.models.custom_cnn import build_custom_cnn

    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=120)
    assert model.output_shape == (None, 120)
    assert model.count_params() > 0


def test_mobilenet_v2_output_shape():
    """Verify MobileNetV2 outputs correct shape (None, 120)."""
    from src.models.mobilenet_model import build_mobilenet_v2

    model = build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=120)
    assert model.output_shape == (None, 120)


def test_model_factory_dispatch():
    """Verify get_model instantiates all registered models."""
    from src.models.model_factory import get_model

    for name in ["custom_cnn", "mobilenet_v2", "efficientnet_b0"]:
        m = get_model(name, num_classes=120)
        assert m.output_shape == (None, 120)

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        get_model("invalid_model_type")
